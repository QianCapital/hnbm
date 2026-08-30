import numpy as np
import copy
import warnings
from numbers import Integral, Real

from tqdm import tqdm
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score, mean_squared_error, log_loss, r2_score
from sklearn.utils.metaestimators import available_if
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import (
    check_array,
    check_consistent_length,
    check_is_fitted,
    column_or_1d,
    has_fit_parameter,
)
from joblib import Parallel, delayed

from .losses import Logistic, MeanSquaredError, PseudoHuber, Quantile


def _classification_mode(estimator):
    """Return whether classifier-only methods should be available."""
    return estimator.mode == "classification"


def _validate_X(X):
    """Validate a dense feature matrix."""
    return check_array(X, dtype=float, ensure_2d=True, accept_sparse=False)


def _validate_X_y(X, y):
    """Validate feature matrix and label vector shapes."""
    X = _validate_X(X)
    y = column_or_1d(y, warn=True)
    y = check_array(y, ensure_2d=False, dtype=None)
    check_consistent_length(X, y)
    return X, y


def _extract_feature_names(X):
    """Return dataframe column labels, or None when they are not all strings.

    Mirrors scikit-learn, which only tracks feature names for containers whose
    columns are entirely strings. ``check_array`` discards this information, so
    it has to be read before any conversion to an array.
    """
    columns = getattr(X, "columns", None)
    if columns is None:
        return None
    names = np.asarray(columns, dtype=object)
    if names.ndim != 1 or not all(isinstance(name, str) for name in names):
        return None
    return names


def _feature_names_mismatch_message(fitted_names, given_names, limit=5):
    """Describe how two sets of feature names differ."""
    def _summarize(label, names):
        shown = [f"- {name}" for name in sorted(names)[:limit]]
        if len(names) > limit:
            shown.append(f"- ... ({len(names) - limit} more)")
        return "\n".join([label, *shown])

    fitted_set, given_set = set(fitted_names), set(given_names)
    sections = []
    unseen = given_set - fitted_set
    if unseen:
        sections.append(_summarize("Feature names unseen at fit time:", unseen))
    missing = fitted_set - given_set
    if missing:
        sections.append(
            _summarize("Feature names seen at fit time, yet now missing:", missing)
        )
    if not sections:
        sections.append(
            "Feature names must be in the same order as they were in fit."
        )
    return "\n".join(sections)


class HNBM(BaseEstimator):
    """
    Heterogeneous Newton Boosting Machine.

    A gradient boosting framework that stochastically selects base learners
    from a heterogeneous pool at each iteration. Subclass HNBM and configure
    ``base_learners_`` and ``probabilities_`` before calling ``fit``.

    Parameters
    ----------
    num_iterations : int, default=100
        Number of boosting iterations.
    learning_rate : float, default=0.1
        Shrinkage applied to each learner's contribution.
    mode : {'classification', 'regression'}, default='classification'
        Training objective.
    random_state : int or None, default=None
        Random seed for base learner selection.
    verbose : bool, default=True
        Whether to show a progress bar during training.
    selection_strategy : {'random', 'greedy'}, default='random'
        Sample one candidate or fit all candidates and select the lowest-loss
        update at each boosting round.
    line_search : bool, default=False
        Whether to choose a contribution weight for every fitted learner.
    early_stopping_rounds : int or None, default=None
        Validation rounds without sufficient improvement before stopping.
        Stopping also truncates the ensemble back to ``best_iteration_``.
    min_delta : float, default=0.0
        Minimum validation-loss improvement that resets early-stopping patience.
    subsample : float, default=1.0
        Fraction of training observations used to fit each base learner.

    Attributes
    ----------
    ensemble_ : list
        Fitted base learners after training.
    n_iter_ : int
        Number of completed boosting iterations.
    base_learners_ : list
        Candidate base learners (must be set before ``fit``).
    probabilities_ : list
        Selection probabilities for each base learner (must be set before ``fit``).
    base_score_ : float
        Optimized constant prediction used before the first boosting round.
    learner_weights_ : list of float
        Contribution weight stored for every fitted learner.
    history_ : dict
        Per-round training loss, validation loss, and selected learner index.
    best_iteration_ : int
        Best validation iteration, or the final iteration without validation.
        The ensemble is truncated to this iteration only when
        ``early_stopping_rounds`` triggers, so a run given an ``eval_set``
        alone keeps every learner and predicts with all of them.
    """

    def __init__(
        self,
        num_iterations=100,
        learning_rate=0.1,
        mode="classification",
        random_state=None,
        verbose=True,
        selection_strategy="random",
        line_search=False,
        early_stopping_rounds=None,
        min_delta=0.0,
        subsample=1.0,
        objective="auto",
        objective_parameter=None,
    ):
        self._validate_hnbm_params(
            num_iterations=num_iterations,
            learning_rate=learning_rate,
            mode=mode,
            random_state=random_state,
            selection_strategy=selection_strategy,
            line_search=line_search,
            early_stopping_rounds=early_stopping_rounds,
            min_delta=min_delta,
            subsample=subsample,
            objective=objective,
            objective_parameter=objective_parameter,
        )

        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.mode = mode
        self.random_state = random_state
        self.verbose = verbose
        self.selection_strategy = selection_strategy
        self.line_search = line_search
        self.early_stopping_rounds = early_stopping_rounds
        self.min_delta = min_delta
        self.subsample = subsample
        self.objective = objective
        self.objective_parameter = objective_parameter

    @property
    def base_learners_(self):
        """Backward-compatible alias for the unfitted learner pool."""
        return self.__dict__.get("_base_learners", ())

    @base_learners_.setter
    def base_learners_(self, value):
        self._base_learners = value

    @property
    def probabilities_(self):
        """Backward-compatible alias for learner-selection probabilities."""
        return self.__dict__.get("_probabilities", ())

    @probabilities_.setter
    def probabilities_(self, value):
        self._probabilities = value

    def __sklearn_is_fitted__(self):
        return bool(getattr(self, "ensemble_", []))

    def __sklearn_tags__(self):
        parent_tags = getattr(super(), "__sklearn_tags__", None)
        if parent_tags is None:
            return self._more_tags()
        tags = parent_tags()
        tags.estimator_type = (
            "classifier" if self.mode == "classification" else "regressor"
        )
        return tags

    def _more_tags(self):
        tags = super()._more_tags()
        tags["binary_only"] = self.mode == "classification"
        return tags

    @staticmethod
    def _validate_hnbm_params(
        *, num_iterations, learning_rate, mode, random_state,
        selection_strategy="random", line_search=False,
        early_stopping_rounds=None, min_delta=0.0,
        subsample=1.0, objective="auto", objective_parameter=None
    ):
        if (
            isinstance(num_iterations, (bool, np.bool_))
            or not isinstance(num_iterations, Integral)
            or num_iterations < 1
        ):
            raise ValueError(
                f"num_iterations must be an integer >= 1, got {num_iterations}."
            )
        if (
            isinstance(learning_rate, (bool, np.bool_))
            or not isinstance(learning_rate, Real)
            or not np.isfinite(learning_rate)
            or learning_rate <= 0
        ):
            raise ValueError(
                f"learning_rate must be a finite number > 0, got {learning_rate}."
            )
        if mode not in ("classification", "regression"):
            raise ValueError("Invalid mode: specify 'classification' or 'regression'.")
        if random_state is not None and (
            isinstance(random_state, (bool, np.bool_))
            or not isinstance(random_state, Integral)
            or random_state < 0
        ):
            raise ValueError(
                "random_state must be a non-negative integer or None, "
                f"got {random_state}."
            )
        if selection_strategy not in ("random", "greedy"):
            raise ValueError("selection_strategy must be 'random' or 'greedy'.")
        if not isinstance(line_search, (bool, np.bool_)):
            raise ValueError("line_search must be a boolean.")
        if early_stopping_rounds is not None and (
            isinstance(early_stopping_rounds, (bool, np.bool_))
            or not isinstance(early_stopping_rounds, Integral)
            or early_stopping_rounds < 1
        ):
            raise ValueError("early_stopping_rounds must be an integer >= 1 or None.")
        if (
            isinstance(min_delta, (bool, np.bool_))
            or not isinstance(min_delta, Real)
            or not np.isfinite(min_delta)
            or min_delta < 0
        ):
            raise ValueError("min_delta must be a finite number >= 0.")
        if (
            isinstance(subsample, (bool, np.bool_))
            or not isinstance(subsample, Real)
            or not np.isfinite(subsample)
            or not 0 < subsample <= 1
        ):
            raise ValueError("subsample must be a finite number in (0, 1].")
        allowed_objectives = (
            ("auto", "log_loss")
            if mode == "classification"
            else ("auto", "squared_error", "pseudo_huber", "quantile")
        )
        if objective not in allowed_objectives:
            raise ValueError(
                f"objective must be one of {allowed_objectives} for {mode}."
            )
        if objective_parameter is not None and (
            isinstance(objective_parameter, (bool, np.bool_))
            or not isinstance(objective_parameter, Real)
            or not np.isfinite(objective_parameter)
        ):
            raise ValueError("objective_parameter must be finite or None.")
        if objective == "pseudo_huber" and (
            objective_parameter is not None and objective_parameter <= 0
        ):
            raise ValueError("Pseudo-Huber delta must be greater than 0.")
        if objective == "quantile" and (
            objective_parameter is not None
            and not 0 < objective_parameter < 1
        ):
            raise ValueError("Quantile must be strictly between 0 and 1.")

    def _check_fitted(self):
        check_is_fitted(self, "ensemble_")
        if not self.ensemble_:
            raise NotFittedError(
                "This HNBM instance is not fitted yet. Call 'fit' with "
                "appropriate arguments."
            )

    def _validate_learner_pool(self):
        """Validate learner selection and weighted-fit compatibility."""
        if not self.base_learners_:
            raise ValueError(
                "No base learners configured. Subclass HNBM and set base_learners_ "
                "and probabilities_ before calling fit."
            )
        if len(self.base_learners_) != len(self.probabilities_):
            raise ValueError(
                "base_learners_ and probabilities_ must have the same length."
            )

        probabilities = np.asarray(self.probabilities_, dtype=float)
        if probabilities.ndim != 1 or not np.all(np.isfinite(probabilities)):
            raise ValueError("probabilities_ must contain finite numeric values.")
        if np.any(probabilities < 0):
            raise ValueError("probabilities_ must be non-negative.")
        probability_sum = probabilities.sum()
        if not np.isclose(probability_sum, 1.0):
            raise ValueError("probabilities_ must sum to 1.")

        for learner in self.base_learners_:
            if not hasattr(learner, "fit") or not hasattr(learner, "predict"):
                raise TypeError("Each base learner must implement fit and predict.")
            clone(learner)
            if not has_fit_parameter(learner, "sample_weight"):
                raise TypeError(
                    f"Base learner {type(learner).__name__} must accept sample_weight "
                    "in fit()."
                )

        # Generator.choice applies a stricter sum check than np.isclose. Use the
        # validated, normalized values so probabilities accepted here cannot fail
        # later solely because of floating-point summation error.
        return probabilities / probability_sum

    def set_params(self, **params):
        self._validate_hnbm_params(
            num_iterations=params.get("num_iterations", self.num_iterations),
            learning_rate=params.get("learning_rate", self.learning_rate),
            mode=params.get("mode", self.mode),
            random_state=params.get("random_state", self.random_state),
            selection_strategy=params.get(
                "selection_strategy", getattr(self, "selection_strategy", "random")
            ),
            line_search=params.get(
                "line_search", getattr(self, "line_search", False)
            ),
            early_stopping_rounds=params.get(
                "early_stopping_rounds", getattr(self, "early_stopping_rounds", None)
            ),
            min_delta=params.get("min_delta", getattr(self, "min_delta", 0.0)),
            subsample=params.get("subsample", getattr(self, "subsample", 1.0)),
            objective=params.get("objective", getattr(self, "objective", "auto")),
            objective_parameter=params.get(
                "objective_parameter", getattr(self, "objective_parameter", None)
            ),
        )
        result = super().set_params(**params)
        if params:
            for attribute in (
                "ensemble_",
                "classes_",
                "n_features_in_",
                "feature_names_in_",
                "n_iter_",
                "learner_weights_",
                "base_score_",
                "history_",
                "best_iteration_",
            ):
                self.__dict__.pop(attribute, None)
        return result

    def _check_feature_names(self, X):
        """Verify that ``X`` carries the same feature names seen during fit."""
        fitted_names = getattr(self, "feature_names_in_", None)
        given_names = _extract_feature_names(X)
        if fitted_names is None and given_names is None:
            return
        if fitted_names is None:
            warnings.warn(
                f"X has feature names, but {type(self).__name__} was fitted "
                "without feature names.",
                UserWarning,
                stacklevel=3,
            )
            return
        if given_names is None:
            warnings.warn(
                f"X does not have valid feature names, but "
                f"{type(self).__name__} was fitted with feature names.",
                UserWarning,
                stacklevel=3,
            )
            return
        if len(fitted_names) != len(given_names) or np.any(
            fitted_names != given_names
        ):
            raise ValueError(
                "The feature names should match those that were passed during "
                "fit.\n"
                + _feature_names_mismatch_message(fitted_names, given_names)
            )

    @staticmethod
    def _validate_sample_weight(sample_weight, n_samples):
        if sample_weight is None:
            return np.ones(n_samples, dtype=float)
        weights = np.asarray(sample_weight, dtype=float)
        if weights.ndim != 1 or weights.shape[0] != n_samples:
            raise ValueError("sample_weight must have one value per sample.")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("sample_weight must be finite and non-negative.")
        if weights.sum() <= 0:
            raise ValueError("sample_weight must have a positive total weight.")
        return weights

    def _loss_value(self, y, raw_prediction, sample_weight=None):
        parameter = self._resolved_objective_parameter()
        if self.objective_ == "pseudo_huber":
            values = PseudoHuber.compute_loss(y, raw_prediction, parameter)
        elif self.objective_ == "quantile":
            values = Quantile.compute_loss(y, raw_prediction, parameter)
        else:
            values = self.loss_.compute_loss(y, raw_prediction)
        return float(np.average(values, weights=sample_weight))

    @property
    def objective_(self):
        if self.objective == "auto":
            return "log_loss" if self.mode == "classification" else "squared_error"
        return self.objective

    def _resolved_objective_parameter(self):
        if self.objective_ == "quantile":
            return 0.5 if self.objective_parameter is None else self.objective_parameter
        if self.objective_ == "pseudo_huber":
            return 1.0 if self.objective_parameter is None else self.objective_parameter
        return self.objective_parameter

    def _compute_derivatives(self, y, raw_prediction):
        parameter = self._resolved_objective_parameter()
        if self.objective_ == "pseudo_huber":
            return PseudoHuber.compute_derivatives(y, raw_prediction, parameter)
        if self.objective_ == "quantile":
            return Quantile.compute_derivatives(y, raw_prediction, parameter)
        return self.loss_.compute_derivatives(y, raw_prediction)

    def _initial_prediction(self, y, sample_weight):
        if self.mode == "regression":
            if self.objective_ == "quantile":
                order = np.argsort(y)
                ordered_y = y[order]
                cumulative = np.cumsum(sample_weight[order])
                threshold = self._resolved_objective_parameter() * cumulative[-1]
                return float(ordered_y[np.searchsorted(cumulative, threshold)])
            return float(np.average(y, weights=sample_weight))
        positive = float(np.average(y > 0, weights=sample_weight))
        eps = np.finfo(float).eps
        positive = np.clip(positive, eps, 1.0 - eps)
        return float(np.log(positive / (1.0 - positive)))

    def _fit_candidate(
        self, prototype, X, target, hessian, iteration, index, sampling_entropy
    ):
        learner = clone(prototype)
        if hasattr(learner, "random_state") and self.random_state is not None:
            seed = np.random.SeedSequence(
                [int(self.random_state), int(iteration), int(index)]
            ).generate_state(1)[0]
            learner.set_params(random_state=int(seed))
        eligible_rows = np.flatnonzero(hessian > 0)
        if eligible_rows.size == 0:
            raise ValueError("A base learner requires at least one positive weight.")
        if self.subsample < 1.0:
            count = max(1, int(np.ceil(self.subsample * eligible_rows.size)))
            sampling_rng = np.random.default_rng(
                np.random.SeedSequence(
                    [int(sampling_entropy), int(iteration), 991]
                )
            )
            rows = sampling_rng.choice(eligible_rows, size=count, replace=False)
        elif eligible_rows.size != X.shape[0]:
            rows = eligible_rows
        else:
            rows = slice(None)
        learner.fit(X[rows], target[rows], sample_weight=hessian[rows])
        prediction = np.asarray(learner.predict(X), dtype=float)
        if prediction.shape != target.shape:
            raise ValueError(
                f"Base learner {type(learner).__name__} returned shape "
                f"{prediction.shape}; expected {target.shape}."
            )
        if not np.all(np.isfinite(prediction)):
            raise FloatingPointError("Base learner predictions became non-finite.")
        return learner, prediction

    def _choose_step(self, y, z, prediction, sample_weight):
        if not self.line_search:
            return float(self.learning_rate)
        steps = self.learning_rate * np.array([0.25, 0.5, 1.0, 1.5, 2.0])
        losses = [
            self._loss_value(y, z + step * prediction, sample_weight)
            for step in steps
        ]
        return float(steps[int(np.argmin(losses))])

    def fit(
        self, X, y, sample_weight=None, eval_set=None,
        eval_metric=None, callbacks=None, candidate_n_jobs=1,
    ):
        """
        Train the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        y : array-like of shape (n_samples,)
            Target values.
        sample_weight : array-like of shape (n_samples,), default=None
            Non-negative observation weights.
        eval_set : tuple (X_validation, y_validation), default=None
            Optional validation pair used for history and early stopping.
        eval_metric : callable, default=None
            Optional ``metric(y, raw_prediction) -> float`` recorded per round.
        callbacks : iterable of callable, default=None
            Functions called with a per-round state dictionary. Returning a
            truthy value requests an orderly stop after the current round.
        candidate_n_jobs : int, default=1
            Threads used to fit greedy candidates. Has no effect for random
            selection. ``-1`` uses all available logical CPUs.

        Returns
        -------
        self
        """
        probabilities = self._validate_learner_pool()
        if eval_metric is not None and not callable(eval_metric):
            raise TypeError("eval_metric must be callable or None.")
        callbacks = () if callbacks is None else tuple(callbacks)
        if any(not callable(callback) for callback in callbacks):
            raise TypeError("Every callback must be callable.")
        if (
            isinstance(candidate_n_jobs, (bool, np.bool_))
            or not isinstance(candidate_n_jobs, Integral)
            or candidate_n_jobs == 0
        ):
            raise ValueError("candidate_n_jobs must be a nonzero integer.")

        feature_names = _extract_feature_names(X)
        X, y = _validate_X_y(X, y)
        sample_weight = self._validate_sample_weight(sample_weight, X.shape[0])
        if self.mode == "classification":
            check_classification_targets(y)
            classes = np.unique(y)
            if classes.shape[0] != 2:
                raise ValueError(
                    "Binary classification requires exactly two classes; got "
                    f"{classes.shape[0]} class(es)."
                )
            y = np.where(y == classes[0], -1.0, 1.0)
        else:
            y = check_array(y, ensure_2d=False, dtype=float)

        if eval_set is not None:
            if not isinstance(eval_set, (tuple, list)) or len(eval_set) != 2:
                raise ValueError("eval_set must be an (X, y) pair.")
            eval_feature_names = _extract_feature_names(eval_set[0])
            X_eval, y_eval_original = _validate_X_y(*eval_set)
            if X_eval.shape[1] != X.shape[1]:
                raise ValueError("Training and validation data must have equal features.")
            if (
                feature_names is not None
                and eval_feature_names is not None
                and (
                    len(feature_names) != len(eval_feature_names)
                    or np.any(feature_names != eval_feature_names)
                )
            ):
                raise ValueError(
                    "eval_set feature names should match the training data.\n"
                    + _feature_names_mismatch_message(
                        feature_names, eval_feature_names
                    )
                )
            if self.mode == "classification":
                unknown = np.setdiff1d(np.unique(y_eval_original), classes)
                if unknown.size:
                    raise ValueError("eval_set contains an unknown class label.")
                y_eval = np.where(y_eval_original == classes[0], -1.0, 1.0)
            else:
                y_eval = check_array(y_eval_original, ensure_2d=False, dtype=float)
        else:
            X_eval = y_eval = None

        rng = np.random.default_rng(self.random_state)
        sampling_entropy = (
            int(self.random_state)
            if self.random_state is not None
            else int(np.random.SeedSequence().entropy)
        )
        base_score = self._initial_prediction(y, sample_weight)
        z = np.full(X.shape[0], base_score)
        z_eval = None if X_eval is None else np.full(X_eval.shape[0], base_score)
        fitted_learners = []
        learner_weights = []
        history = {
            "training_loss": [],
            "validation_loss": [],
            "training_metric": [],
            "validation_metric": [],
            "learner_index": [],
            "learner_weight": [],
        }
        best_loss = np.inf
        best_size = 0
        rounds_without_improvement = 0
        iterations = range(self.num_iterations)
        if self.verbose:
            iterations = tqdm(iterations, desc="Training")

        for iteration in iterations:
            g, h = self._compute_derivatives(y, z)
            newton_target = -np.divide(g, h)
            fit_weight = h * sample_weight
            if not np.all(np.isfinite(newton_target)):
                raise FloatingPointError("Newton targets became non-finite.")
            if self.selection_strategy == "greedy":
                eligible_candidates = np.flatnonzero(probabilities > 0)

                def fit_and_score(idx):
                    prototype = self.base_learners_[idx]
                    learner, prediction = self._fit_candidate(
                        prototype,
                        X,
                        newton_target,
                        fit_weight,
                        iteration,
                        idx,
                        sampling_entropy,
                    )
                    step = self._choose_step(y, z, prediction, sample_weight)
                    loss = self._loss_value(y, z + step * prediction, sample_weight)
                    return loss, idx, learner, prediction, step

                candidates = Parallel(
                    n_jobs=candidate_n_jobs,
                    prefer="threads",
                )(
                    delayed(fit_and_score)(idx) for idx in eligible_candidates
                )
                _, idx, base_learner, learner_prediction, step = min(
                    candidates, key=lambda item: item[0]
                )
                idx = int(idx)
            else:
                idx = int(rng.choice(len(self.base_learners_), p=probabilities))
                base_learner, learner_prediction = self._fit_candidate(
                    self.base_learners_[idx],
                    X,
                    newton_target,
                    fit_weight,
                    iteration,
                    idx,
                    sampling_entropy,
                )
                step = self._choose_step(y, z, learner_prediction, sample_weight)
            candidate_z = z + learner_prediction * step
            if not np.all(np.isfinite(candidate_z)):
                raise FloatingPointError("Boosting predictions became non-finite.")
            z = candidate_z
            fitted_learners.append(base_learner)
            learner_weights.append(step)
            history["learner_index"].append(idx)
            history["learner_weight"].append(step)
            history["training_loss"].append(
                self._loss_value(y, z, sample_weight)
            )
            if eval_metric is not None:
                metric = float(eval_metric(y, z))
                if not np.isfinite(metric):
                    raise FloatingPointError("Training metric became non-finite.")
                history["training_metric"].append(metric)
            if X_eval is not None:
                eval_prediction = np.asarray(
                    base_learner.predict(X_eval), dtype=float
                )
                if eval_prediction.shape != z_eval.shape:
                    raise ValueError(
                        f"Base learner {type(base_learner).__name__} returned "
                        f"validation shape {eval_prediction.shape}; expected "
                        f"{z_eval.shape}."
                    )
                if not np.all(np.isfinite(eval_prediction)):
                    raise FloatingPointError(
                        "Base learner validation predictions became non-finite."
                    )
                z_eval += step * eval_prediction
                validation_loss = self._loss_value(y_eval, z_eval)
                history["validation_loss"].append(validation_loss)
                if eval_metric is not None:
                    metric = float(eval_metric(y_eval, z_eval))
                    if not np.isfinite(metric):
                        raise FloatingPointError(
                            "Validation metric became non-finite."
                        )
                    history["validation_metric"].append(metric)
                if validation_loss < best_loss - self.min_delta:
                    best_loss = validation_loss
                    best_size = len(fitted_learners)
                    rounds_without_improvement = 0
                else:
                    rounds_without_improvement += 1
                if (
                    self.early_stopping_rounds is not None
                    and rounds_without_improvement >= self.early_stopping_rounds
                ):
                    fitted_learners = fitted_learners[:best_size]
                    learner_weights = learner_weights[:best_size]
                    break
            callback_state = {
                "estimator": self,
                "iteration": iteration,
                "learner_index": idx,
                "learner_weight": step,
                "training_loss": history["training_loss"][-1],
                "validation_loss": (
                    history["validation_loss"][-1]
                    if history["validation_loss"] else None
                ),
            }
            callback_results = [callback(callback_state) for callback in callbacks]
            if any(callback_results):
                break

        self.ensemble_ = fitted_learners
        self.learner_weights_ = learner_weights
        self.base_score_ = base_score
        self.history_ = history
        self.n_iter_ = len(fitted_learners)
        self.best_iteration_ = (
            best_size - 1 if X_eval is not None and best_size else self.n_iter_ - 1
        )
        self.n_features_in_ = X.shape[1]
        if feature_names is None:
            self.__dict__.pop("feature_names_in_", None)
        else:
            self.feature_names_in_ = feature_names
        if self.mode == "classification":
            self.classes_ = classes

        return self

    @property
    def loss_(self):
        if self.objective_ == "pseudo_huber":
            return PseudoHuber
        if self.objective_ == "quantile":
            return Quantile
        return Logistic if self.mode == "classification" else MeanSquaredError

    @property
    def num_iterations_(self):
        return self.num_iterations

    @property
    def learning_rate_(self):
        return self.learning_rate

    def _raw_predict(self, X):
        """Return raw model output for classification or regression."""
        self._check_fitted()
        self._check_feature_names(X)
        X = _validate_X(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but model was trained with "
                f"{self.n_features_in_} features."
            )
        preds = np.full(X.shape[0], getattr(self, "base_score_", 0.0))
        weights = getattr(
            self, "learner_weights_", [self.learning_rate] * len(self.ensemble_)
        )
        for learner, weight in zip(self.ensemble_, weights):
            learner_prediction = np.asarray(learner.predict(X), dtype=float)
            if learner_prediction.shape != preds.shape:
                raise ValueError(
                    f"Base learner {type(learner).__name__} returned shape "
                    f"{learner_prediction.shape}; expected {preds.shape}."
                )
            if not np.all(np.isfinite(learner_prediction)):
                raise FloatingPointError("Base learner predictions became non-finite.")
            preds += weight * learner_prediction
        if not np.all(np.isfinite(preds)):
            raise FloatingPointError("Ensemble predictions became non-finite.")
        return preds

    def compact(self, min_abs_weight=0.0, inplace=False):
        """Optionally remove learners with negligible contribution weights.

        Compaction is never performed automatically. A positive threshold can
        change predictions and should be validated on held-out data.
        """
        self._check_fitted()
        if (
            isinstance(min_abs_weight, (bool, np.bool_))
            or not isinstance(min_abs_weight, Real)
            or not np.isfinite(min_abs_weight)
            or min_abs_weight < 0
        ):
            raise ValueError("min_abs_weight must be a finite number >= 0.")
        target = self if inplace else copy.deepcopy(self)
        weights = getattr(
            target,
            "learner_weights_",
            [target.learning_rate] * len(target.ensemble_),
        )
        keep = [abs(weight) > min_abs_weight for weight in weights]
        if not any(keep):
            raise ValueError("Compaction would remove every fitted learner.")
        target.ensemble_ = [
            learner for learner, retained in zip(target.ensemble_, keep) if retained
        ]
        target.learner_weights_ = [
            weight for weight, retained in zip(weights, keep) if retained
        ]
        target.n_iter_ = len(target.ensemble_)
        return target

    @available_if(_classification_mode)
    def decision_function(self, X):
        """Return classification logits."""
        if self.mode != "classification":
            raise ValueError(
                "decision_function is only available in classification mode."
            )
        return self._raw_predict(X)

    def predict(self, X):
        """
        Predict using the model.

        Classification returns 0/1 labels; regression returns continuous values.
        """
        if self.mode == "classification":
            logits = self.decision_function(X)
            return self.classes_[(logits >= 0).astype(int)]
        return self._raw_predict(X)

    @available_if(_classification_mode)
    def predict_proba(self, X):
        """
        Predict class probabilities (classification mode only).

        Returns
        -------
        ndarray of shape (n_samples, 2)
            Probabilities ``[P(y=0), P(y=1)]``.
        """
        if self.mode != "classification":
            raise ValueError("predict_proba is only available in classification mode.")
        logits = self.decision_function(X)
        prob_pos = np.exp(-np.logaddexp(0.0, -logits))
        return np.column_stack([1.0 - prob_pos, prob_pos])

    def score(self, X, y):
        """Return accuracy (classification) or R² (regression)."""
        self._check_fitted()
        _, y = _validate_X_y(X, y)
        if self.mode == "classification":
            return accuracy_score(y, self.predict(X))
        y = check_array(y, ensure_2d=False, dtype=float)
        return r2_score(y, self.predict(X))

    def evaluate(self, X, y):
        """Print and return log loss (classification) or RMSE (regression)."""
        self._check_fitted()
        if self.mode == "classification":
            _, y = _validate_X_y(X, y)
            probabilities = self.predict_proba(X)
            loss = log_loss(y, probabilities, labels=self.classes_)
            print("Log Loss: %.4f" % loss)
        else:
            _, y = _validate_X_y(X, y)
            y = check_array(y, ensure_2d=False, dtype=float)
            preds = self._raw_predict(X)
            loss = np.sqrt(mean_squared_error(y, preds))
            print("RMSE: %.4f" % loss)
        return loss


class HNBMClassifier(ClassifierMixin, HNBM):
    """
    Heterogeneous Newton Boosting Machine for binary classification.

    Subclass ``HNBMClassifier`` and configure ``base_learners_`` and
    ``probabilities_`` before calling ``fit``.

    Parameters
    ----------
    num_iterations : int, default=100
        Number of boosting iterations.
    learning_rate : float, default=0.1
        Shrinkage applied to each learner's contribution.
    random_state : int or None, default=None
        Random seed for base learner selection.
    verbose : bool, default=True
        Whether to show a progress bar during training.
    selection_strategy : {'random', 'greedy'}, default='random'
        Learner-family selection policy.
    line_search : bool, default=False
        Whether to select a contribution weight per boosting round.
    early_stopping_rounds : int or None, default=None
        Optional validation patience.
    min_delta : float, default=0.0
        Minimum validation improvement.
    subsample : float, default=1.0
        Fraction of training rows used by each learner.
    """
    _mode = "classification"

    def __init__(
        self,
        num_iterations=100,
        learning_rate=0.1,
        random_state=None,
        verbose=True,
        selection_strategy="random",
        line_search=False,
        early_stopping_rounds=None,
        min_delta=0.0,
        subsample=1.0,
        objective="auto",
        objective_parameter=None,
    ):
        super().__init__(
            num_iterations=num_iterations,
            learning_rate=learning_rate,
            mode=self._mode,
            random_state=random_state,
            verbose=verbose,
            selection_strategy=selection_strategy,
            line_search=line_search,
            early_stopping_rounds=early_stopping_rounds,
            min_delta=min_delta,
            subsample=subsample,
            objective=objective,
            objective_parameter=objective_parameter,
        )

    def set_params(self, **params):
        if "mode" in params:
            raise ValueError("mode cannot be set on HNBMClassifier.")
        return super().set_params(**params)


class HNBMRegressor(RegressorMixin, HNBM):
    """
    Heterogeneous Newton Boosting Machine for regression.

    Subclass ``HNBMRegressor`` and configure ``base_learners_`` and
    ``probabilities_`` before calling ``fit``.

    Parameters
    ----------
    num_iterations : int, default=100
        Number of boosting iterations.
    learning_rate : float, default=0.1
        Shrinkage applied to each learner's contribution.
    random_state : int or None, default=None
        Random seed for base learner selection.
    verbose : bool, default=True
        Whether to show a progress bar during training.
    selection_strategy : {'random', 'greedy'}, default='random'
        Learner-family selection policy.
    line_search : bool, default=False
        Whether to select a contribution weight per boosting round.
    early_stopping_rounds : int or None, default=None
        Optional validation patience.
    min_delta : float, default=0.0
        Minimum validation improvement.
    subsample : float, default=1.0
        Fraction of training rows used by each learner.
    """
    _mode = "regression"

    def __init__(
        self,
        num_iterations=100,
        learning_rate=0.1,
        random_state=None,
        verbose=True,
        selection_strategy="random",
        line_search=False,
        early_stopping_rounds=None,
        min_delta=0.0,
        subsample=1.0,
        objective="auto",
        objective_parameter=None,
    ):
        super().__init__(
            num_iterations=num_iterations,
            learning_rate=learning_rate,
            mode=self._mode,
            random_state=random_state,
            verbose=verbose,
            selection_strategy=selection_strategy,
            line_search=line_search,
            early_stopping_rounds=early_stopping_rounds,
            min_delta=min_delta,
            subsample=subsample,
            objective=objective,
            objective_parameter=objective_parameter,
        )

    def set_params(self, **params):
        if "mode" in params:
            raise ValueError("mode cannot be set on HNBMRegressor.")
        return super().set_params(**params)
