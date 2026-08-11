import numpy as np
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

from .losses import Logistic, MeanSquaredError


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
    """

    def __init__(
        self,
        num_iterations=100,
        learning_rate=0.1,
        mode="classification",
        random_state=None,
        verbose=True,
    ):
        self._validate_hnbm_params(
            num_iterations=num_iterations,
            learning_rate=learning_rate,
            mode=mode,
            random_state=random_state,
        )

        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.mode = mode
        self.random_state = random_state
        self.verbose = verbose

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
        *, num_iterations, learning_rate, mode, random_state
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
        ):
            raise ValueError(
                f"random_state must be an integer or None, got {random_state}."
            )

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
        )
        result = super().set_params(**params)
        if params:
            for attribute in (
                "ensemble_",
                "classes_",
                "n_features_in_",
                "n_iter_",
            ):
                self.__dict__.pop(attribute, None)
        return result

    def fit(self, X, y):
        """
        Train the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self
        """
        probabilities = self._validate_learner_pool()

        X, y = _validate_X_y(X, y)
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

        rng = np.random.default_rng(self.random_state)
        z = np.zeros(X.shape[0])
        fitted_learners = []
        iterations = range(self.num_iterations)
        if self.verbose:
            iterations = tqdm(iterations, desc="Training")

        for _ in iterations:
            g, h = self.loss_.compute_derivatives(y, z)
            idx = rng.choice(len(self.base_learners_), p=probabilities)
            base_learner = clone(self.base_learners_[idx])
            newton_target = -np.divide(g, h)
            if not np.all(np.isfinite(newton_target)):
                raise FloatingPointError("Newton targets became non-finite.")
            base_learner.fit(X, newton_target, sample_weight=h)
            learner_prediction = np.asarray(base_learner.predict(X), dtype=float)
            if learner_prediction.shape != z.shape:
                raise ValueError(
                    f"Base learner {type(base_learner).__name__} returned shape "
                    f"{learner_prediction.shape}; expected {z.shape}."
                )
            if not np.all(np.isfinite(learner_prediction)):
                raise FloatingPointError("Base learner predictions became non-finite.")
            candidate_z = z + learner_prediction * self.learning_rate
            if not np.all(np.isfinite(candidate_z)):
                raise FloatingPointError("Boosting predictions became non-finite.")
            z = candidate_z
            fitted_learners.append(base_learner)

        self.ensemble_ = fitted_learners
        self.n_iter_ = len(fitted_learners)
        self.n_features_in_ = X.shape[1]
        if self.mode == "classification":
            self.classes_ = classes

        return self

    @property
    def loss_(self):
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
        X = _validate_X(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but model was trained with "
                f"{self.n_features_in_} features."
            )
        preds = np.zeros(X.shape[0])
        for learner in self.ensemble_:
            learner_prediction = np.asarray(learner.predict(X), dtype=float)
            if learner_prediction.shape != preds.shape:
                raise ValueError(
                    f"Base learner {type(learner).__name__} returned shape "
                    f"{learner_prediction.shape}; expected {preds.shape}."
                )
            if not np.all(np.isfinite(learner_prediction)):
                raise FloatingPointError("Base learner predictions became non-finite.")
            preds += self.learning_rate * learner_prediction
        if not np.all(np.isfinite(preds)):
            raise FloatingPointError("Ensemble predictions became non-finite.")
        return preds

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
    """
    _mode = "classification"

    def __init__(
        self,
        num_iterations=100,
        learning_rate=0.1,
        random_state=None,
        verbose=True,
    ):
        super().__init__(
            num_iterations=num_iterations,
            learning_rate=learning_rate,
            mode=self._mode,
            random_state=random_state,
            verbose=verbose,
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
    """
    _mode = "regression"

    def __init__(
        self,
        num_iterations=100,
        learning_rate=0.1,
        random_state=None,
        verbose=True,
    ):
        super().__init__(
            num_iterations=num_iterations,
            learning_rate=learning_rate,
            mode=self._mode,
            random_state=random_state,
            verbose=verbose,
        )

    def set_params(self, **params):
        if "mode" in params:
            raise ValueError("mode cannot be set on HNBMRegressor.")
        return super().set_params(**params)
