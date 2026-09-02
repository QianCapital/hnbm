from collections.abc import Sequence
from numbers import Integral, Real

import numpy as np

from .estimator import HNBM, HNBMClassifier, HNBMRegressor
from .nn_learner import make_shallow_nn_pool


class _NNBoostMixin:
    """Shared configuration for HNBM with shallow neural network base learners."""

    _REBUILD_PARAMS = frozenset({
        "hidden_layer_sizes",
        "activation",
        "alpha",
        "learning_rate_nn",
        "max_iter",
        "tol",
        "random_state",
    })

    def _init_nn_boost_params(
        self,
        hidden_layer_sizes=(16, 32, 64),
        activation="relu",
        alpha=1e-4,
        learning_rate_nn=0.01,
        max_iter=200,
        tol=1e-5,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.alpha = alpha
        self.learning_rate_nn = learning_rate_nn
        self.max_iter = max_iter
        self.tol = tol

    def _validate_nn_boost_params(self, overrides=None):
        overrides = {} if overrides is None else overrides
        hidden_layer_sizes = overrides.get(
            "hidden_layer_sizes", self.hidden_layer_sizes
        )
        activation = overrides.get("activation", self.activation)
        alpha = overrides.get("alpha", self.alpha)
        learning_rate_nn = overrides.get(
            "learning_rate_nn", self.learning_rate_nn
        )
        max_iter = overrides.get("max_iter", self.max_iter)
        tol = overrides.get("tol", self.tol)

        if (
            isinstance(hidden_layer_sizes, (str, bytes))
            or not isinstance(hidden_layer_sizes, (Sequence, np.ndarray))
            or np.ndim(hidden_layer_sizes) != 1
        ):
            raise ValueError(
                "hidden_layer_sizes must be a nonempty sequence of positive "
                "integers."
            )
        if len(hidden_layer_sizes) == 0:
            raise ValueError("hidden_layer_sizes must contain at least one size.")
        if any(
            isinstance(size, (bool, np.bool_))
            or not isinstance(size, Integral)
            or size < 1
            for size in hidden_layer_sizes
        ):
            raise ValueError("Each hidden layer size must be an integer >= 1.")
        if activation not in ("relu", "tanh", "logistic"):
            raise ValueError(
                "activation must be 'relu', 'tanh', or 'logistic', "
                f"got {activation!r}."
            )
        if (
            isinstance(alpha, (bool, np.bool_))
            or not isinstance(alpha, Real)
            or not np.isfinite(alpha)
            or alpha < 0
        ):
            raise ValueError(f"alpha must be a finite number >= 0, got {alpha}.")
        if (
            isinstance(learning_rate_nn, (bool, np.bool_))
            or not isinstance(learning_rate_nn, Real)
            or not np.isfinite(learning_rate_nn)
            or learning_rate_nn <= 0
        ):
            raise ValueError(
                "learning_rate_nn must be a finite number > 0, "
                f"got {learning_rate_nn}."
            )
        if (
            isinstance(max_iter, (bool, np.bool_))
            or not isinstance(max_iter, Integral)
            or max_iter < 1
        ):
            raise ValueError(f"max_iter must be an integer >= 1, got {max_iter}.")
        if (
            isinstance(tol, (bool, np.bool_))
            or not isinstance(tol, Real)
            or not np.isfinite(tol)
            or tol < 0
        ):
            raise ValueError(f"tol must be a finite number >= 0, got {tol}.")

    def _build_base_learners(self):
        self.base_learners_, self.probabilities_ = make_shallow_nn_pool(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            alpha=self.alpha,
            learning_rate=self.learning_rate_nn,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
        )

    def _set_nn_boost_params(self, **params):
        result = super().set_params(**params)
        try:
            self._validate_nn_boost_params()
        except (TypeError, ValueError):
            return result
        rebuild = bool(self._REBUILD_PARAMS.intersection(params))
        if rebuild or not self.base_learners_:
            self._build_base_learners()
        return result

    def fit(
        self, X, y, sample_weight=None, eval_set=None,
        eval_metric=None, callbacks=None, candidate_n_jobs=1,
        *, eval_sample_weight=None,
    ):
        self._validate_nn_boost_params()
        self._build_base_learners()
        return super().fit(
            X,
            y,
            sample_weight=sample_weight,
            eval_set=eval_set,
            eval_metric=eval_metric,
            callbacks=callbacks,
            candidate_n_jobs=candidate_n_jobs,
            eval_sample_weight=eval_sample_weight,
        )


class NNBoost(_NNBoostMixin, HNBM):
    """
    HNBM that uses a pool of shallow neural network base learners.

    At each boosting iteration, a network is drawn from the pool according to
    uniform probabilities over ``hidden_layer_sizes``. Prefer
    :class:`NNBoostClassifier` or :class:`NNBoostRegressor` for task-specific
    models without a ``mode`` parameter.
    """

    def __init__(
        self,
        num_iterations=100,
        learning_rate=0.1,
        hidden_layer_sizes=(16, 32, 64),
        activation="relu",
        alpha=1e-4,
        learning_rate_nn=0.01,
        max_iter=200,
        tol=1e-5,
        mode="classification",
        random_state=None,
        verbose=False,
        selection_strategy="random",
        line_search=False,
        early_stopping_rounds=None,
        min_delta=0.0,
        subsample=1.0,
        objective="auto",
        objective_parameter=None,
    ):
        self._init_nn_boost_params(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            alpha=alpha,
            learning_rate_nn=learning_rate_nn,
            max_iter=max_iter,
            tol=tol,
        )
        super().__init__(
            num_iterations=num_iterations,
            learning_rate=learning_rate,
            mode=mode,
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
        if type(self) is NNBoost:
            import warnings

            warnings.warn(
                "Constructing NNBoost(mode=...) directly is deprecated and will "
                "be removed in a future release. Use NNBoostClassifier or "
                "NNBoostRegressor instead.",
                FutureWarning,
                stacklevel=2,
            )

    def set_params(self, **params):
        return self._set_nn_boost_params(**params)


class NNBoostClassifier(_NNBoostMixin, HNBMClassifier):
    """
    HNBM for classification using shallow neural network base learners.

    Example
    -------
    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.model_selection import train_test_split
    >>> from hnbm import NNBoostClassifier
    >>>
    >>> X, y = load_breast_cancer(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, random_state=42
    ... )
    >>> model = NNBoostClassifier(
    ...     num_iterations=50,
    ...     learning_rate=0.1,
    ...     hidden_layer_sizes=(16, 32),
    ...     random_state=42,
    ...     verbose=False,
    ... )
    >>> model.fit(X_train, y_train)
    >>> model.score(X_test, y_test)
    """

    def __init__(
        self,
        num_iterations=100,
        learning_rate=0.1,
        hidden_layer_sizes=(16, 32, 64),
        activation="relu",
        alpha=1e-4,
        learning_rate_nn=0.01,
        max_iter=200,
        tol=1e-5,
        random_state=None,
        verbose=False,
        selection_strategy="random",
        line_search=False,
        early_stopping_rounds=None,
        min_delta=0.0,
        subsample=1.0,
        objective="auto",
        objective_parameter=None,
    ):
        self._init_nn_boost_params(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            alpha=alpha,
            learning_rate_nn=learning_rate_nn,
            max_iter=max_iter,
            tol=tol,
        )
        super().__init__(
            num_iterations=num_iterations,
            learning_rate=learning_rate,
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
            raise ValueError("mode cannot be set on NNBoostClassifier.")
        return self._set_nn_boost_params(**params)


class NNBoostRegressor(_NNBoostMixin, HNBMRegressor):
    """
    HNBM for regression using shallow neural network base learners.

    Example
    -------
    >>> from sklearn.datasets import load_diabetes
    >>> from sklearn.model_selection import train_test_split
    >>> from hnbm import NNBoostRegressor
    >>>
    >>> X, y = load_diabetes(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, random_state=42
    ... )
    >>> model = NNBoostRegressor(
    ...     num_iterations=50,
    ...     learning_rate=0.1,
    ...     hidden_layer_sizes=(16, 32),
    ...     random_state=42,
    ...     verbose=False,
    ... )
    >>> model.fit(X_train, y_train)
    >>> model.score(X_test, y_test)
    """

    def __init__(
        self,
        num_iterations=100,
        learning_rate=0.1,
        hidden_layer_sizes=(16, 32, 64),
        activation="relu",
        alpha=1e-4,
        learning_rate_nn=0.01,
        max_iter=200,
        tol=1e-5,
        random_state=None,
        verbose=False,
        selection_strategy="random",
        line_search=False,
        early_stopping_rounds=None,
        min_delta=0.0,
        subsample=1.0,
        objective="auto",
        objective_parameter=None,
    ):
        self._init_nn_boost_params(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            alpha=alpha,
            learning_rate_nn=learning_rate_nn,
            max_iter=max_iter,
            tol=tol,
        )
        super().__init__(
            num_iterations=num_iterations,
            learning_rate=learning_rate,
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
            raise ValueError("mode cannot be set on NNBoostRegressor.")
        return self._set_nn_boost_params(**params)
