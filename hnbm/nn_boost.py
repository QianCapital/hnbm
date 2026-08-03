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

    def _validate_nn_boost_params(self):
        if not self.hidden_layer_sizes:
            raise ValueError("hidden_layer_sizes must contain at least one size.")
        if any(size < 1 for size in self.hidden_layer_sizes):
            raise ValueError("Each hidden layer size must be >= 1.")
        if self.activation not in ("relu", "tanh", "logistic"):
            raise ValueError(
                "activation must be 'relu', 'tanh', or 'logistic', "
                f"got {self.activation!r}."
            )
        if self.alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {self.alpha}.")
        if self.learning_rate_nn <= 0:
            raise ValueError(
                f"learning_rate_nn must be > 0, got {self.learning_rate_nn}."
            )
        if self.max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {self.max_iter}.")
        if self.tol < 0:
            raise ValueError(f"tol must be >= 0, got {self.tol}.")

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
        rebuild = bool(self._REBUILD_PARAMS.intersection(params))
        result = super().set_params(**params)
        if rebuild:
            self._validate_nn_boost_params()
            self._build_base_learners()
        return result


class NNBoost(HNBM, _NNBoostMixin):
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
        verbose=True,
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
        )
        self._validate_nn_boost_params()
        self._build_base_learners()

    def set_params(self, **params):
        return self._set_nn_boost_params(**params)


class NNBoostClassifier(HNBMClassifier, _NNBoostMixin):
    """
    HNBM for binary classification using shallow neural network base learners.

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
        verbose=True,
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
        )
        self._validate_nn_boost_params()
        self._build_base_learners()

    def set_params(self, **params):
        if "mode" in params:
            raise ValueError("mode cannot be set on NNBoostClassifier.")
        return self._set_nn_boost_params(**params)


class NNBoostRegressor(HNBMRegressor, _NNBoostMixin):
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
        verbose=True,
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
        )
        self._validate_nn_boost_params()
        self._build_base_learners()

    def set_params(self, **params):
        if "mode" in params:
            raise ValueError("mode cannot be set on NNBoostRegressor.")
        return self._set_nn_boost_params(**params)
