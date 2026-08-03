import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted


def _activation(name):
    if name == "relu":
        return lambda z: np.maximum(0.0, z), lambda z: (z > 0).astype(float)
    if name == "tanh":
        return np.tanh, lambda z: 1.0 - np.tanh(z) ** 2
    if name == "logistic":
        def logistic(z):
            z = np.clip(z, -500.0, 500.0)
            return 1.0 / (1.0 + np.exp(-z))

        return logistic, lambda z: logistic(z) * (1.0 - logistic(z))
    raise ValueError(
        f"activation must be 'relu', 'tanh', or 'logistic', got {name!r}."
    )


def _weighted_mean_std(y, sample_weight):
    weight_sum = np.sum(sample_weight)
    mean = np.sum(sample_weight * y) / weight_sum
    variance = np.sum(sample_weight * (y - mean) ** 2) / weight_sum
    scale = np.sqrt(variance)
    if scale < 1e-8:
        scale = 1.0
    return mean, scale


def _clip_gradients(*grads, max_norm=5.0):
    total_norm = np.sqrt(sum(np.sum(g ** 2) for g in grads))
    if total_norm <= max_norm or total_norm == 0.0:
        return grads
    scale = max_norm / total_norm
    return tuple(g * scale for g in grads)


class ShallowNNRegressor(BaseEstimator, RegressorMixin):
    """
    Single-hidden-layer neural network regressor for HNBM base learning.

    Trains with weighted mean squared error, matching the Newton step targets
    and Hessian weights used by :class:`~hnbm.estimator.HNBM`.
    """

    def __init__(
        self,
        hidden_layer_size=32,
        activation="relu",
        alpha=1e-4,
        learning_rate=0.01,
        max_iter=200,
        tol=1e-5,
        random_state=None,
    ):
        self.hidden_layer_size = hidden_layer_size
        self.activation = activation
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _validate_params(self):
        if self.hidden_layer_size < 1:
            raise ValueError(
                f"hidden_layer_size must be >= 1, got {self.hidden_layer_size}."
            )
        if self.alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {self.alpha}.")
        if self.learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be > 0, got {self.learning_rate}."
            )
        if self.max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {self.max_iter}.")
        if self.tol < 0:
            raise ValueError(f"tol must be >= 0, got {self.tol}.")

    def _init_weights(self, n_features, rng):
        fan_in = n_features
        if self.activation == "relu":
            scale = np.sqrt(2.0 / fan_in)
        else:
            scale = np.sqrt(1.0 / fan_in)
        self.coef_input_ = rng.normal(0.0, scale, (n_features, self.hidden_layer_size))
        self.intercept_hidden_ = np.zeros(self.hidden_layer_size)
        self.coef_output_ = rng.normal(
            0.0, np.sqrt(1.0 / self.hidden_layer_size), (self.hidden_layer_size, 1)
        )
        self.intercept_output_ = np.zeros(1)

    def _forward(self, X):
        self._z_hidden = X @ self.coef_input_ + self.intercept_hidden_
        self._z_hidden = np.clip(self._z_hidden, -50.0, 50.0)
        self._hidden = self._act_fn(self._z_hidden)
        return (self._hidden @ self.coef_output_ + self.intercept_output_).ravel()

    def _loss(self, y_pred, y, sample_weight):
        residual = y_pred - y
        mse = np.sum(sample_weight * residual ** 2) / np.sum(sample_weight)
        reg = self.alpha * (
            np.sum(self.coef_input_ ** 2)
            + np.sum(self.coef_output_ ** 2)
        )
        return mse + reg

    def _backward(self, X, y_pred, y, sample_weight):
        weight_sum = np.sum(sample_weight)
        grad_output = 2.0 * sample_weight * (y_pred - y) / weight_sum
        grad_output_col = grad_output.reshape(-1, 1)

        grad_coef_output = self._hidden.T @ grad_output_col + self.alpha * self.coef_output_
        grad_intercept_output = np.sum(grad_output_col, axis=0)

        grad_hidden = grad_output_col @ self.coef_output_.T
        grad_z_hidden = grad_hidden * self._act_deriv(self._z_hidden)

        grad_coef_input = X.T @ grad_z_hidden + self.alpha * self.coef_input_
        grad_intercept_hidden = np.sum(grad_z_hidden, axis=0)

        return (
            grad_coef_input,
            grad_intercept_hidden,
            grad_coef_output,
            grad_intercept_output,
        )

    def fit(self, X, y, sample_weight=None):
        self._validate_params()

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        y = np.asarray(y, dtype=float).ravel()
        if y.shape[0] != X.shape[0]:
            raise ValueError(
                f"X and y have inconsistent lengths: {X.shape[0]} vs {y.shape[0]}."
            )

        if sample_weight is None:
            sample_weight = np.ones(X.shape[0], dtype=float)
        else:
            sample_weight = np.asarray(sample_weight, dtype=float).ravel()
            if sample_weight.shape[0] != X.shape[0]:
                raise ValueError(
                    "sample_weight must have the same length as y."
                )
            if np.any(sample_weight < 0):
                raise ValueError("sample_weight must be non-negative.")

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)
        self.y_mean_, self.y_scale_ = _weighted_mean_std(y, sample_weight)
        y_scaled = (y - self.y_mean_) / self.y_scale_

        self._act_fn, self._act_deriv = _activation(self.activation)
        rng = np.random.default_rng(self.random_state)
        self._init_weights(X_scaled.shape[1], rng)

        prev_loss = np.inf
        n_iter = 0
        for n_iter in range(1, self.max_iter + 1):
            y_pred = self._forward(X_scaled)
            grads = self._backward(X_scaled, y_pred, y_scaled, sample_weight)
            grads = _clip_gradients(*grads)

            self.coef_input_ -= self.learning_rate * grads[0]
            self.intercept_hidden_ -= self.learning_rate * grads[1]
            self.coef_output_ -= self.learning_rate * grads[2]
            self.intercept_output_ -= self.learning_rate * grads[3]

            loss = self._loss(y_pred, y_scaled, sample_weight)
            if not np.isfinite(loss):
                break
            if abs(prev_loss - loss) < self.tol:
                break
            prev_loss = loss

        self.n_features_in_ = X.shape[1]
        self.n_iter_ = n_iter
        return self

    def predict(self, X):
        check_is_fitted(self, ["scaler_", "coef_input_", "coef_output_", "y_mean_", "y_scale_"])
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but model was trained with "
                f"{self.n_features_in_} features."
            )
        X_scaled = self.scaler_.transform(X)
        self._act_fn, self._act_deriv = _activation(self.activation)
        y_scaled = self._forward(X_scaled)
        return y_scaled * self.y_scale_ + self.y_mean_


def make_shallow_nn_pool(
    hidden_layer_sizes=(16, 32, 64),
    activation="relu",
    alpha=1e-4,
    learning_rate=0.01,
    max_iter=200,
    tol=1e-5,
    random_state=None,
):
    """
    Build a heterogeneous pool of shallow neural network base learners.

    Returns ``(base_learners_, probabilities_)`` suitable for
    :class:`~hnbm.estimator.HNBM`.
    """
    if not hidden_layer_sizes:
        raise ValueError("hidden_layer_sizes must contain at least one size.")
    if any(size < 1 for size in hidden_layer_sizes):
        raise ValueError("Each hidden layer size must be >= 1.")

    base_learners_ = []
    for idx, hidden_layer_size in enumerate(hidden_layer_sizes):
        learner_seed = None if random_state is None else random_state + idx
        base_learners_.append(
            ShallowNNRegressor(
                hidden_layer_size=hidden_layer_size,
                activation=activation,
                alpha=alpha,
                learning_rate=learning_rate,
                max_iter=max_iter,
                tol=tol,
                random_state=learner_seed,
            )
        )

    probability = 1.0 / len(base_learners_)
    probabilities_ = [probability] * len(base_learners_)
    return base_learners_, probabilities_
