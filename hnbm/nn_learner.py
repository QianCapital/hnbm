import numpy as np
from numbers import Integral, Real
from collections.abc import Sequence
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


def _relu(z):
    return np.maximum(0.0, z)


def _relu_derivative(z):
    return (z > 0).astype(float)


def _tanh_derivative(z):
    return 1.0 - np.tanh(z) ** 2


def _logistic(z):
    z = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z))


def _logistic_derivative(z):
    activation = _logistic(z)
    return activation * (1.0 - activation)


def _activation(name):
    if name == "relu":
        return _relu, _relu_derivative
    if name == "tanh":
        return np.tanh, _tanh_derivative
    if name == "logistic":
        return _logistic, _logistic_derivative
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
        if (
            isinstance(self.hidden_layer_size, (bool, np.bool_))
            or not isinstance(self.hidden_layer_size, Integral)
            or self.hidden_layer_size < 1
        ):
            raise ValueError(
                "hidden_layer_size must be an integer >= 1, "
                f"got {self.hidden_layer_size}."
            )
        _activation(self.activation)
        if (
            isinstance(self.alpha, (bool, np.bool_))
            or not isinstance(self.alpha, Real)
            or not np.isfinite(self.alpha)
            or self.alpha < 0
        ):
            raise ValueError(
                f"alpha must be a finite number >= 0, got {self.alpha}."
            )
        if (
            isinstance(self.learning_rate, (bool, np.bool_))
            or not isinstance(self.learning_rate, Real)
            or not np.isfinite(self.learning_rate)
            or self.learning_rate <= 0
        ):
            raise ValueError(
                "learning_rate must be a finite number > 0, "
                f"got {self.learning_rate}."
            )
        if (
            isinstance(self.max_iter, (bool, np.bool_))
            or not isinstance(self.max_iter, Integral)
            or self.max_iter < 1
        ):
            raise ValueError(
                f"max_iter must be an integer >= 1, got {self.max_iter}."
            )
        if (
            isinstance(self.tol, (bool, np.bool_))
            or not isinstance(self.tol, Real)
            or not np.isfinite(self.tol)
            or self.tol < 0
        ):
            raise ValueError(
                f"tol must be a finite number >= 0, got {self.tol}."
            )
        if self.random_state is not None and (
            isinstance(self.random_state, (bool, np.bool_))
            or not isinstance(self.random_state, Integral)
        ):
            raise ValueError(
                "random_state must be an integer or None, "
                f"got {self.random_state}."
            )

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

    def _forward(self, X, return_intermediates=False):
        z_hidden = X @ self.coef_input_ + self.intercept_hidden_
        z_hidden = np.clip(z_hidden, -50.0, 50.0)
        hidden = self._act_fn(z_hidden)
        prediction = (hidden @ self.coef_output_ + self.intercept_output_).ravel()
        if return_intermediates:
            return prediction, z_hidden, hidden
        return prediction

    def _loss(self, y_pred, y, sample_weight):
        residual = y_pred - y
        mse = np.sum(sample_weight * residual ** 2) / np.sum(sample_weight)
        reg = self.alpha * (
            np.sum(self.coef_input_ ** 2)
            + np.sum(self.coef_output_ ** 2)
        )
        return mse + reg

    def _backward(
        self, X, y_pred, y, sample_weight, z_hidden, hidden
    ):
        weight_sum = np.sum(sample_weight)
        grad_output = 2.0 * sample_weight * (y_pred - y) / weight_sum
        grad_output_col = grad_output.reshape(-1, 1)

        grad_coef_output = (
            hidden.T @ grad_output_col + 2.0 * self.alpha * self.coef_output_
        )
        grad_intercept_output = np.sum(grad_output_col, axis=0)

        grad_hidden = grad_output_col @ self.coef_output_.T
        grad_z_hidden = grad_hidden * self._act_deriv(z_hidden)

        grad_coef_input = X.T @ grad_z_hidden + 2.0 * self.alpha * self.coef_input_
        grad_intercept_hidden = np.sum(grad_z_hidden, axis=0)

        return (
            grad_coef_input,
            grad_intercept_hidden,
            grad_coef_output,
            grad_intercept_output,
        )

    def fit(self, X, y, sample_weight=None):
        previous_state = self.__dict__.copy()
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                return self._fit(X, y, sample_weight=sample_weight)
        except Exception:
            self.__dict__.clear()
            self.__dict__.update(previous_state)
            raise

    def _fit(self, X, y, sample_weight=None):
        self._validate_params()

        X, y = check_X_y(X, y, dtype=float, y_numeric=True)

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
        if not np.all(np.isfinite(sample_weight)):
            raise ValueError("sample_weight must contain only finite values.")
        if np.sum(sample_weight) <= 0:
            raise ValueError(
                "sample_weight must contain at least one non-zero number."
            )

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X, sample_weight=sample_weight)
        self.y_mean_, self.y_scale_ = _weighted_mean_std(y, sample_weight)
        y_scaled = (y - self.y_mean_) / self.y_scale_

        self._act_fn, self._act_deriv = _activation(self.activation)
        rng = np.random.default_rng(self.random_state)
        self._init_weights(X_scaled.shape[1], rng)

        prev_loss = np.inf
        n_iter = 0
        for n_iter in range(1, self.max_iter + 1):
            y_pred, z_hidden, hidden = self._forward(
                X_scaled, return_intermediates=True
            )
            grads = self._backward(
                X_scaled,
                y_pred,
                y_scaled,
                sample_weight,
                z_hidden,
                hidden,
            )
            grads = _clip_gradients(*grads)
            if not all(np.all(np.isfinite(gradient)) for gradient in grads):
                raise FloatingPointError("Neural-network gradients became non-finite.")

            current_params = (
                self.coef_input_,
                self.intercept_hidden_,
                self.coef_output_,
                self.intercept_output_,
            )
            candidate_params = tuple(
                parameter - self.learning_rate * gradient
                for parameter, gradient in zip(current_params, grads)
            )
            if not all(
                np.all(np.isfinite(parameter)) for parameter in candidate_params
            ):
                raise FloatingPointError("Neural-network parameters became non-finite.")

            (
                self.coef_input_,
                self.intercept_hidden_,
                self.coef_output_,
                self.intercept_output_,
            ) = candidate_params
            updated_prediction = self._forward(X_scaled)
            loss = self._loss(updated_prediction, y_scaled, sample_weight)
            if not np.isfinite(loss):
                (
                    self.coef_input_,
                    self.intercept_hidden_,
                    self.coef_output_,
                    self.intercept_output_,
                ) = current_params
                raise FloatingPointError("Neural-network loss became non-finite.")
            if abs(prev_loss - loss) < self.tol:
                break
            prev_loss = loss

        self.n_features_in_ = X.shape[1]
        self.n_iter_ = n_iter
        return self

    def predict(self, X):
        check_is_fitted(
            self,
            ["scaler_", "coef_input_", "coef_output_", "y_mean_", "y_scale_"],
        )
        X = check_array(X, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but model was trained with "
                f"{self.n_features_in_} features."
            )
        X_scaled = self.scaler_.transform(X)
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
    if (
        isinstance(hidden_layer_sizes, (str, bytes))
        or not isinstance(hidden_layer_sizes, (Sequence, np.ndarray))
        or np.ndim(hidden_layer_sizes) != 1
    ):
        raise ValueError(
            "hidden_layer_sizes must be a nonempty sequence of positive integers."
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
