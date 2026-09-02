import numpy as np


class MeanSquaredError:
    """Mean Squared Error loss for regression."""

    @staticmethod
    def compute_derivatives(y, f):
        g = 2 * (f - y)
        h = 2.0 * np.ones(y.shape[0])
        return g, h

    @staticmethod
    def compute_loss(y, f):
        return np.square(f - y)


class PseudoHuber:
    """Smooth robust regression loss with a configurable transition scale."""

    @staticmethod
    def compute_derivatives(y, f, delta=1.0):
        residual = f - y
        scaled = residual / delta
        denominator = np.sqrt(1.0 + np.square(scaled))
        g = residual / denominator
        h = np.power(1.0 + np.square(scaled), -1.5)
        return g, np.maximum(h, np.finfo(float).eps)

    @staticmethod
    def compute_loss(y, f, delta=1.0):
        scaled = (f - y) / delta
        return np.square(delta) * (np.sqrt(1.0 + np.square(scaled)) - 1.0)


class Quantile:
    """Quantile loss using a unit-Hessian working approximation."""

    @staticmethod
    def compute_derivatives(y, f, quantile=0.5):
        g = np.where(f >= y, 1.0 - quantile, -quantile)
        return g, np.ones_like(g, dtype=float)

    @staticmethod
    def compute_loss(y, f, quantile=0.5):
        residual = y - f
        return np.maximum(quantile * residual, (quantile - 1.0) * residual)


class Softmax:
    """Multinomial logistic loss with a diagonal Hessian.

    Gradients and Hessians are computed from softmax probabilities. The Hessian
    keeps only the diagonal ``p_k (1 - p_k)`` of the exact
    ``diag(p) - p p^T``, with no damping factor. XGBoost inflates the same
    diagonal by 2 and LightGBM by ``K / (K - 1)``, so the working response here
    is twice XGBoost's; ``learning_rate`` is correspondingly stronger for
    multiclass than for the binary logistic path. See MATH.md 4.2.1.
    """

    @staticmethod
    def probabilities(f):
        scores = np.asarray(f, dtype=float)
        shifted = scores - np.max(scores, axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / np.sum(exp, axis=1, keepdims=True)

    @staticmethod
    def compute_derivatives(y, f):
        probability = Softmax.probabilities(f)
        gradient = probability.copy()
        gradient[np.arange(y.shape[0]), y] -= 1.0
        hessian = np.maximum(
            probability * (1.0 - probability), np.finfo(float).eps
        )
        return gradient, hessian

    @staticmethod
    def compute_loss(y, f):
        scores = np.asarray(f, dtype=float)
        shifted = scores - np.max(scores, axis=1, keepdims=True)
        log_normalizer = np.log(np.sum(np.exp(shifted), axis=1))
        return log_normalizer - shifted[np.arange(y.shape[0]), y]


class Logistic:
    """Logistic loss for binary classification."""

    @staticmethod
    def compute_derivatives(y, f):
        margin = np.multiply(y, f)
        probability_wrong = np.empty_like(margin, dtype=float)
        nonnegative = margin >= 0

        exp_negative = np.exp(-margin[nonnegative])
        probability_wrong[nonnegative] = exp_negative / (1.0 + exp_negative)

        exp_positive = np.exp(margin[~nonnegative])
        probability_wrong[~nonnegative] = 1.0 / (1.0 + exp_positive)

        g = -np.multiply(y, probability_wrong)
        h = np.multiply(probability_wrong, 1.0 - probability_wrong)
        h = np.maximum(h, np.finfo(float).eps)
        return g, h

    @staticmethod
    def compute_loss(y, f):
        return np.logaddexp(0.0, -y * f)
