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
