import numpy as np


class MeanSquaredError:
    """Mean Squared Error loss for regression."""

    @staticmethod
    def compute_derivatives(y, f):
        g = 2 * (f - y)
        h = 2.0 * np.ones(y.shape[0])
        return g, h


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
