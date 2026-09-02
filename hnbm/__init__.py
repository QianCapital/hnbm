from .estimator import HNBM, HNBMClassifier, HNBMRegressor
from .losses import Logistic, MeanSquaredError, PseudoHuber, Quantile, Softmax
from .nn_boost import NNBoost, NNBoostClassifier, NNBoostRegressor
from .nn_learner import ShallowNNRegressor, make_shallow_nn_pool

__all__ = [
    "HNBM",
    "HNBMClassifier",
    "HNBMRegressor",
    "Logistic",
    "MeanSquaredError",
    "PseudoHuber",
    "Quantile",
    "Softmax",
    "NNBoost",
    "NNBoostClassifier",
    "NNBoostRegressor",
    "ShallowNNRegressor",
    "make_shallow_nn_pool",
]
__version__ = "1.2.0"
