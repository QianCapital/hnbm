import inspect

import pytest
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.estimator_checks import check_estimator

from hnbm import HNBMClassifier, HNBMRegressor


class _TreeClassifier(HNBMClassifier):
    def fit(self, X, y, **kwargs):
        self.base_learners_ = [DecisionTreeRegressor(max_depth=2, random_state=0)]
        self.probabilities_ = [1.0]
        return super().fit(X, y, **kwargs)


class _TreeRegressor(HNBMRegressor):
    def fit(self, X, y, **kwargs):
        self.base_learners_ = [DecisionTreeRegressor(max_depth=2, random_state=0)]
        self.probabilities_ = [1.0]
        return super().fit(X, y, **kwargs)


def _expected_failed_checks():
    return {
        "check_sample_weight_equivalence_on_dense_data": (
            "Newton boosting with Hessian weights is not equivalent to "
            "repeating rows according to integer sample weights."
        ),
    }


@pytest.mark.skipif(
    "expected_failed_checks" not in inspect.signature(check_estimator).parameters,
    reason="sklearn 1.6+ expected_failed_checks is required",
)
@pytest.mark.parametrize(
    "estimator",
    [
        _TreeClassifier(num_iterations=8, random_state=0, verbose=False),
        _TreeRegressor(num_iterations=8, random_state=0, verbose=False),
    ],
)
def test_hnbm_passes_sklearn_check_estimator(estimator):
    check_estimator(estimator, expected_failed_checks=_expected_failed_checks())
