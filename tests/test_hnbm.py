import numpy as np
import pytest
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

from hnbm import HNBMRegressor, NNBoostClassifier, NNBoostRegressor
from hnbm.losses import Logistic


class TreeBoostRegressor(HNBMRegressor):
    def __init__(self, num_iterations=2, probabilities=(1.0,), random_state=0):
        self.probabilities = probabilities
        super().__init__(
            num_iterations=num_iterations,
            random_state=random_state,
            verbose=False,
        )
        self.base_learners_ = [DecisionTreeRegressor(max_depth=2)]
        self.probabilities_ = list(probabilities)


class UnweightedRegressor(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X))


def test_logistic_derivatives_are_finite_for_extreme_margins():
    y = np.array([1.0, -1.0])
    predictions = np.array([-1000.0, 1000.0])
    gradient, hessian = Logistic.compute_derivatives(y, predictions)

    assert np.all(np.isfinite(gradient))
    assert np.all(np.isfinite(hessian))
    assert np.all(hessian > 0)


@pytest.mark.parametrize("estimator_class", [NNBoostClassifier, NNBoostRegressor])
def test_nnboost_clone_and_set_params(estimator_class):
    estimator = clone(estimator_class(num_iterations=1, max_iter=2))
    result = estimator.set_params(hidden_layer_sizes=(4, 8))

    assert result is estimator
    assert len(estimator.base_learners_) == 2


def test_nnboost_regressor_works_with_grid_search():
    X, y = make_regression(n_samples=30, n_features=3, random_state=1)
    search = GridSearchCV(
        NNBoostRegressor(num_iterations=1, max_iter=2, random_state=1),
        {"hidden_layer_sizes": [(4,), (8,)]},
        cv=2,
    )
    search.fit(X, y)
    assert search.best_estimator_.ensemble_


def test_nnboost_classifier_fits_extreme_scale_data():
    X, y = make_classification(n_samples=40, n_features=4, random_state=2)
    model = NNBoostClassifier(
        num_iterations=2, max_iter=2, random_state=2
    ).fit(X * 1e6, y)
    assert np.all(np.isfinite(model.predict_proba(X * 1e6)))


@pytest.mark.parametrize(
    "probabilities, message",
    [((0.5,), "sum to 1"), ((-0.1,), "non-negative"), ((float("nan"),), "finite")],
)
def test_invalid_learner_probabilities_are_rejected(probabilities, message):
    X, y = make_regression(n_samples=10, n_features=2, random_state=3)
    with pytest.raises(ValueError, match=message):
        TreeBoostRegressor(probabilities=probabilities).fit(X, y)


def test_pool_rejects_learner_without_weighted_fit():
    X, y = make_regression(n_samples=10, n_features=2, random_state=3)
    model = TreeBoostRegressor()
    model.base_learners_ = [UnweightedRegressor()]
    with pytest.raises(TypeError, match="sample_weight"):
        model.fit(X, y)


def test_prediction_checks_feature_count():
    X, y = make_regression(n_samples=10, n_features=2, random_state=3)
    model = TreeBoostRegressor().fit(X, y)
    with pytest.raises(ValueError, match="features"):
        model.predict(np.ones((2, 3)))
