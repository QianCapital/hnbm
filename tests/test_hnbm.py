import numpy as np
import pytest
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    RegressorMixin,
    clone,
    is_classifier,
    is_regressor,
)
from sklearn.datasets import make_classification, make_regression
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from hnbm import (
    HNBM,
    HNBMClassifier,
    HNBMRegressor,
    NNBoostClassifier,
    NNBoostRegressor,
)
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


class FailingRegressor(BaseEstimator, RegressorMixin):
    def fit(self, X, y, sample_weight=None):
        raise RuntimeError("intentional fit failure")

    def predict(self, X):
        return np.zeros(len(X))


class NonFiniteRegressor(BaseEstimator, RegressorMixin):
    def fit(self, X, y, sample_weight=None):
        return self

    def predict(self, X):
        return np.full(len(X), np.nan)


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
    assert search.best_estimator_.n_iter_ == 1


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


def test_nearly_normalized_probabilities_are_normalized_before_sampling():
    X, y = make_regression(n_samples=10, n_features=2, random_state=3)
    model = TreeBoostRegressor(probabilities=(1.000001,)).fit(X, y)

    assert model.n_iter_ == 2


def test_pool_rejects_learner_without_weighted_fit():
    X, y = make_regression(n_samples=10, n_features=2, random_state=3)
    model = TreeBoostRegressor()
    model.base_learners_ = [UnweightedRegressor()]
    with pytest.raises(TypeError, match="sample_weight"):
        model.fit(X, y)


def test_pool_rejects_pipeline_without_direct_weight_routing():
    X, y = make_regression(n_samples=10, n_features=2, random_state=3)
    model = TreeBoostRegressor()
    model.base_learners_ = [
        Pipeline([("scale", StandardScaler()), ("ridge", Ridge())])
    ]
    with pytest.raises(TypeError, match="sample_weight"):
        model.fit(X, y)


def test_failed_fit_does_not_publish_partial_ensemble():
    X, y = make_regression(n_samples=10, n_features=2, random_state=3)
    model = TreeBoostRegressor(num_iterations=2, random_state=0)
    model.base_learners_ = [FailingRegressor(), DecisionTreeRegressor(max_depth=1)]
    model.probabilities_ = [0.5, 0.5]

    with pytest.raises(RuntimeError, match="intentional"):
        model.fit(X, y)

    assert not hasattr(model, "ensemble_")
    with pytest.raises(NotFittedError, match="not fitted"):
        model.predict(X)


def test_failed_refit_preserves_previous_fitted_model():
    X, y = make_regression(n_samples=10, n_features=2, random_state=3)
    model = TreeBoostRegressor(num_iterations=1).fit(X, y)
    expected = model.predict(X)
    original_ensemble = model.ensemble_
    model.base_learners_ = [FailingRegressor()]

    with pytest.raises(RuntimeError, match="intentional"):
        model.fit(X, y)

    assert model.ensemble_ is original_ensemble
    assert model.predict(X) == pytest.approx(expected)


def test_nonfinite_base_learner_predictions_fail_atomically():
    X, y = make_regression(n_samples=10, n_features=2, random_state=3)
    model = TreeBoostRegressor()
    model.base_learners_ = [NonFiniteRegressor()]

    with pytest.raises(FloatingPointError, match="non-finite"):
        model.fit(X, y)

    assert not hasattr(model, "ensemble_")


def test_prediction_checks_feature_count():
    X, y = make_regression(n_samples=10, n_features=2, random_state=3)
    model = TreeBoostRegressor().fit(X, y)
    with pytest.raises(ValueError, match="features"):
        model.predict(np.ones((2, 3)))


@pytest.mark.parametrize(
    "parameter, value, message",
    [
        ("num_iterations", 1.5, "integer"),
        ("num_iterations", True, "integer"),
        ("learning_rate", float("nan"), "finite"),
        ("learning_rate", float("inf"), "finite"),
        ("random_state", True, "integer"),
    ],
)
def test_invalid_hnbm_parameters_are_rejected(parameter, value, message):
    with pytest.raises(ValueError, match=message):
        NNBoostRegressor(**{parameter: value})


def test_hnbm_set_params_is_transactional():
    model = NNBoostRegressor(num_iterations=2)
    with pytest.raises(ValueError, match="finite"):
        model.set_params(learning_rate=float("nan"))

    assert model.learning_rate == 0.1
    assert model.num_iterations == 2


@pytest.mark.parametrize(
    "parameter, value, message",
    [
        ("max_iter", 1.5, "integer"),
        ("max_iter", True, "integer"),
        ("learning_rate_nn", float("nan"), "finite"),
        ("alpha", float("inf"), "finite"),
        ("tol", float("nan"), "finite"),
        ("hidden_layer_sizes", (True,), "integer"),
    ],
)
def test_invalid_nnboost_parameters_are_rejected(parameter, value, message):
    model = NNBoostRegressor()
    original_pool = model.base_learners_
    with pytest.raises(ValueError, match=message):
        model.set_params(**{parameter: value})

    assert getattr(model, parameter) != value
    assert model.base_learners_ is original_pool


@pytest.mark.parametrize("value", [8, "8", None, [[8]], (size for size in [8])])
def test_hidden_layer_sizes_requires_a_one_dimensional_sequence(value):
    with pytest.raises(ValueError, match="sequence"):
        NNBoostRegressor(hidden_layer_sizes=value)


def test_fitted_iteration_count_matches_completed_boosting_rounds():
    X, y = make_regression(n_samples=30, n_features=3, random_state=1)
    model = NNBoostRegressor(
        num_iterations=3,
        hidden_layer_sizes=(4,),
        max_iter=2,
        random_state=1,
    ).fit(X, y)

    assert model.n_iter_ == 3
    assert model.n_iter_ == len(model.ensemble_)


def test_predict_proba_is_stable_for_extreme_logits():
    model = NNBoostClassifier(num_iterations=1, max_iter=1)
    extreme_learner = type(
        "ExtremeLearner",
        (),
        {"predict": lambda self, X: np.array([-1e4, 1e4])},
    )()
    model.ensemble_ = [extreme_learner]
    model.n_features_in_ = 1

    with np.errstate(over="raise"):
        probabilities = model.predict_proba(np.ones((2, 1)))

    assert probabilities[:, 1] == pytest.approx([0.0, 1.0])


def test_classifier_preserves_arbitrary_binary_labels():
    X, y = make_classification(n_samples=40, n_features=4, random_state=2)
    labels = np.where(y == 0, "cat", "dog")
    model = NNBoostClassifier(
        num_iterations=2, max_iter=2, random_state=2
    ).fit(X, labels)

    assert np.array_equal(model.classes_, ["cat", "dog"])
    assert set(model.predict(X)).issubset({"cat", "dog"})
    assert model.predict_proba(X).shape == (40, 2)


def test_regressor_does_not_advertise_classifier_methods():
    model = NNBoostRegressor()
    assert not hasattr(model, "decision_function")
    assert not hasattr(model, "predict_proba")


def test_task_mixins_precede_hnbm_in_estimator_mro():
    assert HNBMClassifier.__mro__.index(
        ClassifierMixin
    ) < HNBMClassifier.__mro__.index(HNBM)
    assert HNBMRegressor.__mro__.index(
        RegressorMixin
    ) < HNBMRegressor.__mro__.index(HNBM)
    assert is_classifier(NNBoostClassifier())
    assert is_regressor(NNBoostRegressor())


def test_fresh_estimator_is_not_generically_fitted():
    from sklearn.utils.validation import check_is_fitted

    with pytest.raises(NotFittedError):
        check_is_fitted(NNBoostClassifier())
