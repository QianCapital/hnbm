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
    __version__,
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


class WrongValidationShapeRegressor(BaseEstimator, RegressorMixin):
    def fit(self, X, y, sample_weight=None):
        self.training_size_ = len(X)
        return self

    def predict(self, X):
        return np.zeros(self.training_size_)


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
        ("random_state", -1, "non-negative"),
    ],
)
def test_invalid_hnbm_parameters_are_rejected(parameter, value, message):
    with pytest.raises(ValueError, match=message):
        NNBoostRegressor(**{parameter: value})


def test_line_search_requires_boolean():
    with pytest.raises(ValueError, match="boolean"):
        NNBoostRegressor(line_search="yes")


def test_nnboost_exposes_adaptive_parameters_to_clone():
    model = clone(
        NNBoostRegressor(
            selection_strategy="greedy",
            line_search=True,
            early_stopping_rounds=3,
            min_delta=0.01,
            subsample=0.8,
        )
    )

    assert model.selection_strategy == "greedy"
    assert model.line_search is True
    assert model.early_stopping_rounds == 3
    assert model.subsample == pytest.approx(0.8)


def test_package_exposes_staged_version():
    assert __version__ == "0.3.0"


@pytest.mark.parametrize(
    "objective, parameter",
    [("pseudo_huber", 2.0), ("quantile", 0.8)],
)
def test_additive_regression_objectives_fit(objective, parameter):
    X, y = make_regression(n_samples=40, n_features=3, random_state=3)
    model = NNBoostRegressor(
        num_iterations=2,
        hidden_layer_sizes=(4,),
        max_iter=2,
        objective=objective,
        objective_parameter=parameter,
        random_state=3,
    ).fit(X, y)

    assert model.objective_ == objective
    assert np.all(np.isfinite(model.predict(X)))


def test_quantile_objective_uses_weighted_quantile_base_score():
    X = np.arange(10.0).reshape(5, 2)
    y = np.array([0.0, 1.0, 2.0, 3.0, 100.0])
    model = NNBoostRegressor(
        num_iterations=1,
        hidden_layer_sizes=(2,),
        max_iter=1,
        objective="quantile",
        objective_parameter=0.5,
        random_state=3,
    ).fit(X, y)

    assert model.base_score_ == pytest.approx(2.0)


def test_classification_rejects_regression_objective():
    with pytest.raises(ValueError, match="objective"):
        NNBoostClassifier(objective="quantile")


def test_custom_metric_is_recorded_for_training_and_validation():
    X, y = make_regression(n_samples=30, n_features=3, random_state=3)
    metric = lambda truth, raw: np.mean(np.abs(truth - raw))
    model = TreeBoostRegressor(num_iterations=2).fit(
        X[:20], y[:20], eval_set=(X[20:], y[20:]), eval_metric=metric
    )

    assert len(model.history_["training_metric"]) == 2
    assert len(model.history_["validation_metric"]) == 2
    assert len(model.history_["learner_weight"]) == 2


def test_callback_can_stop_training_without_partial_state():
    X, y = make_regression(n_samples=30, n_features=3, random_state=3)
    states = []

    def stop_after_second_round(state):
        states.append(state.copy())
        return state["iteration"] == 1

    model = TreeBoostRegressor(num_iterations=10).fit(
        X, y, callbacks=[stop_after_second_round]
    )

    assert model.n_iter_ == 2
    assert len(states) == 2
    assert states[-1]["estimator"] is model


def test_greedy_candidates_can_fit_in_parallel_deterministically():
    X, y = make_regression(n_samples=50, n_features=3, random_state=3)
    serial = TreeBoostRegressor(num_iterations=3, random_state=3)
    serial.selection_strategy = "greedy"
    serial.base_learners_ = [
        DecisionTreeRegressor(max_depth=1),
        DecisionTreeRegressor(max_depth=3),
    ]
    serial.probabilities_ = [0.5, 0.5]
    parallel = clone(serial)
    parallel.selection_strategy = "greedy"
    parallel.base_learners_ = list(serial.base_learners_)
    parallel.probabilities_ = list(serial.probabilities_)

    serial.fit(X, y, candidate_n_jobs=1)
    parallel.fit(X, y, candidate_n_jobs=2)

    assert parallel.predict(X) == pytest.approx(serial.predict(X))
    assert parallel.history_["learner_index"] == serial.history_["learner_index"]


def test_candidate_n_jobs_rejects_zero():
    X, y = make_regression(n_samples=20, n_features=2, random_state=3)
    with pytest.raises(ValueError, match="nonzero"):
        TreeBoostRegressor().fit(X, y, candidate_n_jobs=0)


def test_optional_compaction_removes_only_small_weight_learners():
    X, y = make_regression(n_samples=30, n_features=2, random_state=3)
    model = TreeBoostRegressor(num_iterations=2, random_state=3).fit(X, y)
    model.learner_weights_ = [0.1, 1e-12]

    compacted = model.compact(min_abs_weight=1e-6)

    assert compacted is not model
    assert compacted.n_iter_ == 1
    assert model.n_iter_ == 2


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


def test_regression_uses_weighted_mean_base_score():
    X = np.arange(8.0).reshape(4, 2)
    y = np.array([0.0, 0.0, 0.0, 10.0])
    model = TreeBoostRegressor(num_iterations=1).fit(
        X, y, sample_weight=np.array([1.0, 1.0, 1.0, 3.0])
    )

    assert model.base_score_ == pytest.approx(5.0)
    assert len(model.learner_weights_) == model.n_iter_


def test_classifier_base_score_matches_class_prior():
    X = np.arange(16.0).reshape(8, 2)
    y = np.array([0, 0, 1, 1, 1, 1, 1, 1])
    model = NNBoostClassifier(
        num_iterations=1, hidden_layer_sizes=(2,), max_iter=1, random_state=2
    ).fit(X, y)

    assert model.base_score_ == pytest.approx(np.log(0.75 / 0.25))


def test_greedy_selection_chooses_best_candidate():
    X, y = make_regression(n_samples=30, n_features=2, noise=0.0, random_state=4)
    model = TreeBoostRegressor(num_iterations=1)
    model.selection_strategy = "greedy"
    model.base_learners_ = [
        DecisionTreeRegressor(max_depth=1),
        DecisionTreeRegressor(max_depth=5),
    ]
    model.probabilities_ = [0.5, 0.5]

    model.fit(X, y)

    assert model.history_["learner_index"] == [1]


def test_greedy_selection_respects_zero_candidate_probability():
    X, y = make_regression(n_samples=30, n_features=2, noise=0.0, random_state=4)
    model = TreeBoostRegressor(num_iterations=1)
    model.selection_strategy = "greedy"
    model.base_learners_ = [
        DecisionTreeRegressor(max_depth=1),
        DecisionTreeRegressor(max_depth=8),
    ]
    model.probabilities_ = [1.0, 0.0]

    model.fit(X, y)

    assert model.history_["learner_index"] == [0]


def test_subsampling_uses_only_positive_weight_rows():
    X, y = make_regression(n_samples=20, n_features=2, random_state=4)
    weights = np.zeros(20)
    weights[-1] = 1.0
    model = TreeBoostRegressor(num_iterations=1)
    model.subsample = 0.1

    model.fit(X, y, sample_weight=weights)

    assert np.all(np.isfinite(model.predict(X)))


def test_invalid_validation_prediction_shape_fails_atomically():
    X, y = make_regression(n_samples=20, n_features=2, random_state=4)
    model = TreeBoostRegressor(num_iterations=1)
    model.base_learners_ = [WrongValidationShapeRegressor()]

    with pytest.raises(ValueError, match="validation shape"):
        model.fit(X[:15], y[:15], eval_set=(X[15:], y[15:]))

    assert not hasattr(model, "ensemble_")


def test_eval_history_and_early_stopping_restore_best_ensemble():
    X, y = make_regression(n_samples=40, n_features=2, random_state=4)
    model = TreeBoostRegressor(num_iterations=20)
    model.early_stopping_rounds = 2
    model.min_delta = 1e100

    model.fit(X[:30], y[:30], eval_set=(X[30:], y[30:]))

    assert model.n_iter_ == 1
    assert len(model.history_["validation_loss"]) == 3


def test_round_specific_random_states_are_distinct_and_reproducible():
    X, y = make_regression(n_samples=30, n_features=2, random_state=4)
    first = TreeBoostRegressor(num_iterations=2, random_state=9).fit(X, y)
    second = TreeBoostRegressor(num_iterations=2, random_state=9).fit(X, y)

    first_seeds = [learner.random_state for learner in first.ensemble_]
    second_seeds = [learner.random_state for learner in second.ensemble_]
    assert first_seeds[0] != first_seeds[1]
    assert first_seeds == second_seeds


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
