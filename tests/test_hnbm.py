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
    NNBoost,
    NNBoostClassifier,
    NNBoostRegressor,
    __version__,
)
from hnbm.losses import Logistic, Softmax


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


class TreeBoostClassifier(HNBMClassifier):
    def __init__(self, num_iterations=2, random_state=0):
        super().__init__(
            num_iterations=num_iterations,
            random_state=random_state,
            verbose=False,
        )
        self.base_learners_ = [DecisionTreeRegressor(max_depth=2, random_state=0)]
        self.probabilities_ = [1.0]


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


def test_softmax_derivatives_match_one_hot_gradient():
    scores = np.zeros((4, 3))
    y = np.array([0, 1, 2, 1])
    gradient, hessian = Softmax.compute_derivatives(y, scores)
    probability = Softmax.probabilities(scores)

    assert probability == pytest.approx(np.full((4, 3), 1.0 / 3.0))
    expected_gradient = probability.copy()
    expected_gradient[np.arange(4), y] -= 1.0
    assert gradient == pytest.approx(expected_gradient)
    assert hessian == pytest.approx(probability * (1.0 - probability))
    assert Softmax.compute_loss(y, scores) == pytest.approx(np.full(4, np.log(3.0)))


def test_softmax_probabilities_are_stable_for_extreme_logits():
    scores = np.array([[1e4, 0.0, -1e4], [-1e4, -1e4, 1e4]])
    with np.errstate(over="raise"):
        probability = Softmax.probabilities(scores)

    assert probability[0] == pytest.approx([1.0, 0.0, 0.0], abs=1e-12)
    assert probability[1] == pytest.approx([0.0, 0.0, 1.0], abs=1e-12)


@pytest.mark.parametrize("estimator_class", [NNBoostClassifier, NNBoostRegressor])
def test_nnboost_clone_and_set_params(estimator_class):
    estimator = clone(estimator_class(num_iterations=1, max_iter=2))
    result = estimator.set_params(hidden_layer_sizes=(4, 8))

    assert result is estimator
    assert len(estimator.base_learners_) == 2


def test_nnboost_direct_construction_is_deprecated():
    with pytest.warns(FutureWarning, match="NNBoostClassifier or NNBoostRegressor"):
        model = NNBoost(num_iterations=1, mode="regression")

    assert model.mode == "regression"
    assert model.set_params(verbose=False) is model


def test_task_specific_nnboost_rejects_mode():
    with pytest.raises(ValueError, match="mode cannot be set"):
        NNBoostClassifier().set_params(mode="regression")
    with pytest.raises(ValueError, match="mode cannot be set"):
        NNBoostRegressor().set_params(mode="classification")


def test_nnboost_rejects_empty_hidden_layer_sizes():
    X, y = make_regression(n_samples=10, n_features=2, random_state=0)
    with pytest.raises(ValueError, match="at least one"):
        NNBoostRegressor(hidden_layer_sizes=()).fit(X, y)


def test_nnboost_rejects_invalid_activation():
    X, y = make_regression(n_samples=10, n_features=2, random_state=0)
    with pytest.raises(ValueError, match="activation"):
        NNBoostRegressor(activation="sigmoid").fit(X, y)


def test_set_params_skips_rebuild_for_unrelated_keys():
    model = NNBoostRegressor()
    model.set_params(hidden_layer_sizes=(4,))
    learners = model.base_learners_
    model.set_params(verbose=False)

    assert model.base_learners_ is learners


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
    X, y = make_regression(n_samples=20, n_features=3, random_state=0)
    model = NNBoostRegressor(**{parameter: value})
    with pytest.raises(ValueError, match=message):
        model.fit(X, y)


def test_line_search_requires_boolean():
    X, y = make_regression(n_samples=20, n_features=3, random_state=0)
    with pytest.raises(ValueError, match="boolean"):
        NNBoostRegressor(line_search="yes").fit(X, y)


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


def test_package_exposes_version():
    assert __version__ == "1.2.0"


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
    X, y = make_classification(n_samples=30, n_features=4, random_state=0)
    with pytest.raises(ValueError, match="objective"):
        NNBoostClassifier(objective="quantile").fit(X, y)


def test_custom_metric_is_recorded_for_training_and_validation():
    X, y = make_regression(n_samples=30, n_features=3, random_state=3)

    def metric(truth, raw):
        return np.mean(np.abs(truth - raw))

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


def test_hnbm_set_params_stores_values_before_fit_validation():
    model = NNBoostRegressor(num_iterations=2)
    result = model.set_params(learning_rate=float("nan"))

    assert result is model
    assert np.isnan(model.learning_rate)
    X, y = make_regression(n_samples=20, n_features=3, random_state=0)
    with pytest.raises(ValueError, match="finite"):
        model.fit(X, y)


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
    X, y = make_regression(n_samples=20, n_features=3, random_state=0)
    model = NNBoostRegressor()
    model.set_params(**{parameter: value})
    with pytest.raises(ValueError, match=message):
        model.fit(X, y)


@pytest.mark.parametrize("value", [8, "8", None, [[8]], (size for size in [8])])
def test_hidden_layer_sizes_requires_a_one_dimensional_sequence(value):
    X, y = make_regression(n_samples=20, n_features=3, random_state=0)
    with pytest.raises(ValueError, match="sequence"):
        NNBoostRegressor(hidden_layer_sizes=value).fit(X, y)


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
    assert len(model.history_["validation_loss"]) == 1


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
    assert not hasattr(model, "staged_decision_function")
    assert not hasattr(model, "staged_predict_proba")


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


def _regression_frame(n_features=4):
    pd = pytest.importorskip("pandas")
    X, y = make_regression(n_samples=60, n_features=n_features, random_state=11)
    columns = [f"f{index}" for index in range(n_features)]
    return pd.DataFrame(X, columns=columns), y


def test_feature_names_recorded_when_fitting_a_dataframe():
    frame, y = _regression_frame()
    model = TreeBoostRegressor(random_state=11).fit(frame, y)

    assert list(model.feature_names_in_) == list(frame.columns)


def test_feature_names_absent_when_fitting_an_array():
    frame, y = _regression_frame()
    model = TreeBoostRegressor(random_state=11).fit(frame.to_numpy(), y)

    assert not hasattr(model, "feature_names_in_")


def test_reordered_columns_are_rejected():
    frame, y = _regression_frame()
    model = TreeBoostRegressor(random_state=11).fit(frame, y)

    with pytest.raises(ValueError, match="same order"):
        model.predict(frame[list(reversed(frame.columns))])


def test_renamed_columns_report_the_difference():
    frame, y = _regression_frame()
    model = TreeBoostRegressor(random_state=11).fit(frame, y)
    renamed = frame.rename(columns={"f0": "elsewhere"})

    with pytest.raises(ValueError, match="unseen at fit time"):
        model.predict(renamed)


def test_many_renamed_columns_are_summarized():
    frame, y = _regression_frame(n_features=8)
    model = TreeBoostRegressor(random_state=11).fit(frame, y)
    renamed = frame.rename(columns={f"f{index}": f"g{index}" for index in range(8)})

    with pytest.raises(ValueError, match=r"\.\.\. \(3 more\)"):
        model.predict(renamed)


def test_non_string_column_names_are_ignored():
    pd = pytest.importorskip("pandas")
    X, y = make_regression(n_samples=20, n_features=3, random_state=0)
    frame = pd.DataFrame(X, columns=[0, 1, 2])
    model = TreeBoostRegressor(random_state=0).fit(frame, y)

    assert not hasattr(model, "feature_names_in_")


def test_mixing_named_and_unnamed_inputs_warns():
    frame, y = _regression_frame()
    named = TreeBoostRegressor(random_state=11).fit(frame, y)
    unnamed = TreeBoostRegressor(random_state=11).fit(frame.to_numpy(), y)

    with pytest.warns(UserWarning, match="does not have valid feature names"):
        named.predict(frame.to_numpy())
    with pytest.warns(UserWarning, match="fitted without feature names"):
        unnamed.predict(frame)


def test_eval_set_feature_names_must_match_training_data():
    frame, y = _regression_frame()
    model = TreeBoostRegressor(random_state=11)

    with pytest.raises(ValueError, match="eval_set feature names"):
        model.fit(frame, y, eval_set=(frame.rename(columns={"f0": "other"}), y))


def test_set_params_clears_recorded_feature_names():
    frame, y = _regression_frame()
    model = TreeBoostRegressor(random_state=11).fit(frame, y)

    model.set_params(num_iterations=3)

    assert not hasattr(model, "feature_names_in_")


def test_eval_metric_receives_original_classification_labels():
    X, y = make_classification(n_samples=40, n_features=4, random_state=2)
    labels = np.where(y == 0, "neg", "pos")
    seen = []

    def metric(truth, raw):
        seen.append(np.unique(truth))
        return 0.0

    TreeBoostClassifier(num_iterations=2, random_state=2).fit(
        X[:30],
        labels[:30],
        eval_set=(X[30:], labels[30:]),
        eval_metric=metric,
    )

    assert all(set(values) == {"neg", "pos"} for values in seen)
    assert len(seen) == 4


def test_eval_sample_weight_changes_validation_loss():
    X, y = make_regression(n_samples=40, n_features=3, random_state=3)
    model = TreeBoostRegressor(num_iterations=2, random_state=3)
    uniform = model.fit(X[:30], y[:30], eval_set=(X[30:], y[30:])).history_[
        "validation_loss"
    ]
    weighted = TreeBoostRegressor(num_iterations=2, random_state=3).fit(
        X[:30],
        y[:30],
        eval_set=(X[30:], y[30:]),
        eval_sample_weight=np.linspace(1.0, 4.0, 10),
    ).history_["validation_loss"]

    assert uniform != pytest.approx(weighted)


def test_eval_sample_weight_requires_eval_set():
    X, y = make_regression(n_samples=20, n_features=2, random_state=3)
    with pytest.raises(ValueError, match="eval_sample_weight"):
        TreeBoostRegressor().fit(X, y, eval_sample_weight=np.ones(len(y)))


def test_staged_predict_matches_final_prediction():
    X, y = make_regression(n_samples=30, n_features=3, random_state=3)
    model = TreeBoostRegressor(num_iterations=3, random_state=3).fit(X, y)
    staged = list(model.staged_predict(X))

    assert len(staged) == model.n_iter_
    assert staged[-1] == pytest.approx(model.predict(X))


def test_staged_classifier_outputs_match_final_prediction():
    X, y = make_classification(n_samples=40, n_features=4, random_state=2)
    model = TreeBoostClassifier(num_iterations=3, random_state=2).fit(X, y)
    labels = list(model.staged_predict(X))
    probabilities = list(model.staged_predict_proba(X))
    logits = list(model.staged_decision_function(X))

    assert np.array_equal(labels[-1], model.predict(X))
    assert probabilities[-1] == pytest.approx(model.predict_proba(X))
    assert logits[-1] == pytest.approx(model.decision_function(X))


def test_permutation_importance_has_one_score_per_feature():
    X, y = make_regression(n_samples=40, n_features=3, random_state=3)
    model = TreeBoostRegressor(num_iterations=3, random_state=3).fit(X, y)
    result = model.permutation_importance(X, y, n_repeats=2, random_state=3)

    assert result.importances_mean.shape == (3,)
    assert np.all(np.isfinite(result.importances_mean))


def test_multiclass_fits_softmax_and_preserves_labels():
    X, y = make_classification(
        n_samples=90,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=6,
    )
    labels = np.array(["a", "b", "c"])[y]
    model = TreeBoostClassifier(num_iterations=5, random_state=6).fit(X, labels)

    probabilities = model.predict_proba(X)
    logits = model.decision_function(X)

    assert model.n_classes_ == 3
    assert np.array_equal(model.classes_, ["a", "b", "c"])
    assert probabilities.shape == (90, 3)
    assert logits.shape == (90, 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert set(model.predict(X)).issubset({"a", "b", "c"})
    assert np.array_equal(model.predict(X), model.classes_[np.argmax(logits, axis=1)])
    assert model.n_iter_ == 5
    assert len(model.ensemble_) == 5
    assert all(len(round_entry) == 3 for round_entry in model.ensemble_)
    assert model.loss_ is Softmax
    staged = list(model.staged_predict_proba(X))
    assert staged[-1] == pytest.approx(probabilities)


def test_multiclass_base_score_matches_class_priors():
    X = np.arange(24.0).reshape(12, 2)
    y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2])
    model = TreeBoostClassifier(num_iterations=1, random_state=2).fit(X, y)

    priors = np.array([6.0, 3.0, 3.0]) / 12.0
    assert model.base_score_ == pytest.approx(np.log(priors))


def test_multiclass_greedy_line_search_and_eval_set():
    X, y = make_classification(
        n_samples=80,
        n_features=5,
        n_informative=4,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=7,
    )
    model = TreeBoostClassifier(num_iterations=3, random_state=7)
    model.selection_strategy = "greedy"
    model.line_search = True
    model.base_learners_ = [
        DecisionTreeRegressor(max_depth=1),
        DecisionTreeRegressor(max_depth=3),
    ]
    model.probabilities_ = [0.5, 0.5]
    model.fit(X[:60], y[:60], eval_set=(X[60:], y[60:]))

    assert model.n_iter_ == 3
    assert len(model.history_["validation_loss"]) == 3
    assert np.all(np.isfinite(model.predict_proba(X)))


def test_early_stopping_truncates_history_to_the_retained_ensemble():
    X, y = make_classification(
        n_samples=90,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=6,
    )
    model = TreeBoostClassifier(num_iterations=20, random_state=6)
    model.early_stopping_rounds = 2
    model.min_delta = 1e100

    # An eval_metric populates the two metric histories as well, so every list
    # in history_ is covered by the alignment check below.
    model.fit(
        X[:60],
        y[:60],
        eval_set=(X[60:], y[60:]),
        eval_metric=lambda truth, raw: float(np.mean(raw)),
    )

    assert model.n_iter_ == 1
    assert len(model.ensemble_) == 1
    assert model.best_iteration_ == 0
    assert set(model.history_) == {
        "training_loss",
        "validation_loss",
        "training_metric",
        "validation_metric",
        "learner_index",
        "learner_weight",
    }
    for recorded in model.history_.values():
        assert len(recorded) == model.n_iter_


def test_callbacks_report_the_early_stopping_round():
    X, y = make_regression(n_samples=40, n_features=2, random_state=4)
    iterations = []
    model = TreeBoostRegressor(num_iterations=20)
    model.early_stopping_rounds = 2
    model.min_delta = 1e100

    model.fit(
        X[:30],
        y[:30],
        eval_set=(X[30:], y[30:]),
        callbacks=[lambda state: iterations.append(state["iteration"])],
    )

    # Three rounds are fitted before patience runs out; the ensemble keeps one.
    assert iterations == [0, 1, 2]
    assert model.n_iter_ == 1


def test_multiclass_eval_set_rejects_unknown_labels():
    X, y = make_classification(
        n_samples=40,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=8,
    )
    model = TreeBoostClassifier(num_iterations=1, random_state=8)
    y_eval = y[30:].copy()
    y_eval[0] = 9

    with pytest.raises(ValueError, match="unknown class"):
        model.fit(X[:30], y[:30], eval_set=(X[30:], y_eval))


def test_binary_ensemble_remains_one_learner_per_round():
    X, y = make_classification(n_samples=40, n_features=4, random_state=2)
    model = TreeBoostClassifier(num_iterations=3, random_state=2).fit(X, y)

    assert model.n_classes_ == 2
    assert model.decision_function(X).ndim == 1
    assert model.predict_proba(X).shape == (40, 2)
    assert all(hasattr(learner, "predict") for learner in model.ensemble_)
    assert model.loss_ is Logistic


def test_single_class_target_is_rejected():
    X = np.arange(20.0).reshape(10, 2)
    model = TreeBoostClassifier(num_iterations=2, random_state=0)

    with pytest.raises(ValueError, match="only one class"):
        model.fit(X, np.ones(10))

    assert not hasattr(model, "ensemble_")


def test_failed_multiclass_fit_keeps_binary_loss_of_fitted_model():
    X, y = make_classification(n_samples=40, n_features=4, random_state=9)
    model = TreeBoostClassifier(num_iterations=2, random_state=9).fit(X, y)
    assert model.loss_ is Logistic

    X_multi, y_multi = make_classification(
        n_samples=60,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=9,
    )
    with pytest.raises(ValueError, match="unknown class"):
        model.fit(
            X_multi, y_multi, eval_set=(X_multi, np.full(y_multi.shape[0], 99))
        )

    assert model.loss_ is Logistic
