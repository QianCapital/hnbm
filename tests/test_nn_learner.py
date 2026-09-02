import pickle

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from hnbm import ShallowNNRegressor, make_shallow_nn_pool


def test_zero_weight_sample_is_equivalent_to_removing_it():
    X = np.array([[0.0], [1.0], [1000.0]])
    y = np.array([0.0, 1.0, -1000.0])
    weights = np.array([1.0, 1.0, 0.0])

    weighted = ShallowNNRegressor(max_iter=5, random_state=4).fit(
        X, y, sample_weight=weights
    )
    removed = ShallowNNRegressor(max_iter=5, random_state=4).fit(X[:2], y[:2])

    assert weighted.predict(X[:2]) == pytest.approx(removed.predict(X[:2]))


def test_all_zero_sample_weights_are_rejected():
    with pytest.raises(ValueError, match="non-zero"):
        ShallowNNRegressor(max_iter=1).fit(
            np.ones((3, 2)), np.ones(3), sample_weight=np.zeros(3)
        )


def test_numerical_failure_raises_and_leaves_new_estimator_unfitted():
    X = np.arange(20.0).reshape(10, 2)
    y = np.arange(10.0)
    model = ShallowNNRegressor(learning_rate=1e308, max_iter=3)

    with pytest.raises(FloatingPointError):
        model.fit(X, y)

    with pytest.raises(NotFittedError, match="not fitted"):
        model.predict(X)


def test_failed_refit_preserves_previous_network():
    X = np.arange(20.0).reshape(10, 2)
    y = np.arange(10.0)
    model = ShallowNNRegressor(max_iter=3, random_state=2).fit(X, y)
    expected = model.predict(X)
    model.learning_rate = 1e308

    with pytest.raises(FloatingPointError):
        model.fit(X, y)

    assert model.predict(X) == pytest.approx(expected)


@pytest.mark.parametrize(
    "parameter, value, message",
    [
        ("hidden_layer_size", True, "integer"),
        ("hidden_layer_size", 1.5, "integer"),
        ("alpha", float("nan"), "finite"),
        ("learning_rate", float("inf"), "finite"),
        ("max_iter", 1.5, "integer"),
        ("tol", float("nan"), "finite"),
        ("random_state", True, "integer"),
    ],
)
def test_invalid_standalone_learner_parameters(parameter, value, message):
    model = ShallowNNRegressor(**{parameter: value})
    with pytest.raises(ValueError, match=message):
        model.fit(np.ones((3, 2)), np.ones(3))


def test_predict_does_not_mutate_fitted_state():
    X = np.arange(20.0).reshape(10, 2)
    y = np.arange(10.0)
    model = ShallowNNRegressor(max_iter=3, random_state=2).fit(X, y)
    state_before = pickle.dumps(model.__dict__)

    model.predict(X)

    assert pickle.dumps(model.__dict__) == state_before


@pytest.mark.parametrize("value", [8, "8", None, [[8]], (size for size in [8])])
def test_pool_requires_a_one_dimensional_size_sequence(value):
    with pytest.raises(ValueError, match="sequence"):
        make_shallow_nn_pool(hidden_layer_sizes=value)


@pytest.mark.parametrize("value", [True, np.bool_(True)])
def test_pool_rejects_boolean_layer_sizes(value):
    with pytest.raises(ValueError, match="integer"):
        make_shallow_nn_pool(hidden_layer_sizes=(value,))


@pytest.mark.parametrize("activation", ["tanh", "logistic"])
def test_non_relu_activations_fit_and_predict(activation):
    X = np.arange(20.0).reshape(10, 2)
    y = np.arange(10.0)
    model = ShallowNNRegressor(
        activation=activation, max_iter=5, random_state=1
    ).fit(X, y)

    assert np.all(np.isfinite(model.predict(X)))


def test_invalid_activation_is_rejected():
    with pytest.raises(ValueError, match="activation"):
        ShallowNNRegressor(activation="sigmoid", max_iter=1).fit(
            np.ones((3, 2)), np.ones(3)
        )


def test_constant_target_uses_unit_scale():
    X = np.arange(10.0).reshape(5, 2)
    y = np.ones(5)
    model = ShallowNNRegressor(max_iter=3, random_state=1).fit(X, y)

    assert model.y_scale_ == pytest.approx(1.0)


def test_sample_weight_must_match_target_length():
    with pytest.raises(ValueError, match="same length"):
        ShallowNNRegressor(max_iter=1).fit(
            np.ones((3, 2)), np.ones(3), sample_weight=np.ones(2)
        )


def test_negative_sample_weight_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        ShallowNNRegressor(max_iter=1).fit(
            np.ones((3, 2)), np.ones(3), sample_weight=np.array([1.0, -1.0, 1.0])
        )


def test_nonfinite_sample_weight_is_rejected():
    with pytest.raises(ValueError, match="finite"):
        ShallowNNRegressor(max_iter=1).fit(
            np.ones((3, 2)), np.ones(3), sample_weight=np.array([1.0, np.nan, 1.0])
        )


def test_predict_checks_feature_count():
    model = ShallowNNRegressor(max_iter=1, random_state=0).fit(
        np.ones((4, 2)), np.arange(4.0)
    )
    with pytest.raises(ValueError, match="features"):
        model.predict(np.ones((2, 3)))


def test_pool_rejects_empty_hidden_layer_sizes():
    with pytest.raises(ValueError, match="at least one"):
        make_shallow_nn_pool(hidden_layer_sizes=())
