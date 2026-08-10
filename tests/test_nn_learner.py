import numpy as np
import pytest

from hnbm import ShallowNNRegressor


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
    with pytest.raises(ValueError, match="positive total"):
        ShallowNNRegressor(max_iter=1).fit(
            np.ones((3, 2)), np.ones(3), sample_weight=np.zeros(3)
        )
