# HNBM

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-compatible-blue.svg)](https://scikit-learn.org/)

**Heterogeneous Newton Boosting Machine (HNBM)** — a scikit-learn-compatible gradient boosting framework that stochastically mixes heterogeneous base learners at each iteration.

Unlike standard gradient boosting libraries that use a single learner type (typically decision trees), HNBM lets you define a pool of base learners with selection probabilities. At each boosting round, a learner is drawn from that pool and fit to the Newton step (gradient divided by Hessian, weighted by the Hessian).

This is the core framework behind [SnapBoost](https://github.com/qiancapital/snapboost), inspired by [SnapBoost: A Heterogeneous Boosting Machine](https://arxiv.org/abs/2006.09745) (Parnell et al., NeurIPS 2020).

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
  - [HNBMClassifier / HNBMRegressor](#hnbmclassifier--hnbmregressor)
  - [HNBM](#hnbm)
  - [Loss functions](#loss-functions)
- [Parameters](#parameters)
- [Docker](#docker)
- [Development](#development)
- [Related projects](#related-projects)
- [License](#license)

---

## Installation

**From PyPI**:

```bash
pip install hnbm
```

**From source**:

```bash
git clone https://github.com/qiancapital-dev/hnbm.git
cd hnbm
pip install .
```

**Requirements**: Python ≥ 3.8, NumPy, scikit-learn, tqdm.

---

## Quick Start

Subclass `HNBMClassifier` or `HNBMRegressor` and configure your base learner pool before training:

### Classification

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from hnbm import HNBMClassifier

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


class TreeClassifier(HNBMClassifier):
    def __init__(self, max_depth=5, **kwargs):
        super().__init__(**kwargs)
        self.base_learners_ = [DecisionTreeRegressor(max_depth=max_depth)]
        self.probabilities_ = [1.0]


model = TreeClassifier(
    num_iterations=100,
    learning_rate=0.1,
    random_state=42,
)
model.fit(X_train, y_train)

print("Accuracy:", model.score(X_test, y_test))
print("Probabilities shape:", model.predict_proba(X_test).shape)  # (n_samples, 2)
model.evaluate(X_test, y_test)  # prints log loss
```

### Regression

```python
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from hnbm import HNBMRegressor

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


class TreeRegressor(HNBMRegressor):
    def __init__(self, max_depth=5, **kwargs):
        super().__init__(**kwargs)
        self.base_learners_ = [DecisionTreeRegressor(max_depth=max_depth)]
        self.probabilities_ = [1.0]


model = TreeRegressor(
    num_iterations=100,
    learning_rate=0.1,
    random_state=42,
)
model.fit(X_train, y_train)

print("R²:", model.score(X_test, y_test))
model.evaluate(X_test, y_test)  # prints RMSE
```

---

## API Reference

### HNBMClassifier / HNBMRegressor

The recommended entry points (similar to `XGBClassifier` / `XGBRegressor`). Subclass one of these and set `base_learners_` (list of unfitted sklearn regressors) and `probabilities_` (list summing to 1) before calling `fit`.

**Methods**

| Method | Classifier | Regressor | Description |
|--------|------------|-----------|-------------|
| `fit(X, y)` | ✓ | ✓ | Train the ensemble |
| `predict(X)` | ✓ | ✓ | Class labels (0/1) or continuous values |
| `predict_proba(X)` | ✓ | | Probabilities, shape `(n_samples, 2)` |
| `decision_function(X)` | ✓ | | Raw logits |
| `score(X, y)` | ✓ | ✓ | Accuracy or R² |
| `evaluate(X, y)` | ✓ | ✓ | Prints and returns log loss or RMSE |

### HNBM

Legacy base class that accepts a `mode` parameter (`"classification"` or `"regression"`). Prefer `HNBMClassifier` or `HNBMRegressor` for new code.

```python
from sklearn.tree import DecisionTreeRegressor
from hnbm import HNBM


class TreeBoost(HNBM):
    def __init__(self, max_depth=5, **kwargs):
        super().__init__(**kwargs)
        self.base_learners_ = [DecisionTreeRegressor(max_depth=max_depth)]
        self.probabilities_ = [1.0]


model = TreeBoost(
    num_iterations=100,
    learning_rate=0.1,
    mode="classification",  # or "regression"
    random_state=42,
)
model.fit(X_train, y_train)
```

### Loss functions

`hnbm.losses` provides `Logistic` (classification) and `MeanSquaredError` (regression), each with a `compute_derivatives(y, f)` method returning gradient and Hessian vectors.

---

## Parameters

### Shared (`HNBMClassifier` / `HNBMRegressor`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_iterations` | `int` | `100` | Number of boosting rounds |
| `learning_rate` | `float` | `0.1` | Shrinkage per learner |
| `random_state` | `int` or `None` | `None` | Seed for learner selection |
| `verbose` | `bool` | `True` | Show tqdm progress bar |

The legacy `HNBM` class also accepts a `mode` parameter (`"classification"` or `"regression"`).

**Label conventions (classification)**: accepts `0`/`1` or `-1`/`+1`. Predictions are returned as `0`/`1`.

---

## Docker

```bash
docker build -t hnbm .
docker run --rm hnbm
```

---

## Development

```bash
git clone https://github.com/qiancapital-dev/hnbm.git
cd hnbm
pip install -r requirements.txt
pip install -e .
```

---

## Related projects

- **[snapboost](https://github.com/QianCapital/snapboost)** — a concrete HNBM using decision trees and RFF ridge regressors

---

## License

MIT — See [LICENSE](LICENSE) for full text.
