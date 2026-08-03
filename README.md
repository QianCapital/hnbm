# HNBM

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-compatible-blue.svg)](https://scikit-learn.org/)

**Heterogeneous Newton Boosting Machine (HNBM)** — a scikit-learn-compatible gradient boosting framework that stochastically mixes heterogeneous base learners at each iteration.

Unlike standard gradient boosting libraries that use a single learner type (typically decision trees), HNBM lets you define a pool of base learners with selection probabilities. At each boosting round, a learner is drawn from that pool and fit to the Newton step (gradient divided by Hessian, weighted by the Hessian).

This is the core framework behind [SnapBoost](https://github.com/qiancapital/snapboost), inspired by [SnapBoost: A Heterogeneous Boosting Machine](https://arxiv.org/abs/2006.09745) (Parnell et al., NeurIPS 2020).

---

## Installation

**From source** (recommended until PyPI release):

```bash
git clone https://github.com/qiancapital-dev/hnbm.git
cd hnbm
pip install .
```

**Requirements**: Python ≥ 3.8, NumPy, scikit-learn, tqdm.

---

## Quick Start

Subclass `HNBM` and configure your base learner pool before training:

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from hnbm import HNBM

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


class TreeBoost(HNBM):
    def __init__(self, max_depth=5, **kwargs):
        super().__init__(**kwargs)
        self.base_learners_ = [DecisionTreeRegressor(max_depth=max_depth)]
        self.probabilities_ = [1.0]


model = TreeBoost(
    num_iterations=100,
    learning_rate=0.1,
    mode="classification",
    random_state=42,
)
model.fit(X_train, y_train)

print("Accuracy:", model.score(X_test, y_test))
model.evaluate(X_test, y_test)
```

---

## API Reference

### `HNBM`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_iterations` | `int` | `100` | Number of boosting rounds |
| `learning_rate` | `float` | `0.1` | Shrinkage per learner |
| `mode` | `str` | `"classification"` | `"classification"` or `"regression"` |
| `random_state` | `int` or `None` | `None` | Seed for learner selection |
| `verbose` | `bool` | `True` | Show tqdm progress bar |

**Methods**

| Method | Mode | Description |
|--------|------|-------------|
| `fit(X, y)` | both | Train the ensemble |
| `predict(X)` | both | Class labels (0/1) or continuous values |
| `predict_proba(X)` | classification | Probabilities, shape `(n_samples, 2)` |
| `decision_function(X)` | classification | Raw logits |
| `score(X, y)` | both | Accuracy or R² |
| `evaluate(X, y)` | both | Prints and returns log loss or RMSE |

**Subclass contract**: set `base_learners_` (list of unfitted sklearn regressors) and `probabilities_` (list summing to 1) before calling `fit`.

### Loss functions

`hnbm.losses` provides `Logistic` (classification) and `MeanSquaredError` (regression), each with a `compute_derivatives(y, f)` method returning gradient and Hessian vectors.

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

- **[snapboost](https://github.com/QianCapital/snapboost)** — a concrete HNBM using decision trees and kernel ridge regressors

---

## License

MIT — See [LICENSE](LICENSE) for full text.
