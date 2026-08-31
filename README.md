# HNBM - Heterogeneous Newton Boosting Machine

[![PyPI version](https://img.shields.io/pypi/v/hnbm.svg)](https://pypi.org/project/hnbm/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-compatible-blue.svg)](https://scikit-learn.org/)

**Heterogeneous Newton Boosting Machine (HNBM)** — a scikit-learn-compatible gradient boosting framework that stochastically mixes heterogeneous base learners at each iteration.

Unlike standard gradient boosting libraries that use a single learner type (typically decision trees), HNBM lets you define a pool of base learners with selection probabilities. At each boosting round, a learner is drawn from that pool and fit to the Newton step (gradient divided by Hessian, weighted by the Hessian).

Built-in support includes **shallow neural network** base learners via `NNBoostClassifier` / `NNBoostRegressor`. You can also plug in a cloneable scikit-learn regressor whose `fit` method explicitly accepts `sample_weight` (for example, decision trees or kernel ridge) by subclassing.

This is the core framework behind [SnapBoost](https://github.com/qiancapital/snapboost), inspired by [SnapBoost: A Heterogeneous Boosting Machine](https://arxiv.org/abs/2006.09745) (Parnell et al., NeurIPS 2020).

## New in 1.0

HNBM 1.0 freezes `HNBMClassifier` / `HNBMRegressor`, reports sklearn estimator
tags (binary-only classification, dense inputs), and delays parameter
validation until `fit`. Constructing `HNBM(mode=...)` or `NNBoost(mode=...)`
directly is deprecated.

## New in 0.3.0: adaptive training

HNBM 0.3.0 adds weighted training, optimized constant base scores, validation
history, early stopping with best-ensemble restoration, deterministic row
subsampling, greedy learner-family selection, and optional per-round line
search. The original stochastic algorithm remains the default with
`selection_strategy="random"`.

```python
model = NNBoostRegressor(
    num_iterations=500,
    learning_rate=0.05,
    selection_strategy="greedy",
    line_search=True,
    subsample=0.8,
    early_stopping_rounds=25,
    random_state=42,
)
model.fit(
    X_train,
    y_train,
    sample_weight=train_weights,
    eval_set=(X_validation, y_validation),
)
print(model.best_iteration_, model.history_["validation_loss"])
```

Additional opt-in extensions include robust and quantile regression objectives,
custom per-round metrics, callbacks, parallel greedy candidate fitting, and
post-fit model compaction. None changes the default objective or training path.

```python
robust = NNBoostRegressor(
    objective="pseudo_huber",
    objective_parameter=2.0,
    random_state=42,
)
robust.fit(
    X_train,
    y_train,
    eval_metric=lambda y, raw: abs(y - raw).mean(),
    callbacks=[lambda state: state["iteration"] >= 499],
    candidate_n_jobs=2,
)

smaller = robust.compact(min_abs_weight=1e-8)
```

---

## Table of Contents

- [Mathematical Overview](#mathematical-overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [Neural networks (NNBoost)](#neural-networks-nnboost)
  - [Custom base learners (subclassing)](#custom-base-learners-subclassing)
- [API Reference](#api-reference)
  - [NNBoostClassifier / NNBoostRegressor](#nnboostclassifier--nnboostregressor)
  - [ShallowNNRegressor](#shallownnregressor)
  - [HNBMClassifier / HNBMRegressor](#hnbmclassifier--hnbmregressor)
  - [HNBM](#hnbm)
  - [Loss functions](#loss-functions)
- [Parameters](#parameters)
- [Docker](#docker)
- [Development](#development)
- [Related projects](#related-projects)
- [License](#license)

---

## Mathematical Overview

HNBM constructs an additive predictor from a probability-weighted pool of
possibly different hypothesis classes:

$$
F_M(x)=F_0+\sum_{m=1}^{M}\eta_m f_m(x),
\qquad f_m\in\mathcal H_{K_m},
\qquad K_m\sim\mathrm{Categorical}(p_1,\ldots,p_K).
$$

Here $F_0$ is a constant initial score, $\eta_m$ is a boosting step size, and
$\mathcal H_{K_m}$ may contain trees, kernels, neural networks, linear models,
or any cloneable weighted regressor. The probabilities satisfy
$p_k\geq0$ and $\sum_kp_k=1$.

At boosting round $m$, HNBM differentiates the loss at the current raw
prediction:

$$
g_i=\left.\frac{\partial\ell(y_i,F)}{\partial F}\right|_{F=F_{m-1}(x_i)},
\qquad
h_i=\left.\frac{\partial^2\ell(y_i,F)}{\partial F^2}\right|_{F=F_{m-1}(x_i)}.
$$

Completing the square in the second-order Taylor approximation shows that the
chosen learner should solve the weighted regression problem

$$
r_i=-\frac{g_i}{h_i},
\qquad
f_m\approx\arg\min_{f\in\mathcal H_{K_m}}
\sum_{i=1}^{n}w_i h_i\bigl(r_i-f(x_i)\bigr)^2,
$$

followed by

$$
F_m(x)=F_{m-1}(x)+\eta_m f_m(x).
$$

Thus $-g_i/h_i$ is the Newton working response and $w_i h_i$ is its effective
sample weight. For squared-error regression, $g_i=2(F-y_i)$ and $h_i=2$, so
$r_i=y_i-F$: ordinary residual boosting is recovered. For binary logistic
classification with $y_i\in\{-1,+1\}$,

$$
\ell(y,F)=\log(1+e^{-yF}),\quad
g=-y\sigma(-yF),\quad
h=\sigma(yF)\sigma(-yF),
$$

and $P(y=+1\mid x)=\sigma(F_M(x))$.

The included NNBoost realization uses a uniform pool of one-hidden-layer
networks. For hidden width $q$ its correction is

$$
f(x)=v^\top a(W^\top\widetilde x+b)+c,
$$

and the network is trained against the Newton responses using Hessian weights
and L2 regularization. Different widths define different subclasses
$\mathcal H_q$; the default random strategy samples one width per round, while
the optional greedy strategy fits every eligible width and keeps the update
with the lowest loss.

Traditional gradient boosting fits $-g_i$ and usually uses a single learner
class. XGBoost uses a related Newton expansion but specializes it to
regularized trees with analytic leaf weights and split gains. HNBM separates
the Newton optimization interface from the learner type, allowing different
inductive biases to coexist in one ensemble.

See [MATH.md](MATH.md) for the complete derivation, objective formulas,
stochastic and greedy HNBM algorithms, NNBoost training equations, convergence
interpretation, and detailed comparisons with gradient boosting, Newton tree
boosting, SnapBoost, and XGBoost.

---

## Installation

**From PyPI**:

```bash
pip install hnbm
```

**From source**:

```bash
git clone https://github.com/qiancapital/hnbm.git
cd hnbm
pip install .
```

**Requirements**: Python ≥ 3.8, NumPy, scikit-learn, tqdm.

---

## Quick Start

### Neural networks (NNBoost)

The fastest way to use HNBM with neural networks is `NNBoostClassifier` or `NNBoostRegressor`. Each boosting round randomly selects a single-hidden-layer network from a pool of hidden sizes.

#### Classification

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from hnbm import NNBoostClassifier

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

model = NNBoostClassifier(
    num_iterations=50,
    learning_rate=0.1,
    hidden_layer_sizes=(16, 32, 64),
    learning_rate_nn=0.01,
    max_iter=100,
    random_state=42,
    verbose=False,
)
model.fit(X_train, y_train)

print("Accuracy:", model.score(X_test, y_test))
model.evaluate(X_test, y_test)  # prints log loss
```

#### Regression

```python
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from hnbm import NNBoostRegressor

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

model = NNBoostRegressor(
    num_iterations=50,
    learning_rate=0.1,
    hidden_layer_sizes=(16, 32),
    random_state=42,
    verbose=False,
)
model.fit(X_train, y_train)

print("R²:", model.score(X_test, y_test))
model.evaluate(X_test, y_test)  # prints RMSE
```

### Custom base learners (subclassing)

Subclass `HNBMClassifier` or `HNBMRegressor` and configure your own base learner pool before training:

#### Classification

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

#### Regression

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

### NNBoostClassifier / NNBoostRegressor

Ready-to-use HNBM models with a pool of shallow neural network base learners. At each iteration, a network is drawn uniformly from `hidden_layer_sizes`.

**Methods** — same as `HNBMClassifier` / `HNBMRegressor` (see below).

A legacy `NNBoost` class is also available with a `mode` parameter; prefer the task-specific classes for new code.

### ShallowNNRegressor

Low-level base learner: a single-hidden-layer network trained with weighted MSE (for Newton step targets and Hessian weights). Supports `relu`, `tanh`, and `logistic` activations.

Use directly in a custom learner pool, or build a pool with `make_shallow_nn_pool`:

```python
from hnbm import HNBMClassifier, make_shallow_nn_pool

class CustomNNBoost(HNBMClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_learners_, self.probabilities_ = make_shallow_nn_pool(
            hidden_layer_sizes=(16, 32, 64),
            activation="relu",
            max_iter=100,
            random_state=self.random_state,
        )
```

### HNBMClassifier / HNBMRegressor

The recommended entry points (similar to `XGBClassifier` / `XGBRegressor`). Subclass one of these and set `base_learners_` (a list of unfitted, cloneable regressors whose `fit` methods explicitly accept `sample_weight`) and `probabilities_` (a list of finite, nonnegative values summing to 1) before calling `fit`.

**Methods**

| Method | Classifier | Regressor | Description |
|--------|------------|-----------|-------------|
| `fit(X, y, sample_weight=None, eval_set=None, ...)` | ✓ | ✓ | Train with optional weights, validation, metrics, callbacks, and candidate parallelism |
| `predict(X)` | ✓ | ✓ | Original class labels or continuous values |
| `predict_proba(X)` | ✓ | | Probabilities, shape `(n_samples, 2)` |
| `decision_function(X)` | ✓ | | Raw logits |
| `score(X, y)` | ✓ | ✓ | Accuracy or R² |
| `evaluate(X, y)` | ✓ | ✓ | Prints and returns log loss or RMSE |

After fitting, `n_iter_` contains the number of completed boosting rounds. The
inner epoch count for each fitted neural-network learner remains available on
that learner's own `n_iter_` attribute.

New fitted attributes in 0.3.0 include `base_score_`, `learner_weights_`,
`history_`, and `best_iteration_`. Validation loss is recorded only when
`eval_set=(X_validation, y_validation)` is supplied. Early stopping requires an
evaluation set and restores the best ensemble before returning.

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

### NNBoost (`NNBoostClassifier` / `NNBoostRegressor`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_iterations` | `int` | `100` | Number of boosting rounds |
| `learning_rate` | `float` | `0.1` | Boosting shrinkage per learner |
| `hidden_layer_sizes` | `tuple of int` | `(16, 32, 64)` | Hidden unit counts in the learner pool |
| `activation` | `str` | `"relu"` | Hidden activation: `"relu"`, `"tanh"`, or `"logistic"` |
| `alpha` | `float` | `1e-4` | L2 penalty on network weights |
| `learning_rate_nn` | `float` | `0.01` | Gradient descent step size per base network |
| `max_iter` | `int` | `200` | Maximum training epochs per base network |
| `tol` | `float` | `1e-5` | Early-stopping tolerance on training loss |
| `random_state` | `int` or `None` | `None` | Seed for learner selection and weight init |
| `verbose` | `bool` | `False` | Show tqdm progress bar |
| `selection_strategy` | `{"random", "greedy"}` | `"random"` | Sample one family or fit all candidates and select the lowest-loss update |
| `line_search` | `bool` | `False` | Select a contribution weight for each fitted learner |
| `subsample` | `float` | `1.0` | Fraction of rows used to fit each base learner |
| `early_stopping_rounds` | positive `int` or `None` | `None` | Validation rounds without improvement before stopping |
| `min_delta` | `float` | `0.0` | Minimum validation-loss improvement that resets patience |
| `objective` | `str` | `"auto"` | `"squared_error"`, `"pseudo_huber"`, or `"quantile"` for regression; `"log_loss"` for classification |
| `objective_parameter` | `float` or `None` | `None` | Pseudo-Huber delta or quantile level |

### Shared (`HNBMClassifier` / `HNBMRegressor`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_iterations` | `int` | `100` | Number of boosting rounds |
| `learning_rate` | `float` | `0.1` | Shrinkage per learner |
| `random_state` | `int` or `None` | `None` | Seed for learner selection |
| `verbose` | `bool` | `True` | Show tqdm progress bar |
| `selection_strategy` | `{"random", "greedy"}` | `"random"` | Learner-family selection policy |
| `line_search` | `bool` | `False` | Select a contribution weight for each round |
| `subsample` | `float` | `1.0` | Fraction of training rows per learner |
| `early_stopping_rounds` | positive `int` or `None` | `None` | Validation patience |
| `min_delta` | `float` | `0.0` | Minimum validation improvement |

The legacy `HNBM` class also accepts a `mode` parameter (`"classification"` or `"regression"`).

**Label conventions (classification)**: accepts any two distinct class labels. Predictions use the original labels, and probability columns follow `classes_` order.

---

## Docker

```bash
docker build -t hnbm .
docker run --rm hnbm
```

---

## Development

Create an environment, install HNBM in editable mode with its test dependencies,
and run the complete validation suite:

```bash
git clone https://github.com/qiancapital/hnbm.git
cd hnbm
python -m pip install -e ".[test]"
python -m pytest -q
python -m compileall -q hnbm tests
```

The pytest command must finish with all tests passing. To run an individual
test module or a single test while developing:

```bash
python -m pytest -q tests/test_hnbm.py
python -m pytest -q tests/test_nn_learner.py
python -m pytest -q tests/test_hnbm.py::test_classifier_preserves_arbitrary_binary_labels
```

CI runs the full test suite on every push and pull request, and again before a
release distribution is built.

---

## Related projects

- **[snapboost](https://github.com/QianCapital/snapboost)** — a concrete HNBM using decision trees and RFF ridge regressors
- **NNBoost** (this package) — a concrete HNBM using shallow neural networks

---

## License

MIT — See [LICENSE](LICENSE) for full text.
