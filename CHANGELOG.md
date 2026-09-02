# Changelog

## 1.2.0

Release date: 2026-09-01

- Add native multiclass classification with softmax Newton boosting. Binary
  logistic classification is unchanged: two classes still use a scalar logit.
- Fit one scalar base learner per class in each multiclass boosting round,
  using the diagonal Hessian `p_k (1 - p_k)`.
- Store `n_classes_` and, for multiclass models, a per-class `base_score_`.
- Report sklearn classifier tags with `multi_class=True`.
- Export `hnbm.losses.Softmax`.
- Document `objective` and `objective_parameter` in the estimator docstrings and
  the README parameter tables.
- Document that the pseudo-Huber `delta` must match the residual scale. The
  Newton working response grows like `residual³ / delta²`, so the default
  `delta=1.0` diverges on targets that are not roughly unit-scale.

### Fixed

- Raise `ValueError` when a classification target contains a single class.
  Previously such a fit produced a `predict_proba` with two columns while
  `classes_` held one label, and `evaluate` raised from `log_loss`.
- Stop a failed `fit` from leaving behind the transient class count used while
  boosting. A failed multiclass fit could make an already-fitted binary model
  report `Softmax` from `loss_`.
- Bind the greedy candidate closure explicitly instead of capturing loop
  variables, removing a latent hazard if candidate fitting is ever deferred.
- Truncate every `history_` list along with the ensemble when early stopping
  rolls back to `best_iteration_`. `history_` previously kept the rounds that
  were discarded, so it ran longer than `ensemble_` and `n_iter_` and silently
  misaligned any per-round series plotted against them.
- Run callbacks for the round that triggers early stopping. The loop used to
  break before the callback block, so a callback accumulating per-round state
  was missing its final entry.
- Correct the softmax Hessian attribution in MATH.md. The text claimed the
  diagonal `p_k (1 - p_k)` is what XGBoost and LightGBM use; they inflate the
  same diagonal by 2 and by `K / (K - 1)` respectively. The implementation is
  unchanged, so the multiclass working response is twice XGBoost's and
  `learning_rate` is a stronger control for multiclass than for binary. This is
  now stated in MATH.md 4.2.1 and in the `Softmax` docstring.

### Compatibility

- Binary `decision_function` remains shape `(n_samples,)`. Multiclass returns
  `(n_samples, n_classes)`.
- `predict_proba` is `(n_samples, n_classes)` for both binary and multiclass.
- Multiclass `ensemble_` entries are length-`n_classes` lists of scalar
  learners. Binary and regression ensembles remain one learner per round.
- Multilabel and multioutput targets still raise `ValueError`.
- Single-class targets now raise `ValueError` instead of fitting a degenerate
  model. This supersedes the binary-only tagging described under 1.0.0.

### Packaging and tooling

- Add PyPI classifiers.
- Ship `LICENSE`, `MATH.md`, and `CONTRIBUTING.md` in the sdist.
- Add `ruff`, `mypy`, `pytest`, and `coverage` configuration, and run lint and
  type checks in CI alongside a wider Python test matrix.

## 1.1.0

Release date: 2026-09-01

- Pass original labels to `eval_metric`, including string class names.
- Accept `eval_sample_weight` for validation loss and early stopping.
- Add `staged_predict`, `staged_predict_proba`, `staged_decision_function`,
  and `permutation_importance`.

## 1.0.1

Release date: 2026-08-30

- Raise a sklearn-compatible error when every `sample_weight` is zero
  (`"at least one non-zero number"`).

## 1.0.0

Release date: 2026-08-30

- Freeze the public `HNBMClassifier` / `HNBMRegressor` constructor surface.
- Report sklearn 1.6+ estimator tags: binary-only classification, dense
  inputs, no native missing values.
- Delay parameter validation until `fit`, matching the sklearn estimator
  contract (`init` / `set_params` no longer raise on invalid values).
- Raise sklearn-compatible errors for multiclass targets and feature-count
  mismatches at predict time.
- Deprecate constructing `HNBM(mode=...)` and `NNBoost(mode=...)` directly.
- Add `check_estimator` coverage for tree-backed classifier and regressor
  subclasses.
- Mark the package as typed (`py.typed`).

### Compatibility

- Invalid constructor values are stored and rejected at `fit` instead of at
  construction. Callers that relied on `__init__` raising should catch the
  error from `fit`.
- Binary classifiers now raise `ValueError` matching
  `"Only binary classification is supported."`
- Requires scikit-learn 1.0 or newer; sklearn 1.6+ is recommended for the
  full tag-based estimator checks.

## 0.3.1

Release date: 2026-08-30

- Record `feature_names_in_` and reject dataframes whose columns are reordered
  or renamed between fit and predict.
- Require `eval_set` feature names to match the training data.

### Compatibility

- Predicting on a dataframe whose columns are reordered or renamed relative to
  fit now raises `ValueError` instead of silently returning wrong values.
  Mixing dataframe and array inputs across fit and predict warns.
- Estimators fitted on arrays are unaffected.

## 0.3.0

Release date: 2026-08-11

- Add optimized regression and classification base scores.
- Add top-level sample weights and validation sets.
- Add validation history, early stopping, and best-ensemble restoration.
- Add random and greedy learner-selection strategies.
- Add opt-in pseudo-Huber and quantile regression objectives.
- Add custom per-round metrics and callback-based orderly stopping.
- Add optional threaded greedy-candidate fitting.
- Add explicit post-fit learner compaction.
- Record per-round learner contribution weights in training history.
- Add optional per-round line search and stored learner weights.
- Add deterministic round-specific learner seeds and row subsampling.
- Restrict subsampling to observations with positive effective Hessian weight.
- Respect zero learner probabilities during greedy candidate selection.
- Reject negative seeds during parameter validation.
- Validate validation-set prediction shape and finiteness atomically.
- Preserve prediction compatibility with models serialized before 0.3.0.

### Compatibility

- Existing constructors and the original random-selection behavior remain the
  defaults.
- Models serialized before 0.3.0 fall back to the historical zero base score
  and global learning rate when new fitted attributes are absent.
- `NNBoost`, `NNBoostClassifier`, and `NNBoostRegressor` now expose all adaptive
  HNBM controls for cloning and hyperparameter search.
