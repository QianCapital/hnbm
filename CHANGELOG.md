# Changelog

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
