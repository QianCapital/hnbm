# Changelog

## 0.3.0

Release date: staged

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
