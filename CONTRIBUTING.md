# Contributing to HNBM

## Development

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

The sklearn estimator checks live in `tests/test_sklearn_checks.py` and run as
part of the default suite. They are slower than unit tests; keep them passing.

## Semantic versioning

HNBM 1.0 freezes `HNBMClassifier` and `HNBMRegressor`:

- Defaults of the random Newton-boosting algorithm stay.
- New constructor arguments are additive and default to the current behavior.
- Removing or renaming a public parameter requires a deprecation cycle and a
  major version.
- `HNBM(mode=...)` and `NNBoost(mode=...)` are deprecated; they will be
  removed in 2.0.

Fitted attributes `ensemble_`, `n_iter_`, `base_score_`, `learner_weights_`,
`history_`, `best_iteration_`, `n_features_in_`, `feature_names_in_`,
`classes_`, and `n_classes_` are part of the 1.0 contract. Multiclass
`ensemble_` entries are per-class learner lists; binary and regression entries
remain a single learner.

## Release checklist

1. Tests pass, including `check_estimator`.
2. Changelog section is dated (not "staged").
3. `__version__` matches the GitHub release tag (`v1.x.y`).
4. SnapBoost pins `hnbm>=1.2` before a SnapBoost 1.2 release that depends on
   this contract.
