# PRC Data Challenge 2026 — Taxi-out prediction

This repository contains the **minimal readable source-code dependency chain** for the model used to generate our final 2026 taxi-out predictions. It contains **10 original Python files** (not a compressed wrapper). The files' executable source is unmodified from the archived production release.

## Model

The target is `TAXITIME_SEC_mvt` (taxi-out duration in seconds) for departure movements identified by `MVT_ID_mvt`.

The production model combines:

1. A first-stage airport-aware predictor and specialized long-tail models developed in C21–C23.
2. Operational congestion/flow features and the dedicated LIRF bad-clock specialist inherited through C26–C28.
3. A CatBoost second-stage **residual regressor**, trained on cross-fitted first-stage errors using operational and categorical features supplied by C32/C33/C35.
4. A final fit of the first-stage model using all valid labelled departures from 2025, followed by the frozen second-stage correction.

**Frozen final settings:** C28 flow weight `1.0`, CatBoost residual blend `0.5`, CatBoost depth `8`, learning rate `0.075`, L2 regularization `15`, `1100` trees. The production code is the authoritative specification.

The code includes checks for data boundaries, feature availability, movement-ID alignment, finite and nonnegative predictions, and output integrity. No predictions are uploaded by the code.

## Repository structure

All 10 `.py` files **must remain next to each other**, at the repository root. The pipeline loads them by original filename and certain dependencies verify their SHA256 hashes. **Do not rename, reformat, or modify the original Python modules** when reproducing the submitted solution.

| File | Role |
| --- | --- |
| `21_clock_aware_tail_research.py` | First-stage architecture, features and long-tail experts |
| `22_bad_clock_regime_research.py` | Clock-anomaly specialist |
| `23_queueing_operations_research_lab.py` | Operational queue/flow modelling |
| `26_ranking_month_block_stress.py` | Parent feature / inference utilities |
| `27_lirf_clock_expert_selection.py` | LIRF specialist selection |
| `28_lirf_flow_expert_tuning.py` | Frozen specialist flow integration |
| `32_crossfit_residual_lgb_research.py` | Residual feature and cross-fit utilities |
| `33_catboost_residual_challenger.py` | CatBoost residual learning components |
| `35_overnight_model_search.py` | Selected residual model implementation and features |
| `37_production_full_2025_to_2026_FIXED.py` | **Production entry point**: audit, fit and 2026 inference |

The `research` names are historical; these modules contain functions that are **executed by the final model**. No separate experimental notebooks or campaign scripts are necessary for the production run.

## Requirements

Use an isolated environment (the original production environment's exact versions should be recorded if available).

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The requirements file lists the packages used by this code. For exact environment replication, dependency **versions** must be pinned to those installed during the submitted run; the unpinned list alone is not a full environment lock.

## Official data

Download the official PRC Data Challenge 2026 Parquet datasets from the organizers, observing their distribution terms. Put the 12 files `training_2025-*.parquet`, plus `final_ranking.parquet` and `final_submitting.parquet`, in `data/` relative to the repository root.

Do **not** commit official source Parquet data, confidential credentials, local environments or generated submission artifacts unless explicitly allowed.

## Reproduce the submitted predictor

Run from the repository root:

```powershell
python 37_production_full_2025_to_2026_FIXED.py --self-test
python 37_production_full_2025_to_2026_FIXED.py --mode audit --threads 4 --data-dir data --ranking final_ranking.parquet --template final_submitting.parquet
python 37_production_full_2025_to_2026_FIXED.py --mode final --authorize-final-fit --threads 4 --data-dir data --ranking final_ranking.parquet --template final_submitting.parquet
```

`--mode audit` reads/validates the official files **without fitting a model**. `--mode final --authorize-final-fit` retrains on 2025 and writes:

```text
campaign37_final/PRC_2026_final_predictions_READY_FOR_MANUAL_SUBMISSION.parquet
campaign37_final/final_fit_manifest.json
```

The final Parquet has exactly two fields: `MVT_ID_mvt`, `TAXITIME_SEC_mvt`; it contains 670,790 rows. The audited original artifact's SHA256 is:

```text
a9b8d0e4bea9fd615e8703b00ea6f7fbf56d1aa89ccf476514b93e432ff7f831
```

The hash documents the actual submitted-candidate artifact, **not** a guarantee that every operating system and third-party package version will produce byte-for-byte identical Parquet serialization.

## Validation and interpretation

During development, the analogous January/February/June/July 2025 block contained 671,118 departure records. In a retrospective comparison, the selected C35 residual blend (`0.5`) had an RMSE of approximately **377.62 seconds**, versus **381.98 seconds** for the C28 baseline on the same rows.

The four-month 2025 block and other historical folds were examined repeatedly during research, so they should **not** be described as an untouched final holdout. The true 2026 target values are not available to participants, so the 2026 RMSE cannot be computed from this repository.

## License

GNU General Public License v3.0. See the repository's existing `LICENSE` file.

## Source data acknowledgement

PRC Data Challenge 2026 / EUROCONTROL. Official data source and challenge conditions: https://prc-data-challenge-2026.netlify.app/
