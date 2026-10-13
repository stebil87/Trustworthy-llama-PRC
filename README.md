# PRC Data Challenge 2026 — Taxi-Out Time Prediction

## Overview

This repository documents our solution to the EUROCONTROL Performance Review Commission (PRC) Data Challenge 2026.

The objective is to predict aircraft taxi-out duration, measured in seconds, for departures at ten European airports. The training data contain movements from 2025; the final prediction dataset includes January, February, June and July 2026.

The solution combines a general prediction model with airport-specific tail modelling, operational congestion features and a second-stage CatBoost residual model.

## Methodology

### 1. Data preparation

We use the official challenge movement and flight data. Departure records are identified by `MVT_ID_mvt`, and the prediction target is `TAXITIME_SEC_mvt`.

The preprocessing stage checks data schemas, timestamps, chronological boundaries, missing values and the integrity of the final submission identifiers.

### 2. Operational feature engineering

Features incorporate available movement and flight information, airport and runway context, temporal patterns, and congestion proxies constructed from eligible observations.

The development process explicitly examines feature availability and the risk of temporal leakage.

### 3. Specialized tail modelling

Taxi-out observations exhibit substantial heterogeneity and long-tailed behaviour.

The modelling pipeline includes specialized treatments for unusually long durations and atypical operational regimes, including clock-related anomalies observed at LIRF.

These components are intended to reduce errors on rare but high-impact observations, which are particularly important under squared-error evaluation.

### 4. Residual modelling

A CatBoost residual learner complements the first-stage predictors.

Its training targets are obtained from out-of-fold first-stage predictions rather than in-sample predictions, to reduce overfitting and information leakage.

The final configuration uses the selected CatBoost development parameters and a residual blending weight of 0.5.

### 5. Final fitting and prediction

The final production pipeline fits on valid labelled departures from the complete 2025 dataset and predicts the official 2026 departure set.

The submitted file contains exactly two fields:

| Field | Meaning |
|---|---|
| `MVT_ID_mvt` | Unique movement identifier |
| `TAXITIME_SEC_mvt` | Predicted taxi-out duration, in seconds |

The submitted predictions are finite and nonnegative and are aligned with the official submission template.

## Validation methodology

We investigated two complementary historical evaluation setups:

1. Chronological out-of-fold evaluation across ten months of 2025.
2. An analogous seasonal evaluation withholding January, February, June and July 2025.

The historical experiments informed feature design, specialized models, residual learning and post-processing decisions.

These historical sets were examined repeatedly during development, so their scores should **not** be interpreted as independent estimates of final 2026 performance.

For the selected residual-model configuration, the retrospective seasonal benchmark achieved approximately **377.62 seconds RMSE**, compared with **381.98 seconds** for the preceding C28 benchmark at the same evaluation setup.

This is not a measured RMSE for the final 2026 prediction dataset. The actual 2026 targets are unavailable to participants.

## Reproduction

### Requirements

- Python 3.11 or a compatible Python environment
- Dependencies listed in `requirements.txt`
- Official PRC Data Challenge 2026 Parquet files

Place the official monthly 2025 training files, `final_ranking.parquet`, and `final_submitting.parquet` inside a local `data/` directory.

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the production pipeline from a working directory containing its imported Python modules:

```bash
python 37_production_full_2025_to_2026_FIXED.py \
  --mode final \
  --authorize-final-fit \
  --data-dir data \
  --ranking final_ranking.parquet \
  --template final_submitting.parquet \
  --threads 4
```

The final predictions are written to:

```text
campaign37_final/PRC_2026_final_predictions_READY_FOR_MANUAL_SUBMISSION.parquet
```

The pipeline also writes a final fit manifest, including data and output checksums.

For exact reproduction, use the same source-code revision, dependency versions and official input files.

## Submission integrity

The verified final prediction artifact contained:

- 670,790 departures.
- The required movement identifiers.
- Finite and nonnegative predictions.
- No missing or additional prediction rows.

**SHA256**

```text
a9b8d0e4bea9fd615e8703b00ea6f7fbf56d1aa89ccf476514b93e432ff7f831
```

The SHA256 identifies the validated production artifact; renaming the file does not change it.

## Research and limitations

We investigated alternative calibration and post-processing methods, as well as an independent direct-prediction CatBoost challenger. These alternatives did not demonstrate sufficiently robust improvements to replace the selected production pipeline.

Performance is especially sensitive to rare extreme observations and data-quality anomalies. The historical validation results must therefore be interpreted with care.

The repository distinguishes production code from experimental research scripts.

## Data availability

Official challenge data are available through the PRC Data Challenge 2026 distribution mechanisms, subject to the organizers' access and licensing conditions.

The full datasets are not redistributed in this repository. No private access credentials should be committed.

Challenge website: https://prc-data-challenge-2026.netlify.app/

## License

The source code is released under the GNU General Public License, version 3 (GPL-3.0).

See `LICENSE` for the complete license text.

## Acknowledgements

We acknowledge EUROCONTROL PRC, OpenSky Network and the organizers of the PRC Data Challenge 2026 for providing the challenge and its data infrastructure.
