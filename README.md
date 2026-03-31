# Gruve-MJ: Diabetic Readmission Classification (AutoGluon + Explainability + App)

## Grader Quick Start (Single Command)
First-time setup is required once (uv environment + dependencies):

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

Then grading inferencing runs with a single command and no code edits:

```bash
./run.sh
```

This executes the final inference pipeline (`src/eda_infer.py`) on `unseen_data.csv` and writes predictions to:
- `outputs/predictions/unseen_predictions.csv`

Live deployed app (no local setup required for evaluator):
- `https://huggingface.co/spaces/QuantCat/Gruve-MJ?logs=container`
- Evaluator can upload `unseen_data.csv` directly in the web UI and run prediction/explainability.

## Project Overview
This repository implements the full assignment pipeline for multiclass diabetic readmission prediction (`readmitted` in `<30`, `>30`, `NO`), including:
- strict split policy (`5% unseen` + `95% working`, then stratified `80/20`),
- EDA with modeling-oriented insights,
- AutoGluon training/evaluation on hold-out test,
- reproducible inferencing for graders,
- explainability (SHAP + LLM/VLM),
- deployable full-stack app (CSV upload + prediction + attribution + chat).

Primary raw inputs:
- `data/raw/diabetic_data.csv`
- `data/raw/IDS_mapping.csv`

## Environment Setup (uv + Python 3.12)
```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
python --version
```

Expected:
- Python `3.12.x`
- dependencies installed from pinned `requirements.txt`

`requirements.txt` was generated from the uv-managed environment:
```bash
uv pip freeze > requirements.txt
```

## Repository Structure
```text
gruve-mj/
├── README.md
├── requirements.txt
├── unseen_data.csv
├── leaderboard.csv
├── ag_models/                               # final trained AutoGluon predictor artifacts
├── data/
│   ├── raw/
│   ├── processed/
│   │   ├── cleaned_data.csv
│   │   ├── train.csv
│   │   ├── test.csv
│   │   └── unseen_data.csv
├── src/
│   ├── prepare_data.ipynb                   # Task 2 data loading/mapping/cleaning/splitting
│   ├── eda.ipynb                            # Task 3 EDA + advanced visuals/stat tests
│   ├── eda_train.py                         # Task 4 final training + hold-out evaluation
│   ├── eda_infer.py                         # one-command reproducible inferencing
│   └── explainability.py                    # B2 SHAP + LLM/VLM reporting
├── outputs/
│   ├── eda/
│   ├── training/
│   └── explainability/
├── app/
│   └── Gruve-MJ/                            # B3 full-stack Space app
└── experiments/                             # archived non-final lineages
```

## Running the Training Script
Run in this order.

1. Data preparation (`Task 2`)
```bash
jupyter nbconvert --to notebook --execute src/prepare_data.ipynb --inplace
```

2. EDA (`Task 3`)
```bash
jupyter nbconvert --to notebook --execute src/eda.ipynb --inplace
```

3. Final model training (`Task 4`)
```bash
python src/eda_train.py \
  --train-path data/processed/train.csv \
  --test-path data/processed/test.csv \
  --model-path ag_models \
  --leaderboard-path leaderboard.csv \
  --output-dir outputs/training \
  --eval-metric f1_macro \
  --presets best_quality \
  --time-limit 7200
```

Saved outputs include:
- `leaderboard.csv`
- `outputs/training/metrics.json`
- `outputs/training/classification_report.csv`
- `outputs/training/confusion_matrix.csv`
- `outputs/training/confusion_matrix_heatmap.png`
- `outputs/training/classwise_f1_bar.png`
- `outputs/training/classwise_recall_bar.png`

## Running Inferencing (Grader, Single Command)
The grader can run inferencing without code edits:

```bash
python src/eda_infer.py \
  --train-path data/processed/train.csv \
  --input-path data/processed/unseen_data.csv \
  --output-path outputs/predictions/unseen_predictions.csv \
  --model-path ag_models
```

This command:
- loads the saved predictor,
- reconstructs feature engineering state from training split,
- predicts on unseen data,
- exports prediction table (`outputs/predictions/unseen_predictions.csv`).

## Model Performance
Official reporting must use the hold-out set (`data/processed/test.csv`) only.

Reference files:
- `leaderboard.csv` (ranked models)
- `outputs/training/metrics.json` (accuracy, macro F1, macro recall, `<30` recall)
- `outputs/training/classification_report.csv` (per-class precision/recall/F1)
- `outputs/training/confusion_matrix.csv`

Current final lineage:
- training script: `src/eda_train.py`
- model artifacts: `ag_models/`
- leaderboard: `leaderboard.csv`

## Model Artifact Download (Large Files)
Due to repository upload size constraints, the full model directory `ag_models/models` can be downloaded from:

- https://drive.google.com/drive/folders/1_CDHzAwUJq9MCvvvmIz-2hH7CjXrveJO?usp=sharing

After download, place it under:
- `ag_models/models`

Then `./run.sh` and `src/eda_infer.py` will work with the same final model lineage.

## EDA-to-Modeling Analysis Summary
Key modeling decisions derived from EDA:
- Severe class imbalance in `readmitted` motivated explicit imbalance handling
  (minority oversampling + class-sensitive sample weighting).
- High-cardinality and sparse clinical/admin fields motivated rare-category grouping
  and missingness-observed indicator design.
- Diagnosis code heterogeneity motivated ICD-family aggregation features to improve
  generalization and interpretability.
- Utilization/length-of-stay/medication interactions motivated composite features
  used in the final AutoGluon training pipeline.
- Dimensionality structure from numeric variables motivated PCA + KMeans-derived
  features used as additional model signals.

## Dependencies
- Python: `3.12.x`
- Environment/tooling: `uv`
- Core libraries: AutoGluon, pandas, numpy, scikit-learn, matplotlib, seaborn, plotly, shap, openai, fastapi, uvicorn
- Exact pinned package versions: `requirements.txt`

## Rubric Alignment Checklist (Mandatory 100%)
### 5.2 Python Environment & Inferencing
- uv + Python 3.12 instructions: provided above.
- pinned requirements from uv freeze: `requirements.txt`.
- single-command inferencing: `src/eda_infer.py` command documented above.

### 5.3 Data Processing & EDA
- ID mapping and loading: `src/prepare_data.ipynb` loads both CSVs and applies mapping for `admission_type_id`, `discharge_disposition_id`, `admission_source_id`.
- missing values (`?`): detected/replaced/documented in preparation and EDA notebooks.
- stratified split policy: 5% unseen + remaining 95% split to stratified 80/20 train/test.
- descriptive stats + distributions + charts + imbalance discussion + modeling implications: `src/eda.ipynb` and `outputs/eda/`.

### 5.4 ML Training, Performance & Artefacts
- AutoGluon TabularPredictor: `src/eda_train.py`.
- leaderboard persisted: `leaderboard.csv`.
- hold-out metrics exported: `outputs/training/*`.
- loadable model artifacts: `ag_models/`.
- reproducible inferencing: `src/eda_infer.py`.

### 5.5 Code Annotation & Explanations
- non-trivial logic commented in scripts.
- function/class docstrings included in Python pipelines.
- notebook/script narrative included for each stage.

## Bonus Coverage
### B1 — Advanced EDA & Visualisations
- interactive plots (Plotly), PCA/t-SNE/hypothesis-testing sections, and modeling-linked interpretations are in `src/eda.ipynb`.

### B2 — LLM/VLM Explainability
Run:
```bash
export OPENAI_API_KEY="..."
python src/explainability.py \
  --input data/processed/test.csv \
  --train-path data/processed/train.csv \
  --model-path ag_models \
  --output-dir outputs/explainability \
  --llm-model gpt-4o-mini
```
Outputs include SHAP global/local tables + LLM/VLM summaries.

### B3 — Full-Stack Application
- path: `app/Gruve-MJ`
- functionality: CSV upload, predictions + confidence, local attributions, chat
- live Space repo: `https://huggingface.co/spaces/QuantCat/Gruve-MJ`
- grader usage: upload `unseen_data.csv` in the app and run inferencing without local environment setup

### B4 — Candidate Innovation
Innovation components include:
- EDA-informed domain feature engineering (diagnosis family features, utilization/medication composites),
- train-only feature-state reconstruction for robust inference parity,
- integrated explainability-to-chat workflow in the deployment app.
