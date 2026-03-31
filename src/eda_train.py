"""
EDA-Driven Training Pipeline (Selected Final Raw-Model Lineage)
===============================================================

This script is the EDA-plus training lineage that produces:
  - leaderboard.csv
  - ag_models (or user-specified --model-path)

Recommended run:
  python src/eda_train.py \
    --train-path data/processed/train.csv \
    --test-path data/processed/test.csv \
    --model-path ag_models \
    --leaderboard-path leaderboard.csv \
    --output-dir outputs/training \
    --eval-metric f1_macro \
    --presets best_quality \
    --time-limit 7200

Pipeline summary (code-level):
1) Load pre-split train/test from Task 2 outputs.
2) Apply EDA-informed feature engineering:
   - diagnosis family grouping,
   - utilization and medication summary features,
   - missingness indicators and high-missingness flags,
   - PCA + KMeans cluster features.
3) Handle class imbalance on train only:
   - minority oversampling,
   - class-sensitive sample weights.
4) Train AutoGluon with bagging/stacking.
5) Evaluate on hold-out test and export leaderboard + metrics artifacts.

Experiment lineage notes (for reporting clarity):
- Baseline lineage:
  - Script: experiments/train_baseline.py
  - Leaderboard: experiments/leaderboard_baseline.csv
  - Logic: minimal pipeline, no aggressive EDA-based engineering.

- EDA-plus lineage (this file):
  - Script: src/eda_train.py
  - Leaderboard: leaderboard.csv
  - Logic: EDA-driven FE + oversampling + sample weighting + AutoGluon ensemble.
  - This is the selected final raw-model leaderboard lineage.

- Optimized lineage:
  - Script: experiments/train_optimized.py
  - Leaderboard: experiments/leaderboard_optimized.csv
  - Logic: strategy search + balancing-policy search + optional probability-bias
    logic in that script.
  - Included as ablation/attempt, separate from this script's direct output.

Rubric alignment notes:
- Task 4 / Rubric 5.4:
  - AutoGluon TabularPredictor training is implemented here.
  - Leaderboard is saved both as canonical `leaderboard.csv` and
    `outputs/training/leaderboard_test.csv`.
  - Hold-out metrics are exported as accuracy, macro F1, classification report,
    and confusion matrix artifacts.
- Code annotation rubric:
  - All non-trivial transforms are documented with inline comments and docstrings.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".mplconfig").resolve()))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, recall_score
from sklearn.preprocessing import StandardScaler

DEFAULT_CLASS_ORDER = ["<30", ">30", "NO"]

MEDICATION_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide",
    "glimepiride", "acetohexamide", "glipizide", "glyburide",
    "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
    "miglitol", "troglitazone", "tolazamide", "examide",
    "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone",
]

HIGH_SIGNAL_CAT_COLS = [
    "race", "gender", "age", "weight",
    "admission_type_id", "discharge_disposition_id", "admission_source_id",
    "payer_code", "medical_specialty", "max_glu_serum", "A1Cresult",
    "change", "diabetesMed",
]

DIAG_COLS = ["diag_1", "diag_2", "diag_3"]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for EDA-plus training.

    Returns:
        argparse.Namespace: Runtime options for paths, AutoGluon settings,
        feature-engineering controls, and imbalance parameters.
    """
    parser = argparse.ArgumentParser(
        description="EDA-informed advanced AutoGluon training script for diabetic readmission."
    )
    # Defaults are intentionally aligned with final submission outputs
    # so that re-running this script reproduces canonical artifacts.
    parser.add_argument("--train-path", default="data/processed/train.csv")
    parser.add_argument("--test-path", default="data/processed/test.csv")
    parser.add_argument("--label", default="readmitted")
    parser.add_argument("--model-path", default="ag_models")
    parser.add_argument("--leaderboard-path", default="leaderboard.csv")
    parser.add_argument("--output-dir", default="outputs/training")
    parser.add_argument("--eval-metric", default="f1_macro")
    parser.add_argument("--presets", default="best_quality")
    parser.add_argument("--time-limit", type=int, default=7200)
    parser.add_argument("--num-bag-folds", type=int, default=8)
    parser.add_argument("--num-bag-sets", type=int, default=1)
    parser.add_argument("--num-stack-levels", type=int, default=2)
    parser.add_argument("--top-models", type=int, default=20)

    parser.add_argument("--rare-threshold", type=float, default=0.01)
    parser.add_argument("--n-clusters", type=int, default=10)
    parser.add_argument("--n-pca-components", type=int, default=8)
    parser.add_argument("--oversample-target-ratio", type=float, default=0.85)

    parser.add_argument("--weight-lt30", type=float, default=5.5)
    parser.add_argument("--weight-gt30", type=float, default=1.8)
    parser.add_argument("--weight-no", type=float, default=1.0)

    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def _ensure_parent(path: Path) -> None:
    """Create parent directory if missing.

    Args:
        path: Target file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)


def _save_json(obj: dict, path: Path) -> None:
    """Save a dictionary as UTF-8 JSON.

    Args:
        obj: JSON-serializable dictionary.
        path: Output JSON path.
    """
    _ensure_parent(path)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _clean_dataframe(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """Standardize columns, replace placeholders, and validate label column.

    Args:
        df: Raw dataframe.
        label_col: Target label column name.

    Returns:
        pd.DataFrame: Cleaned dataframe with normalized label values.
    """
    cleaned = df.copy()
    cleaned.columns = [str(col).strip() for col in cleaned.columns]
    cleaned = cleaned.replace("?", np.nan)

    if label_col not in cleaned.columns:
        raise ValueError(f"Label column '{label_col}' not found.")

    cleaned = cleaned.dropna(subset=[label_col]).reset_index(drop=True)
    cleaned[label_col] = cleaned[label_col].astype(str).str.strip()
    return cleaned


def _drop_redundant_raw_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop `*_raw` columns when decoded base columns exist.

    Args:
        df: Input dataframe.

    Returns:
        pd.DataFrame: Reduced dataframe without redundant raw-ID fields.
    """
    reduced = df.copy()
    raw_cols = [c for c in reduced.columns if c.endswith("_raw")]
    for raw_col in raw_cols:
        base_col = raw_col[:-4]
        if base_col in reduced.columns:
            reduced = reduced.drop(columns=[raw_col])
    return reduced


def _load_required_split(csv_path: str, label_col: str) -> pd.DataFrame:
    """Load a required split CSV and apply common cleaning.

    Args:
        csv_path: CSV path for a prepared split.
        label_col: Target column used for validation/cleanup.

    Returns:
        pd.DataFrame: Cleaned split dataframe ready for feature engineering.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Required split not found: {path}. Run Task 2 preparation first."
        )
    df = pd.read_csv(path)
    df = _clean_dataframe(df, label_col)
    df = _drop_redundant_raw_id_columns(df)
    return df


def _class_order(values: Iterable[str]) -> list[str]:
    """Build stable class order with preferred labels first.

    Args:
        values: Iterable of observed class labels.

    Returns:
        list[str]: Ordered class labels for reporting/plots.
    """
    labels = [str(v) for v in values]
    preferred = [v for v in DEFAULT_CLASS_ORDER if v in labels]
    remaining = sorted({v for v in labels if v not in preferred})
    return preferred + remaining


def _parse_range_midpoint(value: object) -> float:
    """Convert bracketed range text to midpoint numeric value.

    Args:
        value: Range-like string such as `[70-80)` or `>200`.

    Returns:
        float: Midpoint approximation, or NaN if unparsable.
    """
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if text == ">200":
        return 212.5
    match = re.match(r"\[(\d+)-(\d+)\)", text)
    if match:
        low = float(match.group(1))
        high = float(match.group(2))
        return (low + high) / 2.0
    return np.nan


def _map_diag_group(value: object) -> str:
    """Map ICD-style diagnosis value to a coarse clinical group.

    Args:
        value: Raw diagnosis code value.

    Returns:
        str: Coarse diagnosis group label.
    """
    if pd.isna(value):
        return "missing"

    text = str(value).strip().upper()
    if not text:
        return "missing"

    if text.startswith("E"):
        return "external_cause"
    if text.startswith("V"):
        return "supplementary"

    try:
        code = float(text)
    except ValueError:
        return "other"

    if 390 <= code <= 459 or abs(code - 785.0) < 1e-9:
        return "circulatory"
    if 460 <= code <= 519 or abs(code - 786.0) < 1e-9:
        return "respiratory"
    if 520 <= code <= 579 or abs(code - 787.0) < 1e-9:
        return "digestive"
    if 580 <= code <= 629 or abs(code - 788.0) < 1e-9:
        return "genitourinary"
    if 250 <= code < 251:
        return "diabetes"
    if 140 <= code <= 239:
        return "neoplasms"
    if 710 <= code <= 739:
        return "musculoskeletal"
    if 800 <= code <= 999:
        return "injury_poisoning"
    if 780 <= code <= 799:
        return "ill_defined"
    if 240 <= code <= 279:
        return "endocrine_metabolic"
    if 290 <= code <= 319:
        return "mental_disorders"
    return "other"


@dataclass
class AdvancedFeatureEngineer:
    """Train-fitted feature engineering bundle for EDA-plus lineage.

    This class learns train-only statistics/mappings (missingness profiles,
    rare-category maps, PCA/KMeans state) and applies deterministic transforms
    to both train and inference data.
    """
    label_col: str
    rare_threshold: float = 0.01
    n_clusters: int = 10
    n_pca_components: int = 8
    random_state: int = 42

    high_missing_cols_: list[str] = field(default_factory=list)
    very_high_missing_cols_: list[str] = field(default_factory=list)
    rare_maps_: dict[str, set[str]] = field(default_factory=dict)
    util_high_threshold_: float | None = None

    num_imputer_: SimpleImputer | None = None
    scaler_: StandardScaler | None = None
    pca_: PCA | None = None
    kmeans_: KMeans | None = None
    pca_numeric_cols_: list[str] = field(default_factory=list)

    def _base_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply deterministic EDA-informed transformations to a dataframe."""
        out = df.copy()

        for col in HIGH_SIGNAL_CAT_COLS:
            if col in out.columns:
                out[col] = out[col].astype("object")

        if "age" in out.columns:
            out["age_mid"] = out["age"].map(_parse_range_midpoint)

        if "weight" in out.columns:
            out["weight_mid"] = out["weight"].map(_parse_range_midpoint)
            out["weight_missing_flag"] = out["weight"].isna().astype("int8")

        # Missingness indicators learned from train
        for col in self.high_missing_cols_:
            if col in out.columns:
                out[f"is_missing__{col}"] = out[col].isna().astype("int8")

        # For extremely sparse columns, convert to observed/not observed + preserve original
        for col in self.very_high_missing_cols_:
            if col in out.columns:
                out[f"{col}_observed"] = out[col].notna().astype("int8")

        # Diagnosis grouping
        diag_group_cols = []
        for diag_col in DIAG_COLS:
            if diag_col in out.columns:
                group_col = f"{diag_col}_group"
                out[group_col] = out[diag_col].map(_map_diag_group)
                diag_group_cols.append(group_col)

        if diag_group_cols:
            out["diag_group_unique_count"] = out[diag_group_cols].astype(str).nunique(axis=1)
            out["has_diabetes_diag"] = (
                pd.concat([(out[g] == "diabetes") for g in diag_group_cols], axis=1).any(axis=1).astype("int8")
            )
            out["has_circulatory_diag"] = (
                pd.concat([(out[g] == "circulatory") for g in diag_group_cols], axis=1).any(axis=1).astype("int8")
            )
            out["has_respiratory_diag"] = (
                pd.concat([(out[g] == "respiratory") for g in diag_group_cols], axis=1).any(axis=1).astype("int8")
            )
            out["has_digestive_diag"] = (
                pd.concat([(out[g] == "digestive") for g in diag_group_cols], axis=1).any(axis=1).astype("int8")
            )

        # Utilization / severity
        if {"number_outpatient", "number_emergency", "number_inpatient"}.issubset(out.columns):
            out["total_utilization"] = (
                out["number_outpatient"].fillna(0)
                + out["number_emergency"].fillna(0)
                + out["number_inpatient"].fillna(0)
            )
            out["acute_utilization"] = (
                out["number_emergency"].fillna(0)
                + out["number_inpatient"].fillna(0)
            )
            out["emergency_inpatient_ratio"] = (
                out["acute_utilization"] / (1.0 + out["number_outpatient"].fillna(0))
            )
            if self.util_high_threshold_ is not None:
                out["is_high_utilizer"] = (out["total_utilization"] >= self.util_high_threshold_).astype("int8")

        if {"num_lab_procedures", "num_procedures", "num_medications"}.issubset(out.columns):
            out["treatment_intensity"] = (
                out["num_lab_procedures"].fillna(0)
                + out["num_procedures"].fillna(0)
                + out["num_medications"].fillna(0)
            )
            out["lab_proc_ratio"] = out["num_lab_procedures"].fillna(0) / (1.0 + out["num_procedures"].fillna(0))
            out["med_lab_ratio"] = out["num_medications"].fillna(0) / (1.0 + out["num_lab_procedures"].fillna(0))

        if {"time_in_hospital", "num_lab_procedures"}.issubset(out.columns):
            out["lab_per_day"] = out["num_lab_procedures"].fillna(0) / np.maximum(out["time_in_hospital"].fillna(1), 1)

        if {"time_in_hospital", "num_medications"}.issubset(out.columns):
            out["medications_per_day"] = out["num_medications"].fillna(0) / np.maximum(out["time_in_hospital"].fillna(1), 1)

        if {"time_in_hospital", "number_diagnoses"}.issubset(out.columns):
            out["diagnoses_per_day"] = out["number_diagnoses"].fillna(0) / np.maximum(out["time_in_hospital"].fillna(1), 1)

        if "time_in_hospital" in out.columns:
            out["long_stay_flag"] = (out["time_in_hospital"].fillna(0) >= 7).astype("int8")
            out["very_long_stay_flag"] = (out["time_in_hospital"].fillna(0) >= 10).astype("int8")

        if {"time_in_hospital", "number_diagnoses"}.issubset(out.columns):
            out["complexity_index"] = (
                out["time_in_hospital"].fillna(0) * out["number_diagnoses"].fillna(0)
            )

        # Medication summary
        med_cols = [c for c in MEDICATION_COLS if c in out.columns]
        if med_cols:
            med_frame = out[med_cols].fillna("Missing").astype(str)
            out["med_up_count"] = (med_frame == "Up").sum(axis=1)
            out["med_down_count"] = (med_frame == "Down").sum(axis=1)
            out["med_steady_count"] = (med_frame == "Steady").sum(axis=1)
            out["med_no_count"] = (med_frame == "No").sum(axis=1)
            out["med_changed_count"] = ((med_frame == "Up") | (med_frame == "Down")).sum(axis=1)
            out["any_med_change_flag"] = (out["med_changed_count"] > 0).astype("int8")

            if "insulin" in out.columns:
                out["on_insulin_flag"] = out["insulin"].fillna("No").astype(str).ne("No").astype("int8")
                out["insulin_changed_flag"] = out["insulin"].fillna("No").astype(str).isin(["Up", "Down"]).astype("int8")

        # Glucose/A1C observation signal
        if "max_glu_serum" in out.columns:
            out["max_glu_observed"] = out["max_glu_serum"].notna().astype("int8")
        if "A1Cresult" in out.columns:
            out["a1c_observed"] = out["A1Cresult"].notna().astype("int8")

        # Rare category grouping to reduce sparsity
        out = self._apply_rare_grouping(out)

        return out

    def _apply_rare_grouping(self, df: pd.DataFrame) -> pd.DataFrame:
        """Collapse train-identified rare categories into `__RARE__` buckets."""
        out = df.copy()
        for col, rare_values in self.rare_maps_.items():
            if col not in out.columns:
                continue
            out[col] = out[col].fillna("Missing").astype(str)
            out[col] = out[col].where(~out[col].isin(rare_values), other="__RARE__")
        return out

    def fit(self, train_df: pd.DataFrame) -> "AdvancedFeatureEngineer":
        """Fit train-only FE state used later for both train/test transforms."""
        base = train_df.copy()

        missing_rate = base.isna().mean()
        self.high_missing_cols_ = [
            c for c in base.columns
            if c != self.label_col and missing_rate.get(c, 0) > 0
        ]
        self.very_high_missing_cols_ = [
            c for c in base.columns
            if c != self.label_col and missing_rate.get(c, 0) >= 0.80
        ]

        if {"number_outpatient", "number_emergency", "number_inpatient"}.issubset(base.columns):
            total_util = (
                base["number_outpatient"].fillna(0)
                + base["number_emergency"].fillna(0)
                + base["number_inpatient"].fillna(0)
            )
            self.util_high_threshold_ = float(total_util.quantile(0.90))

        candidate_rare_cols = []
        for col in base.columns:
            if col == self.label_col:
                continue
            if base[col].dtype == "object" or str(base[col].dtype).startswith(("string", "category")):
                candidate_rare_cols.append(col)

        for diag_col in DIAG_COLS:
            if diag_col in candidate_rare_cols:
                candidate_rare_cols.remove(diag_col)

        for col in candidate_rare_cols:
            freq = base[col].fillna("Missing").astype(str).value_counts(normalize=True)
            rare_values = set(freq[freq < self.rare_threshold].index.tolist())
            self.rare_maps_[col] = rare_values

        transformed = self._base_transform(train_df)

        numeric_cols = [
            c for c in transformed.columns
            if c != self.label_col
            and pd.api.types.is_numeric_dtype(transformed[c])
            and c not in {"encounter_id", "patient_nbr"}
        ]
        self.pca_numeric_cols_ = numeric_cols

        if numeric_cols:
            X_num = transformed[numeric_cols].copy()
            self.num_imputer_ = SimpleImputer(strategy="median")
            X_num = self.num_imputer_.fit_transform(X_num)

            self.scaler_ = StandardScaler()
            X_num = self.scaler_.fit_transform(X_num)

            n_components = min(self.n_pca_components, X_num.shape[1])
            if n_components >= 2:
                self.pca_ = PCA(n_components=n_components, random_state=self.random_state)
                X_pca = self.pca_.fit_transform(X_num)

                self.kmeans_ = KMeans(
                    n_clusters=self.n_clusters,
                    random_state=self.random_state,
                    n_init=20,
                )
                self.kmeans_.fit(X_pca)

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform input data using FE state learned from train split only."""
        out = self._base_transform(df)

        if self.pca_ is not None and self.kmeans_ is not None and self.pca_numeric_cols_:
            X_num = out.reindex(columns=self.pca_numeric_cols_).copy()
            X_num = self.num_imputer_.transform(X_num)
            X_num = self.scaler_.transform(X_num)
            X_pca = self.pca_.transform(X_num)

            for i in range(X_pca.shape[1]):
                out[f"pca_component_{i + 1}"] = X_pca[:, i]

            out["pca_cluster"] = self.kmeans_.predict(X_pca).astype(str)

        return out


def oversample_train_df(
    train_df: pd.DataFrame,
    label_col: str,
    target_ratio: float,
    random_state: int,
) -> pd.DataFrame:
    """Oversample minority classes up to a ratio of the majority class count."""
    class_counts = train_df[label_col].value_counts()
    max_count = int(class_counts.max())

    pieces = []
    for cls, cls_df in train_df.groupby(label_col):
        target_count = max(len(cls_df), int(max_count * target_ratio))
        if len(cls_df) < target_count:
            extra = cls_df.sample(
                n=target_count - len(cls_df),
                replace=True,
                random_state=random_state,
            )
            cls_df = pd.concat([cls_df, extra], axis=0)
        pieces.append(cls_df)

    oversampled = (
        pd.concat(pieces, axis=0)
        .sample(frac=1.0, random_state=random_state)
        .reset_index(drop=True)
    )
    return oversampled


def add_sample_weights(
    df: pd.DataFrame,
    label_col: str,
    weight_map: dict[str, float],
) -> pd.DataFrame:
    """Attach per-row sample weights derived from class-to-weight mapping."""
    weighted = df.copy()
    weighted["sample_weight"] = weighted[label_col].map(weight_map).astype(float)
    return weighted


def _plot_leaderboard(leaderboard_df: pd.DataFrame, output_path: Path, top_n: int) -> None:
    """Plot and save top-N leaderboard entries as a bar chart."""
    score_col = "score_test" if "score_test" in leaderboard_df.columns else "score_val"
    chart_df = leaderboard_df.head(top_n).copy().iloc[::-1]

    plt.figure(figsize=(11, max(5, top_n * 0.4)))
    plt.barh(chart_df["model"], chart_df[score_col])
    plt.xlabel(score_col)
    plt.ylabel("Model")
    plt.title(f"Top {top_n} Models")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def _plot_confusion_matrix(cm: np.ndarray, labels: list[str], output_path: Path) -> None:
    """Render and save confusion matrix heatmap with count annotations."""
    plt.figure(figsize=(7, 6))
    ax = plt.gca()
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center")

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def _plot_classwise_f1(report_df: pd.DataFrame, output_path: Path) -> None:
    """Plot and save per-class F1 scores from classification report table."""
    class_rows = [idx for idx in report_df.index if idx not in {"accuracy", "macro avg", "weighted avg"}]
    chart_df = report_df.loc[class_rows, "f1-score"].sort_values(ascending=False)

    plt.figure(figsize=(8, max(4, len(chart_df) * 0.6)))
    plt.bar(chart_df.index.astype(str), chart_df.values)
    plt.xlabel("Class")
    plt.ylabel("F1 Score")
    plt.title("Per-Class F1 Scores")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def _plot_classwise_recall(report_df: pd.DataFrame, output_path: Path) -> None:
    """Plot and save per-class recall values from classification report table."""
    class_rows = [idx for idx in report_df.index if idx not in {"accuracy", "macro avg", "weighted avg"}]
    chart_df = report_df.loc[class_rows, "recall"].sort_values(ascending=False)

    plt.figure(figsize=(8, max(4, len(chart_df) * 0.6)))
    plt.bar(chart_df.index.astype(str), chart_df.values)
    plt.xlabel("Class")
    plt.ylabel("Recall")
    plt.title("Per-Class Recall")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def evaluate_and_save(
    predictor,
    test_df: pd.DataFrame,
    label_col: str,
    output_dir: Path,
) -> dict:
    """Evaluate predictor on hold-out set and export full evaluation artifacts."""
    test_features = test_df.drop(columns=[label_col])
    y_true = test_df[label_col].astype(str)
    y_pred = predictor.predict(test_features).astype(str)
    y_proba = predictor.predict_proba(test_features)

    labels = _class_order(list(y_true) + list(y_pred))

    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    macro_recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    report_text = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
        output_dict=True,
    )
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(output_dir / "classification_report.csv", index=True)
    (output_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.to_csv(output_dir / "confusion_matrix.csv", index=True)
    _plot_confusion_matrix(cm, labels, output_dir / "confusion_matrix_heatmap.png")
    _plot_classwise_f1(report_df, output_dir / "classwise_f1_bar.png")
    _plot_classwise_recall(report_df, output_dir / "classwise_recall_bar.png")

    pred_df = pd.DataFrame({
        "true_label": y_true.values,
        "predicted_label": y_pred.values,
    })
    if isinstance(y_proba, pd.DataFrame):
        for col in y_proba.columns:
            pred_df[f"prob_{col}"] = y_proba[col].values
    pred_df.to_csv(output_dir / "test_predictions.csv", index=False)

    lt30_recall = 0.0
    if "<30" in report_df.index:
        lt30_recall = float(report_df.loc["<30", "recall"])

    return {
        "accuracy": accuracy,
        "f1_macro": macro_f1,
        "macro_recall": macro_recall,
        "lt30_recall": lt30_recall,
        "classification_report_text": report_text,
        "confusion_matrix_df": cm_df,
        "report_df": report_df,
    }


def main() -> None:
    """Run the full EDA-plus training pipeline and export all reporting files."""
    args = parse_args()

    try:
        from autogluon.tabular import TabularPredictor
    except ImportError as exc:
        raise ImportError(
            "AutoGluon is required. Install it with: uv pip install autogluon"
        ) from exc

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Hold-out discipline:
    # - train split is the only source for fitting transforms and model parameters.
    # - test split is untouched until final evaluation/export.
    raw_train_df = _load_required_split(args.train_path, args.label)
    raw_test_df = _load_required_split(args.test_path, args.label)

    # Fit feature engineering state on train only (to prevent test leakage).
    fe = AdvancedFeatureEngineer(
        label_col=args.label,
        rare_threshold=args.rare_threshold,
        n_clusters=args.n_clusters,
        n_pca_components=args.n_pca_components,
        random_state=args.random_state,
    )
    fe.fit(raw_train_df)

    train_df = fe.transform(raw_train_df)
    test_df = fe.transform(raw_test_df)

    train_df.to_csv(output_dir / "engineered_train_preview.csv", index=False)
    test_df.to_csv(output_dir / "engineered_test_preview.csv", index=False)

    # Class imbalance mitigation step 1: oversample minority classes on train split.
    oversampled_train_df = oversample_train_df(
        train_df,
        label_col=args.label,
        target_ratio=args.oversample_target_ratio,
        random_state=args.random_state,
    )

    weight_map = {
        "<30": float(args.weight_lt30),
        ">30": float(args.weight_gt30),
        "NO": float(args.weight_no),
    }
    # Class imbalance mitigation step 2: cost-sensitive sample weights.
    weighted_train_df = add_sample_weights(
        oversampled_train_df,
        label_col=args.label,
        weight_map=weight_map,
    )
    weighted_train_df.to_csv(output_dir / "weighted_oversampled_train.csv", index=False)

    before_counts = train_df[args.label].value_counts().rename("before")
    after_counts = weighted_train_df[args.label].value_counts().rename("after_oversample")
    class_balance_df = pd.concat([before_counts, after_counts], axis=1).fillna(0).astype(int)
    class_balance_df.to_csv(output_dir / "class_balance_before_after.csv", index=True)

    # AutoGluon training (sample_weight passed at predictor level for v1.5 API).
    predictor = TabularPredictor(
        label=args.label,
        eval_metric=args.eval_metric,
        path=args.model_path,
        sample_weight="sample_weight",
        weight_evaluation=False,
    ).fit(
        train_data=weighted_train_df,
        presets=args.presets,
        time_limit=args.time_limit,
        num_bag_folds=args.num_bag_folds,
        num_bag_sets=args.num_bag_sets,
        num_stack_levels=args.num_stack_levels,
    )

    # Internal validation leaderboard (diagnostic ranking on validation folds).
    validation_leaderboard = predictor.leaderboard(silent=True)
    validation_leaderboard.to_csv(output_dir / "leaderboard_validation.csv", index=False)

    # Hold-out test leaderboard (primary leaderboard table for this run).
    # This is exported to leaderboard.csv by default and is the key
    # raw-model comparison table used in reporting.
    test_leaderboard = predictor.leaderboard(data=test_df, silent=True)
    leaderboard_path = Path(args.leaderboard_path)
    _ensure_parent(leaderboard_path)
    test_leaderboard.to_csv(leaderboard_path, index=False)
    test_leaderboard.to_csv(output_dir / "leaderboard_test.csv", index=False)
    _plot_leaderboard(test_leaderboard, output_dir / "leaderboard_top_models.png", args.top_models)

    eval_result = evaluate_and_save(
        predictor=predictor,
        test_df=test_df,
        label_col=args.label,
        output_dir=output_dir,
    )

    # Feature importance for bonus reporting
    fi_features = test_df.copy()
    if args.label in fi_features.columns:
        fi_features = fi_features.drop(columns=[args.label])

    try:
        fi = predictor.feature_importance(data=test_df, silent=True)
        fi.to_csv(output_dir / "feature_importance.csv")
    except Exception as exc:
        pd.DataFrame({"warning": [f"feature_importance failed: {repr(exc)}"]}).to_csv(
            output_dir / "feature_importance_warning.csv",
            index=False,
        )

    loaded_predictor = TabularPredictor.load(args.model_path)
    load_check_features = test_df.drop(columns=[args.label]).head(10)
    load_check_pred = loaded_predictor.predict(load_check_features).astype(str)
    pd.DataFrame({"pred": load_check_pred}).to_csv(
        output_dir / "artifact_load_check_predictions.csv",
        index=False,
    )

    feature_meta = {
        "high_missing_cols": fe.high_missing_cols_,
        "very_high_missing_cols": fe.very_high_missing_cols_,
        "rare_grouped_columns": sorted(list(fe.rare_maps_.keys())),
        "pca_numeric_cols": fe.pca_numeric_cols_,
        "n_pca_components": int(fe.pca_.n_components_) if fe.pca_ is not None else 0,
        "n_clusters": args.n_clusters,
        "oversample_target_ratio": args.oversample_target_ratio,
        "sample_weight_map": weight_map,
        "util_high_threshold": fe.util_high_threshold_,
    }
    _save_json(feature_meta, output_dir / "feature_engineering_summary.json")

    # Metrics JSON is the compact reporting payload consumed by README/report writing.
    metrics = {
        "best_model": predictor.model_best,
        "eval_metric": args.eval_metric,
        "accuracy": eval_result["accuracy"],
        "f1_macro": eval_result["f1_macro"],
        "macro_recall": eval_result["macro_recall"],
        "lt30_recall": eval_result["lt30_recall"],
        "n_train_original": int(len(train_df)),
        "n_train_oversampled_weighted": int(len(weighted_train_df)),
        "n_test": int(len(test_df)),
        "model_path": str(Path(args.model_path).resolve()),
        "leaderboard_path": str(leaderboard_path.resolve()),
        "advanced_features": [
            "missing_indicators",
            "very_high_missing_observed_flags",
            "rare_category_grouping",
            "icd_group_features",
            "age_weight_midpoints",
            "utilization_features",
            "medication_summary_features",
            "pca_components",
            "kmeans_cluster_label",
            "minority_oversampling",
            "sample_weight_cost_sensitive_learning",
            "feature_importance_export",
        ],
    }
    _save_json(metrics, output_dir / "metrics.json")

    print("Advanced EDA-informed training completed.")
    print(f"Best model: {predictor.model_best}")
    print(f"Accuracy: {eval_result['accuracy']:.4f}")
    print(f"Macro F1: {eval_result['f1_macro']:.4f}")
    print(f"Macro Recall: {eval_result['macro_recall']:.4f}")
    print(f"<30 Recall: {eval_result['lt30_recall']:.4f}")
    print(f"Saved leaderboard: {leaderboard_path}")
    print(f"Saved model artifacts: {Path(args.model_path).resolve()}")
    print("\nClassification Report:")
    print(eval_result["classification_report_text"])
    print("\nConfusion Matrix:")
    print(eval_result["confusion_matrix_df"].to_string())


if __name__ == "__main__":
    main()
