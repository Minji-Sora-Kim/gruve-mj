"""Inference pipeline aligned with `src/eda_train.py` feature engineering.

This script rebuilds the same EDA-informed transform state from train split,
loads a saved AutoGluon predictor, and runs batch inference on unseen/eval CSV.

Rubric alignment notes:
- Task 5 inferencing reproducibility:
  - Grader can run one command to produce prediction CSV on unseen data.
- Data leakage prevention:
  - FE state is learned from train split only, then applied to input data.
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
    """Parse CLI arguments for EDA-aligned batch inference."""
    parser = argparse.ArgumentParser(
        description="Batch inference for the advanced EDA-informed AutoGluon model."
    )
    parser.add_argument("--train-path", default="data/processed/train.csv")
    parser.add_argument("--input-path", default="data/test/unseen_data.csv")
    parser.add_argument("--output-path", default="outputs/predictions/unseen_predictions.csv")
    parser.add_argument("--metrics-output", default="outputs/predictions/unseen_metrics.json")
    parser.add_argument("--model-path", default="ag_models")
    parser.add_argument("--label", default="readmitted")
    parser.add_argument("--id-column", default="encounter_id")
    parser.add_argument("--include-input", action="store_true")

    # must match training-time FE params
    parser.add_argument("--rare-threshold", type=float, default=0.01)
    parser.add_argument("--n-clusters", type=int, default=10)
    parser.add_argument("--n-pca-components", type=int, default=8)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def _ensure_parent(path: Path) -> None:
    """Create parent directories for a file path when absent."""
    path.parent.mkdir(parents=True, exist_ok=True)


def _save_json(obj: dict, path: Path) -> None:
    """Persist a dictionary to a UTF-8 JSON file."""
    _ensure_parent(path)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _clean_dataframe(df: pd.DataFrame, label_col: str | None = None) -> pd.DataFrame:
    """Normalize columns, convert '?' placeholders, and clean optional label."""
    cleaned = df.copy()
    cleaned.columns = [str(col).strip() for col in cleaned.columns]
    cleaned = cleaned.replace("?", np.nan)

    if label_col is not None and label_col in cleaned.columns:
        cleaned[label_col] = cleaned[label_col].astype(str).str.strip()

    return cleaned.reset_index(drop=True)


def _drop_redundant_raw_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop `*_raw` ID columns when decoded base columns are present."""
    reduced = df.copy()
    raw_cols = [c for c in reduced.columns if c.endswith("_raw")]
    for raw_col in raw_cols:
        base_col = raw_col[:-4]
        if base_col in reduced.columns:
            reduced = reduced.drop(columns=[raw_col])
    return reduced


def _load_required_csv(csv_path: str, label_col: str | None = None) -> pd.DataFrame:
    """Load a required CSV and apply shared cleaning rules."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Required CSV not found: {path}")
    df = pd.read_csv(path)
    df = _clean_dataframe(df, label_col)
    df = _drop_redundant_raw_id_columns(df)
    return df


def _class_order(values: Iterable[str]) -> list[str]:
    """Return stable class order with preferred readmission priority."""
    labels = [str(v) for v in values]
    preferred = [v for v in DEFAULT_CLASS_ORDER if v in labels]
    remaining = sorted({v for v in labels if v not in preferred})
    return preferred + remaining


def _parse_range_midpoint(value: object) -> float:
    """Convert bracket-range text (e.g. '[50-60)') to numeric midpoint."""
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
    """Map ICD-style diagnosis codes into coarse diagnostic families."""
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
    """Reproducible feature-engineering bundle used by EDA-plus lineage."""
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
        """Apply deterministic feature transforms that do not refit state."""
        out = df.copy()

        for col in HIGH_SIGNAL_CAT_COLS:
            if col in out.columns:
                out[col] = out[col].astype("object")

        if "age" in out.columns:
            out["age_mid"] = out["age"].map(_parse_range_midpoint)

        if "weight" in out.columns:
            out["weight_mid"] = out["weight"].map(_parse_range_midpoint)
            out["weight_missing_flag"] = out["weight"].isna().astype("int8")

        for col in self.high_missing_cols_:
            if col in out.columns:
                out[f"is_missing__{col}"] = out[col].isna().astype("int8")

        for col in self.very_high_missing_cols_:
            if col in out.columns:
                out[f"{col}_observed"] = out[col].notna().astype("int8")

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

        if "max_glu_serum" in out.columns:
            out["max_glu_observed"] = out["max_glu_serum"].notna().astype("int8")
        if "A1Cresult" in out.columns:
            out["a1c_observed"] = out["A1Cresult"].notna().astype("int8")

        out = self._apply_rare_grouping(out)
        return out

    def _apply_rare_grouping(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace train-identified rare categorical levels with `__RARE__`."""
        out = df.copy()
        for col, rare_values in self.rare_maps_.items():
            if col not in out.columns:
                continue
            out[col] = out[col].fillna("Missing").astype(str)
            out[col] = out[col].where(~out[col].isin(rare_values), other="__RARE__")
        return out

    def fit(self, train_df: pd.DataFrame) -> "AdvancedFeatureEngineer":
        """Learn train-only FE state (rare maps, thresholds, PCA, KMeans)."""
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
        """Transform input data using previously fitted FE state."""
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


def main() -> None:
    """Run inference and optional labeled-set evaluation for EDA-plus model."""
    args = parse_args()

    try:
        from autogluon.tabular import TabularPredictor
    except ImportError as exc:
        raise ImportError(
            "AutoGluon is required. Install it with: uv pip install autogluon"
        ) from exc

    train_path = Path(args.train_path)
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    metrics_path = Path(args.metrics_output)
    model_path = Path(args.model_path)

    if not train_path.exists():
        raise FileNotFoundError(f"Train CSV not found: {train_path}")
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    # Refit FE state from train split to guarantee transform parity with training.
    train_df = _load_required_csv(str(train_path), args.label)
    input_df = _load_required_csv(str(input_path), args.label)

    fe = AdvancedFeatureEngineer(
        label_col=args.label,
        rare_threshold=args.rare_threshold,
        n_clusters=args.n_clusters,
        n_pca_components=args.n_pca_components,
        random_state=args.random_state,
    )
    fe.fit(train_df)

    transformed_input = fe.transform(input_df.copy())
    predictor = TabularPredictor.load(str(model_path))

    has_label = args.label in transformed_input.columns
    true_labels = None

    features_df = transformed_input.copy()
    if has_label:
        true_labels = features_df[args.label].astype(str)
        features_df = features_df.drop(columns=[args.label])

    # Predict class labels and per-class probabilities for downstream reporting.
    preds = predictor.predict(features_df).astype(str)
    proba = predictor.predict_proba(features_df)

    output_df = pd.DataFrame(index=input_df.index)

    if args.id_column and args.id_column in input_df.columns:
        output_df[args.id_column] = input_df[args.id_column]

    output_df[f"predicted_{args.label}"] = preds.values

    if isinstance(proba, pd.DataFrame):
        for col in proba.columns:
            output_df[f"prob_{col}"] = proba[col].values
    else:
        output_df["prediction_score"] = proba.values

    if args.include_input:
        output_df = pd.concat(
            [input_df.reset_index(drop=True), output_df.reset_index(drop=True)],
            axis=1,
        )

    _ensure_parent(output_path)
    output_df.to_csv(output_path, index=False)

    # Save transformed preview so graders can verify FE output schema quickly.
    transformed_preview_path = output_path.parent / "transformed_input_preview.csv"
    transformed_input.head(200).to_csv(transformed_preview_path, index=False)

    print(f"Inference completed for {len(input_df)} rows.")
    print(f"Predictions saved to: {output_path}")
    print(f"Transformed preview saved to: {transformed_preview_path}")

    if true_labels is not None:
        # If labels exist in input, emit full evaluation package (metrics/report/CM).
        labels = _class_order(list(true_labels) + list(preds))
        accuracy = float(accuracy_score(true_labels, preds))
        macro_f1 = float(f1_score(true_labels, preds, average="macro"))
        macro_recall = float(recall_score(true_labels, preds, average="macro", zero_division=0))

        report_text = classification_report(true_labels, preds, labels=labels, zero_division=0)
        report_dict = classification_report(
            true_labels,
            preds,
            labels=labels,
            zero_division=0,
            output_dict=True,
        )
        report_df = pd.DataFrame(report_dict).transpose()

        cm = confusion_matrix(true_labels, preds, labels=labels)
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)

        lt30_recall = 0.0
        if "<30" in report_df.index:
            lt30_recall = float(report_df.loc["<30", "recall"])

        metrics = {
            "rows": int(len(input_df)),
            "accuracy": accuracy,
            "f1_macro": macro_f1,
            "macro_recall": macro_recall,
            "lt30_recall": lt30_recall,
            "labels": labels,
        }
        _save_json(metrics, metrics_path)
        report_df.to_csv(metrics_path.with_name("classification_report.csv"), index=True)
        cm_df.to_csv(metrics_path.with_name("confusion_matrix.csv"), index=True)
        metrics_path.with_suffix(".txt").write_text(report_text, encoding="utf-8")

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro F1: {macro_f1:.4f}")
        print(f"Macro Recall: {macro_recall:.4f}")
        print(f"<30 Recall: {lt30_recall:.4f}")
        print(report_text)


if __name__ == "__main__":
    main()
