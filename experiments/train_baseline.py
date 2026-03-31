"""Train AutoGluon models for diabetic readmission classification.

Expected workflow:
1. Run `src/prepare_data.ipynb` to create stratified `train.csv` / `test.csv`.
2. Run this script to train, evaluate, and export artifacts.

Experiment lineage (baseline):
- This script represents the baseline lineage for reporting.
- Default leaderboard artifact: `leaderboard.csv`
- Default model directory: `ag_models/`
- Core logic:
  - load prepared train/test splits,
  - apply only minimal cleaning + optional light experiment features,
  - train AutoGluon with no aggressive class rebalancing pipeline.

How this differs from other lineages:
- `src/eda_train.py` -> EDA-plus lineage (`leaderboard_eda_plus.csv`)
  with stronger feature engineering + oversampling + sample weights.
- `src/train_optimized.py` -> optimization-search lineage
  (`leaderboard_optimized_v2.csv` in recent runs) with strategy search.
"""

from __future__ import annotations

# Example run:
# python src/train.py --train-path data/processed/train.csv --test-path data/processed/test.csv --model-path ag_models --leaderboard-path leaderboard.csv --output-dir outputs/training --eval-metric f1_macro --presets best_quality --time-limit 3600 --num-bag-folds 5 --num-bag-sets 1 --num-stack-levels 1

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".mplconfig").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

DEFAULT_CLASS_ORDER = ["<30", ">30", "NO"]

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for training and artifact export."""
    parser = argparse.ArgumentParser(
        description="Train an AutoGluon model for diabetic readmission prediction."
    )
    parser.add_argument("--train-path", default="data/processed/train.csv")
    parser.add_argument("--test-path", default="data/processed/test.csv")
    parser.add_argument("--raw-path", default="data/raw/diabetic_data.csv")
    parser.add_argument("--label", default="readmitted")
    parser.add_argument("--model-path", default="ag_models")
    parser.add_argument("--leaderboard-path", default="leaderboard.csv")
    parser.add_argument("--output-dir", default="outputs/training")
    parser.add_argument("--eval-metric", default="f1_macro")
    parser.add_argument("--presets", default="best_quality")
    parser.add_argument("--time-limit", type=int, default=3600)
    parser.add_argument(
        "--num-bag-folds",
        type=int,
        default=5,
        help="K-fold count for bagging/CV-style optimization on train.csv only.",
    )
    parser.add_argument(
        "--num-bag-sets",
        type=int,
        default=1,
        help="Repeat bagging folds N times (higher = more robust, slower).",
    )
    parser.add_argument(
        "--num-stack-levels",
        type=int,
        default=1,
        help="Stacking levels on top of bagged models.",
    )
    parser.add_argument("--top-models", type=int, default=10)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--allow-raw-fallback",
        action="store_true",
        help="Allow fallback split from raw data if train/test CSVs are missing.",
    )
    parser.add_argument(
        "--experiment",
        choices=["baseline", "missingness", "icd_grouped"],
        default="baseline",
        help="Feature engineering experiment mode.",
    )
    return parser.parse_args()


def _ensure_parent(path: Path) -> None:
    """Create parent directory for a target path if it does not exist."""
    path.parent.mkdir(parents=True, exist_ok=True)


def _icd_group(value: object) -> str:
    """Map ICD-9 diagnosis codes to coarse disease groups."""
    if pd.isna(value):
        return "missing"
    text = str(value).strip().upper()
    if not text:
        return "missing"
    if text.startswith("E"):
        return "external_causes"
    if text.startswith("V"):
        return "supplementary"
    try:
        code = float(text)
    except ValueError:
        return "other"

    if abs(code - 250.0) < 1e-9:
        return "diabetes"
    if 1 <= code <= 139:
        return "infectious_parasitic"
    if 140 <= code <= 239:
        return "neoplasms"
    if 240 <= code <= 279:
        return "endocrine_metabolic"
    if 280 <= code <= 289:
        return "blood_immune"
    if 290 <= code <= 319:
        return "mental_disorders"
    if 320 <= code <= 389:
        return "nervous_system"
    if 390 <= code <= 459 or abs(code - 785.0) < 1e-9:
        return "circulatory"
    if 460 <= code <= 519 or abs(code - 786.0) < 1e-9:
        return "respiratory"
    if 520 <= code <= 579 or abs(code - 787.0) < 1e-9:
        return "digestive"
    if 580 <= code <= 629 or abs(code - 788.0) < 1e-9:
        return "genitourinary"
    if 710 <= code <= 739:
        return "musculoskeletal"
    if 740 <= code <= 759:
        return "congenital"
    if 760 <= code <= 779:
        return "perinatal"
    if 780 <= code <= 799:
        return "ill_defined"
    if 800 <= code <= 999:
        return "injury_poisoning"
    return "other"


def _clean_dataframe(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """Normalize columns, convert '?' placeholders to NaN, and clean label."""
    cleaned = df.copy()
    cleaned.columns = [str(col).strip() for col in cleaned.columns]
    cleaned = cleaned.replace("?", np.nan)

    if label_col not in cleaned.columns:
        raise ValueError(f"Label column '{label_col}' not found in dataset.")

    cleaned = cleaned.dropna(subset=[label_col]).reset_index(drop=True)
    cleaned[label_col] = cleaned[label_col].astype(str).str.strip()
    return cleaned


def _apply_experiment_features(
    df: pd.DataFrame, label_col: str, experiment: str
) -> pd.DataFrame:
    """Apply optional feature engineering for experiment tracking."""
    features = df.copy()
    non_label_cols = [c for c in features.columns if c != label_col]

    if experiment == "missingness":
        for col in non_label_cols:
            if features[col].isna().any():
                features[f"is_missing__{col}"] = features[col].isna().astype("int8")

    if experiment == "icd_grouped":
        for diag_col in ("diag_1", "diag_2", "diag_3"):
            if diag_col in features.columns:
                features[f"{diag_col}_group"] = features[diag_col].map(_icd_group)

    return features


def _drop_redundant_raw_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop *_raw ID columns when mapped categorical columns already exist.

    In this project, *_raw columns are 1:1 duplicates of the decoded ID-label
    columns, so keeping both provides redundant signals.
    """
    reduced = df.copy()
    for raw_col in [c for c in reduced.columns if c.endswith("_raw")]:
        base_col = raw_col[: -len("_raw")]
        if base_col in reduced.columns:
            reduced = reduced.drop(columns=[raw_col])
    return reduced


def _load_or_create_splits(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load stratified train/test splits or optionally create fallback splits.

    Preferred path is to use pre-generated `data/processed/train.csv` and
    `data/processed/test.csv` from `prepare_data.ipynb`, because that notebook
    also enforces the required 5% unseen split protocol.
    """
    train_path = Path(args.train_path)
    test_path = Path(args.test_path)
    raw_path = Path(args.raw_path)

    if train_path.exists() and test_path.exists():
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
    else:
        if not args.allow_raw_fallback:
            raise FileNotFoundError(
                f"Missing '{train_path}' or '{test_path}'. "
                "Run `src/prepare_data.ipynb` first to create the required "
                "stratified splits and unseen sample."
            )
        if not raw_path.exists():
            raise FileNotFoundError(
                f"Missing '{train_path}'/'{test_path}', and raw source '{raw_path}' was not found."
            )

        raw_df = pd.read_csv(raw_path)
        raw_df = _clean_dataframe(raw_df, args.label)
        train_df, test_df = train_test_split(
            raw_df,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=raw_df[args.label],
        )

        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        _ensure_parent(train_path)
        _ensure_parent(test_path)
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

    train_df = _clean_dataframe(train_df, args.label)
    test_df = _clean_dataframe(test_df, args.label)

    raw_id_cols_before = [c for c in train_df.columns if c.endswith("_raw")]
    train_df = _drop_redundant_raw_id_columns(train_df)
    test_df = _drop_redundant_raw_id_columns(test_df)
    dropped_raw_cols = [c for c in raw_id_cols_before if c not in train_df.columns]
    if dropped_raw_cols:
        print(f"Dropped redundant raw ID columns: {dropped_raw_cols}")

    train_df = _apply_experiment_features(train_df, args.label, args.experiment)
    test_df = _apply_experiment_features(test_df, args.label, args.experiment)
    return train_df, test_df


def _class_order(values: Iterable[str]) -> list[str]:
    """Return class order with preferred labels first, then remaining sorted."""
    labels = [str(v) for v in values]
    preferred = [v for v in DEFAULT_CLASS_ORDER if v in labels]
    rest = sorted({v for v in labels if v not in preferred})
    return preferred + rest


def _plot_leaderboard(leaderboard: pd.DataFrame, output_path: Path, top_n: int) -> None:
    """Save horizontal bar chart for top-N leaderboard models."""
    score_col = "score_test" if "score_test" in leaderboard.columns else "score_val"
    chart = leaderboard.head(top_n).copy()
    chart = chart.iloc[::-1]

    plt.figure(figsize=(10, max(5, top_n * 0.4)))
    plt.barh(chart["model"], chart[score_col], color="#1f77b4")
    plt.xlabel(score_col)
    plt.ylabel("Model")
    plt.title(f"Top {top_n} Models")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _plot_confusion_matrix(cm: np.ndarray, labels: list[str], output_path: Path) -> None:
    """Render and save confusion matrix heatmap."""
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
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", color="black")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _plot_classwise_f1(report_df: pd.DataFrame, output_path: Path) -> None:
    """Render and save class-wise F1 bar chart."""
    class_rows = [
        idx
        for idx in report_df.index
        if idx not in {"accuracy", "macro avg", "weighted avg"}
    ]
    classwise = report_df.loc[class_rows, "f1-score"].sort_values(ascending=False)

    plt.figure(figsize=(8, max(4, len(classwise) * 0.5)))
    plt.bar(classwise.index.astype(str), classwise.values, color="#2ca02c")
    plt.ylabel("F1 Score")
    plt.xlabel("Class")
    plt.title("Classwise F1 Scores")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    """Run training, hold-out evaluation, and artifact/plot export."""
    args = parse_args()

    try:
        from autogluon.tabular import TabularPredictor
    except ImportError as exc:
        raise ImportError(
            "AutoGluon is required. Install with `uv pip install autogluon.tabular`."
        ) from exc

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df = _load_or_create_splits(args)

    predictor = TabularPredictor(
        label=args.label,
        eval_metric=args.eval_metric,
        path=args.model_path,
    ).fit(
        # K-fold bagging is performed only within train.csv.
        # This is intentional to optimize model parameters/ensembling without
        # touching hold-out test.csv or unseen_data.csv.
        train_data=train_df,
        presets=args.presets,
        time_limit=args.time_limit,
        num_bag_folds=args.num_bag_folds,
        num_bag_sets=args.num_bag_sets,
        num_stack_levels=args.num_stack_levels,
    )
    loaded_predictor = TabularPredictor.load(args.model_path)
    _ = loaded_predictor.predict(test_features.head(5))

    leaderboard = predictor.leaderboard(test_data=test_df, silent=True)
    leaderboard_path = Path(args.leaderboard_path)
    _ensure_parent(leaderboard_path)
    leaderboard.to_csv(leaderboard_path, index=False)
    leaderboard.to_csv(output_dir / "leaderboard_full.csv", index=False)
    _plot_leaderboard(leaderboard, output_dir / "leaderboard_top_models.png", args.top_models)

    test_features = test_df.drop(columns=[args.label])
    y_true = test_df[args.label].astype(str)
    y_pred = predictor.predict(test_features).astype(str)
    labels = _class_order(list(y_true) + list(y_pred))

    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
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
    (output_dir / "classification_report.txt").write_text(report_text)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.to_csv(output_dir / "confusion_matrix.csv", index=True)
    _plot_confusion_matrix(cm, labels, output_dir / "confusion_matrix_heatmap.png")
    _plot_classwise_f1(report_df, output_dir / "classwise_f1_bar.png")

    metrics = {
        "best_model": predictor.model_best,
        "eval_metric": args.eval_metric,
        "accuracy": accuracy,
        "f1_macro": macro_f1,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "experiment": args.experiment,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print("Training completed.")
    print(f"Best model: {predictor.model_best}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 macro: {macro_f1:.4f}")
    print(f"Leaderboard: {leaderboard_path}")
    print(f"Artifacts: {output_dir}")


if __name__ == "__main__":
    main()
