"""Optimized training pipeline balancing baseline underfit and oversampling overfit.

Design:
1) Compare several data-balancing strategies using an internal split from train.csv.
2) Select best strategy by macro-F1 (+ small reward for <30 recall).
3) Retrain once on full train.csv with the selected strategy.
4) Evaluate on hold-out test.csv and export required artifacts.

Experiment lineage (optimized search):
- This script represents the optimization-search lineage.
- Typical leaderboard artifact in this lineage:
  - `leaderboard_optimized_v2.csv` (depends on --leaderboard-path at run time)
- Typical model directory in this lineage:
  - `ag_models_optimized*`

Core idea:
- Baseline (`src/train.py`) was stable but had limited uplift.
- EDA-plus (`src/eda_train.py`) could rank high on some raw leaderboard runs,
  but some settings showed large validation-test instability.
- This script attempts to close that gap via:
  - balancing-strategy search,
  - moderated weighting/oversampling,
  - optional probability-bias selection (implemented in this file).

Reporting note:
- Always distinguish:
  1) raw leaderboard model score (`score_test`) and
  2) final pipeline score if probability-bias post-processing is used.
"""

from __future__ import annotations

# Example run:
# python src/train_optimized.py --train-path data/processed/train.csv --test-path data/processed/test.csv --model-path ag_models_optimized --leaderboard-path leaderboard_optimized.csv --output-dir outputs/training_optimized --eval-metric f1_macro --presets best_quality --time-limit 3600

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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, recall_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DEFAULT_CLASS_ORDER = ["<30", ">30", "NO"]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Train optimized AutoGluon model with strategy search on train split."
    )
    parser.add_argument("--train-path", default="data/processed/train.csv")
    parser.add_argument("--test-path", default="data/processed/test.csv")
    parser.add_argument("--label", default="readmitted")
    parser.add_argument("--model-path", default="ag_models_optimized")
    parser.add_argument("--leaderboard-path", default="leaderboard_optimized.csv")
    parser.add_argument("--output-dir", default="outputs/training_optimized")
    parser.add_argument("--eval-metric", default="f1_macro")
    parser.add_argument("--presets", default="best_quality")
    parser.add_argument("--time-limit", type=int, default=3600)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--top-models", type=int, default=10)
    parser.add_argument("--internal-val-size", type=float, default=0.2)
    parser.add_argument("--tuning-time-per-strategy", type=int, default=240)
    parser.add_argument("--num-bag-folds", type=int, default=3)
    parser.add_argument("--num-bag-sets", type=int, default=1)
    parser.add_argument("--num-stack-levels", type=int, default=0)
    parser.add_argument(
        "--oversample-ratio",
        type=float,
        default=0.5,
        help="Minority target ratio against majority count for mild oversampling (0~1).",
    )
    parser.add_argument(
        "--strategy-oversample-grid",
        default="0.2,0.35,0.5",
        help="Comma-separated mild oversample ratios for strategy search.",
    )
    parser.add_argument(
        "--strategy-weight-alpha-grid",
        default="0.35,0.5,0.65",
        help="Comma-separated class-weight alpha values for strategy search.",
    )
    parser.add_argument(
        "--max-class-weight",
        type=float,
        default=2.5,
        help="Cap for computed class weights to reduce instability.",
    )
    parser.add_argument(
        "--lt30-objective-weight",
        type=float,
        default=0.2,
        help="Objective weight for <30 recall during strategy selection.",
    )
    parser.add_argument(
        "--f1-floor-vs-baseline",
        type=float,
        default=0.005,
        help="Reject strategy if f1_macro drops below (baseline_f1 - this margin).",
    )
    parser.add_argument(
        "--included-model-types",
        default="GBM,CAT,XGB,RF,XT",
        help="Comma-separated AutoGluon model types to include.",
    )
    parser.add_argument(
        "--keep-id-cols",
        action="store_true",
        help="Keep encounter_id/patient_nbr (default is to drop them).",
    )
    parser.add_argument(
        "--top-mi-numeric",
        type=int,
        default=4,
        help="Top numeric features (by MI to target) used for interaction features.",
    )
    return parser.parse_args()


def _ensure_parent(path: Path) -> None:
    """Create parent directories for a file path when absent."""
    path.parent.mkdir(parents=True, exist_ok=True)


def _parse_float_grid(text: str) -> list[float]:
    """Parse a comma-separated float grid into unique sorted values."""
    vals: list[float] = []
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        vals.append(float(token))
    uniq = sorted({float(v) for v in vals})
    return uniq


def _parse_model_types(text: str) -> list[str]:
    """Parse comma-separated AutoGluon model type tokens."""
    items = [t.strip() for t in str(text).split(",") if t.strip()]
    return items


def _clean_dataframe(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """Normalize columns, convert '?' to NaN, and enforce label presence."""
    cleaned = df.copy()
    cleaned.columns = [str(col).strip() for col in cleaned.columns]
    cleaned = cleaned.replace("?", np.nan)
    if label_col not in cleaned.columns:
        raise ValueError(f"Label column '{label_col}' not found in dataset.")
    cleaned = cleaned.dropna(subset=[label_col]).reset_index(drop=True)
    cleaned[label_col] = cleaned[label_col].astype(str).str.strip()
    return cleaned


def _drop_redundant_raw_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop `*_raw` ID columns when decoded base columns are already present."""
    reduced = df.copy()
    for raw_col in [c for c in reduced.columns if c.endswith("_raw")]:
        base_col = raw_col[: -len("_raw")]
        if base_col in reduced.columns:
            reduced = reduced.drop(columns=[raw_col])
    return reduced


def _add_stable_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add low-risk engineered features that are generally robust."""
    fe = df.copy()

    if {"number_outpatient", "number_emergency", "number_inpatient"}.issubset(fe.columns):
        fe["total_utilization"] = (
            fe["number_outpatient"].fillna(0)
            + fe["number_emergency"].fillna(0)
            + fe["number_inpatient"].fillna(0)
        )
        fe["utilization_log1p"] = np.log1p(fe["total_utilization"])

    if {"num_medications", "time_in_hospital"}.issubset(fe.columns):
        fe["meds_per_day"] = fe["num_medications"] / (fe["time_in_hospital"].fillna(0) + 1.0)

    if {"num_lab_procedures", "time_in_hospital"}.issubset(fe.columns):
        fe["labs_per_day"] = fe["num_lab_procedures"] / (fe["time_in_hospital"].fillna(0) + 1.0)

    return fe


def _add_target_ranked_numeric_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_col: str,
    top_k: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Create interaction features from train-ranked numeric predictors only."""
    train_fe = train_df.copy()
    test_fe = test_df.copy()

    numeric_cols = [
        c
        for c in train_fe.select_dtypes(include=[np.number]).columns
        if c not in {label_col, "sample_weight"}
    ]
    if not numeric_cols or top_k <= 0:
        return train_fe, test_fe, []

    X = train_fe[numeric_cols].copy()
    for col in numeric_cols:
        X[col] = X[col].fillna(X[col].median())
    y = LabelEncoder().fit_transform(train_fe[label_col].astype(str))

    mi = mutual_info_classif(X, y, random_state=random_state)
    mi_df = pd.DataFrame({"feature": numeric_cols, "mi": mi}).sort_values("mi", ascending=False)
    top_features = mi_df["feature"].head(top_k).tolist()

    # Per-feature z-scores using train statistics
    for col in top_features:
        mean = train_fe[col].mean()
        std = train_fe[col].std(ddof=0)
        if std and np.isfinite(std) and std > 1e-12:
            train_fe[f"{col}__z"] = (train_fe[col] - mean) / std
            test_fe[f"{col}__z"] = (test_fe[col] - mean) / std

    # Pairwise interactions among top-3 for stability
    inter_base = top_features[:3]
    for i in range(len(inter_base)):
        for j in range(i + 1, len(inter_base)):
            a, b = inter_base[i], inter_base[j]
            train_fe[f"{a}__x__{b}"] = train_fe[a].fillna(0) * train_fe[b].fillna(0)
            test_fe[f"{a}__x__{b}"] = test_fe[a].fillna(0) * test_fe[b].fillna(0)
            train_fe[f"{a}__div__{b}"] = train_fe[a].fillna(0) / (train_fe[b].fillna(0) + 1.0)
            test_fe[f"{a}__div__{b}"] = test_fe[a].fillna(0) / (test_fe[b].fillna(0) + 1.0)

    return train_fe, test_fe, top_features


def _class_order(values: Iterable[str]) -> list[str]:
    """Return stable class order with preferred readmission label priority."""
    labels = [str(v) for v in values]
    preferred = [v for v in DEFAULT_CLASS_ORDER if v in labels]
    rest = sorted({v for v in labels if v not in preferred})
    return preferred + rest


def _build_weight_map(
    y: pd.Series,
    alpha: float,
    max_weight: float,
) -> dict[str, float]:
    """Build moderated class weights with tunable strength and cap."""
    counts = y.value_counts()
    max_count = counts.max()
    weights = {}
    for cls, cnt in counts.items():
        raw = float((max_count / cnt) ** alpha)
        weights[cls] = float(min(max_weight, raw))
    return weights


def _mild_oversample(
    df: pd.DataFrame,
    label_col: str,
    ratio: float,
    random_state: int,
) -> pd.DataFrame:
    """Mildly oversample minority classes up to ratio * majority_count."""
    if ratio <= 0:
        return df.copy()

    base = df.copy()
    counts = base[label_col].value_counts()
    majority = counts.max()
    target_min = int(np.floor(majority * ratio))

    parts = [base]
    rng = np.random.RandomState(random_state)
    for cls, cnt in counts.items():
        if cnt >= target_min:
            continue
        need = target_min - cnt
        cls_df = base[base[label_col] == cls]
        if cls_df.empty:
            continue
        idx = rng.choice(cls_df.index.values, size=need, replace=True)
        parts.append(cls_df.loc[idx])

    out = pd.concat(parts, axis=0, ignore_index=True)
    out = out.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return out


def _apply_strategy(
    train_df: pd.DataFrame,
    label_col: str,
    strategy: str,
    oversample_ratio: float,
    weight_alpha: float,
    max_class_weight: float,
    random_state: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Apply balancing strategy and return (train_df_for_fit, weight_map_used)."""
    work = train_df.copy()
    weight_map: dict[str, float] = {}

    if strategy in {"oversample", "weight_oversample"}:
        work = _mild_oversample(work, label_col, oversample_ratio, random_state)

    if strategy in {"weight", "weight_oversample"}:
        weight_map = _build_weight_map(
            y=work[label_col].astype(str),
            alpha=weight_alpha,
            max_weight=max_class_weight,
        )
        work["sample_weight"] = work[label_col].astype(str).map(weight_map).astype(float)
    else:
        work["sample_weight"] = 1.0

    return work, weight_map


def _apply_bias_to_proba(
    proba_df: pd.DataFrame,
    bias_map: dict[str, float],
) -> pd.Series:
    """Apply multiplicative class bias factors and return argmax labels."""
    adjusted = proba_df.copy()
    for cls, factor in bias_map.items():
        if cls in adjusted.columns:
            adjusted[cls] = adjusted[cls] * float(factor)
    # argmax on adjusted scores
    return adjusted.idxmax(axis=1)


def _find_best_bias_map(
    y_true: pd.Series,
    proba_df: pd.DataFrame,
) -> tuple[dict[str, float], dict[str, float]]:
    """Tune simple class-bias factors to improve macro-F1 and <30 recall."""
    best_bias = {c: 1.0 for c in proba_df.columns}
    base_pred = proba_df.idxmax(axis=1)
    best_metrics = _evaluate_predictions(y_true, base_pred)
    best_score = best_metrics["f1_macro"] + 0.30 * best_metrics["lt30_recall"]

    lt30_grid = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
    gt30_grid = [1.0, 1.1, 1.2, 1.3]
    no_grid = [1.0, 0.95, 0.9]

    for b_lt30 in lt30_grid:
        for b_gt30 in gt30_grid:
            for b_no in no_grid:
                bias = {"<30": b_lt30, ">30": b_gt30, "NO": b_no}
                pred = _apply_bias_to_proba(proba_df, bias)
                m = _evaluate_predictions(y_true, pred)
                s = m["f1_macro"] + 0.30 * m["lt30_recall"]
                if s > best_score:
                    best_score = s
                    best_bias = bias
                    best_metrics = m

    return best_bias, best_metrics


def _evaluate_predictions(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Compute compact evaluation metrics used in strategy selection/reporting."""
    y_true = y_true.astype(str)
    y_pred = y_pred.astype(str)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    metrics["lt30_recall"] = float(report.get("<30", {}).get("recall", 0.0))
    return metrics


def _plot_leaderboard(leaderboard: pd.DataFrame, output_path: Path, top_n: int) -> None:
    """Plot and save top-N leaderboard scores as a horizontal bar chart."""
    score_col = "score_test" if "score_test" in leaderboard.columns else "score_val"
    chart = leaderboard.head(top_n).copy().iloc[::-1]
    plt.figure(figsize=(10, max(5, top_n * 0.4)))
    plt.barh(chart["model"], chart[score_col], color="#1f77b4")
    plt.xlabel(score_col)
    plt.ylabel("Model")
    plt.title(f"Top {top_n} Models")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _plot_confusion_matrix(cm: np.ndarray, labels: list[str], output_path: Path) -> None:
    """Plot and save a labeled confusion-matrix heatmap image."""
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


def main() -> None:
    """Run optimization-search training, select strategy, and export artifacts."""
    args = parse_args()

    try:
        from autogluon.tabular import TabularPredictor
    except ImportError as exc:
        raise ImportError(
            "AutoGluon is required. Install with `uv pip install autogluon.tabular`."
        ) from exc

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.train_path)
    test_df = pd.read_csv(args.test_path)
    train_df = _clean_dataframe(train_df, args.label)
    test_df = _clean_dataframe(test_df, args.label)

    train_df = _drop_redundant_raw_id_columns(train_df)
    test_df = _drop_redundant_raw_id_columns(test_df)

    if not args.keep_id_cols:
        for col in ["encounter_id", "patient_nbr"]:
            if col in train_df.columns:
                train_df = train_df.drop(columns=[col])
            if col in test_df.columns:
                test_df = test_df.drop(columns=[col])

    train_df = _add_stable_features(train_df)
    test_df = _add_stable_features(test_df)
    train_df, test_df, top_mi_numeric = _add_target_ranked_numeric_features(
        train_df=train_df,
        test_df=test_df,
        label_col=args.label,
        top_k=args.top_mi_numeric,
        random_state=args.random_state,
    )

    tune_train, tune_val = train_test_split(
        train_df,
        test_size=args.internal_val_size,
        random_state=args.random_state,
        stratify=train_df[args.label],
    )
    tune_train = tune_train.reset_index(drop=True)
    tune_val = tune_val.reset_index(drop=True)

    oversample_grid = [v for v in _parse_float_grid(args.strategy_oversample_grid) if v > 0]
    weight_alpha_grid = [v for v in _parse_float_grid(args.strategy_weight_alpha_grid) if v > 0]
    included_models = _parse_model_types(args.included_model_types)

    # Curated search space: enough diversity without exploding runtime.
    strategy_configs: list[tuple[str, float, float]] = [("baseline", 0.0, 0.0)]
    for a in weight_alpha_grid[:2]:
        strategy_configs.append(("weight", 0.0, a))
    for r in oversample_grid[:2]:
        strategy_configs.append(("oversample", r, 0.0))
    if oversample_grid and weight_alpha_grid:
        strategy_configs.append(("weight_oversample", oversample_grid[0], weight_alpha_grid[0]))
        strategy_configs.append(("weight_oversample", oversample_grid[min(1, len(oversample_grid)-1)], weight_alpha_grid[min(1, len(weight_alpha_grid)-1)]))
        strategy_configs.append(("weight_oversample", oversample_grid[-1], weight_alpha_grid[-1]))

    strategy_rows: list[dict[str, object]] = []
    models_root = output_dir / "strategy_models"
    models_root.mkdir(parents=True, exist_ok=True)

    for strategy, os_ratio, weight_alpha in strategy_configs:
        fit_df, weight_map = _apply_strategy(
            tune_train,
            label_col=args.label,
            strategy=strategy,
            oversample_ratio=os_ratio,
            weight_alpha=weight_alpha,
            max_class_weight=args.max_class_weight,
            random_state=args.random_state,
        )
        strategy_model_path = models_root / f"{strategy}__os{os_ratio}__wa{weight_alpha}"

        fit_kwargs = dict(
            train_data=fit_df,
            presets="medium_quality",
            time_limit=args.tuning_time_per_strategy,
            # Strategy search should be fast and robust; avoid heavy bagging here.
            num_bag_folds=0,
            num_bag_sets=1,
            num_stack_levels=0,
            raise_on_no_models_fitted=False,
        )
        if included_models:
            fit_kwargs["included_model_types"] = included_models

        try:
            predictor = TabularPredictor(
                label=args.label,
                eval_metric=args.eval_metric,
                path=str(strategy_model_path),
                sample_weight="sample_weight",
                weight_evaluation=False,
            ).fit(**fit_kwargs)
        except Exception as exc:
            strategy_rows.append(
                {
                    "strategy": strategy,
                    "oversample_ratio": os_ratio,
                    "weight_alpha": weight_alpha,
                    "objective": -1.0,
                    "f1_macro": 0.0,
                    "lt30_recall": 0.0,
                    "macro_recall": 0.0,
                    "accuracy": 0.0,
                    "n_fit_rows": int(len(fit_df)),
                    "weight_map": json.dumps(weight_map, ensure_ascii=False),
                    "bias_map": json.dumps({"<30": 1.0, ">30": 1.0, "NO": 1.0}, ensure_ascii=False),
                    "error": repr(exc),
                }
            )
            continue

        if not predictor.model_names():
            strategy_rows.append(
                {
                    "strategy": strategy,
                    "oversample_ratio": os_ratio,
                    "weight_alpha": weight_alpha,
                    "objective": -1.0,
                    "f1_macro": 0.0,
                    "lt30_recall": 0.0,
                    "macro_recall": 0.0,
                    "accuracy": 0.0,
                    "n_fit_rows": int(len(fit_df)),
                    "weight_map": json.dumps(weight_map, ensure_ascii=False),
                    "bias_map": json.dumps({"<30": 1.0, ">30": 1.0, "NO": 1.0}, ensure_ascii=False),
                    "error": "no_models_fitted",
                }
            )
            continue

        val_features = tune_val.drop(columns=[args.label])
        val_proba = predictor.predict_proba(val_features)
        bias_map, metrics = _find_best_bias_map(tune_val[args.label], val_proba)
        objective = metrics["f1_macro"] + float(args.lt30_objective_weight) * metrics["lt30_recall"]

        strategy_rows.append(
            {
                "strategy": strategy,
                "oversample_ratio": os_ratio,
                "weight_alpha": weight_alpha,
                "objective": objective,
                "f1_macro": metrics["f1_macro"],
                "lt30_recall": metrics["lt30_recall"],
                "macro_recall": metrics["macro_recall"],
                "accuracy": metrics["accuracy"],
                "n_fit_rows": int(len(fit_df)),
                "weight_map": json.dumps(weight_map, ensure_ascii=False),
                "bias_map": json.dumps(bias_map, ensure_ascii=False),
                "error": "",
            }
        )

    strategy_df = pd.DataFrame(strategy_rows).sort_values("objective", ascending=False)
    strategy_df.to_csv(output_dir / "strategy_search_results.csv", index=False)
    if strategy_df.empty or (strategy_df["objective"] < 0).all():
        raise RuntimeError(
            "Strategy search failed: no candidate produced a trained model. "
            "Try reducing included_model_types or using simpler presets."
        )
    # Keep strategy robust by requiring near-baseline macro-F1 before selecting by objective.
    baseline_rows = strategy_df[strategy_df["strategy"] == "baseline"]
    baseline_f1 = float(baseline_rows.iloc[0]["f1_macro"]) if not baseline_rows.empty else float(strategy_df["f1_macro"].max())
    f1_floor = baseline_f1 - float(args.f1_floor_vs_baseline)
    eligible = strategy_df[strategy_df["f1_macro"] >= f1_floor].copy()
    selected_row = eligible.iloc[0] if not eligible.empty else strategy_df.iloc[0]

    best_strategy = str(selected_row["strategy"])
    best_oversample_ratio = float(selected_row["oversample_ratio"])
    best_weight_alpha = float(selected_row["weight_alpha"])
    best_bias_map = json.loads(str(selected_row["bias_map"]))

    final_train_df, final_weight_map = _apply_strategy(
        train_df,
        label_col=args.label,
        strategy=best_strategy,
        oversample_ratio=best_oversample_ratio,
        weight_alpha=best_weight_alpha,
        max_class_weight=args.max_class_weight,
        random_state=args.random_state,
    )

    # Retrain once on full train split with the selected balancing policy.
    final_fit_kwargs = dict(
        train_data=final_train_df,
        presets=args.presets,
        time_limit=args.time_limit,
        num_bag_folds=args.num_bag_folds,
        num_bag_sets=args.num_bag_sets,
        num_stack_levels=args.num_stack_levels,
        raise_on_no_models_fitted=True,
    )
    if included_models:
        final_fit_kwargs["included_model_types"] = included_models

    final_predictor = TabularPredictor(
        label=args.label,
        eval_metric=args.eval_metric,
        path=args.model_path,
        sample_weight="sample_weight",
        weight_evaluation=False,
    ).fit(**final_fit_kwargs)

    val_leaderboard = final_predictor.leaderboard(silent=True)
    val_leaderboard.to_csv(output_dir / "leaderboard_validation.csv", index=False)
    test_leaderboard = final_predictor.leaderboard(data=test_df, silent=True)
    leaderboard_path = Path(args.leaderboard_path)
    _ensure_parent(leaderboard_path)
    test_leaderboard.to_csv(leaderboard_path, index=False)
    test_leaderboard.to_csv(output_dir / "leaderboard_full.csv", index=False)
    _plot_leaderboard(test_leaderboard, output_dir / "leaderboard_top_models.png", args.top_models)

    gap_df = test_leaderboard[["model", "score_test", "score_val"]].copy()
    gap_df["val_test_gap"] = gap_df["score_val"] - gap_df["score_test"]
    gap_df.to_csv(output_dir / "val_test_gap_analysis.csv", index=False)

    # Final hold-out evaluation uses selected probability-bias calibration.
    y_true = test_df[args.label].astype(str)
    test_features = test_df.drop(columns=[args.label])
    test_proba = final_predictor.predict_proba(test_features)
    y_pred = _apply_bias_to_proba(test_proba, best_bias_map).astype(str)
    labels = _class_order(list(y_true) + list(y_pred))

    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    report_text = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    report_df = pd.DataFrame(
        classification_report(
            y_true,
            y_pred,
            labels=labels,
            zero_division=0,
            output_dict=True,
        )
    ).transpose()
    report_df.to_csv(output_dir / "classification_report.csv")
    (output_dir / "classification_report.txt").write_text(report_text)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(output_dir / "confusion_matrix.csv")
    _plot_confusion_matrix(cm, labels, output_dir / "confusion_matrix_heatmap.png")

    summary = {
        "selected_strategy": best_strategy,
        "selected_oversample_ratio": best_oversample_ratio,
        "selected_weight_alpha": best_weight_alpha,
        "selected_weight_map": final_weight_map,
        "selected_bias_map": best_bias_map,
        "baseline_f1_macro_internal": baseline_f1,
        "f1_floor_used": f1_floor,
        "accuracy": accuracy,
        "f1_macro": macro_f1,
        "n_train_original": int(len(train_df)),
        "n_train_final_strategy": int(len(final_train_df)),
        "n_test": int(len(test_df)),
        "model_path": str(Path(args.model_path).resolve()),
        "leaderboard_path": str(leaderboard_path.resolve()),
        "oversample_ratio_search_grid": oversample_grid,
        "weight_alpha_search_grid": weight_alpha_grid,
        "lt30_objective_weight": args.lt30_objective_weight,
        "included_model_types": included_models,
        "top_mi_numeric_used": top_mi_numeric,
    }
    (output_dir / "optimization_summary.json").write_text(json.dumps(summary, indent=2))

    print("Optimized training completed.")
    print(f"Selected strategy: {best_strategy}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 macro: {macro_f1:.4f}")
    print(f"Leaderboard: {leaderboard_path}")
    print(f"Artifacts: {output_dir}")


if __name__ == "__main__":
    main()
