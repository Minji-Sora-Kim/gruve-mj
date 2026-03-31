"""Generate explainability artifacts for a trained AutoGluon tabular model.

Outputs include:
- model feature importance,
- optional SHAP global/local summaries when available,
- representative prediction-level explanations, and
- concise narrative summaries for reporting.

Bonus alignment notes:
- B2: SHAP global/local explanations + optional LLM/VLM narrative outputs.
- B3 integration: output files are directly consumable by the app chat/explanation
  endpoints for evaluator-facing interpretability.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
from typing import Callable

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".mplconfig").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for explainability input/model/output options."""
    parser = argparse.ArgumentParser(
        description="Generate global/local explanations for a trained AutoGluon model."
    )
    parser.add_argument("--input", default="data/processed/test.csv")
    parser.add_argument("--train-path", default="data/processed/train.csv")
    parser.add_argument("--model-path", default="ag_models")
    parser.add_argument("--label", default="readmitted")
    parser.add_argument("--output-dir", default="outputs/explainability")
    parser.add_argument("--sample-size", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--background-size", type=int, default=100)
    parser.add_argument("--global-shap-sample", type=int, default=80)
    parser.add_argument("--local-shap-batch-size", type=int, default=128)
    parser.add_argument("--feature-importance-sample", type=int, default=500)
    parser.add_argument("--top-features", type=int, default=15)
    parser.add_argument("--llm-model", default="gpt-4o-mini")
    parser.add_argument("--rare-threshold", type=float, default=0.01)
    parser.add_argument("--n-clusters", type=int, default=10)
    parser.add_argument("--n-pca-components", type=int, default=8)
    parser.add_argument(
        "--eda-chart-paths",
        default="outputs/training/confusion_matrix_heatmap.png,outputs/training/classwise_f1_bar.png",
        help="Comma-separated chart/image paths for optional VLM interpretation. "
        "PNG/JPG/WebP files are used directly; HTML paths are skipped.",
    )
    return parser.parse_args()


def _clean_dataframe(df: pd.DataFrame, label_col: str | None = None) -> pd.DataFrame:
    """Normalize headers, convert '?' placeholders, and strip label values."""
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    out = out.replace("?", np.nan)
    if label_col is not None and label_col in out.columns:
        out[label_col] = out[label_col].astype(str).str.strip()
    return out.reset_index(drop=True)


def _drop_redundant_raw_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop *_raw columns when decoded base ID column already exists."""
    out = df.copy()
    for raw_col in [c for c in out.columns if c.endswith("_raw")]:
        base_col = raw_col[: -len("_raw")]
        if base_col in out.columns:
            out = out.drop(columns=[raw_col])
    return out


def _expected_feature_columns(predictor) -> list[str]:
    """Get model-required input columns from predictor metadata."""
    try:
        meta = getattr(predictor, "feature_metadata_in", None)
        if meta is not None and hasattr(meta, "get_features"):
            cols = list(meta.get_features())
            if cols:
                return cols
    except Exception:
        pass
    try:
        cols = list(predictor.features())
        if cols:
            return cols
    except Exception:
        pass
    return []


def _select_representative_indices(
    y_true: pd.Series | None,
    y_pred: pd.Series,
    sample_size: int,
    random_state: int,
) -> list[int]:
    """Choose representative rows for local explanations.

    Priority:
    - correctly predicted `NO`
    - correctly predicted `<30`
    - one misclassified case
    - random fill up to sample_size
    """
    selected: list[int] = []
    if y_true is not None:
        for cls in ("NO", "<30"):
            candidates = y_true.index[(y_true == cls) & (y_pred == cls)].tolist()
            if candidates:
                selected.append(candidates[0])
        misclassified = y_true.index[y_true != y_pred].tolist()
        if misclassified:
            selected.append(misclassified[0])

    selected = list(dict.fromkeys(selected))
    if len(selected) < sample_size:
        remaining = [idx for idx in y_pred.index if idx not in selected]
        if remaining:
            rng = np.random.default_rng(random_state)
            take_n = min(sample_size - len(selected), len(remaining))
            selected.extend(rng.choice(remaining, size=take_n, replace=False).tolist())

    return selected[:sample_size]


def _plot_global_importance(
    importance_df: pd.DataFrame,
    output_path: Path,
    top_n: int,
    value_col: str,
    title: str,
) -> None:
    """Save a top-N global importance horizontal bar chart."""
    plot_df = importance_df.copy()
    if value_col in plot_df.columns:
        plot_df[value_col] = pd.to_numeric(plot_df[value_col], errors="coerce")
        plot_df = plot_df.replace([np.inf, -np.inf], np.nan).dropna(subset=[value_col])

    top = plot_df.head(top_n).copy()
    top = top.iloc[::-1]

    plt.figure(figsize=(10, max(5, top_n * 0.42)))
    if top.empty or (top[value_col].abs().sum() == 0):
        plt.text(
            0.5,
            0.5,
            f"No non-zero {value_col} values available for this run.",
            ha="center",
            va="center",
            fontsize=11,
        )
        plt.axis("off")
    else:
        plt.barh(top["feature"], top[value_col], color="#1f77b4")
        plt.xlabel(value_col)
        plt.ylabel("Feature")
        plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _plot_local_contributions(
    local_df: pd.DataFrame,
    sample_id: int,
    output_path: Path,
    top_n: int,
) -> None:
    """Save local SHAP contribution chart for one sample."""
    sample = local_df[local_df["sample_id"] == sample_id].copy()
    sample["shap_value"] = pd.to_numeric(sample["shap_value"], errors="coerce")
    sample = sample.replace([np.inf, -np.inf], np.nan).dropna(subset=["shap_value"])
    sample["abs_value"] = sample["shap_value"].abs()
    sample = sample.sort_values("abs_value", ascending=False).head(top_n)
    sample = sample.iloc[::-1]

    plt.figure(figsize=(9, max(4, top_n * 0.4)))
    if sample.empty or (sample["abs_value"].sum() == 0):
        plt.text(
            0.5,
            0.5,
            f"No non-zero local SHAP values for sample_id={sample_id}.",
            ha="center",
            va="center",
            fontsize=11,
        )
        plt.axis("off")
    else:
        colors = np.where(sample["shap_value"] >= 0, "#d62728", "#2ca02c")
        plt.barh(sample["feature"], sample["shap_value"], color=colors)
        plt.axvline(0, color="black", linewidth=1)
        plt.xlabel("SHAP value")
        plt.ylabel("Feature")
        plt.title(f"Local SHAP Contributions (sample_id={sample_id})")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _build_predict_proba_fn(
    predictor,
    feature_columns: list[str],
    shap_schema: dict[str, dict[str, object]] | None = None,
) -> Callable[[pd.DataFrame | np.ndarray], np.ndarray]:
    """Create a SHAP-compatible function that returns class probabilities."""
    def predict_fn(data: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(data, pd.DataFrame):
            batch = data.loc[:, feature_columns]
        else:
            batch = pd.DataFrame(data, columns=feature_columns)
        if shap_schema is not None:
            batch = _decode_shap_frame(batch, shap_schema)
        proba = predictor.predict_proba(batch)
        if isinstance(proba, pd.Series):
            proba = proba.to_frame(name=str(proba.name or "positive"))
        return proba.to_numpy()

    return predict_fn


def _build_shap_schema(df: pd.DataFrame) -> dict[str, dict[str, object]]:
    """Build encoding schema so SHAP can run on numeric-only matrices."""
    schema: dict[str, dict[str, object]] = {}
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            median = float(pd.to_numeric(series, errors="coerce").median())
            if not np.isfinite(median):
                median = 0.0
            schema[col] = {"kind": "numeric", "median": median}
        else:
            values = series.fillna("__MISSING__").astype(str)
            categories = sorted(values.unique().tolist())
            if not categories:
                categories = ["__MISSING__"]
            schema[col] = {"kind": "categorical", "categories": categories}
    return schema


def _encode_shap_frame(df: pd.DataFrame, schema: dict[str, dict[str, object]]) -> pd.DataFrame:
    """Encode mixed-type dataframe into numeric dataframe for SHAP."""
    enc = pd.DataFrame(index=df.index)
    for col, col_schema in schema.items():
        if col_schema["kind"] == "numeric":
            median = float(col_schema["median"])
            enc[col] = pd.to_numeric(df[col], errors="coerce").fillna(median).astype(float)
        else:
            cats = list(col_schema["categories"])
            val_to_idx = {v: i for i, v in enumerate(cats)}
            values = df[col].fillna("__MISSING__").astype(str)
            fallback = val_to_idx.get("__MISSING__", 0)
            enc[col] = values.map(lambda v: val_to_idx.get(v, fallback)).astype(float)
    return enc


def _decode_shap_frame(df: pd.DataFrame, schema: dict[str, dict[str, object]]) -> pd.DataFrame:
    """Decode numeric SHAP matrix back to predictor-consumable mixed dataframe."""
    dec = pd.DataFrame(index=df.index)
    for col, col_schema in schema.items():
        if col_schema["kind"] == "numeric":
            dec[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
        else:
            cats = list(col_schema["categories"])
            idx = pd.to_numeric(df[col], errors="coerce").fillna(0).round().astype(int)
            idx = idx.clip(lower=0, upper=max(0, len(cats) - 1))
            values = idx.map(lambda i: cats[int(i)])
            values = values.replace("__MISSING__", np.nan)
            dec[col] = values.astype("object")
    return dec


def _extract_global_shap_importance(
    shap_values,
    feature_names: list[str],
) -> pd.DataFrame:
    """Aggregate SHAP values into global mean absolute importance."""
    values = np.asarray(shap_values.values)
    if values.ndim == 3:
        mean_abs = np.mean(np.abs(values), axis=(0, 2))
    elif values.ndim == 2:
        mean_abs = np.mean(np.abs(values), axis=0)
    else:
        raise ValueError(f"Unsupported SHAP value shape: {values.shape}")

    df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
    return df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def _extract_local_shap(
    shap_values,
    explain_df: pd.DataFrame,
    predicted_labels: pd.Series,
    class_names: list[str],
) -> pd.DataFrame:
    """Convert per-sample SHAP tensor to long-format local contribution table."""
    values = np.asarray(shap_values.values)
    records: list[dict[str, object]] = []

    for i, sample_id in enumerate(explain_df.index.tolist()):
        pred_label = str(predicted_labels.loc[sample_id])
        class_idx = class_names.index(pred_label) if pred_label in class_names else 0

        if values.ndim == 3:
            contribution = values[i, :, class_idx]
        elif values.ndim == 2:
            contribution = values[i, :]
        else:
            raise ValueError(f"Unsupported SHAP value shape: {values.shape}")

        for feature, value in zip(explain_df.columns.tolist(), contribution):
            records.append(
                {
                    "sample_id": int(sample_id),
                    "predicted_label": pred_label,
                    "feature": feature,
                    "shap_value": float(value),
                }
            )

    return pd.DataFrame(records)


def _build_llm_summary(
    global_importance: pd.DataFrame,
    representative_df: pd.DataFrame,
    predictions: pd.Series,
    probabilities: pd.DataFrame,
    local_shap_df: pd.DataFrame | None,
    label_col: str,
) -> str:
    """Build an LLM-ready narrative in card-style summary format."""
    lines: list[str] = []
    lines.append("Prediction / Confidence / Top factors / Caution")
    lines.append("")
    lines.append("Global summary:")
    top_global = global_importance.head(8)["feature"].tolist()
    lines.append(f"- Most influential features: {', '.join(top_global)}.")
    lines.append("")
    lines.append("Local explanation cards:")

    for sample_id in representative_df.index.tolist():
        pred = str(predictions.loc[sample_id])
        confidence = float(probabilities.loc[sample_id].max())
        line = f"- Sample {sample_id}: Prediction={pred}, Confidence={confidence:.3f}."

        if label_col in representative_df.columns:
            truth = str(representative_df.loc[sample_id, label_col])
            line += f" TrueLabel={truth}."

        if local_shap_df is not None and not local_shap_df.empty:
            local_rows = local_shap_df[local_shap_df["sample_id"] == sample_id].copy()
            local_rows["abs_value"] = local_rows["shap_value"].abs()
            local_rows = local_rows.sort_values("abs_value", ascending=False)
            top_factors = local_rows.head(4)["feature"].tolist()
            if top_factors:
                line += f" Top factors: {', '.join(top_factors)}."

        line += " Caution: treat this as decision support, not standalone clinical advice."
        lines.append(line)

    return "\n".join(lines)


def _format_feature_pairs(df: pd.DataFrame, value_col: str, top_n: int = 8) -> str:
    """Format top feature-value pairs for prompt context."""
    if df.empty or value_col not in df.columns:
        return "- (no feature values available)"
    top = df.head(top_n)
    lines = []
    for row in top.itertuples(index=False):
        feature = str(getattr(row, "feature", "unknown"))
        value = float(getattr(row, value_col, 0.0))
        lines.append(f"- {feature}: {value:.6f}")
    return "\n".join(lines)


def _build_llm_rewrite_prompt(
    rule_based_summary: str,
    global_importance: pd.DataFrame,
    local_shap_df: pd.DataFrame | None,
    representative_df: pd.DataFrame,
    predictions: pd.Series,
    probabilities: pd.DataFrame,
    y_true: pd.Series | None,
    label_col: str,
) -> str:
    """Build a richer evaluator-facing prompt for LLM narrative generation."""
    global_block = _format_feature_pairs(global_importance, "mean_abs_shap", top_n=10)

    pred_dist = (predictions.value_counts(normalize=True) * 100).round(2)
    pred_dist_block = "\n".join(f"- {k}: {v:.2f}%" for k, v in pred_dist.items())
    if not pred_dist_block:
        pred_dist_block = "- (prediction distribution unavailable)"

    metric_block = "- Hold-out-like metrics unavailable in this run context."
    if y_true is not None and len(y_true) == len(predictions):
        try:
            acc = accuracy_score(y_true.astype(str), predictions.astype(str))
            macro_f1 = f1_score(y_true.astype(str), predictions.astype(str), average="macro")
            metric_block = (
                f"- Accuracy: {acc:.4f}\n"
                f"- Macro F1: {macro_f1:.4f}\n"
                f"- Rows evaluated: {len(predictions)}"
            )
        except Exception:
            pass

    representative_lines: list[str] = []
    for sample_id in representative_df.index.tolist():
        pred = str(predictions.loc[sample_id]) if sample_id in predictions.index else "unknown"
        confidence = (
            float(probabilities.loc[sample_id].max())
            if sample_id in probabilities.index
            else float("nan")
        )
        truth_text = ""
        if label_col in representative_df.columns:
            truth_text = f", true={representative_df.loc[sample_id, label_col]}"

        pos_text = "n/a"
        neg_text = "n/a"
        if local_shap_df is not None and not local_shap_df.empty:
            rows = local_shap_df[local_shap_df["sample_id"] == sample_id].copy()
            if not rows.empty:
                pos_rows = rows[rows["shap_value"] > 0].sort_values("shap_value", ascending=False).head(3)
                neg_rows = rows[rows["shap_value"] < 0].sort_values("shap_value", ascending=True).head(3)
                if not pos_rows.empty:
                    pos_text = ", ".join(
                        f"{r.feature} ({float(r.shap_value):+.4f})"
                        for r in pos_rows.itertuples(index=False)
                    )
                if not neg_rows.empty:
                    neg_text = ", ".join(
                        f"{r.feature} ({float(r.shap_value):+.4f})"
                        for r in neg_rows.itertuples(index=False)
                    )

        representative_lines.append(
            f"- sample_id={sample_id}, pred={pred}, confidence={confidence:.3f}{truth_text}\n"
            f"  - top_positive_shap: {pos_text}\n"
            f"  - top_negative_shap: {neg_text}"
        )
    representative_block = "\n".join(representative_lines) if representative_lines else "- (none)"

    return (
        "You are writing an evaluator-facing explainability report for a multiclass medical readmission model.\n"
        "Write in simple, precise English for non-ML reviewers.\n"
        "Do not invent facts. Use only the provided evidence.\n\n"
        "Return STRICTLY in markdown with these sections:\n"
        "1) Executive Summary (3 bullets)\n"
        "2) Global Drivers (top features and what they imply for modelling)\n"
        "3) Local Case Explanations (for each representative sample)\n"
        "4) Reliability & Risks (imbalance risk, uncertainty, potential leakage checks)\n"
        "5) Action Plan (exactly 5 prioritized actions for next iteration)\n\n"
        "Evidence block:\n"
        "[Metrics]\n"
        f"{metric_block}\n\n"
        "[Prediction distribution]\n"
        f"{pred_dist_block}\n\n"
        "[Global SHAP importance]\n"
        f"{global_block}\n\n"
        "[Representative local SHAP snapshots]\n"
        f"{representative_block}\n\n"
        "[Baseline narrative draft]\n"
        f"{rule_based_summary}\n"
    )


def _build_vlm_prompt(chart_paths: list[Path]) -> str:
    """Build a chart-grounded VLM prompt for actionable interpretation."""
    chart_list = "\n".join(f"- {p}" for p in chart_paths) if chart_paths else "- (none)"
    return (
        "You are reviewing model evaluation charts for a multiclass readmission classifier.\n"
        "Interpret only what is visible in charts. If a value is unreadable, explicitly say so.\n"
        "Use simple English and connect observations to modelling decisions.\n\n"
        "Return STRICTLY in markdown with these sections:\n"
        "1) What Each Chart Shows\n"
        "2) Quantified Findings (at least 3; include approximate numbers when visible)\n"
        "3) Class-Imbalance Impact and Error Pattern\n"
        "4) Modelling Recommendations (exactly 5 prioritized actions)\n"
        "5) Expected Metric Impact (what should improve and why)\n\n"
        "Charts to interpret:\n"
        f"{chart_list}\n"
    )


def _ensure_openai_client():
    """Create OpenAI client when OPENAI_API_KEY is configured."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        from openai import OpenAI

        return OpenAI(api_key=api_key)
    except Exception:
        return None


def _call_openai_text_summary(prompt: str, llm_model: str) -> str:
    """Call OpenAI text model and return (summary, error_message)."""
    client = _ensure_openai_client()
    if client is None:
        return "", "OPENAI_API_KEY is missing or OpenAI client import failed."
    try:
        resp = client.responses.create(
            model=llm_model,
            input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
        )
        return (resp.output_text or "").strip(), ""
    except Exception:
        return "", "OpenAI text request failed."


def _call_openai_vlm_summary(prompt: str, chart_paths: list[Path], llm_model: str) -> str:
    """Call OpenAI with chart images and return (summary, error_message)."""
    client = _ensure_openai_client()
    if client is None:
        return "", "OPENAI_API_KEY is missing or OpenAI client import failed."

    content: list[dict[str, str]] = [{"type": "input_text", "text": prompt}]
    for chart_path in chart_paths:
        if not chart_path.exists():
            continue
        suffix = chart_path.suffix.lower()
        if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
            continue
        mime = "image/png" if suffix == ".png" else "image/jpeg"
        if suffix == ".webp":
            mime = "image/webp"
        data = base64.b64encode(chart_path.read_bytes()).decode("utf-8")
        content.append({"type": "input_image", "image_url": f"data:{mime};base64,{data}"})

    if len(content) == 1:
        return "", "No valid chart images were found for VLM input."

    try:
        resp = client.responses.create(
            model=llm_model,
            input=[{"role": "user", "content": content}],
        )
        return (resp.output_text or "").strip(), ""
    except Exception:
        return "", "OpenAI VLM request failed."


def main() -> None:
    """Run explainability workflow and export all artifacts."""
    args = parse_args()

    try:
        from autogluon.tabular import TabularPredictor
    except ImportError as exc:
        raise ImportError(
            "AutoGluon is required. Install with `uv pip install autogluon.tabular`."
        ) from exc

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clear stale summary/metadata files first so every run rewrites final outputs.
    for stale_name in [
        "llm_summary.txt",
        "llm_summary_rule_based.txt",
        "vlm_chart_summary.txt",
        "run_metadata.json",
    ]:
        stale_path = output_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model path '{model_path}' does not exist. Run `src/eda_train.py` first."
        )

    raw_df = pd.read_csv(input_path)
    raw_df = _clean_dataframe(raw_df, args.label)
    raw_df = _drop_redundant_raw_id_columns(raw_df)
    predictor = TabularPredictor.load(str(model_path))

    expected_features = _expected_feature_columns(predictor)
    model_df = raw_df.copy()
    model_X = model_df.drop(columns=[args.label]) if args.label in model_df.columns else model_df.copy()

    feature_transform_applied = False
    missing_required = [c for c in expected_features if c not in model_X.columns]
    if missing_required:
        try:
            from eda_infer import AdvancedFeatureEngineer
        except Exception as exc:
            raise KeyError(
                "Model expects engineered columns not present in input, and "
                "AdvancedFeatureEngineer import failed. "
                f"Missing columns sample: {missing_required[:10]}"
            ) from exc

        train_path = Path(args.train_path)
        if not train_path.exists():
            raise FileNotFoundError(
                f"Training split required for FE refit not found: {train_path}"
            )
        train_df = pd.read_csv(train_path)
        train_df = _clean_dataframe(train_df, args.label)
        train_df = _drop_redundant_raw_id_columns(train_df)

        fe = AdvancedFeatureEngineer(
            label_col=args.label,
            rare_threshold=args.rare_threshold,
            n_clusters=args.n_clusters,
            n_pca_components=args.n_pca_components,
            random_state=args.random_state,
        )
        fe.fit(train_df)
        model_df = fe.transform(raw_df.copy())
        model_X = model_df.drop(columns=[args.label]) if args.label in model_df.columns else model_df.copy()
        feature_transform_applied = True

        still_missing = [c for c in expected_features if c not in model_X.columns]
        if still_missing:
            raise KeyError(
                "Engineered feature generation completed, but model-required columns "
                f"are still missing. Missing columns sample: {still_missing[:10]}"
            )

    if expected_features:
        model_X = model_X.reindex(columns=expected_features)

    y_true = model_df[args.label].astype(str) if args.label in model_df.columns else None
    y_pred = predictor.predict(model_X).astype(str)
    proba = predictor.predict_proba(model_X)
    if isinstance(proba, pd.Series):
        proba = proba.to_frame(name=str(proba.name or "positive"))

    representative_ids = _select_representative_indices(
        y_true=y_true,
        y_pred=y_pred,
        sample_size=args.sample_size,
        random_state=args.random_state,
    )
    representative_df = raw_df.loc[representative_ids].copy()
    representative_df.to_csv(output_dir / "representative_samples.csv", index=True)

    fi_data = model_df.copy()
    if len(fi_data) > args.feature_importance_sample:
        fi_data = fi_data.sample(args.feature_importance_sample, random_state=args.random_state)

    try:
        fi = predictor.feature_importance(fi_data, silent=True)
        fi = fi.reset_index().rename(columns={"index": "feature"})
    except Exception:
        fi = pd.DataFrame({"feature": model_X.columns.tolist(), "importance": np.nan})

    fi.to_csv(output_dir / "feature_importance.csv", index=False)
    if "importance" in fi.columns and fi["importance"].notna().any():
        fi_sorted = fi.sort_values("importance", ascending=False).reset_index(drop=True)
        _plot_global_importance(
            fi_sorted,
            output_dir / "feature_importance_top.png",
            top_n=args.top_features,
            value_col="importance",
            title="Global Feature Importance",
        )
    else:
        fi_sorted = fi.copy()

    local_shap_df: pd.DataFrame | None = None
    shap_global_df: pd.DataFrame | None = None
    shap_error = ""
    total_local_records = 0
    n_rows_explained = 0
    try:
        import shap

        background = model_X.sample(min(args.background_size, len(model_X)), random_state=args.random_state)

        shap_schema = _build_shap_schema(background)
        background_enc = _encode_shap_frame(background, shap_schema)

        predict_fn = _build_predict_proba_fn(
            predictor,
            model_X.columns.tolist(),
            shap_schema=shap_schema,
        )
        explainer = shap.Explainer(predict_fn, background_enc)
        class_names = [str(c) for c in proba.columns.tolist()]

        # Requirement: generate local SHAP explanations for all rows.
        local_path = output_dir / "shap_local_contributions.csv"
        if local_path.exists():
            local_path.unlink()

        batch_size = max(1, int(args.local_shap_batch_size))
        all_indices = model_X.index.tolist()
        global_abs_sum: dict[str, float] = {}
        representative_local_frames: list[pd.DataFrame] = []
        for start in range(0, len(all_indices), batch_size):
            batch_ids = all_indices[start : start + batch_size]
            batch_X = model_X.loc[batch_ids]
            batch_enc = _encode_shap_frame(batch_X, shap_schema)
            shap_batch = explainer(batch_enc)

            local_batch_df = _extract_local_shap(
                shap_values=shap_batch,
                explain_df=batch_X,
                predicted_labels=y_pred,
                class_names=class_names,
            )
            local_batch_df.to_csv(
                local_path,
                mode="a",
                header=not local_path.exists(),
                index=False,
            )

            total_local_records += int(len(local_batch_df))
            n_rows_explained += int(len(batch_X))
            tmp = local_batch_df[["feature", "shap_value"]].copy()
            tmp["abs_shap"] = tmp["shap_value"].abs()
            grouped = tmp.groupby("feature", as_index=False)["abs_shap"].sum()
            for row in grouped.itertuples(index=False):
                global_abs_sum[str(row.feature)] = global_abs_sum.get(str(row.feature), 0.0) + float(row.abs_shap)

            rep_local = local_batch_df[local_batch_df["sample_id"].isin(representative_ids)]
            if not rep_local.empty:
                representative_local_frames.append(rep_local)

        # Global SHAP derived from all-row local SHAP aggregation.
        shap_global_df = pd.DataFrame(
            {
                "feature": list(global_abs_sum.keys()),
                "mean_abs_shap": [v / max(1, n_rows_explained) for v in global_abs_sum.values()],
            }
        ).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        shap_global_df.to_csv(output_dir / "shap_global_importance.csv", index=False)
        _plot_global_importance(
            shap_global_df,
            output_dir / "shap_global_importance.png",
            top_n=args.top_features,
            value_col="mean_abs_shap",
            title="Global SHAP Importance (All Rows)",
        )

        local_shap_df = (
            pd.concat(representative_local_frames, axis=0, ignore_index=True)
            if representative_local_frames
            else pd.DataFrame(columns=["sample_id", "predicted_label", "feature", "shap_value"])
        )
        local_shap_df.to_csv(output_dir / "shap_local_representative.csv", index=False)

        for sample_id in representative_ids:
            _plot_local_contributions(
                local_df=local_shap_df,
                sample_id=sample_id,
                output_path=output_dir / f"shap_local_{sample_id}.png",
                top_n=10,
            )
    except Exception as exc:
        shap_error = str(exc)

    summary_global = shap_global_df if shap_global_df is not None else fi_sorted.rename(
        columns={"importance": "mean_abs_shap"}
    )
    rule_based_summary = _build_llm_summary(
        global_importance=summary_global,
        representative_df=representative_df,
        predictions=y_pred,
        probabilities=proba,
        local_shap_df=local_shap_df,
        label_col=args.label,
    )
    llm_summary = rule_based_summary

    # Optional LLM rewrite for easy-English evaluator narrative.
    llm_prompt = _build_llm_rewrite_prompt(
        rule_based_summary=rule_based_summary,
        global_importance=summary_global,
        local_shap_df=local_shap_df,
        representative_df=representative_df,
        predictions=y_pred,
        probabilities=proba,
        y_true=y_true,
        label_col=args.label,
    )
    llm_rewrite, llm_error = _call_openai_text_summary(llm_prompt, args.llm_model)
    if llm_rewrite:
        llm_summary = llm_rewrite
    elif llm_error:
        llm_summary += f"\n\nLLM rewrite status: fallback to rule-based summary ({llm_error})"

    if shap_error:
        llm_summary += (
            "\n\nSHAP status: SHAP explanation was skipped due to runtime/import issue: "
            f"{shap_error}"
        )

    (output_dir / "llm_summary.txt").write_text(llm_summary, encoding="utf-8")
    (output_dir / "llm_summary_rule_based.txt").write_text(rule_based_summary, encoding="utf-8")

    # Optional VLM-style chart interpretation from provided chart/image paths.
    chart_paths = [
        Path(p.strip()) for p in str(args.eda_chart_paths).split(",") if p.strip()
    ]
    vlm_prompt = _build_vlm_prompt(chart_paths)
    vlm_summary, vlm_error = _call_openai_vlm_summary(vlm_prompt, chart_paths, args.llm_model)
    if vlm_summary:
        vlm_text = vlm_summary
    else:
        reason = vlm_error if vlm_error else "No VLM output returned."
        vlm_text = f"VLM summary unavailable. Fallback reason: {reason}"
    (output_dir / "vlm_chart_summary.txt").write_text(vlm_text, encoding="utf-8")

    run_metadata = {
        "model_path": str(model_path),
        "input_path": str(input_path),
        "rows": int(len(raw_df)),
        "representative_ids": [int(i) for i in representative_ids],
        "shap_available": shap_error == "",
        "shap_error": shap_error,
        "feature_transform_applied": feature_transform_applied,
        "expected_feature_count": int(len(expected_features)),
        "missing_required_count_before_transform": int(len(missing_required)),
        "shap_local_records": int(total_local_records),
        "shap_rows_explained": int(n_rows_explained),
        "openai_enabled": bool(os.getenv("OPENAI_API_KEY", "").strip()),
        "llm_model": args.llm_model,
        "llm_rewrite_generated": bool(llm_rewrite),
        "llm_error": llm_error,
        "eda_chart_paths": [str(p) for p in chart_paths],
        "vlm_summary_generated": bool(vlm_summary),
        "vlm_error": vlm_error,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")

    print("Explainability artifacts generated.")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
