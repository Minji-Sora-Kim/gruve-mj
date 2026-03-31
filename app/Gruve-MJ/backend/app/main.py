"""FastAPI backend for CSV inferencing, SHAP attributions, and LLM chat."""

from __future__ import annotations

import base64
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


WORKSPACE_ROOT = Path(os.getenv("WORKSPACE_ROOT", "/app/context"))
DEFAULT_MODEL_CANDIDATES = [
    WORKSPACE_ROOT / "ag_models",
    WORKSPACE_ROOT / "backend" / "models" / "ag_models",
    WORKSPACE_ROOT / "models" / "ag_models",
    WORKSPACE_ROOT / "experiments" / "ag_models_optimized",
    WORKSPACE_ROOT / "experiments" / "ag_models_baseline",
]
DEFAULT_BACKGROUND_PATH = WORKSPACE_ROOT / "data/processed/train.csv"
LABEL_COL = os.getenv("LABEL_COL", "readmitted")
FRONTEND_DIST_DIR = Path(os.getenv("FRONTEND_DIST_DIR", "/app/frontend_dist"))


@dataclass
class ModelRuntime:
    """Runtime objects shared across inference endpoints."""

    predictor: Any
    model_path: Path
    feature_columns: list[str]
    background_df: pd.DataFrame
    class_labels: list[str]
    shap_explainer: Any | None = None


class ExplainRowRequest(BaseModel):
    """Request payload for row-level explanation."""

    row_id: int | None = None
    row: dict[str, Any]
    predicted_label: str | None = None
    top_n: int = Field(default=8, ge=3, le=20)
    llm_model: str = "gpt-4o-mini"


class ChatRequest(BaseModel):
    """Request payload for Q&A over prediction/explainability context."""

    question: str = Field(min_length=3, max_length=4000)
    context: dict[str, Any] | None = None
    llm_model: str = "gpt-4o-mini"


class VLMSummaryRequest(BaseModel):
    """Request payload for chart interpretation using VLM-capable model."""

    chart_paths: list[str] = Field(default_factory=list)
    prompt: str = (
        "Interpret these EDA charts in simple English. "
        "Highlight class-imbalance implications and modelling actions."
    )
    llm_model: str = "gpt-4o-mini"


def _resolve_model_path() -> Path:
    """Resolve model path from environment or candidate directories."""
    env_path = os.getenv("MODEL_PATH")
    if env_path:
        model_path = Path(env_path)
        if model_path.exists():
            return model_path
        # Allow relative MODEL_PATH values resolved from WORKSPACE_ROOT.
        rel_model_path = WORKSPACE_ROOT / env_path
        if rel_model_path.exists():
            return rel_model_path
    for candidate in DEFAULT_MODEL_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No model directory found. Set MODEL_PATH or provide one of: "
        + ", ".join(str(p) for p in DEFAULT_MODEL_CANDIDATES)
    )


def _require_runtime() -> ModelRuntime:
    """Return loaded runtime or raise a 503 with actionable setup guidance."""
    runtime = getattr(app.state, "runtime", None)
    if runtime is None:
        runtime_error = getattr(app.state, "runtime_error", "Runtime is not initialized.")
        raise HTTPException(
            status_code=503,
            detail=(
                f"{runtime_error} "
                "Upload model artifacts under /app/context/backend/models/ag_models "
                "or set MODEL_PATH to a valid predictor directory."
            ),
        )
    return runtime


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize columns and convert '?' placeholders to missing values."""
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    out = out.replace("?", np.nan)
    return out


def _drop_redundant_raw_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop *_raw columns when decoded base ID column already exists."""
    out = df.copy()
    for raw_col in [c for c in out.columns if c.endswith("_raw")]:
        base_col = raw_col[: -len("_raw")]
        if base_col in out.columns:
            out = out.drop(columns=[raw_col])
    return out


def _build_predict_proba_fn(runtime: ModelRuntime):
    """Build SHAP-compatible predict_proba callable."""

    def predict_fn(batch: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(batch, np.ndarray):
            batch_df = pd.DataFrame(batch, columns=runtime.feature_columns)
        else:
            batch_df = batch.copy()
        batch_df = batch_df.reindex(columns=runtime.feature_columns)
        proba = runtime.predictor.predict_proba(batch_df)
        if isinstance(proba, pd.Series):
            proba = proba.to_frame(name=str(proba.name or "positive"))
        return proba.to_numpy()

    return predict_fn


def _ensure_openai_client():
    """Create OpenAI client when OPENAI_API_KEY is available."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        from openai import OpenAI

        return OpenAI(api_key=api_key)
    except Exception:
        return None


def _call_llm_text(prompt: str, llm_model: str) -> str:
    """Call OpenAI text model; return empty string on failure."""
    client = _ensure_openai_client()
    if client is None:
        return ""
    try:
        resp = client.responses.create(
            model=llm_model,
            input=[
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}],
                }
            ],
        )
        return (resp.output_text or "").strip()
    except Exception:
        return ""


def _call_vlm_with_images(prompt: str, image_paths: list[Path], llm_model: str) -> str:
    """Call VLM-capable endpoint with local images encoded as data URLs."""
    client = _ensure_openai_client()
    if client is None:
        return ""
    if not image_paths:
        return ""

    content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
    for path in image_paths:
        if not path.exists():
            continue
        mime = "image/png"
        if path.suffix.lower() in {".jpg", ".jpeg"}:
            mime = "image/jpeg"
        elif path.suffix.lower() == ".webp":
            mime = "image/webp"
        data = base64.b64encode(path.read_bytes()).decode("utf-8")
        content.append(
            {
                "type": "input_image",
                "image_url": f"data:{mime};base64,{data}",
            }
        )
    if len(content) <= 1:
        return ""

    try:
        resp = client.responses.create(
            model=llm_model,
            input=[{"role": "user", "content": content}],
        )
        return (resp.output_text or "").strip()
    except Exception:
        return ""


def _load_runtime() -> ModelRuntime:
    """Load predictor and background data for SHAP explanations."""
    try:
        from autogluon.tabular import TabularPredictor
    except ImportError as exc:
        raise RuntimeError(
            "AutoGluon not installed in backend environment."
        ) from exc

    model_path = _resolve_model_path()
    predictor = TabularPredictor.load(str(model_path))

    background_df = pd.DataFrame()
    if DEFAULT_BACKGROUND_PATH.exists():
        raw_bg = pd.read_csv(DEFAULT_BACKGROUND_PATH)
        raw_bg = _drop_redundant_raw_id_columns(_clean_dataframe(raw_bg))
        if LABEL_COL in raw_bg.columns:
            raw_bg = raw_bg.drop(columns=[LABEL_COL])
        background_df = raw_bg.sample(min(120, len(raw_bg)), random_state=42)

    feature_columns = list(background_df.columns)
    if not feature_columns:
        # Fallback: derive from predictor metadata if background is unavailable.
        feature_columns = list(getattr(predictor, "feature_metadata_in", {}).keys()) if hasattr(
            predictor, "feature_metadata_in"
        ) else []

    class_labels = [str(c) for c in getattr(predictor, "class_labels", [])]
    if not class_labels:
        # Infer classes lazily from a tiny prediction pass if needed.
        class_labels = []

    return ModelRuntime(
        predictor=predictor,
        model_path=model_path,
        feature_columns=feature_columns,
        background_df=background_df,
        class_labels=class_labels,
        shap_explainer=None,
    )


def _standardize_features(df: pd.DataFrame, runtime: ModelRuntime) -> pd.DataFrame:
    """Apply shared cleanup and align columns to model feature schema."""
    out = _drop_redundant_raw_id_columns(_clean_dataframe(df))
    if LABEL_COL in out.columns:
        out = out.drop(columns=[LABEL_COL])
    if runtime.feature_columns:
        out = out.reindex(columns=runtime.feature_columns)
    return out


def _ensure_shap_explainer(runtime: ModelRuntime, fallback_background: pd.DataFrame) -> Any:
    """Build SHAP explainer once and cache it on runtime."""
    if runtime.shap_explainer is not None:
        return runtime.shap_explainer
    bg = runtime.background_df if not runtime.background_df.empty else fallback_background
    bg = bg.copy()
    if bg.empty:
        raise RuntimeError("Cannot build SHAP explainer: no background dataset available.")
    bg = bg.sample(min(80, len(bg)), random_state=42)
    predict_fn = _build_predict_proba_fn(runtime)
    runtime.shap_explainer = shap.Explainer(predict_fn, bg)
    return runtime.shap_explainer


def _row_local_shap(
    runtime: ModelRuntime,
    row_df: pd.DataFrame,
    predicted_label: str | None,
    top_n: int,
) -> list[dict[str, Any]]:
    """Compute top local SHAP contributors for one row."""
    explainer = _ensure_shap_explainer(runtime, fallback_background=row_df)
    shap_values = explainer(row_df)
    values = np.asarray(shap_values.values)

    if values.ndim == 3:
        classes = runtime.class_labels or [str(c) for c in runtime.predictor.predict_proba(row_df).columns]
        class_idx = classes.index(predicted_label) if predicted_label in classes else 0
        contrib = values[0, :, class_idx]
    elif values.ndim == 2:
        contrib = values[0, :]
    else:
        raise RuntimeError(f"Unsupported SHAP output shape: {values.shape}")

    feat_names = row_df.columns.tolist()
    records = pd.DataFrame(
        {"feature": feat_names, "shap_value": contrib, "abs_value": np.abs(contrib)}
    ).sort_values("abs_value", ascending=False)
    top = records.head(top_n)
    return [
        {
            "feature": str(r.feature),
            "shap_value": float(r.shap_value),
            "direction": "increase" if float(r.shap_value) >= 0 else "decrease",
        }
        for r in top.itertuples(index=False)
    ]


def _safe_json_row(row: pd.Series) -> dict[str, Any]:
    """Convert pandas row to JSON-safe dict."""
    result: dict[str, Any] = {}
    for key, value in row.items():
        if pd.isna(value):
            result[str(key)] = None
        elif isinstance(value, (np.integer, np.int64)):
            result[str(key)] = int(value)
        elif isinstance(value, (np.floating, np.float64)):
            result[str(key)] = float(value)
        else:
            result[str(key)] = value
    return result


def _select_representative_row_ids(rows: list[dict[str, Any]]) -> list[int]:
    """Pick representative rows for LLM/chat: class exemplars + uncertain + error."""
    if not rows:
        return []

    selected: list[int] = []

    # High-confidence exemplar per predicted class.
    by_class: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_class.setdefault(str(row.get("predicted_label", "unknown")), []).append(row)
    for cls, cls_rows in by_class.items():
        best = sorted(cls_rows, key=lambda r: float(r.get("confidence", 0.0)), reverse=True)[0]
        selected.append(int(best["row_id"]))

    # Add one uncertain case (lowest confidence).
    uncertain = sorted(rows, key=lambda r: float(r.get("confidence", 0.0)))[0]
    selected.append(int(uncertain["row_id"]))

    # Add one misclassified row if labels available.
    misclassified = [
        r for r in rows
        if r.get("true_label") is not None and str(r.get("true_label")) != str(r.get("predicted_label"))
    ]
    if misclassified:
        selected.append(int(misclassified[0]["row_id"]))

    # Keep order and uniqueness.
    return list(dict.fromkeys(selected))


app = FastAPI(title="Gruve Inference API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    """Initialize predictor runtime once during app startup."""
    app.state.runtime = None
    app.state.runtime_error = ""
    try:
        app.state.runtime = _load_runtime()
    except Exception as exc:
        # Keep app alive so health endpoint explains what is missing instead of
        # causing container launch timeout.
        app.state.runtime_error = str(exc)
    app.state.representative_row_ids = set()


@app.get("/health")
def health() -> dict[str, Any]:
    """Health check endpoint with active model metadata."""
    runtime: ModelRuntime | None = getattr(app.state, "runtime", None)
    if runtime is None:
        return {
            "status": "degraded",
            "ready": False,
            "runtime_error": getattr(app.state, "runtime_error", "Runtime is not initialized."),
            "model_candidates": [str(p) for p in DEFAULT_MODEL_CANDIDATES],
        }
    return {
        "status": "ok",
        "ready": True,
        "model_path": str(runtime.model_path),
        "feature_count": len(runtime.feature_columns),
    }


@app.post("/api/predict")
async def predict_csv(
    file: UploadFile = File(...),
    include_input: bool = Form(default=False),
    explain_rows: int = Form(default=0),
    explain_top_n: int = Form(default=8),
    max_rows: int = Form(default=300),
) -> dict[str, Any]:
    """Run inference for uploaded CSV and attach row-level SHAP summaries."""
    runtime = _require_runtime()
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    payload = await file.read()
    try:
        raw_df = pd.read_csv(io.BytesIO(payload))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {exc}") from exc

    input_df = _clean_dataframe(raw_df)
    input_df = _drop_redundant_raw_id_columns(input_df)
    has_label = LABEL_COL in input_df.columns

    features_df = _standardize_features(input_df, runtime)
    if features_df.empty:
        raise HTTPException(status_code=400, detail="No valid feature columns after preprocessing.")

    preds = runtime.predictor.predict(features_df).astype(str)
    proba = runtime.predictor.predict_proba(features_df)
    if isinstance(proba, pd.Series):
        proba = proba.to_frame(name=str(proba.name or "positive"))
    class_labels = [str(c) for c in proba.columns]
    if not runtime.class_labels:
        runtime.class_labels = class_labels

    result_rows: list[dict[str, Any]] = []
    output_limit = min(max(1, max_rows), len(features_df))
    output_indices = features_df.index[:output_limit].tolist()
    explain_count = output_limit if explain_rows <= 0 else min(max(0, explain_rows), output_limit)
    explain_indices = set(output_indices[:explain_count])

    for idx in output_indices:
        row_payload: dict[str, Any] = {
            "row_id": int(idx),
            "predicted_label": str(preds.loc[idx]),
            "confidence": float(proba.loc[idx].max()),
            "probabilities": {str(c): float(proba.loc[idx, c]) for c in class_labels},
            "local_shap": [],
        }
        if has_label:
            row_payload["true_label"] = str(input_df.loc[idx, LABEL_COL])
        if include_input:
            row_payload["input"] = _safe_json_row(input_df.loc[idx])
        if idx in explain_indices:
            try:
                row_df = features_df.loc[[idx]]
                row_payload["local_shap"] = _row_local_shap(
                    runtime=runtime,
                    row_df=row_df,
                    predicted_label=row_payload["predicted_label"],
                    top_n=explain_top_n,
                )
            except Exception as exc:
                row_payload["local_shap_error"] = str(exc)
        result_rows.append(row_payload)

    representative_ids = _select_representative_row_ids(result_rows)
    rep_set = set(representative_ids)
    app.state.representative_row_ids = rep_set
    for row in result_rows:
        row["is_representative"] = int(row["row_id"]) in rep_set

    summary: dict[str, Any] = {
        "rows_total": int(len(features_df)),
        "rows_returned": int(len(result_rows)),
        "class_distribution_pred": preds.value_counts(normalize=True).to_dict(),
        "model_path": str(runtime.model_path),
        "explain_rows": explain_count,
        "representative_rows": len(representative_ids),
    }

    if has_label:
        from sklearn.metrics import accuracy_score, f1_score

        y_true = input_df[LABEL_COL].astype(str)
        summary["accuracy"] = float(accuracy_score(y_true, preds))
        summary["f1_macro"] = float(f1_score(y_true, preds, average="macro"))

    representative_rows = [r for r in result_rows if int(r["row_id"]) in rep_set]
    return {
        "summary": summary,
        "class_labels": class_labels,
        "rows": result_rows,
        "representative_row_ids": representative_ids,
        "representative_rows": representative_rows,
    }


@app.post("/api/explain-row")
def explain_row(req: ExplainRowRequest) -> dict[str, Any]:
    """Generate local SHAP + LLM narrative for representative rows only."""
    runtime = _require_runtime()
    rep_ids: set[int] = set(getattr(app.state, "representative_row_ids", set()))
    if req.row_id is None:
        raise HTTPException(status_code=400, detail="row_id is required for representative-row validation.")
    if int(req.row_id) not in rep_ids:
        raise HTTPException(
            status_code=403,
            detail="LLM explanation is restricted to representative rows only.",
        )

    row_df = pd.DataFrame([req.row])
    row_df = _standardize_features(row_df, runtime)
    if row_df.empty:
        raise HTTPException(status_code=400, detail="Row has no valid model features.")

    pred = req.predicted_label
    if not pred:
        pred = str(runtime.predictor.predict(row_df).iloc[0])
    proba = runtime.predictor.predict_proba(row_df)
    if isinstance(proba, pd.Series):
        proba = proba.to_frame(name=str(proba.name or "positive"))
    confidence = float(proba.max(axis=1).iloc[0])

    contributions = _row_local_shap(runtime, row_df, pred, req.top_n)
    prompt = (
        "Explain the prediction in plain English for a non-technical healthcare reviewer.\n"
        f"Prediction: {pred}\n"
        f"Confidence: {confidence:.3f}\n"
        f"Top SHAP contributions: {json.dumps(contributions, ensure_ascii=False)}\n"
        "Keep answer under 8 bullet points."
    )
    llm_text = _call_llm_text(prompt, req.llm_model)
    if not llm_text:
        llm_text = (
            f"Prediction is '{pred}' (confidence {confidence:.3f}). "
            "Positive SHAP values increased this class score; negative values decreased it. "
            "Review top factors and confirm with clinical context."
        )

    return {
        "predicted_label": pred,
        "confidence": confidence,
        "local_shap": contributions,
        "llm_explanation": llm_text,
    }


@app.post("/api/chat")
def chat(req: ChatRequest) -> dict[str, Any]:
    """Answer questions for representative-row context only."""
    context = req.context or {}
    selected_row = context.get("selected_row") if isinstance(context, dict) else None
    row_id = None
    if isinstance(selected_row, dict) and "row_id" in selected_row:
        try:
            row_id = int(selected_row["row_id"])
        except Exception:
            row_id = None

    rep_ids: set[int] = set(getattr(app.state, "representative_row_ids", set()))
    if row_id is None or row_id not in rep_ids:
        return {
            "answer": (
                "Chat is limited to representative rows only. "
                "Select a row marked as representative and try again."
            )
        }

    ctx = json.dumps(context, ensure_ascii=False)
    prompt = (
        "You are assisting a grader evaluating a diabetic readmission model.\n"
        "Answer in clear English with concrete references to provided context.\n"
        "If context is missing, explicitly say what is missing.\n\n"
        f"Context JSON:\n{ctx}\n\n"
        f"Question:\n{req.question}"
    )
    answer = _call_llm_text(prompt, req.llm_model)
    if not answer:
        answer = (
            "OPENAI_API_KEY is not configured or LLM call failed. "
            "I can still answer from structured context if you include prediction, confidence, and SHAP factors."
        )
    return {"answer": answer}


@app.post("/api/vlm-summary")
def vlm_summary(req: VLMSummaryRequest) -> dict[str, Any]:
    """Interpret EDA charts with VLM when API key is available."""
    resolved_paths: list[Path] = []
    for raw in req.chart_paths[:5]:
        p = Path(raw)
        resolved = p if p.is_absolute() else (WORKSPACE_ROOT / p)
        if resolved.exists():
            resolved_paths.append(resolved)

    vlm_text = _call_vlm_with_images(req.prompt, resolved_paths, req.llm_model)
    if not vlm_text:
        vlm_text = (
            "VLM summary unavailable (missing OPENAI_API_KEY or request failure). "
            "Use /api/chat with explicit chart statistics for deterministic fallback reasoning."
        )
    return {
        "summary": vlm_text,
        "resolved_chart_paths": [str(p) for p in resolved_paths],
    }


# Mount static frontend for single-container deployment targets (e.g., HF Spaces).
# API routes are declared above, so they keep priority over the catch-all mount.
if FRONTEND_DIST_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST_DIR), html=True), name="frontend")
