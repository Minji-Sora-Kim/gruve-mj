---
title: Gruve Readmission Inference App
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Gruve Full-Stack Inference (HF Spaces Docker)

This app provides:
- CSV upload for `unseen_data.csv`
- prediction + confidence scores
- row-level SHAP contributions
- representative-row LLM explanation + chat
- VLM-style chart interpretation endpoint

## 1) Deploy to Hugging Face Spaces (Free)

1. Create a new **Space** on Hugging Face.
2. Choose **SDK = Docker**.
3. Push this `app/` folder contents to the Space repository root.
4. In Space Settings → Variables and secrets, add:
   - `OPENAI_API_KEY` (Secret, optional for LLM/VLM)

## 2) Required folders in the Space repo

Backend searches model/data under `WORKSPACE_ROOT=/app/context`.
Make sure these are present in the Space repo (inside the app root or copied in before push):

- `ag_models/` (final submission model)
- `data/processed/train.csv` (for SHAP background/reference)

If model artifacts are too large for normal Git, use Git LFS.

## 3) Local test (same Docker image)

```bash
docker build -t gruve-space-app ./app
docker run --rm -p 7860:7860 -e OPENAI_API_KEY="$OPENAI_API_KEY" gruve-space-app
```

Open:
- `http://localhost:7860`

## 4) Notes

- SHAP is computed for all returned rows; this can be slow on large files.
- LLM/chat is restricted to representative rows by design.
- If `OPENAI_API_KEY` is missing, app falls back to deterministic non-LLM responses.
