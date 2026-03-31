FROM node:22-alpine AS frontend_builder

WORKDIR /web
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV WORKSPACE_ROOT=/app/context
ENV MODEL_PATH=/app/context/ag_models
ENV LABEL_COL=readmitted
ENV FRONTEND_DIST_DIR=/app/frontend_dist

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Backend app code
COPY backend/app /app/backend/app

# Frontend static build
COPY --from=frontend_builder /web/dist /app/frontend_dist

# Copy full Space context. If model/data folders are included in the Space repo,
# backend will discover them under /app/context via WORKSPACE_ROOT.
COPY . /app/context

EXPOSE 7860

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "7860"]
