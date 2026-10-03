FROM node:22-alpine AS dashboard
WORKDIR /app/dashboard
COPY dashboard/package*.json ./
RUN npm ci
COPY dashboard/ ./
RUN npm run build

FROM python:3.11-slim AS runtime
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 BLACKBOX_DB=/app/data/traces.db \
    BLACKBOX_METRICS=/app/data/metrics.json BLACKBOX_MODEL_DIR=/app/data/models HF_HOME=/app/data/cache/hf
COPY pyproject.toml README.md ./
COPY blackbox/ ./blackbox/
# CPU torch keeps the image small; diagnosis runs in ~100 ms per run on CPU.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir .
COPY --from=dashboard /app/dashboard/dist ./dashboard/dist
RUN mkdir -p /app/data
EXPOSE 8000
CMD ["uvicorn", "blackbox.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
