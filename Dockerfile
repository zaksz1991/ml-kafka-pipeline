# Dockerfile
# ----------
# Builds the API service container. The Kafka producer/consumer
# (kafka_client.py) is ALSO built from this same image — docker-compose.yml
# overrides the CMD for the 'producer' and 'consumer' services so we don't
# need three separate Dockerfiles for what is fundamentally the same
# Python environment and dependency set.

FROM python:3.9-slim

# Prevents Python from buffering stdout/stderr — critical for seeing
# logs in real time via `docker-compose logs -f`, especially for the
# Kafka consumer's continuous console output.
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# ── Install dependencies first (separate layer for build caching) ─────────
# This ordering means `docker build` only re-installs dependencies when
# requirements.txt actually changes, not on every code edit — significantly
# speeding up iterative development.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application code ───────────────────────────────────────────────
COPY train_model.py .
COPY api_server.py .
COPY kafka_client.py .

# ── Train and serialize the model at build time ─────────────────────────
# Baking the model into the image (rather than training at container
# startup) means: (1) startup is fast and deterministic, (2) the exact
# model version is pinned to the image tag, which matters for rollback
# and reproducibility in production.
RUN python train_model.py

# API service port
EXPOSE 8000

# Default command runs the API server.
# docker-compose.yml overrides this CMD for the producer/consumer services.
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
