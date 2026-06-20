"""
api_server.py
-------------
Technical Phase — Implementation

FastAPI service that loads the serialized Iris model ONCE at startup and
exposes a POST /predict endpoint for real-time inference on JSON feature
arrays.

Design notes:
- Model loading happens in a module-level block, not inside the request
  handler. Loading a model from disk on every request would add
  unnecessary I/O latency to every single prediction — the model should
  live in memory for the lifetime of the process.
- Pydantic validates the incoming feature array's shape and types before
  it ever reaches the model, returning a clean 422 error for malformed
  input rather than a confusing internal stack trace.
- This service is intentionally synchronous and stateless per-request —
  Kafka (see kafka_client.py) is the layer that adds asynchronous,
  decoupled processing on top of this same prediction logic.
"""

import json
import logging
import time
from typing import List, Dict

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load model and labels at module import time (process startup)
# ---------------------------------------------------------------------------
MODEL_PATH = "model.pkl"
LABELS_PATH = "class_labels.json"

logger.info("Loading model from '%s' ...", MODEL_PATH)
model = joblib.load(MODEL_PATH)

with open(LABELS_PATH) as f:
    label_data = json.load(f)
    CLASS_NAMES = label_data["class_names"]
    FEATURE_NAMES = label_data["feature_names"]

logger.info("Model loaded. Classes: %s", CLASS_NAMES)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Iris Prediction API",
    description="Production-style FastAPI service serving a LogisticRegression Iris classifier.",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class PredictionRequest(BaseModel):
    """
    Schema for POST /predict.

    Expects exactly 4 features in the order: sepal_length, sepal_width,
    petal_length, petal_width (cm) — matching the Iris dataset's
    feature_names order.
    """

    features: List[float] = Field(
        ...,
        description="Exactly 4 floats: [sepal_length, sepal_width, petal_length, petal_width] in cm.",
        examples=[[5.1, 3.5, 1.4, 0.2]],
    )

    @field_validator("features")
    @classmethod
    def validate_feature_count(cls, v: List[float]) -> List[float]:
        if len(v) != 4:
            raise ValueError(
                f"Expected exactly 4 features, got {len(v)}. "
                f"Order: {FEATURE_NAMES}"
            )
        return v


class PredictionResponse(BaseModel):
    """Schema for the POST /predict response."""

    predicted_class: str
    predicted_class_index: int
    probabilities: Dict[str, float]
    inference_time_ms: float


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", summary="Health check")
def root():
    """Simple health-check endpoint confirming the service and model are live."""
    return {
        "status": "ok",
        "service": "iris-prediction-api",
        "model_loaded": model is not None,
        "classes": CLASS_NAMES,
        "expected_feature_order": FEATURE_NAMES,
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict Iris species from 4 numeric features",
)
def predict(request: PredictionRequest):
    """
    Run inference on a single Iris feature vector and return the predicted
    species along with class probabilities.

    Example request body:
        {"features": [5.1, 3.5, 1.4, 0.2]}

    Example response:
        {
          "predicted_class": "setosa",
          "predicted_class_index": 0,
          "probabilities": {"setosa": 0.97, "versicolor": 0.02, "virginica": 0.01},
          "inference_time_ms": 1.23
        }
    """
    try:
        start = time.perf_counter()

        X = np.array(request.features).reshape(1, -1)

        pred_index = int(model.predict(X)[0])
        pred_proba = model.predict_proba(X)[0]

        elapsed_ms = (time.perf_counter() - start) * 1000

        probabilities = {
            CLASS_NAMES[i]: round(float(p), 4) for i, p in enumerate(pred_proba)
        }

        logger.info(
            "Prediction | features=%s | predicted=%s | time=%.2fms",
            request.features,
            CLASS_NAMES[pred_index],
            elapsed_ms,
        )

        return PredictionResponse(
            predicted_class=CLASS_NAMES[pred_index],
            predicted_class_index=pred_index,
            probabilities=probabilities,
            inference_time_ms=round(elapsed_ms, 3),
        )

    except Exception as exc:
        logger.error("Prediction failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {type(exc).__name__}: {exc}",
        )


# ---------------------------------------------------------------------------
# Entry point (for direct execution: python api_server.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
