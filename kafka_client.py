"""
kafka_client.py
----------------
Discovery Phase — Modular Setup / Technical Phase — Kafka Integration

Implements both the Producer and Consumer sides of the asynchronous
prediction pipeline, plus a small CLI to run either role.

Architecture:
    Producer:  reads feature vectors -> publishes to 'ml-requests' topic
    Consumer:  subscribes to 'ml-requests' -> calls the local FastAPI
               /predict endpoint -> publishes the result to
               'ml-predictions' -> also prints it to the console

Why a Consumer calls the REST API rather than loading the model directly:
    This keeps a single source of truth for inference logic (api_server.py).
    The Kafka layer is purely about decoupling and message transport; it
    does not duplicate model-loading or prediction code. In a larger
    system this also means the consumer could be scaled independently
    of the API, or the API could be swapped/upgraded without touching
    the streaming layer.

Topics:
    ml-requests     — raw feature vectors awaiting prediction
    ml-predictions  — completed predictions, keyed by request_id

Usage:
    python kafka_client.py producer   # sends 10 sample requests
    python kafka_client.py consumer   # listens and processes forever
"""

import json
import logging
import sys
import time
import uuid

import requests
from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import NoBrokersAvailable

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# In docker-compose, the broker is reachable at the service name 'kafka';
# locally (outside Docker) it would be 'localhost:9092'. We read from an
# env var so the same code works in both contexts without modification.
import os

KAFKA_BROKER = os.environ.get("KAFKA_BROKER", "kafka:9092")
API_URL = os.environ.get("API_URL", "http://api:8000/predict")

REQUEST_TOPIC = "ml-requests"
PREDICTION_TOPIC = "ml-predictions"

# Sample Iris feature vectors spanning all three classes, used for the
# 10-request producer demo required by the Action Phase.
SAMPLE_REQUESTS = [
    [5.1, 3.5, 1.4, 0.2],   # setosa
    [4.9, 3.0, 1.4, 0.2],   # setosa
    [4.7, 3.2, 1.3, 0.2],   # setosa
    [7.0, 3.2, 4.7, 1.4],   # versicolor
    [6.4, 3.2, 4.5, 1.5],   # versicolor
    [5.9, 3.0, 4.2, 1.5],   # versicolor
    [6.3, 3.3, 6.0, 2.5],   # virginica
    [5.8, 2.7, 5.1, 1.9],   # virginica
    [7.1, 3.0, 5.9, 2.1],   # virginica
    [6.7, 3.0, 5.2, 2.3],   # virginica
]


def _connect_producer(retries: int = 10, delay: float = 3.0) -> KafkaProducer:
    """
    Connect to the Kafka broker with retry logic.

    Kafka can take several seconds to become ready after container startup,
    especially the first time Zookeeper + Kafka initialize together. A
    naive single-attempt connection would fail if the producer starts
    before the broker is ready, which is common in docker-compose's
    default (non-health-check-gated) startup order.
    """
    for attempt in range(1, retries + 1):
        try:
            producer = KafkaProducer(
                bootstrap_servers=[KAFKA_BROKER],
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                key_serializer=lambda k: k.encode("utf-8") if k else None,
            )
            logger.info("Connected to Kafka broker at %s", KAFKA_BROKER)
            return producer
        except NoBrokersAvailable:
            logger.warning(
                "Kafka broker not available (attempt %d/%d). Retrying in %.0fs...",
                attempt, retries, delay,
            )
            time.sleep(delay)
    raise ConnectionError(
        f"Could not connect to Kafka broker at {KAFKA_BROKER} after {retries} attempts."
    )


def _connect_consumer(retries: int = 10, delay: float = 3.0) -> KafkaConsumer:
    """Connect a KafkaConsumer to the REQUEST_TOPIC, with the same retry logic."""
    for attempt in range(1, retries + 1):
        try:
            consumer = KafkaConsumer(
                REQUEST_TOPIC,
                bootstrap_servers=[KAFKA_BROKER],
                value_deserializer=lambda v: json.loads(v.decode("utf-8")),
                key_deserializer=lambda k: k.decode("utf-8") if k else None,
                auto_offset_reset="earliest",
                enable_auto_commit=True,
                group_id="ml-prediction-consumers",
            )
            logger.info(
                "Connected to Kafka broker at %s, subscribed to '%s'",
                KAFKA_BROKER, REQUEST_TOPIC,
            )
            return consumer
        except NoBrokersAvailable:
            logger.warning(
                "Kafka broker not available (attempt %d/%d). Retrying in %.0fs...",
                attempt, retries, delay,
            )
            time.sleep(delay)
    raise ConnectionError(
        f"Could not connect to Kafka broker at {KAFKA_BROKER} after {retries} attempts."
    )


# ---------------------------------------------------------------------------
# Producer
# ---------------------------------------------------------------------------

def run_producer():
    """
    Publish 10 sample Iris feature vectors to the 'ml-requests' topic.

    Each message carries a unique request_id (so the corresponding
    prediction can later be correlated) and a submitted_at timestamp
    (used to measure end-to-end request-to-prediction latency).
    """
    producer = _connect_producer()

    logger.info("Sending %d sample requests to topic '%s' ...", len(SAMPLE_REQUESTS), REQUEST_TOPIC)

    for i, features in enumerate(SAMPLE_REQUESTS, start=1):
        request_id = str(uuid.uuid4())
        message = {
            "request_id": request_id,
            "features": features,
            "submitted_at": time.time(),
        }

        producer.send(REQUEST_TOPIC, key=request_id, value=message)
        logger.info("[%d/%d] Sent request_id=%s | features=%s", i, len(SAMPLE_REQUESTS), request_id[:8], features)

        time.sleep(0.5)  # small delay so console output is readable during demo

    producer.flush()
    producer.close()
    logger.info("✅ All %d requests sent and flushed.", len(SAMPLE_REQUESTS))


# ---------------------------------------------------------------------------
# Consumer
# ---------------------------------------------------------------------------

def run_consumer():
    """
    Continuously consume from 'ml-requests', call the prediction API,
    publish the result to 'ml-predictions', and print it to the console.

    This is the asynchronous processing loop: it runs indefinitely,
    decoupled from whatever produced the request (a single producer
    script, a web form, a batch job, etc).
    """
    consumer = _connect_consumer()
    producer = _connect_producer()  # used to publish results onward

    logger.info("Consumer ready. Waiting for messages on '%s' ...", REQUEST_TOPIC)

    for message in consumer:
        request = message.value
        request_id = request["request_id"]
        features = request["features"]
        submitted_at = request["submitted_at"]

        try:
            # Call the prediction API — the single source of truth for
            # inference logic, shared with the synchronous REST path.
            api_start = time.time()
            response = requests.post(
                API_URL, json={"features": features}, timeout=10
            )
            response.raise_for_status()
            prediction = response.json()
            api_elapsed_ms = (time.time() - api_start) * 1000

            end_to_end_ms = (time.time() - submitted_at) * 1000

            result = {
                "request_id": request_id,
                "features": features,
                "predicted_class": prediction["predicted_class"],
                "probabilities": prediction["probabilities"],
                "api_call_ms": round(api_elapsed_ms, 2),
                "end_to_end_ms": round(end_to_end_ms, 2),
                "processed_at": time.time(),
            }

            producer.send(PREDICTION_TOPIC, key=request_id, value=result)
            producer.flush()

            # Console output — this is what the Action Phase screenshot captures
            print(
                f"[CONSUMED] request_id={request_id[:8]} | "
                f"features={features} | "
                f"predicted={result['predicted_class']:<12} | "
                f"end-to-end={result['end_to_end_ms']:.1f}ms"
            )

        except requests.exceptions.RequestException as exc:
            logger.error("API call failed for request_id=%s: %s", request_id, exc)
        except Exception as exc:
            logger.error("Unexpected error processing request_id=%s: %s", request_id, exc, exc_info=True)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("producer", "consumer"):
        print("Usage: python kafka_client.py [producer|consumer]")
        sys.exit(1)

    role = sys.argv[1]

    if role == "producer":
        run_producer()
    else:
        run_consumer()
