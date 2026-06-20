"""
train_model.py
---------------
Discovery Phase — Modular Setup

Trains a LogisticRegression classifier on the classic Iris dataset and
serializes it to model.pkl using joblib, ready to be loaded by the FastAPI
service at container startup.

Design notes:
- This script is run ONCE at build time (or whenever the model needs
  retraining), producing a static artifact (model.pkl) that the API
  loads. This separation — training pipeline vs. serving pipeline — is
  standard practice: training is expensive and infrequent, serving is
  cheap and frequent. They should not be coupled.
- We also save the target class names alongside the model so the API
  can return human-readable species names ("setosa") rather than raw
  integer labels (0).
"""

import json

import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

MODEL_PATH = "model.pkl"
LABELS_PATH = "class_labels.json"


def train_and_serialize():
    """Train a LogisticRegression model on Iris and serialize it to disk."""

    # ── Load data ────────────────────────────────────────────────────────
    iris = load_iris()
    X, y = iris.data, iris.target
    class_names = list(iris.target_names)  # ['setosa', 'versicolor', 'virginica']
    feature_names = list(iris.feature_names)

    print(f"Loaded Iris dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Features: {feature_names}")
    print(f"Classes : {class_names}")

    # ── Train/test split ────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Train ────────────────────────────────────────────────────────────
    # max_iter raised from sklearn's default (100) to ensure convergence;
    # Iris features are on different scales which can slow convergence.
    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X_train, y_train)

    # ── Evaluate ─────────────────────────────────────────────────────────
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\nTest accuracy: {acc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # ── Serialize model ──────────────────────────────────────────────────
    joblib.dump(model, MODEL_PATH)
    print(f"\n✅ Model serialized to '{MODEL_PATH}'")

    # ── Serialize class labels separately ───────────────────────────────
    # Keeping labels in a small JSON file (rather than re-deriving them)
    # ensures the API and training script stay in sync even if retrained
    # on a different dataset ordering.
    with open(LABELS_PATH, "w") as f:
        json.dump(
            {"class_names": class_names, "feature_names": feature_names},
            f,
            indent=2,
        )
    print(f"✅ Class labels serialized to '{LABELS_PATH}'")

    return model, acc


if __name__ == "__main__":
    train_and_serialize()
