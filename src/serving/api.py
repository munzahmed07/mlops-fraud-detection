# src/serving/api.py

import time
import logging
import pandas as pd
import mlflow.pyfunc
from fastapi import FastAPI
from pydantic import BaseModel

from src.monitoring.prediction_logger import log_prediction

# --------------------
# Logging config
# --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# --------------------
# Load production model from MLflow
# --------------------
MODEL_NAME = "fraud-detection-model"
MODEL_ALIAS = "production"
MODEL_URI = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

logging.info(f"Loading model from: {MODEL_URI}")
model = mlflow.pyfunc.load_model(MODEL_URI)

# --------------------
# Load expected feature schema
# --------------------
FEATURE_SCHEMA_PATH = "data/features/X_train.csv"

# We only read the header to get column names
EXPECTED_COLUMNS = list(pd.read_csv(FEATURE_SCHEMA_PATH, nrows=1).columns)

logging.info(f"Loaded {len(EXPECTED_COLUMNS)} expected features.")

# --------------------
# FastAPI app
# --------------------
app = FastAPI(title="Fraud Detection API")

# --------------------
# Input schema
# --------------------


class Transaction(BaseModel):
    # User can send ANY subset of features
    features: dict


# --------------------
# Health check
# --------------------
@app.get("/health")
def health():
    return {"status": "ok"}


# --------------------
# Prediction endpoint
# --------------------
@app.post("/predict")
def predict(txn: Transaction):
    start_time = time.time()

    # --------------------------------------------------
    # Build a full feature row matching training schema
    # --------------------------------------------------
    # Start with ALL zeros
    row = {col: 0 for col in EXPECTED_COLUMNS}

    # Fill only what user provided
    for k, v in txn.features.items():
        if k in row:
            row[k] = v

    X = pd.DataFrame([row])

    # --------------------
    # Run prediction
    # --------------------
    preds = model.predict(X)

    # Handle return type safely
    if hasattr(preds, "__len__"):
        fraud_score = float(preds[0])
    else:
        fraud_score = float(preds)

    fraud_label = int(fraud_score >= 0.5)
    latency = time.time() - start_time

    # --------------------
    # Persistent logging
    # --------------------
    log_prediction(fraud_score, fraud_label, latency)

    # Console logging
    logging.info(
        f"PREDICTION | score={fraud_score:.4f} | "
        f"label={fraud_label} | latency={latency:.4f}s"
    )

    return {
        "fraud_probability": fraud_score,
        "fraud_prediction": fraud_label,
        "latency_sec": latency
    }
