# src/monitoring/prediction_logger.py

import os
import csv
from datetime import datetime

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "predictions.csv")

HEADERS = ["timestamp", "fraud_probability", "fraud_label", "latency_sec"]


def log_prediction(fraud_prob, fraud_label, latency):
    os.makedirs(LOG_DIR, exist_ok=True)

    file_exists = os.path.isfile(LOG_FILE)

    with open(LOG_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(HEADERS)

        writer.writerow([
            datetime.utcnow().isoformat(),
            fraud_prob,
            fraud_label,
            latency
        ])
