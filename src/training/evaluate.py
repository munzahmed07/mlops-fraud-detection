# src/training/evaluate.py

import os
import logging
import joblib
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

FEATURES_DIR = "data/features"
MODELS_DIR = "models"

X_TEST_PATH = os.path.join(FEATURES_DIR, "X_test.csv")
y_TEST_PATH = os.path.join(FEATURES_DIR, "y_test.csv")

LR_MODEL_PATH = os.path.join(MODELS_DIR, "logistic_regression.pkl")
RF_MODEL_PATH = os.path.join(MODELS_DIR, "random_forest.pkl")
XGB_MODEL_PATH = os.path.join(MODELS_DIR, "xgboost.pkl")


def load_test_data():
    logging.info("Loading test data...")
    X_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(y_TEST_PATH).squeeze()
    return X_test, y_test


def load_models():
    logging.info("Loading trained models...")
    models = {
        "Logistic Regression": joblib.load(LR_MODEL_PATH),
        "Random Forest": joblib.load(RF_MODEL_PATH),
        "XGBoost": joblib.load(XGB_MODEL_PATH)
    }
    return models


def evaluate_model(name, model, X_test, y_test):
    logging.info(f"Evaluating {name}...")

    # Probabilities for ROC-AUC
    y_proba = model.predict_proba(X_test)[:, 1]

    # Default threshold = 0.5
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = {
        "roc_auc": roc_auc_score(y_test, y_proba),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }

    return metrics


def main():
    logging.info("===== MODEL EVALUATION STARTED =====")

    X_test, y_test = load_test_data()
    models = load_models()

    results = {}

    for name, model in models.items():
        metrics = evaluate_model(name, model, X_test, y_test)
        results[name] = metrics

        logging.info(
            f"{name} | "
            f"ROC-AUC: {metrics['roc_auc']:.4f} | "
            f"Precision: {metrics['precision']:.4f} | "
            f"Recall: {metrics['recall']:.4f} | "
            f"F1: {metrics['f1']:.4f}"
        )

    logging.info("===== MODEL EVALUATION COMPLETED =====")

    logging.info("Final comparison:")
    for name, m in results.items():
        logging.info(f"{name}: {m}")


if __name__ == "__main__":
    main()
