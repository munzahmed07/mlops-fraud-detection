# src/training/train_with_mlflow.py

import os
import logging
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

FEATURES_DIR = "data/features"
X_TRAIN_PATH = os.path.join(FEATURES_DIR, "X_train.csv")
X_TEST_PATH = os.path.join(FEATURES_DIR, "X_test.csv")
y_TRAIN_PATH = os.path.join(FEATURES_DIR, "y_train.csv")
y_TEST_PATH = os.path.join(FEATURES_DIR, "y_test.csv")


def load_data():
    X_train = pd.read_csv(X_TRAIN_PATH)
    X_test = pd.read_csv(X_TEST_PATH)
    y_train = pd.read_csv(y_TRAIN_PATH).squeeze()
    y_test = pd.read_csv(y_TEST_PATH).squeeze()
    return X_train, X_test, y_train, y_test


def eval_and_log(model, model_name, X_test, y_test):
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = {
        "roc_auc": roc_auc_score(y_test, y_proba),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }

    for k, v in metrics.items():
        mlflow.log_metric(k, v)

    return metrics


def main():
    logging.info("===== MLFLOW TRAINING STARTED =====")

    mlflow.set_experiment("fraud-detection")

    X_train, X_test, y_train, y_test = load_data()

    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, class_weight="balanced", n_jobs=-1
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            n_jobs=-1,
            class_weight="balanced",
            random_state=42,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
        ),
    }

    for name, model in models.items():
        with mlflow.start_run(run_name=name):

            logging.info(f"Training {name}...")
            model.fit(X_train, y_train)

            # Log parameters
            mlflow.log_params(model.get_params())

            # Evaluate + log metrics
            metrics = eval_and_log(model, name, X_test, y_test)

            # Log model artifact
            mlflow.sklearn.log_model(model, artifact_path="model")

            logging.info(
                f"{name} | ROC-AUC: {metrics['roc_auc']:.4f} | "
                f"F1: {metrics['f1']:.4f}"
            )

    logging.info("===== MLFLOW TRAINING COMPLETED =====")


if __name__ == "__main__":
    main()
