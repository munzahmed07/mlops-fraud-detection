# src/training/train.py

import os
import logging
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

FEATURES_DIR = "data/features"
MODELS_DIR = "models"

X_TRAIN_PATH = os.path.join(FEATURES_DIR, "X_train.csv")
y_TRAIN_PATH = os.path.join(FEATURES_DIR, "y_train.csv")

LR_MODEL_PATH = os.path.join(MODELS_DIR, "logistic_regression.pkl")
RF_MODEL_PATH = os.path.join(MODELS_DIR, "random_forest.pkl")
XGB_MODEL_PATH = os.path.join(MODELS_DIR, "xgboost.pkl")


def load_training_data():
    logging.info("Loading training features...")
    X_train = pd.read_csv(X_TRAIN_PATH)
    y_train = pd.read_csv(y_TRAIN_PATH).squeeze()
    return X_train, y_train


def train_logistic_regression(X, y):
    logging.info("Training Logistic Regression...")
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        n_jobs=-1
    )
    model.fit(X, y)
    return model


def train_random_forest(X, y):
    logging.info("Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        n_jobs=-1,
        class_weight="balanced",
        random_state=42
    )
    model.fit(X, y)
    return model


def train_xgboost(X, y):
    logging.info("Training XGBoost...")
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )
    model.fit(X, y)
    return model


def save_model(model, path):
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, path)
    logging.info(f"Model saved to: {path}")


def main():
    logging.info("===== MODEL TRAINING STARTED =====")

    X_train, y_train = load_training_data()

    lr_model = train_logistic_regression(X_train, y_train)
    save_model(lr_model, LR_MODEL_PATH)

    rf_model = train_random_forest(X_train, y_train)
    save_model(rf_model, RF_MODEL_PATH)

    xgb_model = train_xgboost(X_train, y_train)
    save_model(xgb_model, XGB_MODEL_PATH)

    logging.info("===== MODEL TRAINING COMPLETED =====")


if __name__ == "__main__":
    main()
