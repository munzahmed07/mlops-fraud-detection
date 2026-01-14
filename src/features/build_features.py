# src/features/build_features.py

import os
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

INPUT_PATH = "data/processed/fraud_clean.csv"
FEATURES_DIR = "data/features"

X_TRAIN_PATH = os.path.join(FEATURES_DIR, "X_train.csv")
X_TEST_PATH = os.path.join(FEATURES_DIR, "X_test.csv")
y_TRAIN_PATH = os.path.join(FEATURES_DIR, "y_train.csv")
y_TEST_PATH = os.path.join(FEATURES_DIR, "y_test.csv")
SCALER_PATH = os.path.join(FEATURES_DIR, "scaler.pkl")

# 🔹 DEV MODE: limit data size so pipeline is runnable locally
MAX_ROWS = 150_000   # change later if you have a stronger machine


def load_data():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Cleaned data not found: {INPUT_PATH}")

    logging.info("Loading cleaned fraud data...")
    df = pd.read_csv(INPUT_PATH)

    # Sample for local dev
    if len(df) > MAX_ROWS:
        logging.info(f"Sampling {MAX_ROWS} rows for local development...")
        df = df.sample(MAX_ROWS, random_state=42)

    return df


def separate_target(df):
    if "isfraud" not in df.columns:
        raise ValueError("Target column 'isfraud' not found.")

    X = df.drop(columns=["isfraud"])
    y = df["isfraud"]

    return X, y


def encode_categoricals(df):
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    logging.info(f"Encoding {len(cat_cols)} categorical columns...")

    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df_encoded


def scale_numeric_features(X_train, X_test):
    """
    Use float32 to reduce memory footprint by 50%
    """
    scaler = StandardScaler()

    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")

    logging.info("Scaling features...")
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler


def save_artifacts(X_train, X_test, y_train, y_test, scaler):
    os.makedirs(FEATURES_DIR, exist_ok=True)

    pd.DataFrame(X_train).to_csv(X_TRAIN_PATH, index=False)
    pd.DataFrame(X_test).to_csv(X_TEST_PATH, index=False)
    y_train.to_csv(y_TRAIN_PATH, index=False)
    y_test.to_csv(y_TEST_PATH, index=False)

    joblib.dump(scaler, SCALER_PATH)

    logging.info("Feature datasets and scaler saved.")


def main():
    logging.info("===== FEATURE ENGINEERING STARTED =====")

    df = load_data()
    logging.info(f"Working dataset shape: {df.shape}")

    X, y = separate_target(df)

    X_encoded = encode_categoricals(X)
    logging.info(f"Feature matrix shape after encoding: {X_encoded.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    logging.info(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

    X_train_scaled, X_test_scaled, scaler = scale_numeric_features(
        X_train, X_test
    )

    save_artifacts(X_train_scaled, X_test_scaled, y_train, y_test, scaler)

    logging.info("===== FEATURE ENGINEERING COMPLETED =====")


if __name__ == "__main__":
    main()
