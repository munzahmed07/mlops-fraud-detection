# src/ingestion/ingest.py

import os
import sys
import logging
import pandas as pd

# ---------------------------
# Basic logging configuration
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"

TRANSACTION_FILE = "train_transaction.csv"
IDENTITY_FILE = "train_identity.csv"
OUTPUT_FILE = "fraud_base.csv"


def check_files_exist():
    """Ensure required raw files are present."""
    txn_path = os.path.join(RAW_DATA_DIR, TRANSACTION_FILE)
    id_path = os.path.join(RAW_DATA_DIR, IDENTITY_FILE)

    if not os.path.exists(txn_path):
        logging.error(f"Missing file: {txn_path}")
        sys.exit(1)

    if not os.path.exists(id_path):
        logging.error(f"Missing file: {id_path}")
        sys.exit(1)

    return txn_path, id_path


def load_data(txn_path, id_path):
    """Load raw CSV files."""
    logging.info("Loading raw datasets...")
    transactions = pd.read_csv(txn_path)
    identities = pd.read_csv(id_path)

    logging.info(f"Transactions shape: {transactions.shape}")
    logging.info(f"Identities shape: {identities.shape}")

    return transactions, identities


def validate_schema(transactions):
    """Validate critical columns."""
    required_columns = ["TransactionID", "isFraud"]

    for col in required_columns:
        if col not in transactions.columns:
            logging.error(f"Required column missing: {col}")
            sys.exit(1)

    logging.info("Schema validation passed.")


def merge_data(transactions, identities):
    """
    Left join identity data.
    We keep all transactions even if identity info is missing.
    """
    logging.info("Merging transaction and identity data...")
    merged = transactions.merge(
        identities,
        on="TransactionID",
        how="left"
    )

    logging.info(f"Merged dataset shape: {merged.shape}")
    return merged


def save_processed_data(df):
    """Save merged data for downstream pipeline."""
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    output_path = os.path.join(PROCESSED_DATA_DIR, OUTPUT_FILE)

    df.to_csv(output_path, index=False)
    logging.info(f"Processed data saved to: {output_path}")


def log_basic_stats(df):
    """Log important dataset stats."""
    total_rows = len(df)
    fraud_count = df["isFraud"].sum()
    fraud_rate = fraud_count / total_rows

    logging.info(f"Total records: {total_rows}")
    logging.info(f"Fraud cases: {fraud_count}")
    logging.info(f"Fraud rate: {fraud_rate:.4f}")


def main():
    logging.info("===== FRAUD DATA INGESTION STARTED =====")

    txn_path, id_path = check_files_exist()

    transactions, identities = load_data(txn_path, id_path)

    validate_schema(transactions)

    merged_df = merge_data(transactions, identities)

    log_basic_stats(merged_df)

    save_processed_data(merged_df)

    logging.info("===== FRAUD DATA INGESTION COMPLETED =====")


if __name__ == "__main__":
    main()
