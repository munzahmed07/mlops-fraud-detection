# src/preprocessing/clean.py

import os
import logging
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

INPUT_PATH = "data/processed/fraud_base.csv"
OUTPUT_PATH = "data/processed/fraud_clean.csv"


def load_data():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")
    logging.info("Loading merged fraud dataset...")
    return pd.read_csv(INPUT_PATH)


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Make column names consistent."""
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )
    return df


def drop_high_missing_columns(df: pd.DataFrame, threshold: float = 0.90) -> pd.DataFrame:
    """
    Drop columns with too many missing values.
    If more than `threshold` fraction is missing, the column is useless.
    """
    missing_ratio = df.isna().mean()
    to_drop = missing_ratio[missing_ratio > threshold].index.tolist()

    logging.info(
        f"Dropping {len(to_drop)} columns with >{int(threshold*100)}% missing values.")
    return df.drop(columns=to_drop)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple, safe strategy:
    - Numeric columns → fill with median
    - Categorical columns → fill with 'Unknown'
    """
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(exclude=[np.number]).columns

    for col in num_cols:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    for col in cat_cols:
        df[col] = df[col].fillna("Unknown")

    logging.info("Missing values handled.")
    return df


def remove_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove columns with only one unique value.
    They add no information to models.
    """
    nunique = df.nunique()
    constant_cols = nunique[nunique <= 1].index.tolist()

    logging.info(f"Removing {len(constant_cols)} constant columns.")
    return df.drop(columns=constant_cols)


def save_clean_data(df: pd.DataFrame):
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    logging.info(f"Cleaned data saved to: {OUTPUT_PATH}")


def main():
    logging.info("===== DATA CLEANING STARTED =====")

    df = load_data()

    logging.info(f"Initial shape: {df.shape}")

    df = standardize_column_names(df)
    df = drop_high_missing_columns(df, threshold=0.90)
    df = handle_missing_values(df)
    df = remove_constant_columns(df)

    logging.info(f"Final shape after cleaning: {df.shape}")

    save_clean_data(df)

    logging.info("===== DATA CLEANING COMPLETED =====")


if __name__ == "__main__":
    main()
