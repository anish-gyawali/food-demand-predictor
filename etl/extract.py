import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_from_csv(filepath: str) -> pd.DataFrame:
    """
    Read raw CSV data from disk and return as a DataFrame.

    This function does nothing except read the file.
    No cleaning, no transformation, no filtering.
    The raw data must be preserved exactly as received.

    Args:
        filepath: path to the raw CSV file

    Returns:
        pd.DataFrame: raw, unmodified data
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")

    logger.info(f"Extracting data from: {filepath}")

    df = pd.read_csv(filepath)

    logger.info(f"Extracted {len(df)} rows and {len(df.columns)} columns")
    logger.info(f"Columns: {list(df.columns)}")

    return df


def get_raw_data_summary(df: pd.DataFrame) -> None:
    """
    Print a summary of the raw data for inspection.
    Used during development and debugging.
    """
    print("\n--- RAW DATA SUMMARY ---")
    print(f"Shape: {df.shape}")
    print(f"\nColumn names:\n{list(df.columns)}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nNull counts:\n{df.isnull().sum()}")
    print(f"\nFirst 3 rows:\n{df.head(3)}")
    print("------------------------\n")


if __name__ == "__main__":
    # When you run this file directly, it extracts and prints a summary
    # Usage: python -m etl.extract
    RAW_FILE = "data/raw/taco_sales.csv"
    df = extract_from_csv(RAW_FILE)
    get_raw_data_summary(df)