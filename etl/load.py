import pandas as pd
import logging
from sqlalchemy import text
from db.connection import get_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_raw_orders(clean_df: pd.DataFrame) -> None:
    """
    Load the cleaned orders DataFrame into the raw_orders table.

    We use 'replace' strategy here during development — it wipes
    the table and reloads it fresh every time the pipeline runs.

    In production you would use 'append' with deduplication logic,
    but 'replace' is correct for a batch pipeline on a fixed dataset.

    Input:  clean_df from build_clean_orders() — 147,711 rows
    Output: raw_orders table in PostgreSQL — same 147,711 rows
    """
    engine = get_engine()

    logger.info(f"Loading {len(clean_df):,} rows into raw_orders table...")

    # Drop the auto-generated 'id' concern — let PostgreSQL handle SERIAL
    # We only write the columns that exist in our DataFrame
    clean_df.to_sql(
        name="raw_orders",
        con=engine,
        if_exists="replace",   # wipe and reload
        index=False,           # do not write DataFrame index as a column
        chunksize=5000,        # write in batches of 5000 rows (memory efficient)
        method="multi",        # use multi-row INSERT (faster than row by row)
    )

    logger.info("raw_orders table loaded successfully")
    _verify_load(engine, "raw_orders", len(clean_df))


def load_features(features_df: pd.DataFrame) -> None:
    """
    Load the aggregated features DataFrame into the features table.

    This is the table the ML model will read from during training.

    Input:  features_df from build_features() — 31,354 rows
    Output: features table in PostgreSQL — same 31,354 rows
    """
    engine = get_engine()

    logger.info(f"Loading {len(features_df):,} rows into features table...")

    features_df.to_sql(
        name="features",
        con=engine,
        if_exists="replace",
        index=False,
        chunksize=5000,
        method="multi",
    )

    logger.info("features table loaded successfully")
    _verify_load(engine, "features", len(features_df))


def _verify_load(engine, table_name: str, expected_rows: int) -> None:
    """
    After loading, query the table and confirm the row count matches.
    This is a basic data quality check — never skip it in real pipelines.
    """
    with engine.connect() as conn:
        result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
        actual_rows = result.scalar()

    if actual_rows == expected_rows:
        logger.info(f"Verified: {table_name} has {actual_rows:,} rows (matches expected)")
    else:
        logger.error(
            f"Row count mismatch in {table_name}: "
            f"expected {expected_rows:,}, got {actual_rows:,}"
        )
        raise ValueError(f"Load verification failed for {table_name}")


if __name__ == "__main__":
    from etl.extract import extract_from_csv
    from etl.transform import build_clean_orders, build_features

    raw_df      = extract_from_csv("data/raw/taco_sales.csv")
    clean_df    = build_clean_orders(raw_df)
    features_df = build_features(clean_df)

    load_raw_orders(clean_df)
    load_features(features_df)