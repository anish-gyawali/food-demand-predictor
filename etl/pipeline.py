import logging
from etl.extract import extract_from_csv
from etl.transform import build_clean_orders, build_features
from etl.load import load_raw_orders, load_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

RAW_FILE = "data/raw/taco_sales.csv"


def run_pipeline() -> None:
    """
    Full ETL pipeline: CSV -> PostgreSQL

    Sequence:
        1. Extract  — read raw CSV into DataFrame
        2. Transform — clean, parse, engineer features
        3. Load     — write both tables to PostgreSQL

    This is the single entry point for the entire ETL phase.
    In production this function would be triggered by a scheduler.
    """
    logger.info("=== ETL PIPELINE STARTED ===")

    # EXTRACT
    logger.info("Phase 1: Extract")
    raw_df = extract_from_csv(RAW_FILE)

    # TRANSFORM
    logger.info("Phase 2: Transform")
    clean_df    = build_clean_orders(raw_df)
    features_df = build_features(clean_df)

    # LOAD
    logger.info("Phase 3: Load")
    load_raw_orders(clean_df)
    load_features(features_df)

    logger.info("=== ETL PIPELINE COMPLETED SUCCESSFULLY ===")
    logger.info(f"raw_orders loaded:  {len(clean_df):,} rows")
    logger.info(f"features loaded:    {len(features_df):,} rows")


if __name__ == "__main__":
    run_pipeline()