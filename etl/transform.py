import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse Order Time and Delivery Time from string to datetime.

    Raw format: DD-MM-YYYY HH:MM
    We tell pandas the exact format so it does not guess.
    Guessing is slow and sometimes wrong.
    """
    df = df.copy()  # never mutate the input DataFrame

    df["Order Time"] = pd.to_datetime(df["Order Time"], format="%d-%m-%Y %H:%M")
    df["Delivery Time"] = pd.to_datetime(df["Delivery Time"], format="%d-%m-%Y %H:%M")

    logger.info("Parsed Order Time and Delivery Time to datetime")
    return df


def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract useful time-based features from the Order Time column.

    These features are what the ML model will use to understand
    patterns like 'Friday evenings are busier than Monday mornings'.

    Input:  Order Time as datetime
    Output: new columns added to the DataFrame
    """
    df = df.copy()

    df["order_date"] = df["Order Time"].dt.date
    df["order_hour"] = df["Order Time"].dt.hour
    df["day_of_week"] = df["Order Time"].dt.dayofweek  # 0=Monday, 6=Sunday
    df["month"] = df["Order Time"].dt.month

    logger.info("Extracted time features: order_date, order_hour, day_of_week, month")
    return df


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names to snake_case lowercase.

    Raw column names have spaces and special characters.
    Databases and Python code both prefer snake_case.

    Example: 'Restaurant Name' -> 'restaurant_name'
             'Price ($)'       -> 'price'
             'Tip ($)'         -> 'tip'
    """
    df = df.copy()

    df = df.rename(columns={
        "Order ID": "order_id",
        "Restaurant Name": "restaurant_name",
        "Location": "location",
        "Order Time": "order_time",
        "Delivery Time": "delivery_time",
        "Delivery Duration (min)": "delivery_duration_min",
        "Taco Size": "taco_size",
        "Taco Type": "taco_type",
        "Toppings Count": "toppings_count",
        "Distance (km)": "distance_km",
        "Price ($)": "price",
        "Tip ($)": "tip",
        "Weekend Order": "is_weekend"
    })

    logger.info("Standardized column names to snake_case")
    return df


def build_clean_orders(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce a clean, enriched orders table.

    This is the intermediate state between raw data and features.
    Each row is still one order, but now with clean names,
    correct types, and derived time columns.

    This table maps to: raw_orders in PostgreSQL
    """
    logger.info("Building clean orders table...")

    df = parse_datetime_columns(df)
    df = extract_time_features(df)
    df = clean_column_names(df)

    # Select and order the columns we want to keep
    clean_df = df[[
        "order_id",
        "restaurant_name",
        "location",
        "order_time",
        "order_date",
        "order_hour",
        "day_of_week",
        "month",
        "is_weekend",
        "taco_size",
        "taco_type",
        "toppings_count",
        "distance_km",
        "delivery_duration_min",
        "price",
        "tip"
    ]]

    logger.info(f"Clean orders table: {clean_df.shape[0]} rows, {clean_df.shape[1]} columns")
    return clean_df


def build_features(clean_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate clean orders into ML-ready features.

    This is the most important transformation in the pipeline.
    We collapse individual orders into hourly demand summaries.

    Input:  1 row = 1 order (1000 rows)
    Output: 1 row = 1 hour at 1 restaurant (fewer rows, aggregated)

    The 'total_orders' column we create here is the TARGET VARIABLE
    that the ML model will learn to predict.

    Group by: restaurant + date + hour
    Because we want to answer: "how many orders did this restaurant
    get at 7pm on a Tuesday?"
    """
    logger.info("Building features table by aggregating orders...")

    features_df = clean_df.groupby(
        ["restaurant_name", "location", "order_date", "order_hour", "day_of_week", "month", "is_weekend"]
    ).agg(
        total_orders=("order_id", "count"),              # TARGET: how many orders in this hour
        avg_price=("price", "mean"),                     # average order value in this hour
        avg_tip=("tip", "mean"),                         # average tip
        avg_delivery_duration=("delivery_duration_min", "mean"),  # avg how long delivery took
        avg_distance=("distance_km", "mean"),            # avg delivery distance
        avg_toppings=("toppings_count", "mean"),         # avg toppings (proxy for order complexity)
    ).reset_index()

    # Round the averages to 2 decimal places for clean storage
    numeric_cols = ["avg_price", "avg_tip", "avg_delivery_duration", "avg_distance", "avg_toppings"]
    features_df[numeric_cols] = features_df[numeric_cols].round(2)

    logger.info(f"Features table: {features_df.shape[0]} rows (hourly aggregations)")
    logger.info(f"Unique restaurants: {features_df['restaurant_name'].nunique()}")
    logger.info(f"Date range: {features_df['order_date'].min()} to {features_df['order_date'].max()}")

    return features_df


if __name__ == "__main__":
    from etl.extract import extract_from_csv

    raw_df = extract_from_csv("data/raw/taco_sales.csv")

    clean_df = build_clean_orders(raw_df)
    print("\n--- CLEAN ORDERS (first 3 rows) ---")
    print(clean_df.head(3))
    print(f"Shape: {clean_df.shape}")

    features_df = build_features(clean_df)
    print("\n--- FEATURES TABLE (first 3 rows) ---")
    print(features_df.head(3))
    print(f"Shape: {features_df.shape}")
    print(f"\ntotal_orders range: {features_df['total_orders'].min()} to {features_df['total_orders'].max()}")