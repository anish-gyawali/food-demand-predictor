import pandas as pd
import numpy as np
import pickle
import logging
import os
from sqlalchemy import text
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from db.connection import get_engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# Path where the trained model will be saved
MODEL_PATH = "model/artifacts/model.pkl"
FEATURES_PATH = "model/artifacts/feature_columns.pkl"


def load_features_from_db() -> pd.DataFrame:
    """
    Pull the features table from PostgreSQL into a DataFrame.

    This is the exact data the model will train on.
    We read from the database, not from CSV, because the database
    is the single source of truth after the ETL pipeline runs.
    """
    engine = get_engine()
    logger.info("Loading features from PostgreSQL...")

    with engine.connect() as conn:
        df = pd.read_sql(
            text("SELECT * FROM features ORDER BY order_date, order_hour"),
            conn
        )

    logger.info(f"Loaded {len(df):,} rows from features table")
    return df


def prepare_training_data(df: pd.DataFrame):
    """
    Split the features DataFrame into X (inputs) and y (target).

    X = the columns the model uses to make predictions
    y = the column the model is trying to predict

    We also do a time-based train/test split here.
    Why time-based and not random?

    Because we are predicting the future. If we split randomly,
    the model might train on December data and test on January data
    — which means it trained on 'future' data relative to the test.
    That would give falsely optimistic results.

    Instead: train on Jan-Oct, test on Nov-Dec.
    This mirrors real-world usage exactly.
    """

    # These are the columns the model will see as input
    FEATURE_COLUMNS = [
        "order_hour",
        "day_of_week",
        "month",
        "is_weekend",
        "avg_price",
        "avg_tip",
        "avg_delivery_duration",
        "avg_distance",
        "avg_toppings",
    ]

    # This is what the model predicts
    TARGET_COLUMN = "total_orders"

    # Convert boolean to integer (ML models need numbers, not True/False)
    df["is_weekend"] = df["is_weekend"].astype(int)

    # Convert order_date to datetime for time-based split
    df["order_date"] = pd.to_datetime(df["order_date"])

    # Time-based split: train on first 10 months, test on last 2
    split_date = pd.Timestamp("2024-11-01")
    train_df = df[df["order_date"] < split_date]
    test_df  = df[df["order_date"] >= split_date]

    logger.info(f"Training set: {len(train_df):,} rows "
                f"({train_df['order_date'].min().date()} to "
                f"{train_df['order_date'].max().date()})")
    logger.info(f"Test set:     {len(test_df):,} rows "
                f"({test_df['order_date'].min().date()} to "
                f"{test_df['order_date'].max().date()})")

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]
    X_test  = test_df[FEATURE_COLUMNS]
    y_test  = test_df[TARGET_COLUMN]

    return X_train, X_test, y_train, y_test, FEATURE_COLUMNS


def train_model(X_train, y_train) -> RandomForestRegressor:
    """
    Train the Random Forest model.

    Parameters explained:
    - n_estimators=100  : build 100 decision trees
    - max_depth=10      : each tree can be at most 10 levels deep
                          (prevents overfitting)
    - min_samples_leaf=4: each leaf node must have at least 4 samples
                          (smooths predictions)
    - random_state=42   : makes results reproducible
    - n_jobs=-1         : use all CPU cores for training
    """
    logger.info("Training Random Forest Regressor...")
    logger.info(f"Training on {len(X_train):,} rows × {len(X_train.columns)} features")

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1
    )

    # THIS IS THE CORE STEP
    # X_train: matrix of input features  shape=(N, 9)
    # y_train: vector of target values   shape=(N,)
    # The model learns the mapping: inputs → total_orders
    model.fit(X_train, y_train)

    logger.info("Model training complete")
    return model


def evaluate_model(model, X_test, y_test) -> dict:
    """
    Measure how well the model predicts on data it never saw during training.

    Metrics explained:
    - MAE  (Mean Absolute Error): average prediction error in order units
            MAE of 1.2 means predictions are off by 1.2 orders on average
    - RMSE (Root Mean Squared Error): penalizes large errors more than MAE
    - R²   (R-squared): how much variance the model explains
            R²=1.0 is perfect, R²=0.0 means model learned nothing
    """
    logger.info("Evaluating model on test set...")

    y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    metrics = {
    "mae":  round(float(mae), 4),
    "rmse": round(float(rmse), 4),
    "r2":   round(float(r2), 4)
   }

    logger.info(f"MAE:  {mae:.4f}  (avg prediction error in orders)")
    logger.info(f"RMSE: {rmse:.4f}")
    logger.info(f"R²:   {r2:.4f}  (1.0 = perfect, 0.0 = learned nothing)")

    return metrics


def show_feature_importance(model, feature_columns: list) -> None:
    """
    Print which input features the model relied on most.

    This tells us: which signals actually drive demand?
    High importance = model found this feature very predictive.
    Low importance = this feature barely affected predictions.
    """
    importance_df = pd.DataFrame({
        "feature":   feature_columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    logger.info("Feature importances:")
    for _, row in importance_df.iterrows():
        bar = "█" * int(row["importance"] * 100)
        logger.info(f"  {row['feature']:<25} {row['importance']:.4f}  {bar}")


def save_model(model, feature_columns: list) -> None:
    """
    Save the trained model to disk as a .pkl file.

    pkl = Python pickle format. It serializes the entire model object
    — all 100 decision trees, all learned parameters — into a binary file.

    We also save the feature column list separately.
    This is critical: when the API loads the model, it must pass
    features in the exact same order and with the exact same names
    that the model was trained on. If the order changes, predictions
    will be wrong or the model will throw an error.
    """
    os.makedirs("model/artifacts", exist_ok=True)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to: {MODEL_PATH}")

    with open(FEATURES_PATH, "wb") as f:
        pickle.dump(feature_columns, f)
    logger.info(f"Feature columns saved to: {FEATURES_PATH}")


def run_training() -> None:
    """
    Full training pipeline:
    PostgreSQL features → trained model → model.pkl
    """
    logger.info("=== MODEL TRAINING STARTED ===")

    # Step 1: Pull data from database
    df = load_features_from_db()

    # Step 2: Prepare X and y with time-based split
    X_train, X_test, y_train, y_test, feature_columns = prepare_training_data(df)

    # Step 3: Train
    model = train_model(X_train, y_train)

    # Step 4: Evaluate
    metrics = evaluate_model(model, X_test, y_test)

    # Step 5: Show what the model learned
    show_feature_importance(model, feature_columns)

    # Step 6: Save to disk
    save_model(model, feature_columns)

    logger.info("=== MODEL TRAINING COMPLETED ===")
    logger.info(f"Final metrics: {metrics}")


if __name__ == "__main__":
    run_training()