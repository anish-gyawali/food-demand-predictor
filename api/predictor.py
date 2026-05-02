import pickle
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MODEL_PATH    = "model/artifacts/model.pkl"
FEATURES_PATH = "model/artifacts/feature_columns.pkl"


class DemandPredictor:
    """
    Wraps the trained model and handles all prediction logic.

    Lifecycle:
    - Created once when the API starts
    - load() reads model.pkl and feature_columns.pkl from disk
    - predict() is called for every incoming request

    Why a class and not just functions?
    The model object is heavy (100 decision trees).
    We load it once into this class instance and reuse it
    for every request. Loading it per-request would be
    very slow and wasteful.
    """

    def __init__(self):
        self.model = None
        self.feature_columns = None
        self.is_loaded = False

    def load(self) -> None:
        """
        Load model and feature columns from disk into memory.
        Called once at API startup.
        """
        logger.info(f"Loading model from {MODEL_PATH}...")

        with open(MODEL_PATH, "rb") as f:
            self.model = pickle.load(f)

        with open(FEATURES_PATH, "rb") as f:
            self.feature_columns = pickle.load(f)

        self.is_loaded = True
        logger.info("Model loaded successfully")
        logger.info(f"Expected features: {self.feature_columns}")

    def predict(self, request_data: dict) -> dict:
        """
        Run the model and return a prediction.

        Input:  dict of feature values from the API request
        Output: dict with predicted_orders and demand_level

        The critical step here is building the feature array
        in the exact same column order the model was trained on.
        We use self.feature_columns (saved during training) to
        guarantee this order is always correct.
        """
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded. Call load() first.")

        # Build a single-row DataFrame using the saved column order
        # This guarantees the model receives features in the right order
        input_df = pd.DataFrame([{
            "order_hour":            request_data["order_hour"],
            "day_of_week":           request_data["day_of_week"],
            "month":                 request_data["month"],
            "is_weekend":            int(request_data["is_weekend"]),
            "avg_price":             request_data["avg_price"],
            "avg_tip":               request_data["avg_tip"],
            "avg_delivery_duration": request_data["avg_delivery_duration"],
            "avg_distance":          request_data["avg_distance"],
            "avg_toppings":          request_data["avg_toppings"],
        }])[self.feature_columns]  # reorder columns to match training

        # Run the model
        raw_prediction = self.model.predict(input_df)[0]

        # Round to nearest integer — you cannot have 7.3 orders
        predicted_orders = max(0, int(round(raw_prediction)))

        # Convert numeric prediction to human-readable demand level
        demand_level = self._get_demand_level(predicted_orders)

        logger.info(
            f"Prediction: {predicted_orders} orders "
            f"(hour={request_data['order_hour']}, "
            f"day={request_data['day_of_week']}, "
            f"demand={demand_level})"
        )

        return {
            "predicted_orders": predicted_orders,
            "demand_level":     demand_level,
        }

    def _get_demand_level(self, predicted_orders: int) -> str:
        """
        Translate a raw order count into a human-readable label.
        Thresholds based on what we saw in the data:
        - max observed was ~25 orders/hour
        - dinner peak averaged ~10 orders/hour
        """
        if predicted_orders <= 2:
            return "low"
        elif predicted_orders <= 7:
            return "medium"
        else:
            return "high"


# Single instance shared across all requests
# This is the object FastAPI will use
predictor = DemandPredictor()