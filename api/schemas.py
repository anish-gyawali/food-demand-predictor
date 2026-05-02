from pydantic import BaseModel, Field
from typing import Optional


class PredictionRequest(BaseModel):
    """
    What the user must send to get a prediction.

    Every field maps directly to a feature the model was trained on.
    Pydantic validates types automatically — if order_hour is a string,
    FastAPI rejects the request before it ever reaches the model.
    """
    restaurant_name: str = Field(
        ...,
        example="El Taco Loco",
        description="Name of the restaurant"
    )
    order_hour: int = Field(
        ...,
        ge=0,
        le=23,
        example=19,
        description="Hour of day (0-23)"
    )
    day_of_week: int = Field(
        ...,
        ge=0,
        le=6,
        example=4,
        description="Day of week (0=Monday, 6=Sunday)"
    )
    month: int = Field(
        ...,
        ge=1,
        le=12,
        example=12,
        description="Month (1-12)"
    )
    is_weekend: bool = Field(
        ...,
        example=False,
        description="True if Saturday or Sunday"
    )
    avg_price: float = Field(
        ...,
        gt=0,
        example=7.25,
        description="Average order price in dollars"
    )
    avg_tip: float = Field(
        ...,
        ge=0,
        example=2.50,
        description="Average tip amount"
    )
    avg_delivery_duration: float = Field(
        ...,
        gt=0,
        example=28.5,
        description="Average delivery duration in minutes"
    )
    avg_distance: float = Field(
        ...,
        gt=0,
        example=8.3,
        description="Average delivery distance in km"
    )
    avg_toppings: float = Field(
        ...,
        ge=0,
        example=3.0,
        description="Average number of toppings per order"
    )

class BatchPredictionRequest(BaseModel):
    """
    User sends restaurant context once.
    We predict all 24 hours automatically.
    """
    restaurant_name: str = Field(
        ...,
        example="El Taco Loco",
        description="Name of the restaurant"
    )
    day_of_week: int = Field(
        ...,
        ge=0,
        le=6,
        example=4,
        description="Day of week (0=Monday, 6=Sunday)"
    )
    month: int = Field(
        ...,
        ge=1,
        le=12,
        example=12,
        description="Month (1-12)"
    )
    is_weekend: bool = Field(
        ...,
        example=False
    )
    avg_price: float = Field(..., gt=0, example=7.25)
    avg_tip: float = Field(..., ge=0, example=2.50)
    avg_delivery_duration: float = Field(..., gt=0, example=28.5)
    avg_distance: float = Field(..., gt=0, example=8.3)
    avg_toppings: float = Field(..., ge=0, example=3.0)


class HourlyPrediction(BaseModel):
    """Single hour result inside a batch response."""
    hour: int
    predicted_orders: int
    demand_level: str

class PredictionResponse(BaseModel):
    """
    What the API sends back after running the model.
    """
    restaurant_name: str
    order_hour: int
    day_of_week: int
    predicted_orders: int
    demand_level: str        # "low", "medium", "high" — human readable
    model_version: str

class BatchPredictionResponse(BaseModel):
    """
    Full day forecast for one restaurant.
    """
    restaurant_name: str
    day_of_week: int
    month: int
    is_weekend: bool
    total_predicted_orders: int       # sum across all 24 hours
    peak_hour: int                    # hour with highest predicted demand
    hourly_predictions: list[HourlyPrediction]
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool