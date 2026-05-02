import logging
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from api.schemas import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse
)
from api.predictor import predictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs on API startup and shutdown.

    On startup:  load the model into memory
    On shutdown: nothing to clean up for now

    Using lifespan instead of @app.on_event because
    on_event is deprecated in newer FastAPI versions.
    """
    logger.info("API starting up — loading model...")
    predictor.load()
    logger.info("API ready to serve requests")
    yield
    logger.info("API shutting down")


app = FastAPI(
    title="Food Demand Predictor API",
    description="Predicts hourly order demand for taco restaurants",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health", response_model=HealthResponse)
def health_check():
    """
    Check if the API is running and the model is loaded.
    Always build this endpoint — it is used by load balancers
    and monitoring systems in production.
    """
    return HealthResponse(
        status="ok",
        model_loaded=predictor.is_loaded
    )


@app.post("/predict", response_model=PredictionResponse)
def predict_demand(request: PredictionRequest):
    """
    Predict hourly order demand for a restaurant.

    Full request lifecycle:
    1. FastAPI receives POST request
    2. Pydantic validates all fields in PredictionRequest
    3. predictor.predict() builds feature array and runs model
    4. Response is serialized to JSON and returned

    If any field is missing or wrong type,
    Pydantic returns a 422 error before the model is ever called.
    """
    try:
        result = predictor.predict(request.model_dump())

        return PredictionResponse(
            restaurant_name=request.restaurant_name,
            order_hour=request.order_hour,
            day_of_week=request.day_of_week,
            predicted_orders=result["predicted_orders"],
            demand_level=result["demand_level"],
            model_version="1.0.0"
        )

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_demand_batch(request: BatchPredictionRequest):
    """
    Predict hourly demand for all 24 hours of a given day.

    Send restaurant context once, receive a full day forecast.
    More efficient than calling /predict 24 times.

    Use this for:
    - Daily staff scheduling
    - Ingredient prep planning
    - Identifying peak windows for the day
    """
    try:
        result = predictor.predict_batch(request.model_dump())

        return BatchPredictionResponse(
            restaurant_name=request.restaurant_name,
            day_of_week=request.day_of_week,
            month=request.month,
            is_weekend=request.is_weekend,
            total_predicted_orders=result["total_predicted_orders"],
            peak_hour=result["peak_hour"],
            hourly_predictions=result["hourly_predictions"],
            model_version="1.0.0"
        )

    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))