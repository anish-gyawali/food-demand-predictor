# Food Delivery Demand Predictor

A complete **Data Engineering + Machine Learning** system that predicts hourly 
order demand for food delivery restaurants.

Built to demonstrate the full lifecycle of a real data system:
raw data → ETL pipeline → PostgreSQL → ML model → REST API.

---

## System Architecture

```
CSV Data (147,711 orders)
        │
        ▼
ETL Pipeline (Extract → Transform → Load)
        │
        ▼
PostgreSQL
├── raw_orders  (147,711 rows — one per transaction)
└── features    ( 31,354 rows — one per restaurant/hour)
        │
        ▼
Model Training (Random Forest Regressor)
├── Train: Jan–Oct 2024  (26,109 rows)
└── Test:  Nov–Dec 2024  ( 5,245 rows)
        │
        ▼
Model Artifacts
├── model.pkl
└── feature_columns.pkl
        │
        ▼
FastAPI Prediction Service
├── POST /predict        (single hour)
└── POST /predict/batch  (full 24-hour forecast)
```

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Data storage | PostgreSQL 16 (Docker) | Stores cleaned orders and features |
| ETL | Python, pandas | Extract, transform, load pipeline |
| ML model | scikit-learn RandomForest | Demand regression |
| API | FastAPI + uvicorn | Serves predictions over HTTP |
| Environment | Docker Compose | Reproducible local infrastructure |
| Version control | Git + GitHub | Branch-per-phase workflow |

---

## Project Structure

```
food-demand-predictor/
│
├── data/
│   ├── raw/                  # Source CSV (gitignored)
│   └── simulate_data.py      # Generates realistic order volume
│
├── etl/
│   ├── extract.py            # Read CSV → raw DataFrame
│   ├── transform.py          # Clean, parse, engineer features
│   ├── load.py               # Write to PostgreSQL
│   └── pipeline.py           # Orchestrates full ETL sequence
│
├── model/
│   ├── train.py              # Pull features, train, evaluate, save
│   └── artifacts/            # model.pkl + feature_columns.pkl
│
├── api/
│   ├── main.py               # FastAPI app and routes
│   ├── schemas.py            # Request/response validation
│   └── predictor.py          # Loads model, runs predictions
│
├── db/
│   ├── connection.py         # SQLAlchemy engine factory
│   └── init.sql              # Table definitions
│
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Quickstart

### Prerequisites

- Python 3.10+
- Docker Desktop

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/food-demand-predictor.git
cd food-demand-predictor

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### 2. Configure environment

```bash
cp .env.example .env
# .env is pre-filled with Docker defaults — no changes needed for local dev
```

### 3. Start PostgreSQL

```bash
docker compose up -d
docker compose ps   # confirm status is healthy
```

### 4. Generate the dataset

```bash
python data/simulate_data.py
# Generates data/raw/taco_sales.csv with 147,711 realistic orders
```

### 5. Run the ETL pipeline

```bash
python -m etl.pipeline
# Extract → Transform → Load
# Loads 147,711 rows into raw_orders
# Loads  31,354 rows into features
```

### 6. Train the model

```bash
python -m model.train
# Reads features from PostgreSQL
# Trains RandomForestRegressor
# Saves model.pkl and feature_columns.pkl
```

### 7. Start the API

```bash
uvicorn api.main:app --reload --port 8000
```

Open `http://localhost:8000/docs` for the interactive API documentation.

---

## API Reference

### `GET /health`

Confirms the API is running and the model is loaded.

```json
{
  "status": "ok",
  "model_loaded": true
}
```

---

### `POST /predict`

Predict demand for a single hour.

**Request:**
```json
{
  "restaurant_name": "El Taco Loco",
  "order_hour": 19,
  "day_of_week": 4,
  "month": 12,
  "is_weekend": false,
  "avg_price": 7.25,
  "avg_tip": 2.50,
  "avg_delivery_duration": 28.5,
  "avg_distance": 8.3,
  "avg_toppings": 3.0
}
```

**Response:**
```json
{
  "restaurant_name": "El Taco Loco",
  "order_hour": 19,
  "day_of_week": 4,
  "predicted_orders": 9,
  "demand_level": "high",
  "model_version": "1.0.0"
}
```

---

### `POST /predict/batch`

Predict demand for all 24 hours of a given day. 
Send restaurant context once, receive a full day forecast.

**Request:**
```json
{
  "restaurant_name": "El Taco Loco",
  "day_of_week": 4,
  "month": 12,
  "is_weekend": false,
  "avg_price": 7.25,
  "avg_tip": 2.50,
  "avg_delivery_duration": 28.5,
  "avg_distance": 8.3,
  "avg_toppings": 3.0
}
```

**Response:**
```json
{
  "restaurant_name": "El Taco Loco",
  "day_of_week": 4,
  "month": 12,
  "is_weekend": false,
  "total_predicted_orders": 93,
  "peak_hour": 19,
  "hourly_predictions": [
    {"hour": 0,  "predicted_orders": 1, "demand_level": "low"},
    {"hour": 1,  "predicted_orders": 1, "demand_level": "low"},
    {"hour": 12, "predicted_orders": 7, "demand_level": "medium"},
    {"hour": 19, "predicted_orders": 9, "demand_level": "high"},
    "..."
  ],
  "model_version": "1.0.0"
}
```

---

## Model Performance

| Metric | Value | Meaning |
|---|---|---|
| MAE | 1.30 | Predictions off by 1.3 orders on average |
| RMSE | 1.84 | Penalized error (sensitive to large misses) |
| R² | 0.73 | Model explains 73% of demand variance |

**Feature importances:**

| Feature | Importance | Insight |
|---|---|---|
| order_hour | 79% | Time of day drives demand more than anything else |
| is_weekend | 3.6% | Weekends slightly busier |
| day_of_week | 3.5% | Day matters less than hour |
| avg_toppings | 3.5% | Order complexity correlates with volume |
| month | 0.2% | No strong seasonal pattern in this dataset |

---

## ETL vs ELT — What This Project Uses and Why

This project implements **ETL** (Extract → Transform → Load):

- **Extract:** pandas reads the raw CSV exactly as-is
- **Transform:** Python cleans, parses datetimes, engineers hourly features
- **Load:** cleaned data written to PostgreSQL in batches

**Why ETL and not ELT?**
The feature engineering (grouping by restaurant + date + hour, computing 
averages) is naturally expressed in pandas. Doing this in raw SQL would be 
more verbose and harder to test. ETL is the right choice when transformation 
logic is complex or requires ML-oriented processing.

**ELT** (load raw first, transform inside the DB with SQL/dbt) would be 
appropriate if we were working at larger scale with a cloud data warehouse 
like Snowflake or BigQuery.

---

## What I Learned Building This

- How a real ETL pipeline moves data from CSV to a relational database
- Why time-based train/test splits matter for time-series prediction
- How a trained ML model becomes a REST API (the full lifecycle)
- How to structure a multi-phase project with clean git history
- The practical difference between ETL and ELT

---

## Git History

Each phase was developed on its own branch and merged to main:

```
etl-phase   → Extract, Transform, Load pipeline
model-phase → Feature prep, training, evaluation
api-phase   → FastAPI service, single + batch prediction
docs-phase  → README and documentation
```

---

## Author

Built by Anish Gyawali as a practical learning project in Data Engineering and ML Engineering.
