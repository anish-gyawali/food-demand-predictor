# Food Delivery Demand Predictor

A complete data engineering + ML system that predicts food delivery order volume
for local restaurants using historical order data.

## System Architecture

- **ETL Pipeline**: Ingests raw CSV data, cleans it, loads into PostgreSQL
- **ML Model**: Trained on historical features, predicts hourly order demand
- **REST API**: FastAPI service that serves real-time predictions

## Tech Stack

- Python, pandas, scikit-learn
- PostgreSQL
- FastAPI + uvicorn

## Project Phases

- [ ] Phase 1: Project setup
- [ ] Phase 2: ETL pipeline
- [ ] Phase 3: Feature engineering
- [ ] Phase 4: Model training
- [ ] Phase 5: FastAPI prediction service

## Setup

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env      # fill in your DB credentials
```