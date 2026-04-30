
-- Drop tables if they exist so this script is safe to re-run
DROP TABLE IF EXISTS features;
DROP TABLE IF EXISTS raw_orders;

-- One row per transaction (output of build_clean_orders)
CREATE TABLE raw_orders (
    id                      SERIAL PRIMARY KEY,
    order_id                INTEGER,
    restaurant_name         VARCHAR(100),
    location                VARCHAR(100),
    order_time              TIMESTAMP,
    order_date              DATE,
    order_hour              SMALLINT,
    day_of_week             SMALLINT,
    month                   SMALLINT,
    is_weekend              BOOLEAN,
    taco_size               VARCHAR(20),
    taco_type               VARCHAR(50),
    toppings_count          SMALLINT,
    distance_km             NUMERIC(6,2),
    delivery_duration_min   SMALLINT,
    price                   NUMERIC(6,2),
    tip                     NUMERIC(6,2),
    created_at              TIMESTAMP DEFAULT NOW()
);

-- One row per restaurant + date + hour (output of build_features)
-- This is the table the ML model trains on
CREATE TABLE features (
    id                      SERIAL PRIMARY KEY,
    restaurant_name         VARCHAR(100),
    location                VARCHAR(100),
    order_date              DATE,
    order_hour              SMALLINT,
    day_of_week             SMALLINT,
    month                   SMALLINT,
    is_weekend              BOOLEAN,
    total_orders            SMALLINT,       -- TARGET VARIABLE
    avg_price               NUMERIC(6,2),
    avg_tip                 NUMERIC(6,2),
    avg_delivery_duration   NUMERIC(6,2),
    avg_distance            NUMERIC(6,2),
    avg_toppings            NUMERIC(6,2),
    created_at              TIMESTAMP DEFAULT NOW()
);