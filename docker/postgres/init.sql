CREATE TABLE IF NOT EXISTS raw_orders (
    id SERIAL PRIMARY KEY,
    order_id VARCHAR(50),
    restaurant_id VARCHAR(50),
    order_date DATE,
    order_time TIME,
    order_hour INTEGER,
    day_of_week INTEGER,
    is_weekend BOOLEAN,
    items_count INTEGER,
    order_value NUMERIC(10, 2),
    delivery_time_minutes INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS features (
    id SERIAL PRIMARY KEY,
    restaurant_id VARCHAR(50),
    order_date DATE,
    order_hour INTEGER,
    day_of_week INTEGER,
    is_weekend BOOLEAN,
    total_orders INTEGER,
    avg_order_value NUMERIC(10, 2),
    avg_delivery_time NUMERIC(10, 2),
    created_at TIMESTAMP DEFAULT NOW()
);