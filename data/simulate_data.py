# data/simulate_data.py
#
# Purpose: Generate a realistic taco orders dataset with proper demand patterns.
# This runs ONCE to produce data/raw/taco_sales.csv
# It replaces the sparse Kaggle dataset with volume that ML can actually learn from.

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

random.seed(42)
np.random.seed(42)

# --- Configuration ---
RESTAURANTS = [
    ("El Taco Loco",   "New York"),
    ("Taco Haven",     "Austin"),
    ("Casa del Taco",  "San Antonio"),
    ("Taco Fiesta",    "Los Angeles"),
    ("The Taco Spot",  "Chicago"),
]

TACO_TYPES = ["Beef Taco", "Chicken Taco", "Pork Taco", "Fish Taco", "Veggie Taco"]
TACO_SIZES = ["Regular", "Large"]

START_DATE = datetime(2024, 1, 1)
END_DATE   = datetime(2025, 1, 1)  # 1 full year

# --- Demand pattern: probability weight per hour (0-23) ---
# This is what makes the data realistic.
# High weights at lunch and dinner, near-zero at 2am.
HOUR_WEIGHTS = {
    0: 0.1,  1: 0.05, 2: 0.02, 3: 0.02, 4: 0.02, 5: 0.05,
    6: 0.2,  7: 0.4,  8: 0.5,  9: 0.6,  10: 0.8, 11: 1.5,
    12: 2.0, 13: 1.8, 14: 1.0, 15: 0.8, 16: 0.9, 17: 1.4,
    18: 2.5, 19: 3.0, 20: 2.8, 21: 2.0, 22: 1.2, 23: 0.5,
}


def orders_for_hour(hour: int, is_weekend: bool) -> int:
    """
    How many orders does a restaurant get in a given hour?
    Weekends get a 1.4x multiplier.
    """
    base = HOUR_WEIGHTS[hour]
    if is_weekend:
        base *= 1.4
    # Poisson distribution makes the count feel random but realistic
    count = np.random.poisson(lam=base * 3)
    return max(0, count)


def generate_orders() -> pd.DataFrame:
    records = []
    order_id = 100000

    current_date = START_DATE
    while current_date < END_DATE:
        is_weekend = current_date.weekday() >= 5  # Saturday=5, Sunday=6

        for restaurant_name, location in RESTAURANTS:
            for hour in range(24):
                n_orders = orders_for_hour(hour, is_weekend)

                for _ in range(n_orders):
                    minute = random.randint(0, 59)
                    order_time = current_date.replace(hour=hour, minute=minute)

                    taco_type = random.choice(TACO_TYPES)
                    taco_size = random.choice(TACO_SIZES)
                    toppings  = random.randint(0, 6)

                    base_price = 4.25 if taco_size == "Regular" else 6.50
                    price = round(base_price + toppings * 0.75 + random.uniform(-0.5, 0.5), 2)

                    duration = random.randint(10, 45)
                    delivery_time = order_time + timedelta(minutes=duration)
                    distance = round(random.uniform(0.5, 25.0), 2)
                    tip = round(random.uniform(0.5, 5.0), 2)

                    records.append({
                        "Order ID":                 order_id,
                        "Restaurant Name":          restaurant_name,
                        "Location":                 location,
                        "Order Time":               order_time.strftime("%d-%m-%Y %H:%M"),
                        "Delivery Time":            delivery_time.strftime("%d-%m-%Y %H:%M"),
                        "Delivery Duration (min)":  duration,
                        "Taco Size":                taco_size,
                        "Taco Type":                taco_type,
                        "Toppings Count":           toppings,
                        "Distance (km)":            distance,
                        "Price ($)":                price,
                        "Tip ($)":                  tip,
                        "Weekend Order":            is_weekend,
                    })

                    order_id += 1

        current_date += timedelta(days=1)

    return pd.DataFrame(records)


if __name__ == "__main__":
    print("Generating realistic taco orders dataset...")
    df = generate_orders()

    print(f"Total orders generated: {len(df):,}")
    print(f"Date range: {df['Order Time'].min()} to {df['Order Time'].max()}")
    print(f"Restaurants: {df['Restaurant Name'].nunique()}")

    # Quick sanity check on demand patterns
    df["_hour"] = pd.to_datetime(df["Order Time"], format="%d-%m-%Y %H:%M").dt.hour
    print("\nOrders by hour (sample):")
    print(df.groupby("_hour")["Order ID"].count().to_string())

    df = df.drop(columns=["_hour"])
    df.to_csv("data/raw/taco_sales.csv", index=False)
    print("\nSaved to data/raw/taco_sales.csv")