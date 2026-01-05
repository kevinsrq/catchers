import polars as pl
import catchers
from datetime import datetime, timedelta
import math
import random

# Set seed for reproducibility
random.seed(42)

# 1. Generate generic time series data
# 3 different IDs with 100 points each
n_points = 100
ids = ["A", "B", "C"]

data = []
start_date = datetime(2024, 1, 1)

for i, uid in enumerate(ids):
    rw_val = 0.0
    for j in range(n_points):
        # Create different patterns for each ID
        if uid == "A":
            # Sinusoidal
            target = math.sin(j * 0.2) + random.gauss(0, 0.1)
        elif uid == "B":
            # Random walk
            rw_val += random.gauss(0, 0.1)
            target = rw_val
        else:
            # Constant with noise
            target = 10.0 + random.gauss(0, 0.5)
            
        data.append({
            "id": uid,
            "datetime": start_date + timedelta(hours=j),
            "target": float(target)
        })

df = pl.DataFrame(data)

print("--- Input Data Sample ---")
print(df.head())

# 2. Apply catch_all on the target column grouped by id
print("\n--- Aggregated Catchers Features per ID ---")
features_df = (
    df.sort("datetime")
    .group_by("id")
    .agg(
        pl.col("target").catchers.catch_all().alias("features")
    )
    .unnest("features")
)

# Sorting by ID for clean output
features_df = features_df.sort("id")

print(features_df)

# Show columns to verify
print("\n--- Columns in Result ---")
print(features_df.columns)
