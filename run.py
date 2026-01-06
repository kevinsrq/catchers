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


print("--- Script Completed Syntax Checks ---")
try:
    # 2. Apply catch_all on the target column grouped by id
    print("\n--- Aggregated Catchers Features per ID ---")
    catchers_df = (
        df.sort("datetime")
        .group_by("id")
        .agg(
            pl.col("target").catchers.catch_all().alias("features")
        )
        .unnest("features")
    ).sort("id")
    print(catchers_df)

    # 3. Apply fresh.catch_all on the target column grouped by id
    print("\n--- Aggregated Fresh Features per ID ---")
    fresh_df = (
        df.sort("datetime")
        .group_by("id")
        .agg(
            pl.col("target").fresh.catch_all().alias("features")
        )
        .unnest("features")
    ).sort("id")
    print(fresh_df)

    # Show columns to verify
    print("\n--- Columns in Result ---")
    print("Catchers:", catchers_df.columns)
    print("Fresh:", fresh_df.columns)

    # 4. Validating Rolling Windows
    print("\n--- Rolling Window Validation (Fresh) ---")
    # Using 10-point rolling window instead of time if datetime is not evenly spaced, 
    # but here we have hourly data so "10h" is fine or index based.
    # Let's use index-based rolling for simplicity or known window size.
    rolling_df = (
        df.sort("datetime")
        .rolling(index_column="datetime", period="10h")
        .agg(
            pl.col("target").fresh.catch_all().alias("rolling_features")
        )
        .unnest("rolling_features")
    )
    print(rolling_df.tail())
    print("Rolling Columns:", rolling_df.columns)

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

print("--- Script Finished Successfully ---")
