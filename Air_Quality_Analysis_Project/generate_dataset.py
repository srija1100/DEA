import pandas as pd
import numpy as np

print("Starting dataset generation...")

dates = pd.date_range(start="2014-01-01", end="2023-12-31 23:00:00", freq="h")

np.random.seed(42)

data = {
    "date": dates,
    "pm2_5": np.random.randint(20, 250, len(dates)),
    "pm10": np.random.randint(40, 350, len(dates)),
    "no2": np.random.randint(10, 180, len(dates)),
    "so2": np.random.randint(5, 120, len(dates)),
    "co": np.round(np.random.uniform(0.3, 5.0, len(dates)), 2),
    "ozone": np.random.randint(10, 200, len(dates)),
}

df = pd.DataFrame(data)

df["aqi"] = (
    df["pm2_5"] * 0.4 +
    df["pm10"] * 0.2 +
    df["no2"] * 0.15 +
    df["so2"] * 0.1 +
    df["co"] * 8 +
    df["ozone"] * 0.15
).astype(int)

df.to_csv("air_pollution_large_dataset.csv", index=False)

print("Dataset created successfully!")
print("Total rows:", len(df))
