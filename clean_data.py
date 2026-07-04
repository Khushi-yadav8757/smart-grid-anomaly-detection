"""
Smart Grid Project - Data Cleaning 
-----------------------------------------
"""

import pandas as pd
import numpy as np

RAW_FILE = "household_power_consumption.csv"

print("Loading raw data... this may take a minute for large files")

df = pd.read_csv(
    RAW_FILE,
    sep=",",                # Excel-exported CSV uses commas
    na_values=["?", ""],    # treat "?" or blank cells as missing
    low_memory=False
)

print(f"Loaded {len(df):,} rows")
print("Columns found:", list(df.columns))

# ---------------------------------------------------------
# STEP 2: Combine Date + Time into one datetime column
# ---------------------------------------------------------
# Date format in the Excel file looks like "16-12-2006" (DD-MM-YYYY)
df["datetime"] = pd.to_datetime(
    df["Date"].astype(str) + " " + df["Time"].astype(str),
    format="%d-%m-%Y %H:%M:%S",
    errors="coerce"          # if a row doesn't match, mark it as NaT instead of crashing
)

# Drop rows where the datetime could not be parsed at all
df = df[df["datetime"].notna()]

df.drop(columns=["Date", "Time"], inplace=True)
df.set_index("datetime", inplace=True)

# ---------------------------------------------------------
# STEP 3: Convert all readings to numeric (float) type
# ---------------------------------------------------------
numeric_cols = [
    "Global_active_power", "Global_reactive_power", "Voltage",
    "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ---------------------------------------------------------
# STEP 4: Handle missing values
# ---------------------------------------------------------
missing_before = df[numeric_cols].isna().sum().sum()
print(f"Missing values found: {missing_before:,}")

# Forward-fill then backward-fill: uses the nearest known reading
# This works well for time-series data (better than dropping rows)
df[numeric_cols] = df[numeric_cols].ffill().bfill()

missing_after = df[numeric_cols].isna().sum().sum()
print(f"Missing values after cleaning: {missing_after}")

# ---------------------------------------------------------
# STEP 5: Feature engineering (useful for the Isolation Forest model)
# ---------------------------------------------------------
# Rolling average and standard deviation over the last 15 readings (minutes)
df["power_rolling_mean_15"] = df["Global_active_power"].rolling(window=15, min_periods=1).mean()
df["power_rolling_std_15"] = df["Global_active_power"].rolling(window=15, min_periods=1).std().fillna(0)

# Time-based features
df["hour"] = df.index.hour
df["day_of_week"] = df.index.dayofweek          # 0 = Monday, 6 = Sunday
df["is_peak_hour"] = df["hour"].apply(lambda h: 1 if 18 <= h <= 22 else 0)

# ---------------------------------------------------------
# STEP 6: Save the full cleaned (minute-level) dataset
# ---------------------------------------------------------
df.reset_index(inplace=True)
df.to_csv("cleaned_minute_level.csv", index=False)
print("Saved: cleaned_minute_level.csv")

# ---------------------------------------------------------
# STEP 7: Also create a resampled (15-minute average) version
#         -> smaller file, easier to work with for training/demo
# ---------------------------------------------------------
df_resampled = df.set_index("datetime").resample("15min").mean(numeric_only=True)
df_resampled.reset_index(inplace=True)
df_resampled.to_csv("cleaned_15min_resampled.csv", index=False)
print("Saved: cleaned_15min_resampled.csv")

print("\nDone! Two files are ready in this folder:")
print("  1. cleaned_minute_level.csv      (full detail, ~2 million rows)")
print("  2. cleaned_15min_resampled.csv   (smaller, easier to test with)")