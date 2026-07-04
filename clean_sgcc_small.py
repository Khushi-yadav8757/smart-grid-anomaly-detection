"""
Smart Grid Project - SGCC Theft Dataset Cleaning
"""
# ---------------------------------------------------------

import pandas as pd
import numpy as np

RAW_FILE = "datasetsmall.csv"

print("Loading raw data... this may take a minute for large files")

df = pd.read_csv(
    RAW_FILE,
    sep=",",
    na_values=["?", ""],   # treat "?" or blank cells as missing
    low_memory=False
)

print(f"Loaded {len(df):,} rows")
print("Columns found (first 10):", list(df.columns[:10]), "...")

# ---------------------------------------------------------
# STEP 2: Identify id/label columns vs date columns
# ---------------------------------------------------------
# CONS_NO = consumer ID, FLAG = ground-truth label (0 = normal, 1 = theft)
id_cols = ["CONS_NO", "FLAG"]
date_cols = [c for c in df.columns if c not in id_cols]

print(f"Found {len(date_cols)} date columns")

# ---------------------------------------------------------
# STEP 3: Convert WIDE format -> LONG format
# ---------------------------------------------------------
# Each row becomes: CONS_NO, FLAG, date, consumption
df_long = df.melt(
    id_vars=id_cols,
    value_vars=date_cols,
    var_name="date",
    value_name="consumption"
)

# Parse the date column into a proper datetime
df_long["date"] = pd.to_datetime(df_long["date"], errors="coerce")

# Drop rows where the date could not be parsed at all
df_long = df_long[df_long["date"].notna()]

# ---------------------------------------------------------
# STEP 4: Convert consumption to numeric (float) type
# ---------------------------------------------------------
df_long["consumption"] = pd.to_numeric(df_long["consumption"], errors="coerce")

# ---------------------------------------------------------
# STEP 5: Handle missing values
# ---------------------------------------------------------
missing_before = df_long["consumption"].isna().sum()
print(f"Missing values found: {missing_before:,}")

# Sort so forward/backward fill happens correctly per consumer, over time
df_long.sort_values(["CONS_NO", "date"], inplace=True)

# Forward-fill then backward-fill WITHIN each consumer's own readings
# (so one consumer's missing days don't get filled from another consumer's data)
df_long["consumption"] = (
    df_long.groupby("CONS_NO")["consumption"]
    .transform(lambda s: s.ffill().bfill())
)

# Any consumer with ALL missing values will still be NaN after ffill/bfill -> fill with 0
df_long["consumption"] = df_long["consumption"].fillna(0)

missing_after = df_long["consumption"].isna().sum()
print(f"Missing values after cleaning: {missing_after}")

# ---------------------------------------------------------
# STEP 6: Feature engineering (useful for the Isolation Forest model)
# ---------------------------------------------------------
# Rolling average and standard deviation over the last 7 days, PER CONSUMER
df_long["consumption_rolling_mean_7"] = (
    df_long.groupby("CONS_NO")["consumption"]
    .transform(lambda s: s.rolling(window=7, min_periods=1).mean())
)
df_long["consumption_rolling_std_7"] = (
    df_long.groupby("CONS_NO")["consumption"]
    .transform(lambda s: s.rolling(window=7, min_periods=1).std().fillna(0))
)

# Time-based features
df_long["day_of_week"] = df_long["date"].dt.dayofweek     # 0 = Monday, 6 = Sunday
df_long["month"] = df_long["date"].dt.month
df_long["is_weekend"] = df_long["day_of_week"].apply(lambda d: 1 if d >= 5 else 0)

# ---------------------------------------------------------
# STEP 7: Save the full cleaned (long-format) dataset
# ---------------------------------------------------------
df_long.to_csv("cleaned_sgcc_small_long.csv", index=False)
print("Saved: cleaned_sgcc_small_long.csv")

# ---------------------------------------------------------
# STEP 8: Also create a per-consumer summary (one row per consumer)
# ---------------------------------------------------------
df_summary = df_long.groupby(["CONS_NO", "FLAG"]).agg(
    avg_consumption=("consumption", "mean"),
    max_consumption=("consumption", "max"),
    min_consumption=("consumption", "min"),
    std_consumption=("consumption", "std"),
    total_days=("consumption", "count")
).reset_index()

df_summary.to_csv("cleaned_sgcc_small_summary.csv", index=False)
print("Saved: cleaned_sgcc_small_summary.csv")

print("\nDone! Two files are ready in this folder:")
print("  1. cleaned_sgcc_small_long.csv      (full detail, one row per consumer per day)")
print("  2. cleaned_sgcc_small_summary.csv   (one row per consumer, quick overview)")