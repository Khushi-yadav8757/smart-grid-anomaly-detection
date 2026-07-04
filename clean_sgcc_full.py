"""
Smart Grid Project - SGCC Theft Dataset Cleaning (Memory-Safe / Batched Version)
-----------------------------------------
"""

import pandas as pd
import numpy as np

RAW_FILE = "data set.csv"
OUT_LONG = "cleaned_sgcc_full_long.csv"
OUT_SUMMARY = "cleaned_sgcc_full_summary.csv"
BATCH_SIZE = 2000   # number of consumers processed at a time

print("Reading header to identify columns...")
header = pd.read_csv(RAW_FILE, nrows=0).columns.tolist()
id_cols = ["CONS_NO", "FLAG"]
date_cols = [c for c in header if c not in id_cols]
print(f"Found {len(date_cols)} date columns")

# ---------------------------------------------------------
# Read only CONS_NO to know total number of consumers/rows
# ---------------------------------------------------------
total_rows = sum(1 for _ in open(RAW_FILE, encoding="utf-8")) - 1  # minus header
print(f"Total consumers (rows) in file: {total_rows:,}")

first_write = True
summary_rows = []
total_missing_before = 0
total_missing_after = 0

# ---------------------------------------------------------
# Process the file in batches of rows (consumers)
# ---------------------------------------------------------
reader = pd.read_csv(
    RAW_FILE,
    sep=",",
    na_values=["?", ""],
    low_memory=False,
    chunksize=BATCH_SIZE
)

batch_num = 0
for chunk in reader:
    batch_num += 1
    print(f"Processing batch {batch_num} ({len(chunk):,} consumers)...")

    # Wide -> Long for this batch only
    chunk_long = chunk.melt(
        id_vars=id_cols,
        value_vars=date_cols,
        var_name="date",
        value_name="consumption"
    )

    # Parse date
    chunk_long["date"] = pd.to_datetime(chunk_long["date"], errors="coerce")
    chunk_long = chunk_long[chunk_long["date"].notna()]

    # Numeric conversion
    chunk_long["consumption"] = pd.to_numeric(chunk_long["consumption"], errors="coerce")

    missing_before = chunk_long["consumption"].isna().sum()
    total_missing_before += missing_before

    # Sort for correct fill/rolling order
    chunk_long.sort_values(["CONS_NO", "date"], inplace=True)

    # Fill missing values within each consumer
    chunk_long["consumption"] = (
        chunk_long.groupby("CONS_NO")["consumption"]
        .transform(lambda s: s.ffill().bfill())
    )
    chunk_long["consumption"] = chunk_long["consumption"].fillna(0)

    missing_after = chunk_long["consumption"].isna().sum()
    total_missing_after += missing_after

    # Feature engineering (rolling stats over 7 days, per consumer)
    chunk_long["consumption_rolling_mean_7"] = (
        chunk_long.groupby("CONS_NO")["consumption"]
        .transform(lambda s: s.rolling(window=7, min_periods=1).mean())
    )
    chunk_long["consumption_rolling_std_7"] = (
        chunk_long.groupby("CONS_NO")["consumption"]
        .transform(lambda s: s.rolling(window=7, min_periods=1).std().fillna(0))
    )

    # Time-based features
    chunk_long["day_of_week"] = chunk_long["date"].dt.dayofweek
    chunk_long["month"] = chunk_long["date"].dt.month
    chunk_long["is_weekend"] = chunk_long["day_of_week"].apply(lambda d: 1 if d >= 5 else 0)

    # Downcast dtypes to save memory/disk space
    chunk_long["consumption"] = chunk_long["consumption"].astype("float32")
    chunk_long["consumption_rolling_mean_7"] = chunk_long["consumption_rolling_mean_7"].astype("float32")
    chunk_long["consumption_rolling_std_7"] = chunk_long["consumption_rolling_std_7"].astype("float32")

    # Append this batch to the output CSV (write header only once)
    chunk_long.to_csv(OUT_LONG, mode="w" if first_write else "a", header=first_write, index=False)

    # Build per-consumer summary for this batch
    batch_summary = chunk_long.groupby(["CONS_NO", "FLAG"]).agg(
        avg_consumption=("consumption", "mean"),
        max_consumption=("consumption", "max"),
        min_consumption=("consumption", "min"),
        std_consumption=("consumption", "std"),
        total_days=("consumption", "count")
    ).reset_index()
    summary_rows.append(batch_summary)

    first_write = False

print(f"\nTotal missing values found: {total_missing_before:,}")
print(f"Total missing values after cleaning: {total_missing_after}")
print(f"Saved: {OUT_LONG}")

# Combine all batch summaries into one file
df_summary = pd.concat(summary_rows, ignore_index=True)
df_summary.to_csv(OUT_SUMMARY, index=False)
print(f"Saved: {OUT_SUMMARY}")

print("\nDone! Two files are ready in this folder:")
print(f"  1. {OUT_LONG}      (full detail, one row per consumer per day)")
print(f"  2. {OUT_SUMMARY}   (one row per consumer, quick overview)")