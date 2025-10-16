# Purpose: Create engineered features for Assessment 3 and save engineered dataset(s)
# Output: datasets/cleaned/engineered_dataset1.csv

import pandas as pd
import numpy as np
from pathlib import Path

# ---------- Helpers ----------
def find_repo_root(start: Path) -> Path:
    """Walk up until a 'datasets' folder is found; return that as repo root."""
    p = start.resolve()
    for parent in [p] + list(p.parents):
        if (parent / "datasets").exists():
            return parent
    raise RuntimeError("Could not locate repo root (folder containing 'datasets').")

def half_hour_bin(hours: float) -> float:
    """Round to nearest 0.5 hours for reproducible joins."""
    if pd.isna(hours):
        return np.nan
    return float(np.round(hours * 2) / 2)

# ---------- Locate files ----------
here = Path(__file__).resolve().parent
repo_root = find_repo_root(here)

ds1_path = repo_root / "datasets" / "cleaned" / "cleaned_dataset1.csv"
ds2_path = repo_root / "datasets" / "cleaned" / "cleaned_dataset2_seasons.csv"
out1_path = repo_root / "datasets" / "cleaned" / "engineered_dataset1.csv"

print(f"[i] Repo root: {repo_root}")
print(f"[i] Loading: {ds1_path.name}, {ds2_path.name}")

# ---------- Load ----------
df1 = pd.read_csv(ds1_path)  # bat landings
df2 = pd.read_csv(ds2_path)  # 30-min periods (rats presence etc.)

print(f"[i] df1 shape: {df1.shape}")
print(f"[i] df2 shape: {df2.shape}")

# ---------- Normalize season coding ----------
# In our data: dataset1 uses {0,1}, dataset2 might contain a few 2s -> merge them into 1 (Spring)
if "season" in df2.columns:
    df2["season"] = df2["season"].replace({2: 1})

# ---------- CORE FEATURE ENGINEERING (df1 = bat-level) ----------
# 1) Bat efficiency: how quickly bats get reward after landing
#    (0 if no reward; small values if reward took long; higher=more efficient)
df1["bat_efficiency"] = df1["reward"] / (df1["bat_landing_to_food"].astype(float) + 1.0)

# 2) Risk–reward interaction: does risk-taking pay off?
df1["risk_reward_interaction"] = (df1["risk"].astype(float) * df1["reward"].astype(float))

# 3) Night period category from hours_after_sunset (Early/Mid/Late)
bins = [0, 2, 4, 24]
labels = ["Early", "Mid", "Late"]
df1["night_period"] = pd.cut(df1["hours_after_sunset"].astype(float), bins=bins, labels=labels, include_lowest=True)

# 4) (Optional numeric coding for regression dummies later)
#    We'll keep the categorical, but we can one-hot later during modeling.

# ---------- RAT-INTENSITY FEATURES FROM df2 ----------
# 5) Rat intensity per period: how long rats are present per arrival
#    (captures "pressure" from rats during an interval)
df2["rat_intensity"] = df2["rat_minutes"].astype(float) / (df2["rat_arrival_number"].astype(float) + 1.0)

# 6) Create a half-hour bin for hours_after_sunset in both datasets to align periods
df2["hour_bin"] = df2["hours_after_sunset"].astype(float).apply(half_hour_bin)
df1["hour_bin"] = df1["hours_after_sunset"].astype(float).apply(half_hour_bin)

# 7) Aggregate rat pressure by (month, season, hour_bin) from df2
agg = (
    df2.groupby(["month", "season", "hour_bin"], dropna=True)
       .agg(
           rat_intensity_mean=("rat_intensity", "mean"),
           rat_minutes_mean=("rat_minutes", "mean"),
           rat_arrival_mean=("rat_arrival_number", "mean"),
       )
       .reset_index()
)

print(f"[i] Aggregated rat-pressure rows: {len(agg)}")

# 8) Merge aggregated rat pressure into df1 by (month, season, hour_bin)
merge_keys = ["month", "season", "hour_bin"]
for k in merge_keys:
    if k not in df1.columns:
        raise KeyError(f"df1 is missing required column for merge: {k}")
    if k not in agg.columns:
        raise KeyError(f"agg is missing required column for merge: {k}")

df1_before = len(df1)
df1 = df1.merge(agg, on=merge_keys, how="left")
matched = df1["rat_intensity_mean"].notna().sum()
print(f"[i] Merged rat-pressure features into df1: matched {matched}/{df1_before} rows")

# ---------- Sanity checks ----------
new_cols = ["bat_efficiency", "risk_reward_interaction", "night_period",
            "rat_intensity_mean", "rat_minutes_mean", "rat_arrival_mean"]
print("\n[i] New columns created:")
for c in new_cols:
    print("   -", c)

print("\n[i] Null counts (new cols):")
print(df1[new_cols].isna().sum())

# ---------- Save ----------
df1.to_csv(out1_path, index=False)
print(f"\n[✔] Saved engineered dataset: {out1_path}")
print(f"[✔] Final shape: {df1.shape}")

# ---------- Quick preview ----------
preview_cols = [
    "bat_landing_to_food", "risk", "reward", "hours_after_sunset", "season",
    "bat_efficiency", "risk_reward_interaction", "night_period",
    "rat_intensity_mean", "rat_minutes_mean", "rat_arrival_mean"
]
preview_cols = [c for c in preview_cols if c in df1.columns]
print("\n[i] Preview (first 5 rows):")
print(df1[preview_cols].head(5))
