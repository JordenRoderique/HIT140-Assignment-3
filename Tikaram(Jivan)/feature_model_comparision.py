# --- Investigation A: Baseline vs Enhanced Linear Regression ---
# Feature Engineered Model Comparison

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from pathlib import Path

# --- Auto path finder: works no matter where we run it ---
def find_repo_root(start: Path) -> Path:
    p = start.resolve()
    for parent in [p] + list(p.parents):
        if (parent / "datasets").exists():
            return parent
    raise RuntimeError("Repo root not found")

repo_root = find_repo_root(Path(__file__).parent)
base_path = repo_root / "datasets" / "cleaned" / "cleaned_dataset1.csv"
enh_path  = repo_root / "datasets" / "cleaned" / "engineered_dataset1.csv"

base = pd.read_csv(base_path)
enh  = pd.read_csv(enh_path)
print(f"[i] Loaded: {base_path.name}, {enh_path.name}")

# Clean up: remove rare season label (2)
base = base[base['season'] < 2]
enh  = enh[enh['season'] < 2]

# Response variable
y_base = base['bat_landing_to_food']
y_enh  = enh['bat_landing_to_food']

# --- BASELINE MODEL (Jorden's) ---
X_base = base[['seconds_after_rat_arrival','hours_after_sunset','risk','reward','season']]
X_base = sm.add_constant(X_base)
model_base = sm.OLS(y_base, X_base, missing='drop').fit()

# --- ENHANCED MODEL (our engineered features) ---
X_enh = enh[['seconds_after_rat_arrival','hours_after_sunset','risk','reward',
             'bat_efficiency','risk_reward_interaction',
             'rat_intensity_mean','rat_minutes_mean','rat_arrival_mean']]
X_enh = sm.add_constant(X_enh)
model_enh = sm.OLS(y_enh, X_enh, missing='drop').fit()

# --- RMSE helper ---
def rmse(y, pred): 
    return np.sqrt(mean_squared_error(y.loc[pred.index], pred))

# --- Compare performance ---
results = pd.DataFrame({
    'Model': ['Baseline', 'Enhanced'],
    'R2': [model_base.rsquared, model_enh.rsquared],
    'Adj_R2': [model_base.rsquared_adj, model_enh.rsquared_adj],
    'RMSE': [rmse(y_base, model_base.fittedvalues),
             rmse(y_enh, model_enh.fittedvalues)]
}).round(4)

print("\n📊 Model Performance Comparison:")
print(results.to_string(index=False))

print("\n🔍 Significant Predictors (Enhanced Model):")
print(model_enh.summary().tables[1])
