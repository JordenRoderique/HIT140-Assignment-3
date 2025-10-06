# --- Investigation B: Seasonal Models (Winter vs Spring) ---
# Tikaram (Jivan) – Seasonal Behaviour Comparison

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from pathlib import Path

# 🧭 Automatically find repo root (folder containing "datasets")
repo_root = Path(__file__).resolve().parents[1]
data_path = repo_root / "datasets" / "cleaned" / "engineered_dataset1.csv"

print(f"[i] Loading dataset from: {data_path}")
enh = pd.read_csv(data_path)

# 🧹 Clean and prepare
enh['season'] = pd.to_numeric(enh['season'], errors='coerce')
df = enh.copy()

# Split dataset by season
df_winter = df[df['season'] == 0]
df_spring = df[df['season'] == 1]

# Predictor columns (same as Investigation A)
X_cols = [
    'seconds_after_rat_arrival', 'hours_after_sunset', 'risk', 'reward',
    'bat_efficiency', 'risk_reward_interaction',
    'rat_intensity_mean', 'rat_minutes_mean', 'rat_arrival_mean'
]

# Helper function to fit regression
def fit_lr(subset, label):
    X = sm.add_constant(subset[X_cols])
    y = subset['bat_landing_to_food']
    model = sm.OLS(y, X, missing='drop').fit()
    print(f"\n🌤️  {label} Results")
    print(f"R² = {model.rsquared:.3f} | Adj R² = {model.rsquared_adj:.3f}")
    return model

# Fit seasonal models
model_winter = fit_lr(df_winter, "Winter (Season = 0)")
model_spring = fit_lr(df_spring, "Spring (Season = 1)")

# Compare coefficients side by side
coef_compare = pd.DataFrame({
    'Variable': X_cols,
    'Winter β': model_winter.params[X_cols].round(3),
    'Spring β': model_spring.params[X_cols].round(3)
})

print("\n🧮  Coefficient Comparison (Winter vs Spring):")
print(coef_compare.to_string(index=False))

# --- Visual: Rat intensity vs Bat behaviour by season ---
sns.set(style="whitegrid")
plot = sns.lmplot(
    data=df,
    x='rat_intensity_mean',
    y='bat_landing_to_food',
    hue='season',
    palette='Set2',               # changed colour palette
    scatter_kws={'alpha': 0.5},
    line_kws={'lw': 2}
)
plt.title("Bat Feeding Delay vs Rat Intensity by Season")
plt.xlabel("Mean Rat Intensity")
plt.ylabel("Bat Landing → Food Delay (sec)")

# --- Save the figure under Tikaram(Jivan)/figures ---
save_path = repo_root / "Tikaram(Jivan)" / "figures"
save_path.mkdir(parents=True, exist_ok=True)

output_file = save_path / "figure1_seasonal_comparison.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"[✔] Figure saved at: {output_file}")

plt.show()
