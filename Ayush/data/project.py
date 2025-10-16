# src/investigation_b_pipeline.py
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from math import sqrt
from scipy import stats
from statsmodels.nonparametric.smoothers_lowess import lowess
import statsmodels.formula.api as smf

# ---------------------------
# 0) Global config
# ---------------------------
# Resolve base directory to the script location
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
OUT_DIR = BASE_DIR.parent / "outputs"
FIG_DIR = OUT_DIR / "figure"  # Changed from "figures" to "figure"
STAT_DIR = OUT_DIR / "stats"

# Data files
DATA1 = DATA_DIR / "dataset1.csv"
DATA2 = DATA_DIR / "dataset2.csv"

# Ensure output folders exist
FIG_DIR.mkdir(parents=True, exist_ok=True)
STAT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
rng = np.random.default_rng(SEED)

SEASON_MAP = {0: "Winter", 1: "Spring"}
MONTH_NAMES = {0: "Dec", 1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr"}

plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120
})
COLORS = {"Winter": "#1f77b4", "Spring": "#2ca02c"}

# ---------------------------
# 1) Helpers
# ---------------------------
def wilson_ci(successes: int, n: int, conf: float = 0.95):
    """95% Wilson score CI for a binomial proportion."""
    if n == 0:
        return np.nan, np.nan
    z = stats.norm.ppf(1 - (1 - conf) / 2)
    phat = successes / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2*n)) / denom
    margin = (z * sqrt((phat*(1-phat) + z**2/(4*n)) / n)) / denom
    return center - margin, center + margin

def iqr_fences(x: np.ndarray):
    """Return Q1, Q3, IQR and Tukey 1.5*IQR outlier fences."""
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
    return q1, q3, iqr, lower, upper

def save_json(obj, path: Path):
    path.write_text(json.dumps(obj, indent=2))

# ---------------------------
# 2) Load & harmonize data
# ---------------------------
def load_data():
    df1 = pd.read_csv(DATA1)
    df2 = pd.read_csv(DATA2)

    df1 = df1.dropna(subset=["season"]).copy()
    df1["season"] = df1["season"].astype(float).astype(int)
    df1["season_label"] = df1["season"].map(SEASON_MAP)
    df1["month_name"] = df1["month"].map(MONTH_NAMES)

    month_to_season = (
        df1.groupby("month")["season"]
           .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan)
           .to_dict()
    )
    df2["season"] = df2["month"].map(month_to_season)
    df2 = df2.dropna(subset=["season"]).copy()
    df2["season"] = df2["season"].astype(int)
    df2["season_label"] = df2["season"].map(SEASON_MAP)
    return df1, df2

# ---------------------------
# 3) Descriptives
# ---------------------------
def summarize_outcomes(df1: pd.DataFrame):
    rows = []
    for s in ["Winter", "Spring"]:
        sub = df1[df1["season_label"] == s]
        n = len(sub)
        r_success = int(sub["risk"].sum())
        r_hat = r_success / n
        r_lo, r_hi = wilson_ci(r_success, n)
        w_success = int(sub["reward"].sum())
        w_hat = w_success / n
        w_lo, w_hi = wilson_ci(w_success, n)
        rows.append({
            "season": s, "n": n,
            "risk_rate": r_hat, "risk_lo": r_lo, "risk_hi": r_hi,
            "reward_rate": w_hat, "reward_lo": w_lo, "reward_hi": w_hi
        })
    out = pd.DataFrame(rows)
    out.to_csv(STAT_DIR / "risk_reward_by_season.csv", index=False)
    return out

def landing_delay_stats(df1: pd.DataFrame):
    stats_dict = {}
    for s in ["Winter", "Spring"]:
        x = df1.loc[df1["season_label"] == s, "bat_landing_to_food"].dropna().values
        q1, q3, iqr, lo, hi = iqr_fences(x)
        stats_dict[s] = {
            "n": int(len(x)), "Q1": float(q1), "Q3": float(q3),
            "IQR": float(iqr), "lower_fence": float(lo), "upper_fence": float(hi),
            "median": float(np.median(x)), "mean": float(np.mean(x))
        }

    w = df1.loc[df1["season_label"] == "Winter", "bat_landing_to_food"].dropna().values
    sp = df1.loc[df1["season_label"] == "Spring", "bat_landing_to_food"].dropna().values
    nboot = 5000
    boots = []
    for _ in range(nboot):
        w_s = rng.choice(w, size=len(w), replace=True)
        s_s = rng.choice(sp, size=len(sp), replace=True)
        boots.append(np.median(s_s) - np.median(w_s))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    stats_dict["effect_size_median_diff"] = {
        "diff": float(np.median(sp) - np.median(w)),
        "ci95_lo": float(lo),
        "ci95_hi": float(hi)
    }
    save_json(stats_dict, STAT_DIR / "landing_delay_stats.json")
    return stats_dict

def rat_bat_activity(df2: pd.DataFrame):
    ag = (df2.groupby("season_label")
               .agg(rat_minutes_mean=("rat_minutes", "mean"),
                    rat_minutes_sd=("rat_minutes", "std"),
                    rat_minutes_n=("rat_minutes", "count"),
                    rat_arr_mean=("rat_arrival_number", "mean"),
                    rat_arr_sd=("rat_arrival_number", "std"),
                    rat_arr_n=("rat_arrival_number", "count"),
                    bat_land_mean=("bat_landing_number", "mean"),
                    bat_land_sd=("bat_landing_number", "std"),
                    bat_land_n=("bat_landing_number", "count")))
    ag.to_csv(STAT_DIR / "rat_bat_activity_by_season.csv")
    return ag

# ---------------------------
# 4) Visuals (w/ outlier handling documented)
# ---------------------------
def plot_risk_reward_with_cis(rr_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    x = np.arange(len(rr_df))
    # Risk
    ax.errorbar(x-0.1, rr_df["risk_rate"], 
                yerr=[rr_df["risk_rate"]-rr_df["risk_lo"], rr_df["risk_hi"]-rr_df["risk_rate"]],
                fmt="o", color=COLORS["Winter"], ecolor="#555", capsize=4, label="Risk-taking")
    # Reward
    ax.errorbar(x+0.1, rr_df["reward_rate"], 
                yerr=[rr_df["reward_rate"]-rr_df["reward_lo"], rr_df["reward_hi"]-rr_df["reward_rate"]],
                fmt="o", color="#ff7f0e", ecolor="#555", capsize=4, label="Reward")
    ax.set_xticks(x)
    ax.set_xticklabels(rr_df["season"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Proportion (95% Wilson CI)")
    ax.set_title("Risk-taking & Reward by Season")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "risk_reward_wilson_ci_by_season.png")
    plt.close(fig)

def plot_landing_delay_violin_box(df1: pd.DataFrame):
    data = [df1.loc[df1["season_label"]==s, "bat_landing_to_food"].dropna() for s in ["Winter","Spring"]]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    parts = ax.violinplot(data, showmeans=False, showmedians=False, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor("#c6dbef"); pc.set_edgecolor("#4a90e2"); pc.set_alpha(0.85)
    # Box overlay uses 1.5*IQR whiskers → outliers shown as points
    ax.boxplot(data, labels=["Winter","Spring"], widths=0.18, patch_artist=True,
               boxprops=dict(facecolor="white", edgecolor="#333"),
               medianprops=dict(color="red"),
               whiskerprops=dict(color="#333"),
               capprops=dict(color="#333"),
               flierprops=dict(marker="o", markerfacecolor="#e74c3c", markersize=3, alpha=0.6))
    ax.set_ylabel("Seconds (landing → food)")
    ax.set_title("Landing → Food Delay by Season (outliers shown)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "landing_delay_violin_box.png")
    plt.close(fig)

def plot_landing_delay_effectsize_bootstrap(stats_dict: dict):
    diff = stats_dict["effect_size_median_diff"]["diff"]
    lo = stats_dict["effect_size_median_diff"]["ci95_lo"]
    hi = stats_dict["effect_size_median_diff"]["ci95_hi"]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.axvline(0, color="#999", ls="--")
    ax.errorbar([0], [diff], yerr=[[diff-lo], [hi-diff]], fmt="o", color="#e74c3c", capsize=5)
    ax.set_xlim(-1, 1)
    ax.set_xticks([])
    ax.set_ylabel("Median(Spring) – Median(Winter) seconds")
    ax.set_title("Effect size: Landing→Food delay (bootstrap 95% CI)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "landing_delay_effectsize_bootstrap.png")
    plt.close(fig)

def plot_rat_activity_means_ci(ag: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    seasons = ["Winter", "Spring"]
    x = np.arange(len(seasons))
    bar_w = 0.36

    # Means & 95% CI (mean ± 1.96*SE). We keep outliers—CIs show uncertainty.
    mins = ag.loc[seasons, "rat_minutes_mean"]; mins_ci = 1.96 * (ag.loc[seasons, "rat_minutes_sd"] / np.sqrt(ag.loc[seasons, "rat_minutes_n"]))
    arrs = ag.loc[seasons, "rat_arr_mean"]; arrs_ci = 1.96 * (ag.loc[seasons, "rat_arr_sd"] / np.sqrt(ag.loc[seasons, "rat_arr_n"]))

    ax.bar(x - bar_w/2, mins, bar_w, color=[COLORS[s] for s in seasons], label="Avg rat minutes")
    ax.bar(x + bar_w/2, arrs, bar_w, color="#9467bd", label="Avg rat arrivals")
    ax.errorbar(x - bar_w/2, mins, yerr=mins_ci, fmt="none", ecolor="#333", capsize=4)
    ax.errorbar(x + bar_w/2, arrs, yerr=arrs_ci, fmt="none", ecolor="#333", capsize=4)
    ax.set_xticks(x); ax.set_xticklabels(seasons)
    ax.set_ylabel("Mean per 30-min interval ± 95% CI")
    ax.set_title("Rat Activity by Season")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rat_activity_means_ci.png")
    plt.close(fig)

def plot_ratmins_vs_batlandings_lowess(df2: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for s in ["Winter", "Spring"]:
        sub = df2[df2["season_label"] == s]
        ax.scatter(sub["rat_minutes"], sub["bat_landing_number"], s=18, alpha=0.5, color=COLORS[s], label=s)
        if len(sub) > 10:
            # LOWESS downweights outliers (robust smoother)
            fitted = lowess(sub["bat_landing_number"], sub["rat_minutes"], frac=0.3, it=1, return_sorted=True)
            ax.plot(fitted[:, 0], fitted[:, 1], color=COLORS[s], lw=2)
    ax.set_xlabel("Rat minutes (per 30-min interval)")
    ax.set_ylabel("Bat landings (per 30-min interval)")
    ax.set_title("Relationship: Rat Presence vs Bat Landings (LOWESS)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "scatter_ratmins_vs_batlandings_lowess.png")
    plt.close(fig)

def plot_reward_rate_monthly(df1: pd.DataFrame):
    order = ["Dec","Jan","Feb","Mar","Apr"]
    trend = (df1.groupby(["month_name","season_label"])["reward"]
                .mean().reset_index())
    trend["month_order"] = trend["month_name"].map({m:i for i,m in enumerate(order)})
    trend = trend.sort_values("month_order")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for s in ["Winter", "Spring"]:
        sub = trend[trend["season_label"] == s]
        ax.plot(sub["month_order"], sub["reward"], marker="o", color=COLORS[s], label=s)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Reward rate")
    ax.set_title("Monthly Reward Rate (Dec → Apr)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "reward_rate_monthly.png")
    plt.close(fig)

# ---------------------------
# 5) Statistical tests
# ---------------------------
def run_tests(df1: pd.DataFrame, df2: pd.DataFrame):
    out = {}

    # Chi-square: risk vs season; reward vs season
    risk_ct = pd.crosstab(df1["season_label"], df1["risk"])
    reward_ct = pd.crosstab(df1["season_label"], df1["reward"])
    chi_risk = stats.chi2_contingency(risk_ct)
    chi_reward = stats.chi2_contingency(reward_ct)
    out["chi2_risk"] = {"chi2": float(chi_risk[0]), "p": float(chi_risk[1]), "dof": int(chi_risk[2])}
    out["chi2_reward"] = {"chi2": float(chi_reward[0]), "p": float(chi_reward[1]), "dof": int(chi_reward[2])}

    # Mann–Whitney U: landing->food delay (robust to outliers)
    w = df1[df1["season_label"]=="Winter"]["bat_landing_to_food"].dropna()
    s = df1[df1["season_label"]=="Spring"]["bat_landing_to_food"].dropna()
    U, p = stats.mannwhitneyu(w, s, alternative="two-sided")
    out["mw_delay"] = {"U": float(U), "p": float(p),
                       "winter_median": float(np.median(w)),
                       "spring_median": float(np.median(s))}

    # Mann–Whitney U for activity measures (df2)
    for col in ["rat_minutes", "rat_arrival_number", "bat_landing_number"]:
        wv = df2[df2["season_label"]=="Winter"][col].dropna()
        sv = df2[df2["season_label"]=="Spring"][col].dropna()
        U, p = stats.mannwhitneyu(wv, sv, alternative="two-sided")
        out[f"mw_{col}"] = {"U": float(U), "p": float(p),
                            "winter_median": float(np.median(wv)), "spring_median": float(np.median(sv)),
                            "winter_mean": float(wv.mean()), "spring_mean": float(sv.mean())}

    # Spearman rho: rat_minutes vs bat_landing_number (rank-based, less sensitive to outliers)
    rho_all, p_all = stats.spearmanr(df2["rat_minutes"], df2["bat_landing_number"])
    out["spearman_all"] = {"rho": float(rho_all), "p": float(p_all), "n": int(len(df2))}
    for s_label in ["Winter", "Spring"]:
        sub = df2[df2["season_label"] == s_label]
        rho, p = stats.spearmanr(sub["rat_minutes"], sub["bat_landing_number"])
        out[f"spearman_{s_label.lower()}"] = {"rho": float(rho), "p": float(p), "n": int(len(sub))}

    save_json(out, STAT_DIR / "investigation_b_tests.json")
    return out

# ---------------------------
# 6) Optional models
# ---------------------------
def models_optional(df1: pd.DataFrame):
    """Logistic models to adjust for time-of-night; linear model for delay."""
    results = {}

    # Logistic: risk ~ season + hours_after_sunset
    df = df1.dropna(subset=["risk","season","hours_after_sunset"]).copy()
    df["season"] = df["season"].astype(int)
    m1 = smf.logit("risk ~ season + hours_after_sunset", data=df).fit(disp=False)
    results["logit_risk"] = {
        "params": {k: float(v) for k, v in m1.params.items()},
        "pvalues": {k: float(v) for k, v in m1.pvalues.items()}
    }

    # Logistic: reward ~ season + risk + hours_after_sunset
    df = df1.dropna(subset=["reward","risk","season","hours_after_sunset"]).copy()
    df["season"] = df["season"].astype(int)
    m2 = smf.logit("reward ~ season + risk + hours_after_sunset", data=df).fit(disp=False)
    results["logit_reward"] = {
        "params": {k: float(v) for k, v in m2.params.items()},
        "pvalues": {k: float(v) for k, v in m2.pvalues.items()}
    }

    # Linear (optional): landing delay ~ season + rat_minutes (merged proxy)
    
    lm_df = df1.dropna(subset=["bat_landing_to_food","season"]).copy()
    lm_df["season"] = lm_df["season"].astype(int)
    m3 = smf.ols("bat_landing_to_food ~ C(season)", data=lm_df).fit()
    results["ols_delay"] = {
        "params": {k: float(v) for k, v in m3.params.items()},
        "pvalues": {k: float(v) for k, v in m3.pvalues.items()},
        "r2": float(m3.rsquared)
    }

    save_json(results, STAT_DIR / "models_optional.json")
    return results

# ---------------------------
# 7) Orchestrate (main)
# ---------------------------
def main():
    df1, df2 = load_data()

    rr = summarize_outcomes(df1)
    delay_stats = landing_delay_stats(df1)
    activity = rat_bat_activity(df2)

    # Call your plotting functions here
    plot_risk_reward_with_cis(rr)
    plot_landing_delay_violin_box(df1)
    plot_landing_delay_effectsize_bootstrap(delay_stats)
    plot_rat_activity_means_ci(activity)
    plot_ratmins_vs_batlandings_lowess(df2)
    plot_reward_rate_monthly(df1)

    tests = run_tests(df1, df2)
    models = models_optional(df1)

    print("Done. Figures in:", FIG_DIR.resolve())
    print("Stats JSON in:", STAT_DIR.resolve())

if __name__ == "__main__":
    main()