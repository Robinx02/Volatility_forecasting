import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.graphics.tsaplots import plot_acf
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv(r"D:\MBFM_project\master_features.csv", index_col="date", parse_dates=True)

out = r"D:\MBFM_project\figures"
import os; os.makedirs(out, exist_ok=True)

#these are the crisis annotations

crises = {
    "GFC\n2008":      "2008-09-15",
    "Euro\n2011":     "2011-08-01",
    "Taper\n2013":    "2013-05-22",
    "CNY\n2015":      "2015-08-11",
    "Demone-\ntiz. 2016": "2016-11-08",
    "IL&FS\n2018":    "2018-09-21",
    "COVID\n2020":    "2020-03-01",
    "RU-UA\n2022":    "2022-02-24",
}


# FIG 1 — RV time series with crisis annotations

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(df.index, df["rv"], lw=0.8, color="#2c7bb6", label="Realised Volatility")
ax.fill_between(df.index, df["rv"], alpha=0.15, color="#2c7bb6")

for label, dt in crises.items():
    x = pd.Timestamp(dt)
    if x in df.index or (x > df.index.min() and x < df.index.max()):
        ax.axvline(x, color="crimson", lw=0.8, ls="--", alpha=0.7)
        ax.text(x, df["rv"].max() * 0.97, label, fontsize=6.5, ha="center", va="top", color="crimson",bbox=dict(fc="white", ec="none", alpha=0.6, pad=1))

ax.set_ylabel("Annualised RV")
ax.set_xlabel("")
ax.set_title("INR/USD Realised Volatility (2005–2026)")
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.xticks(rotation=45)
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{out}/fig1_rv_series.png", dpi=200)
plt.show()
print("Fig 1 saved")

# FIG 2 — ACF of RV to 60 lags

fig, ax = plt.subplots(figsize=(10, 3.5))
plot_acf(df["rv"], lags=60, ax=ax, color="#2c7bb6", vlines_kwargs={"lw": 1.2})
ax.set_title("ACF of Realised Volatility (lags 0–60)")
ax.set_xlabel("Lag (days)")
ax.set_ylabel("Autocorrelation")
plt.tight_layout()
plt.savefig(f"{out}/fig2_rv_acf.png", dpi=200)
plt.show()
print("Fig 2 saved")

# TABLE 1 — Descriptive statistics

desc_cols = ["rv", "rv_lag1", "vix_level", "brent_ret", "dxy_level", "india10y", "us10y", "rate_spread", "callmoney", "nifty_ret", "em_ret", "ret_t"]

desc = df[desc_cols].describe().T[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
desc["skew"] = df[desc_cols].skew()
desc["kurt"] = df[desc_cols].kurt()
desc = desc.round(4)
print("\nTable 1 — Descriptive Statistics")
print(desc.to_string())
desc.to_csv(f"{out}/table1_descriptives.csv")
print("Table 1 saved")


# FIG 3 — Correlation heatmap (ML features only)

ml_features = ["rv_lag1", "rv_5d", "rv_22d", "sq_return", "vix_level", "vix_chg", "brent_ret", "dxy_level", "dxy_ret", "india10y", "india10y_chg", "us10y", "us10y_chg", "rate_spread", "callmoney", "nifty_ret", "em_ret"]

corr = df[ml_features].corr()

fig, ax = plt.subplots(figsize=(11, 9))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
plt.colorbar(im, ax=ax, fraction=0.03)
ax.set_xticks(range(len(ml_features)))
ax.set_yticks(range(len(ml_features)))
ax.set_xticklabels(ml_features, rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(ml_features, fontsize=8)
for i in range(len(ml_features)):
    for j in range(len(ml_features)):
        ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center",
                fontsize=5.5, color="black")
ax.set_title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig(f"{out}/fig3_corr_heatmap.png", dpi=200)
plt.show()
print("Fig 3 saved")


# FIG 4 — Rolling 252-day correlation of RV with key features

roll_features = ["vix_level", "dxy_ret", "brent_ret", "nifty_ret", "em_ret", "rate_spread"]
colors = ["#e41a1c","#377eb8","#4daf4a","#984ea3","#ff7f00","#a65628"]

fig, ax = plt.subplots(figsize=(14, 4))
for feat, col in zip(roll_features, colors):
    rc = df["rv"].rolling(252).corr(df[feat])
    ax.plot(df.index, rc, lw=0.9, label=feat, color=col)

ax.axhline(0, color="black", lw=0.6, ls="--")
ax.set_ylabel("Rolling Correlation with RV")
ax.set_title("252-day Rolling Correlation: RV vs Key Features")
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.xticks(rotation=45)
ax.legend(fontsize=8, ncol=3)
plt.tight_layout()
plt.savefig(f"{out}/fig4_rolling_corr.png", dpi=200)
plt.show()
print("Fig 4 saved")

# FIG 5 — RV distribution: full vs high-vol regime (>75th pct)

q75 = df["rv"].quantile(0.75)
rv_low  = df.loc[df["rv"] <= q75, "rv"]
rv_high = df.loc[df["rv"] >  q75, "rv"]

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(rv_low,  bins=60, alpha=0.6, color="#2c7bb6", label=f"Low vol (≤p75={q75:.3f})", density=True)
ax.hist(rv_high, bins=60, alpha=0.6, color="crimson",  label=f"High vol (>p75)",          density=True)
ax.axvline(q75, color="black", lw=1, ls="--", label="75th pct")
ax.set_xlabel("Annualised RV")
ax.set_ylabel("Density")
ax.set_title("RV Distribution: Low vs High Volatility Regimes")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{out}/fig5_rv_regimes.png", dpi=200)
plt.show()
print("Fig 5 saved")

print(f"\nHigh-vol regime: {len(rv_high)} days ({100*len(rv_high)/len(df):.1f}% of sample)")
print(f"75th pct RV threshold: {q75:.4f}")
print(f"\nAll figures saved to {out}")