import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

DATA   = r"D:\MBFM_project\master_features.csv"
FCAST  = r"D:\MBFM_project\results\forecasts"
FIGDIR = r"D:\MBFM_project\figures"
RESDIR = r"D:\MBFM_project\results"
os.makedirs(FIGDIR, exist_ok=True)
os.makedirs(RESDIR, exist_ok=True)

TRAIN     = 1500
RV_THRESH = 0.070

df     = pd.read_csv(DATA, index_col=0, parse_dates=True)
actual = df["rv"].values[TRAIN:]
dates  = df.index[TRAIN:]

MODELS = {
    "GARCH(1,1)" : "garch11.npy",
    "EGARCH"     : "egarch.npy",
    "GJR-GARCH"  : "gjr_garch.npy",
    "HAR-RV"     : "har_rv.npy",
    "HAR-RV-X"   : "har_rv_x.npy",
    "Random Forest" : "random_forest.npy",
    "XGBoost" : "xgboost.npy",
    "LSTM"   : "lstm.npy",
    "TFT"   : "tft.npy",
}

def load(fname):
    fc = np.load(os.path.join(FCAST, fname))
    if len(fc) == len(df):
        fc = fc[TRAIN:]
    fc = np.where((fc > 0) & ~np.isnan(fc), fc, np.nan)
    return fc

forecasts = {name: load(fname) for name, fname in MODELS.items()}

def qlike(a, f):
    m = ~np.isnan(f) & (f > 0)
    return np.mean(a[m]/f[m] - np.log(a[m]/f[m]) - 1)

def rmse(a, f):
    m = ~np.isnan(f)
    return np.sqrt(np.mean((f[m] - a[m])**2))

def mae(a, f):
    m = ~np.isnan(f)
    return np.mean(np.abs(f[m] - a[m]))

#  Taking the Diebold-Mariano test

def dm_test(a, f1, f2, loss="qlike"):
    m = ~np.isnan(f1) & ~np.isnan(f2) & (f1 > 0) & (f2 > 0)
    a_, f1_, f2_ = a[m], f1[m], f2[m]
    if loss == "qlike":
        L1 = a_/f1_ - np.log(a_/f1_) - 1
        L2 = a_/f2_ - np.log(a_/f2_) - 1
    else:
        L1 = (a_ - f1_)**2
        L2 = (a_ - f2_)**2
    d    = L1 - L2
    dbar = d.mean()
    T    = len(d)

    # Newey West variance with the lag = int(T^(1/3)) 
    
    lag  = int(T ** (1/3))
    gam0 = np.var(d, ddof=0)
    nw   = gam0
    for k in range(1, lag + 1):
        gk  = np.cov(d[k:], d[:-k], ddof=0)[0, 1]
        nw += 2 * (1 - k / (lag + 1)) * gk
    se   = np.sqrt(max(nw, 1e-12) / T)
    stat = dbar / se
    pval = 2 * (1 - stats.norm.cdf(abs(stat)))   # two-sided
    return stat, pval

# now taking the Mincer Zarnowitz

def mz_test(a, f):
    m = ~np.isnan(f) & (f > 0)
    a_, f_ = a[m], f[m]
    X  = np.column_stack([np.ones(len(f_)), f_])
    b  = np.linalg.lstsq(X, a_, rcond=None)[0]
    res = a_ - X @ b
    s2  = np.var(res, ddof=2)
    cov = s2 * np.linalg.inv(X.T @ X)
    
    # F-test: H0: alpha=0 and beta=1

    R  = np.array([[1, 0], [0, 1]])
    r  = np.array([0.0, 1.0])
    Rb_r = R @ b - r
    F  = (Rb_r @ np.linalg.inv(R @ cov @ R.T) @ Rb_r) / 2
    pf = 1 - stats.f.cdf(F, 2, len(a_) - 2)
    return float(b[0]), float(b[1]), float(pf)

# TABLE 4: main accuracy + DM vs HAR-RV

harv = forecasts["HAR-RV"]
rows = []
for name, fc in forecasts.items():
    m    = ~np.isnan(fc) & (fc > 0)
    r_   = rmse(actual, fc)
    m_   = mae(actual, fc)
    q_   = qlike(actual, fc)
    if name == "HAR-RV":
        dm_s, dm_p = np.nan, np.nan
    else:
        dm_s, dm_p = dm_test(actual, fc, harv)
    rows.append({"Model": name, "OOS_n": m.sum(), "RMSE": r_, "MAE": m_, "QLIKE": q_, "DM_stat": dm_s, "DM_pval": dm_p})

t4 = pd.DataFrame(rows).set_index("Model")
print("\n=== TABLE 4: Main OOS Accuracy ===")
print(t4.round(6).to_string())
t4.to_csv(os.path.join(RESDIR, "table4_main_accuracy.csv"))

# TABLE 5: regime split

high = actual > RV_THRESH
low  = ~high
print(f"\nHigh-vol days: {high.sum()}  Low-vol days: {low.sum()}")

rows5 = []
for name, fc in forecasts.items():
    for regime, mask_r, label in [(high, high, "high"), (low, low, "low")]:
        m = mask_r & ~np.isnan(fc) & (fc > 0)
        if m.sum() == 0:
            continue
        rows5.append({
            "Model"  : name,
            "Regime" : label,
            "n"      : m.sum(),
            "RMSE"   : rmse(actual[m], fc[m]),
            "QLIKE"  : qlike(actual[m], fc[m]),
        })

t5 = pd.DataFrame(rows5)
print("\n=== TABLE 5: Regime Split ===")
print(t5.round(6).to_string(index=False))
t5.to_csv(os.path.join(RESDIR, "table5_regime_split.csv"), index=False)

# TABLE 6: Mincer-Zarnowitz

rows6 = []
for name, fc in forecasts.items():
    alpha, beta, fp = mz_test(actual, fc)
    rows6.append({"Model": name, "alpha": alpha, "beta": beta, "F_pval": fp})

t6 = pd.DataFrame(rows6).set_index("Model")
print("\n=== TABLE 6: Mincer-Zarnowitz ===")
print(t6.round(6).to_string())
t6.to_csv(os.path.join(RESDIR, "table6_mz.csv"))

# FIGURE 6: OOS forecast comparison (2018-2022 window)

fig, ax = plt.subplots(figsize=(14, 5))
w = (dates >= "2018-01-01") & (dates <= "2022-12-31")
ax.plot(dates[w], actual[w], color="black", lw=1.2, label="Actual RV", zorder=5)

colors = {
    "GARCH(1,1)"    : "#e41a1c",
    "HAR-RV"        : "#377eb8",
    "XGBoost"       : "#4daf4a",
    "LSTM"          : "#984ea3",
    "TFT"           : "#ff7f00",
}
for name, color in colors.items():
    fc = forecasts[name]
    ax.plot(dates[w], fc[w], lw=0.8, alpha=0.75, color=color, label=name)

ax.axvspan(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-09-30"), alpha=0.12, color="grey", label="COVID window")
ax.set_title("OOS Volatility Forecasts — Selected Models (2018–2022)")
ax.set_ylabel("Annualised Realised Volatility")
ax.legend(fontsize=8, ncol=3)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "fig6_forecast_comparison.png"), dpi=300)
print("\nSaved fig6_forecast_comparison.png")

# FIGURE: regime split bar chart

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
model_names = list(forecasts.keys())

for ax, metric in zip(axes, ["RMSE", "QLIKE"]):
    hi_vals = [t5[(t5.Model == n) & (t5.Regime == "high")][metric].values[0]
               if len(t5[(t5.Model == n) & (t5.Regime == "high")]) > 0 else 0
               for n in model_names]
    lo_vals = [t5[(t5.Model == n) & (t5.Regime == "low")][metric].values[0]
               if len(t5[(t5.Model == n) & (t5.Regime == "low")]) > 0 else 0
               for n in model_names]
    x = np.arange(len(model_names))
    ax.bar(x - 0.2, hi_vals, 0.4, label="High vol", color="#e41a1c", alpha=0.8)
    ax.bar(x + 0.2, lo_vals, 0.4, label="Low vol",  color="#377eb8", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=35, ha="right", fontsize=8)
    ax.set_title(f"Regime Split — {metric}")
    ax.set_ylabel(metric)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "fig7_regime_split.png"), dpi=300)
print("Saved fig7_regime_split.png")

print("\nAll evaluation complete.")
print(f"Tables saved in {RESDIR}")
print(f"Figures saved in {FIGDIR}")