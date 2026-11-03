import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import os

df = pd.read_csv(r"D:\MBFM_project\master_features.csv", index_col="date", parse_dates=True)

FEATURES = [
    "rv_lag1","rv_5d","rv_22d","sq_return",
    "vix_level","vix_chg","brent_ret",
    "dxy_level","dxy_ret",
    "india10y","india10y_chg",
    "us10y","us10y_chg","rate_spread",
    "callmoney","nifty_ret","em_ret",
    "is_mpc_day","is_fomc_day"
]
TARGET = "rv"
TRAIN_SIZE = 1500
REFIT_EVERY = 252

X = df[FEATURES].values
y = df[TARGET].values
dates = df.index

forecasts = np.full(len(y), np.nan)
steps_since_refit = 0
model = None

for t in range(TRAIN_SIZE, len(y)):
    if model is None or steps_since_refit >= REFIT_EVERY:
        model = xgb.XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            random_state=42, n_jobs=-1, verbosity=0
        )
        model.fit(X[:t], y[:t])
        steps_since_refit = 0
    forecasts[t] = model.predict(X[t:t+1])[0]
    steps_since_refit += 1

oos = ~np.isnan(forecasts)
a, f = y[oos], forecasts[oos]
rmse  = np.sqrt(np.mean((f - a)**2))
mae   = np.mean(np.abs(f - a))
qlike = np.mean(a/f - np.log(a/f) - 1)
print(f"{'XGBoost':<15} {oos.sum():>6}  {rmse:>10.6f}  {mae:>10.6f}  {qlike:>10.6f}")

np.save(r"D:\MBFM_project\results\forecasts\xgboost.npy", forecasts)

# ── SHAP ────────────────────────────────────────────────────────────────────
# Refit final model on full data for SHAP (representative of last window)
final_model = xgb.XGBRegressor(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
    random_state=42, n_jobs=-1, verbosity=0
)
final_model.fit(X[:TRAIN_SIZE + (((len(y) - TRAIN_SIZE) // REFIT_EVERY)) * REFIT_EVERY], 
                y[:TRAIN_SIZE + (((len(y) - TRAIN_SIZE) // REFIT_EVERY)) * REFIT_EVERY])

X_oos = X[oos]
explainer = shap.TreeExplainer(final_model)
shap_vals = explainer.shap_values(X_oos)

os.makedirs(r"D:\MBFM_project\results\shap", exist_ok=True)
np.save(r"D:\MBFM_project\results\shap\shap_values.npy", shap_vals)
np.save(r"D:\MBFM_project\results\shap\shap_feature_names.npy", np.array(FEATURES))

rv_oos   = y[oos]
rv_75    = np.percentile(rv_oos, 75)
hi_mask  = rv_oos > rv_75
lo_mask  = ~hi_mask
vix_idx  = FEATURES.index("vix_level")
oos_dates = dates[oos]

# Fig: mean |SHAP| bar
mean_abs = np.abs(shap_vals).mean(axis=0)
order = np.argsort(mean_abs)[::-1]
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh([FEATURES[i] for i in order[::-1]], mean_abs[order[::-1]])
ax.set_xlabel("Mean |SHAP|")
ax.set_title("XGBoost Feature Importance (SHAP)")
plt.tight_layout()
plt.savefig(r"D:\MBFM_project\figures\fig_shap_bar.png", dpi=150)
plt.close()

# Fig: VIX SHAP time series vs rv
fig, ax1 = plt.subplots(figsize=(12, 4))
ax2 = ax1.twinx()
ax1.plot(oos_dates, shap_vals[:, vix_idx], color="steelblue", lw=0.8, label="VIX SHAP")
ax2.plot(oos_dates, rv_oos, color="firebrick", lw=0.6, alpha=0.6, label="RV")
ax1.set_ylabel("SHAP(vix_level)"); ax2.set_ylabel("Realized Volatility")
ax1.set_title("VIX SHAP Value Over Time")
fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
plt.tight_layout()
plt.savefig(r"D:\MBFM_project\figures\fig_shap_vix_ts.png", dpi=150)
plt.close()

# Fig: VIX dependence plot
fig, ax = plt.subplots(figsize=(6, 5))
sc = ax.scatter(X_oos[:, vix_idx], shap_vals[:, vix_idx],
                c=rv_oos, cmap="RdYlGn_r", s=4, alpha=0.6)
plt.colorbar(sc, ax=ax, label="RV")
ax.set_xlabel("vix_level"); ax.set_ylabel("SHAP(vix_level)")
ax.set_title("VIX Dependence Plot")
plt.tight_layout()
plt.savefig(r"D:\MBFM_project\figures\fig_shap_vix_dep.png", dpi=150)
plt.close()

# Fig: SHAP regime comparison
mean_hi = np.abs(shap_vals[hi_mask]).mean(axis=0)
mean_lo = np.abs(shap_vals[lo_mask]).mean(axis=0)
order2 = np.argsort(mean_hi)[::-1]
x = np.arange(len(FEATURES))
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - 0.2, mean_hi[order2], 0.4, label="High-vol", color="firebrick")
ax.bar(x + 0.2, mean_lo[order2], 0.4, label="Low-vol",  color="steelblue")
ax.set_xticks(x)
ax.set_xticklabels([FEATURES[i] for i in order2], rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Mean |SHAP|"); ax.set_title("SHAP by Volatility Regime")
ax.legend()
plt.tight_layout()
plt.savefig(r"D:\MBFM_project\figures\fig_shap_regime.png", dpi=150)
plt.close()

print("SHAP figures saved.")