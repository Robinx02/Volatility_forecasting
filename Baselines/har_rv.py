import pandas as pd
import numpy as np
import os
import statsmodels.api as sm

DATA  = r"D:\MBFM_project\master_features.csv"
FCAST = r"D:\MBFM_project\results\forecasts"
os.makedirs(FCAST, exist_ok=True)

df    = pd.read_csv(DATA, index_col=0, parse_dates=True)
rv    = df["rv"].values
rv_d  = df["rv_lag1"].values
rv_w  = df["rv_5d"].values
rv_m  = df["rv_22d"].values

TRAIN = 1500
REFIT = 252
OOS   = len(df) - TRAIN

fcast   = np.full(OOS, np.nan)
res_har = None

for i, t in enumerate(range(TRAIN, len(df))):
    if i == 0 or i % REFIT == 0:
        start = 22
        y = rv[start:t]
        X = sm.add_constant(
            np.column_stack([rv_d[start:t], rv_w[start:t], rv_m[start:t]])
        )
        try:
            res_har = sm.OLS(y, X).fit()
        except Exception as e:
            print(f"  fit failed at i={i}: {e}")

    if res_har is not None:
        try:
            x_new    = np.array([[1, rv_d[t], rv_w[t], rv_m[t]]])
            fcast[i] = float(res_har.predict(x_new)[0])
        except Exception as e:
            print(f"  predict failed at i={i}: {e}")

fcast = np.clip(fcast, 1e-8, None)
out   = os.path.join(FCAST, "har_rv.npy")
np.save(out, fcast)

if res_har is not None:
    param_df = pd.DataFrame({
        "coef" : res_har.params,
        "tstat": res_har.tvalues,
        "pval" : res_har.pvalues
    }, index=["const", "rv_d", "rv_w", "rv_m"])
    print("\nHAR-RV last-fit parameters:")
    print(param_df.round(6).to_string())

actual = rv[TRAIN:]
mask   = ~np.isnan(fcast) & (fcast > 0)
a, f   = actual[mask], fcast[mask]
rmse   = np.sqrt(np.mean((f - a) ** 2))
mae    = np.mean(np.abs(f - a))
qlike  = np.mean(a / f - np.log(a / f) - 1)

print("\n" + "=" * 45)
print("HAR-RV")
print(f"  OOS n  : {mask.sum()}")
print(f"  RMSE   : {rmse:.6f}")
print(f"  MAE    : {mae:.6f}")
print(f"  QLIKE  : {qlike:.6f}")
print(f"  Saved  : {out}")
print("=" * 45)