import pandas as pd
import numpy as np
import os
from arch import arch_model

DATA  = r"D:\MBFM_project\master_features.csv"
FCAST = r"D:\MBFM_project\results\forecasts"
os.makedirs(FCAST, exist_ok=True)

df    = pd.read_csv(DATA, index_col=0, parse_dates=True)
rv    = df["rv"].values
ret   = df["ret_t"].values * 100
dates = df.index

TRAIN = 1500
REFIT = 252
OOS   = len(df) - TRAIN

fcast = np.full(OOS, np.nan)
res   = None

for i, t in enumerate(range(TRAIN, len(df))):
    if i == 0 or i % REFIT == 0:
        try:
            # o=1 adds the GJR asymmetry term
            res = arch_model(ret[:t], vol="Garch", p=1, o=1, q=1, dist="normal").fit(disp="off", show_warning=False)
        except Exception as e:
            print(f"  fit failed at i={i}: {e}")
    if res is not None:
        try:
            fc = res.forecast(horizon=1, reindex=False)
            var_fc = fc.variance.values[-1, 0]
            fcast[i] = np.sqrt(var_fc / 10000) * np.sqrt(252)
        except Exception:
            pass

out = os.path.join(FCAST, "gjr_garch.npy")
np.save(out, fcast)

actual = rv[TRAIN:]
mask   = ~np.isnan(fcast) & (fcast > 0)
a, f   = actual[mask], fcast[mask]
rmse   = np.sqrt(np.mean((f - a) ** 2))
mae    = np.mean(np.abs(f - a))
qlike  = np.mean(a / f - np.log(a / f) - 1)

print("=" * 45)
print("GJR-GARCH(1,1)")
print(f"  OOS n  : {mask.sum()}")
print(f"  RMSE   : {rmse:.6f}")
print(f"  MAE    : {mae:.6f}")
print(f"  QLIKE  : {qlike:.6f}")
print(f"  Saved  : {out}")
print("=" * 45)

if res is not None:
    param_df = pd.DataFrame({
        "coef" : res.params,
        "tstat": res.tvalues,
        "pval" : res.pvalues,
        "se"   : res.std_err,
    })
    param_df.to_csv(os.path.join(FCAST, "..", "gjr_garch_params.csv"))
    print("\nGJR-GARCH(1,1) parameters:")
    print(param_df.round(6).to_string())
    print(f"Log-likelihood: {res.loglikelihood:.4f}")
    print(f"AIC: {res.aic:.4f}  BIC: {res.bic:.4f}")