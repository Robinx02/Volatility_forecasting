import pandas as pd
import numpy as np
import os
from arch import arch_model

DATA  = os.environ.get("MBFM_DATA",  r"D:\MBFM_project\master_features.csv")
FCAST = os.environ.get("MBFM_FCAST", r"D:\MBFM_project\results\forecasts")
os.makedirs(FCAST, exist_ok=True)

df    = pd.read_csv(DATA, index_col=0, parse_dates=True)
rv    = df["rv"].values
ret   = df["ret_t"].values * 100         

TRAIN = 1500
REFIT = 252
OOS   = len(df) - TRAIN

NAME = "GARCH(1,1)"
KEY  = "garch11"
SPEC = dict(vol="Garch", p=1, q=1)                       # arch_model settings for this variant

full_model = arch_model(ret, dist="normal", **SPEC)

fcast = np.full(OOS, np.nan)
res = None
cond_vol = None

for i, t in enumerate(range(TRAIN, len(df))):
    if i == 0 or i % REFIT == 0:
        try:
            res = arch_model(ret[:t], dist="normal", **SPEC).fit(disp="off", show_warning=False)
            cond_vol = np.asarray(full_model.fix(res.params).conditional_volatility)
            if i == 0:
                var_old = res.forecast(horizon=1, reindex=False).variance.values[-1, 0]
                ok = np.isclose(cond_vol[t] ** 2, var_old, rtol=1e-4)
                print(f"indexing check (filtered sigma_t == 1-step forecast at first OOS day): {'OK' if ok else 'MISMATCH - tell Claude'}")
        except Exception as e:
            print(f"  fit failed at i={i}: {e}")
    if cond_vol is not None:
        var_fc   = cond_vol[t] ** 2                       # (pct return)^2
        fcast[i] = np.sqrt(var_fc / 10000) * np.sqrt(252) # annualised std, same scale as rv

n_distinct = len(np.unique(np.round(fcast[~np.isnan(fcast)], 10)))
print(f"  distinct forecast values: {n_distinct}  (old fixed-forecast version: 15; should be about {OOS})")

t_chk = TRAIN + 100
w_start = TRAIN + ((t_chk - TRAIN) // REFIT) * REFIT
r_w = arch_model(ret[:w_start], dist="normal", **SPEC).fit(disp="off", show_warning=False)
ret_alt = ret.copy(); ret_alt[t_chk:] = 0.0
cv_a = np.asarray(arch_model(ret,     dist="normal", **SPEC).fix(r_w.params).conditional_volatility)
cv_b = np.asarray(arch_model(ret_alt, dist="normal", **SPEC).fix(r_w.params).conditional_volatility)
print(f"  no-look-ahead check: {'OK' if np.isclose(cv_a[t_chk], cv_b[t_chk], rtol=1e-8) else 'LOOK-AHEAD DETECTED - tell Claude'}")

out = os.path.join(FCAST, KEY + ".npy")
np.save(out, fcast)

actual = rv[TRAIN:]
mask   = ~np.isnan(fcast) & (fcast > 0)
a, f   = actual[mask], fcast[mask]
rmse   = np.sqrt(np.mean((f - a) ** 2))
mae    = np.mean(np.abs(f - a))
qlike  = np.mean(a / f - np.log(a / f) - 1)

print("=" * 45)
print(NAME)
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
    param_df.to_csv(os.path.join(FCAST, "..", KEY + "_params.csv"))
    print("\n" + NAME + " parameters (last window):")
    print(param_df.round(6).to_string())
    print(f"Log-likelihood: {res.loglikelihood:.4f}")
    print(f"AIC: {res.aic:.4f}  BIC: {res.bic:.4f}")
