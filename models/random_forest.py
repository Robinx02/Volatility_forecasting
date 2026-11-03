import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

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

forecasts = np.full(len(y), np.nan)

train_end = TRAIN_SIZE
steps_since_refit = 0
model = None

for t in range(TRAIN_SIZE, len(y)):
    if model is None or steps_since_refit >= REFIT_EVERY:
        model = RandomForestRegressor(
            n_estimators=200, max_features="sqrt",
            min_samples_leaf=5, random_state=42, n_jobs=-1
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
print(f"{'Random Forest':<15} {oos.sum():>6}  {rmse:>10.6f}  {mae:>10.6f}  {qlike:>10.6f}")

np.save(r"D:\MBFM_project\results\forecasts\random_forest.npy", forecasts)