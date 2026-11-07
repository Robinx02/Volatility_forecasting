import os, sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import torch

sys.path.insert(0, r"D:\MBFM_project\models\lstm")
sys.path.insert(0, r"D:\MBFM_project\models\temporal_fusion_transformer")
from lstm_train import make_sequences as lstm_seq, train as lstm_train
from tft_train  import make_sequences as tft_seq,  train as tft_train, KNOWN_FEATURES

DATA   = r"D:\MBFM_project\master_features.csv"
RESDIR = r"D:\MBFM_project\results\robustness"
os.makedirs(RESDIR, exist_ok=True)

TRAIN   = 1500
REFIT   = 252
HORIZON = 5
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ML_FEATURES = [ "rv_lag1","rv_5d","rv_22d","sq_return", "vix_level","vix_chg","brent_ret", "dxy_level","dxy_ret", "india10y","india10y_chg", "us10y","us10y_chg","rate_spread", "callmoney","nifty_ret","em_ret", "is_mpc_day","is_fomc_day",]

df = pd.read_csv(DATA, index_col=0, parse_dates=True)
y = df["rv"].values.astype(np.float32)
X_ml = df[ML_FEATURES].values.astype(np.float32)
X_kn = df[KNOWN_FEATURES].values.astype(np.float32)
rv_floor = float(np.percentile(y[:TRAIN], 1))
actual = y[TRAIN + HORIZON:]
N = len(y) - TRAIN - HORIZON

def qlike(a, f):
    m = ~np.isnan(f) & (f > 0)
    return np.mean(a[m]/f[m] - np.log(a[m]/f[m]) - 1)

def rmse(a, f):
    m = ~np.isnan(f)
    return np.sqrt(np.mean((f[m] - a[m])**2))

results = {}

print("running HAR-RV")
from statsmodels.tools import add_constant

fcast = np.full(N, np.nan)
b = None; wait = 0
for i, t in enumerate(range(TRAIN, len(y) - HORIZON)):
    if b is None or wait >= REFIT:
        X_ = add_constant(df[["rv_lag1","rv_5d","rv_22d"]].iloc[:t].values)
        b  = np.linalg.lstsq(X_, y[:t], rcond=None)[0]
        wait = 0
    xf       = np.array([1, df["rv_lag1"].iloc[t], df["rv_5d"].iloc[t], df["rv_22d"].iloc[t]])
    fcast[i] = max(float(xf @ b), 1e-8)
    wait += 1
results["HAR-RV"] = fcast

print("running Random Forest")
fcast = np.full(N, np.nan)
model = None; wait = 0
for i, t in enumerate(range(TRAIN, len(y) - HORIZON)):
    if model is None or wait >= REFIT:
        model = RandomForestRegressor(n_estimators=200, max_features="sqrt", min_samples_leaf=5, random_state=42, n_jobs=-1)
        model.fit(X_ml[:t], y[:t]); wait = 0
    fcast[i] = max(model.predict(X_ml[t:t+1])[0], rv_floor)
    wait += 1
results["Random Forest"] = fcast

print("running XGBoost")
fcast = np.full(N, np.nan)
model = None; wait = 0
for i, t in enumerate(range(TRAIN, len(y) - HORIZON)):
    if model is None or wait >= REFIT:
        model = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=5, random_state=42, n_jobs=-1, verbosity=0)
        model.fit(X_ml[:t], y[:t]); wait = 0
    fcast[i] = max(model.predict(X_ml[t:t+1])[0], rv_floor)
    wait += 1
results["XGBoost"] = fcast

print("running LSTM")
fcast  = np.full(N, np.nan)
model  = None; scaler = None; wait = 0; SEQ = 30
for i, t in enumerate(range(TRAIN, len(y) - HORIZON)):
    if model is None or wait >= REFIT:
        scaler = StandardScaler()
        Xsc    = scaler.fit_transform(X_ml[:t])
        Xs, ys = lstm_seq(Xsc, y[:t], SEQ)
        if len(Xs) < 100:
            wait += 1; continue
        print(f"  lstm refit i={i}  seqs={len(Xs)}")
        model = lstm_train(Xs, ys, device=DEVICE); wait = 0
    x_in = torch.from_numpy(
        scaler.transform(X_ml[t-SEQ:t])[np.newaxis].astype(np.float32)
    ).to(DEVICE)
    with torch.no_grad():
        fcast[i] = max(model(x_in).item(), rv_floor)
    wait += 1
results["LSTM"] = np.maximum(fcast, rv_floor)

print("running TFT")
fcast  = np.full(N, np.nan)
model  = None; so = None; sk = None; wait = 0; ENC = 63
for i, t in enumerate(range(TRAIN, len(y) - HORIZON)):
    if model is None or wait >= REFIT:
        so = StandardScaler(); sk = StandardScaler()
        Xo = so.fit_transform(X_ml[:t])
        Xk = sk.fit_transform(X_kn[:t])
        Xos, Xks, ys = tft_seq(Xo, Xk, y[:t], ENC)
        if len(ys) < 100:
            wait += 1; continue
        print(f"  tft refit i={i}  seqs={len(ys)}")
        model = tft_train(Xos, Xks, ys, DEVICE, ENC); wait = 0
    xo = torch.from_numpy(so.transform(X_ml[t-ENC:t])[np.newaxis].astype(np.float32)).to(DEVICE)
    xk = torch.from_numpy(sk.transform(X_kn[t-ENC:t+1])[np.newaxis].astype(np.float32)).to(DEVICE)
    with torch.no_grad():
        p50, *_ = model(xo, xk)
    fcast[i] = max(p50.item(), rv_floor)
    wait += 1
results["TFT"] = np.maximum(fcast, rv_floor)

rows = []
for name, fc in results.items():
    rows.append({"Model": name, "RMSE": rmse(actual, fc), "QLIKE": qlike(actual, fc)})

t = pd.DataFrame(rows).set_index("Model")
print("\n=== R4: 5-day-ahead horizon ===")
print(t.round(6).to_string())
t.to_csv(os.path.join(RESDIR, "r4_5day_ahead.csv"))
print("saved r4_5day_ahead.csv")