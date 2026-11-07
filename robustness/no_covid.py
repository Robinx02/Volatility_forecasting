import os
import numpy as np
import pandas as pd

DATA   = r"D:\MBFM_project\master_features.csv"
FCAST  = r"D:\MBFM_project\results\forecasts"
RESDIR = r"D:\MBFM_project\results\robustness"
os.makedirs(RESDIR, exist_ok=True)

TRAIN = 1500
MODELS = {
    "GARCH(1,1)" : "garch11.npy",
    "EGARCH" : "egarch.npy",
    "GJR-GARCH" : "gjr_garch.npy",
    "HAR-RV" : "har_rv.npy",
    "HAR-RV-X" : "har_rv_x.npy",
    "Random Forest": "random_forest.npy",
    "XGBoost" : "xgboost.npy",
    "LSTM" : "lstm.npy",
    "TFT" : "tft.npy",
}

df = pd.read_csv(DATA, index_col=0, parse_dates=True)
actual = df["rv"].values[TRAIN:]
dates_oos = df.index[TRAIN:]
mask = ~((dates_oos >= "2020-01-01") & (dates_oos <= "2020-09-30"))

def qlike(a, f):
    m = ~np.isnan(f) & (f > 0)
    return np.mean(a[m]/f[m] - np.log(a[m]/f[m]) - 1)

def rmse(a, f):
    m = ~np.isnan(f)
    return np.sqrt(np.mean((f[m] - a[m])**2))

rows = []
for name, fname in MODELS.items():
    fc = np.load(os.path.join(FCAST, fname))
    if len(fc) == len(df):
        fc = fc[TRAIN:]
    fc = np.where((fc > 0) & ~np.isnan(fc), fc, np.nan)
    rows.append({
        "Model": name,
        "n"    : mask.sum(),
        "RMSE" : rmse(actual[mask], fc[mask]),
        "QLIKE": qlike(actual[mask], fc[mask]),
    })

t = pd.DataFrame(rows).set_index("Model")
print("\n=== R1: No COVID (2020-01 to 2020-09 excluded) ===")
print(t.round(6).to_string())
t.to_csv(os.path.join(RESDIR, "r1_no_covid.csv"))
print("saved r1_no_covid.csv")