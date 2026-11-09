import os
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lstm_train import make_sequences, train

DATA  = r"D:\MBFM_project\master_features.csv"
FCAST = r"D:\MBFM_project\results\forecasts"
os.makedirs(FCAST, exist_ok=True)

TRAIN   = 1500
REFIT   = 252
SEQ_LEN = 30
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if DEVICE.type != "cuda":
    print("GPU not detected. Exiting.")
    exit()

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

FEATURES = ["rv_lag1", "rv_5d", "rv_22d", "sq_return", "vix_level", "vix_chg", "brent_ret", "dxy_level", "dxy_ret", "india10y", "india10y_chg", "us10y", "us10y_chg", "rate_spread", "callmoney", "nifty_ret", "em_ret", "is_mpc_day", "is_fomc_day",]

df    = pd.read_csv(DATA, index_col=0, parse_dates=True)
X_all = df[FEATURES].values.astype(np.float32)
y_all = df["rv"].values.astype(np.float32)
OOS   = len(df) - TRAIN

rv_floor = float(np.percentile(y_all[:TRAIN] ,1))
print(f"rv_floor (1st pct train RV) : {rv_floor:.6f}")
print(f"device={DEVICE}  features={len(FEATURES)}  OOS={OOS}")

fcast  = np.full(OOS, np.nan)
model  = None
scaler = None

for i, t in enumerate(range(TRAIN, len(df))):
    if i == 0 or i % REFIT == 0:
        scaler        = StandardScaler()
        X_scaled      = scaler.fit_transform(X_all[:t])
        X_seq, y_seq  = make_sequences(X_scaled, y_all[:t], SEQ_LEN)

        if len(X_seq) < 100:
            continue

        print(f"refit  i={i:4d}  train_seqs={len(X_seq)}")
        model = train(X_seq, y_seq, device =DEVICE)

    if model is None:
        continue

    x_in = torch.from_numpy(
        scaler.transform(X_all[t - SEQ_LEN : t])[np.newaxis].astype(np.float32)
    ).to(DEVICE)

    with torch.no_grad():
        fcast[i] = max(model(x_in).item(), 1e-8)

#applying the floor to any reaming near-zero values so that it doesnt blow up QLIKE score againnnnn!!!!!!!!!!!!!

fcast = np.maximum(fcast , rv_floor)

out = os.path.join(FCAST, "lstm.npy")
np.save(out, fcast)

actual = y_all[TRAIN:]
mask   = ~np.isnan(fcast) & (fcast > 0)
a, f_  = actual[mask], fcast[mask]
rmse   = np.sqrt(np.mean((f_ - a) ** 2))
mae    = np.mean(np.abs(f_ - a))
qlike  = np.mean(a / f_ - np.log(a / f_) - 1)

print(f"\nLSTM")
print(f"  device : {DEVICE}")
print(f"  OOS n  : {mask.sum()}")
print(f"  RMSE   : {rmse:.6f}")
print(f"  MAE    : {mae:.6f}")
print(f"  QLIKE  : {qlike:.6f}")
print(f"  saved  : {out}")

fig , axes = plt.subplots(1,2 , figsize = (14,4))

oos_dates = df.index[TRAIN:]
axes[0].plot(oos_dates , actual , color = "black" , lw=0.8 , label = "actual rv")
axes[0].plot(oos_dates , fcast , color = "red" , lw= 0.7 , alpha = 0.8 , label = "lstm forecast")
axes[0].set_title("LSTM OOS Forecast vs Actual")
axes[0].set_ylabel("Annualised RV")
axes[0].legend(fontsize = 8)
axes[0].grid(alpha = 0.3)

axes[1].plot(oos_dates , np.abs(fcast-actual) , color = "steelblue" , lw = 0.6)
axes[1].set_title("Absolute Forecast Error")
axes[1].set_ylabel("|forecast - actual")
axes[1].grid(alpha=0.3)

#this is figure 16

plt.tight_layout()
out_fig = r"D:\MBFM_project\figures\fig_lstm_forecast.png"
plt.savefig(out_fig , dpi = 150)
print(f"saved {out_fig}")
