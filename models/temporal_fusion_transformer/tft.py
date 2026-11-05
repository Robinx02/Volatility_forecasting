import os
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tft_train import make_sequences, train, OBS_FEATURES, KNOWN_FEATURES

DATA   = r"D:\MBFM_project\master_features.csv"
FCAST  = r"D:\MBFM_project\results\forecasts"
TFTDIR = r"D:\MBFM_project\results\tft"
os.makedirs(FCAST,  exist_ok=True)
os.makedirs(TFTDIR, exist_ok=True)

TRAIN   = 1500
REFIT   = 252
ENC_LEN = 63
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if DEVICE.type != "cuda":
    print("GPU not detected. Exiting.")
    exit()

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

df      = pd.read_csv(DATA, index_col=0, parse_dates=True)
X_obs   = df[OBS_FEATURES].values.astype(np.float32)
X_known = df[KNOWN_FEATURES].values.astype(np.float32)
y_all   = df["rv"].values.astype(np.float32)
OOS     = len(df) - TRAIN

#rv floor same fix so that the QLIKE doesnt blowup
rv_floor =float(np.percentile(y_all[:TRAIN] ,1))
print(f"rv_floor (1st pct train RV) : {rv_floor:.6f}")

print(f"device={DEVICE}  obs={len(OBS_FEATURES)}  known={len(KNOWN_FEATURES)}  OOS={OOS}")

fcast        = np.full(OOS, np.nan)
attn_store   = []
varw_store   = []
model        = None
scaler_obs   = None
scaler_known = None

for i, t in enumerate(range(TRAIN, len(df))):
    if i == 0 or i % REFIT == 0:
        scaler_obs   = StandardScaler()
        scaler_known = StandardScaler()

        Xo_sc = scaler_obs.fit_transform(X_obs[:t])
        Xk_sc = scaler_known.fit_transform(X_known[:t])

        Xo_seq, Xk_seq, y_seq = make_sequences(Xo_sc, Xk_sc, y_all[:t], ENC_LEN)

        if len(y_seq) < 100:
            continue

        print(f"refit  i={i:4d}  train_seqs={len(y_seq)}")
        model = train(Xo_seq, Xk_seq, y_seq, DEVICE, ENC_LEN)

    if model is None:
        continue

    # build input: encoder = [t-ENC_LEN : t], decoder step = [t]
    xo_in = scaler_obs.transform(X_obs[t - ENC_LEN : t])
    xk_in = scaler_known.transform(X_known[t - ENC_LEN : t + 1])

    xo_t  = torch.from_numpy(xo_in[np.newaxis].astype(np.float32)).to(DEVICE)
    xk_t  = torch.from_numpy(xk_in[np.newaxis].astype(np.float32)).to(DEVICE)

    with torch.no_grad():
        p50, _, _, attn, var_w = model(xo_t, xk_t)

    fcast[i] = max(p50.item(), 1e-8)

    # store last attention row (forecast position) and variable weights
    attn_store.append(attn[0, -1].cpu().numpy())    # (ENC_LEN+1,)
    varw_store.append(var_w[0].cpu().numpy())        # (n_obs,)

out = os.path.join(FCAST, "tft.npy")
np.save(out, fcast)

np.save(os.path.join(TFTDIR, "attention_weights.npy"), np.array(attn_store, dtype=np.float32))
np.save(os.path.join(TFTDIR, "variable_weights.npy"), np.array(varw_store, dtype=np.float32))
np.save(os.path.join(TFTDIR, "obs_feature_names.npy"), np.array(OBS_FEATURES))

actual = y_all[TRAIN:]
mask   = ~np.isnan(fcast) & (fcast > 0)
a, f_  = actual[mask], fcast[mask]
rmse   = np.sqrt(np.mean((f_ - a) ** 2))
mae    = np.mean(np.abs(f_ - a))
qlike  = np.mean(a / f_ - np.log(a / f_) - 1)

print(f"\nTFT")
print(f"  device : {DEVICE}")
print(f"  OOS n  : {mask.sum()}")
print(f"  RMSE   : {rmse:.6f}")
print(f"  MAE    : {mae:.6f}")
print(f"  QLIKE  : {qlike:.6f}")
print(f"  saved  : {out}")

# taking out the figure 

oos_dates = df.index[TRAIN:]

# figure 1 is the forecast vs actual and error
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
axes[0].plot(oos_dates, actual, color="black", lw=0.8, label="actual rv")
axes[0].plot(oos_dates, fcast,  color="steelblue", lw=0.7, alpha=0.8, label="tft p50")
axes[0].set_title("TFT OOS Forecast vs Actual")
axes[0].set_ylabel("Annualised RV")
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.3)
axes[1].plot(oos_dates, np.abs(fcast - actual), color="firebrick", lw=0.6)
axes[1].set_title("Absolute Forecast Error")
axes[1].set_ylabel("|forecast - actual|")
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(r"D:\MBFM_project\figures\fig_tft_forecast.png", dpi=150)
plt.close()
print("saved fig_tft_forecast.png")

# figure 2 is the mean temporal attention vs lag
attn_arr  = np.array(attn_store, dtype=np.float32)   # (OOS, T_total)
T_attn    = attn_arr.shape[1]
mean_attn = attn_arr.mean(axis=0)
p10_attn  = np.percentile(attn_arr, 10, axis=0)
p90_attn  = np.percentile(attn_arr, 90, axis=0)
lags      = np.arange(-T_attn + 1, 1) 


fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(lags, mean_attn, color="steelblue", lw=1.5, label="mean attention")
ax.fill_between(lags, p10_attn, p90_attn, alpha=0.2, color="steelblue", label="10–90 pct")
ax.set_xlabel("Lag (days)")
ax.set_ylabel("Attention weight")
ax.set_title("TFT Mean Temporal Attention")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(r"D:\MBFM_project\figures\fig_tft_attention.png", dpi=150)
plt.close()
print("saved fig_tft_attention.png")


#figure 3 is the mean variable selection weights bar

varw_arr  = np.array(varw_store, dtype=np.float32)  # (OOS, n_obs)
mean_varw = varw_arr.mean(axis=0)
order = np.argsort(mean_varw)[::-1]

fig, ax = plt.subplots(figsize=(8, 5))
ax.barh([OBS_FEATURES[i] for i in order[::-1]], mean_varw[order[::-1]], color="steelblue")
ax.set_xlabel("Mean variable selection weight")
ax.set_title("TFT Variable Selection Weights")
plt.tight_layout()
plt.savefig(r"D:\MBFM_project\figures\fig_tft_varweights.png", dpi=150)
plt.close()
print("saved fig_tft_varweights.png")