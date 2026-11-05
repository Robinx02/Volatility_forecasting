import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SHAP_DIR = r"D:\MBFM_project\results\shap"
TFT_DIR  = r"D:\MBFM_project\results\tft"
FIGDIR   = r"D:\MBFM_project\figures"
DATA     = r"D:\MBFM_project\master_features.csv"
TRAIN    = 1500

shap_vals = np.load(rf"{SHAP_DIR}\shap_values.npy")
shap_names = np.load(rf"{SHAP_DIR}\shap_feature_names.npy", allow_pickle=True)
tft_varw = np.load(rf"{TFT_DIR}\variable_weights.npy")
tft_names = np.load(rf"{TFT_DIR}\obs_feature_names.npy",  allow_pickle=True)
tft_attn = np.load(rf"{TFT_DIR}\attention_weights.npy")

df = pd.read_csv(DATA, index_col=0, parse_dates=True)
rv_oos = df["rv"].values[TRAIN:]
oos_dates = df.index[TRAIN:]

shap_mean = np.abs(shap_vals).mean(axis=0)
tft_mean = tft_varw.mean(axis=0)

shap_norm = shap_mean / shap_mean.sum()
tft_norm = tft_mean  / tft_mean.sum()

common = [n for n in tft_names if n in shap_names]
shap_c = np.array([shap_norm[list(shap_names).index(n)] for n in common])
tft_c  = np.array([tft_norm [list(tft_names).index(n)]  for n in common])

order = np.argsort(shap_c)[::-1]
x = np.arange(len(common))

fig, ax = plt.subplots(figsize=(11, 5))
ax.bar(x - 0.2, shap_c[order], 0.4, label="XGBoost (SHAP)", color="#4daf4a", alpha=0.85)
ax.bar(x + 0.2, tft_c[order],  0.4, label="TFT (var-sel)",  color="#ff7f00", alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels([common[i] for i in order], rotation=40, ha="right", fontsize=8)
ax.set_ylabel("Normalised importance")
ax.set_title("Feature Importance: XGBoost SHAP vs TFT Variable Selection")
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(rf"{FIGDIR}\fig11_shap_vs_tft_varweights.png", dpi=300)
plt.close()
print("saved fig11_shap_vs_tft_varweights.png")
