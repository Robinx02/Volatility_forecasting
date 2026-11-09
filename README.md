# Forecasting INR/USD Realized Volatility: Do Machine Learning Models Beat the HAR-RV Benchmark?

**Abhishek Raj · IISER Bhopal · Money, Banking & Financial Markets**

---

## Summary

This paper tests whether machine learning models can forecast INR/USD realized volatility more accurately than HAR-RV, the standard econometric benchmark. Using 5,199 daily observations from April 2005 to October 2026 and a strict expanding-window walk-forward design, nine models are compared across 3,699 out-of-sample forecasts.

**Main finding:** HAR-RV wins. Its QLIKE of 0.0022 is less than half that of the best ML model (XGBoost, 0.0053), and the gap holds across volatility regimes, after excluding COVID-19, and in both high- and low-volatility periods. Machine learning models become competitive only when the target is 5-day volatility or the horizon is five days.

---

## Results

| Model | RMSE | MAE | QLIKE | DM p-value vs HAR-RV |
|---|---|---|---|---|
| GARCH(1,1) | 0.03361 | 0.02503 | 0.15429 | <0.001 |
| EGARCH | 0.03136 | 0.02314 | 0.13160 | <0.001 |
| GJR-GARCH | 0.03517 | 0.02619 | 0.15846 | <0.001 |
| **HAR-RV** | **0.00347** | **0.00194** | **0.00224** | benchmark |
| HAR-RV-X | 0.00348 | 0.00198 | 0.00226 | 0.155 |
| Random Forest | 0.00936 | 0.00408 | 0.01075 | <0.001 |
| XGBoost | 0.00681 | 0.00275 | 0.00532 | 0.005 |
| LSTM | 0.00742 | 0.00491 | 0.01269 | <0.001 |
| TFT | 0.01754 | 0.00901 | 0.02782 | <0.001 |

QLIKE is the primary metric (standard for realized volatility evaluation). DM tests use Newey-West standard errors with lag 15, two-sided, null of equal predictive accuracy versus HAR-RV.

**Key findings:**
- HAR-RV dominates all models. XGBoost is the best ML model but still 2.4× worse on QLIKE
- HAR-RV-X is not significantly better than HAR-RV (DM p = 0.155) — VIX adds noise at the OLS level
- TFT is the only model that passes the Mincer-Zarnowitz calibration test (F p = 0.638), but this reflects absence of bias rather than precision
- SHAP attributes 92.7% of XGBoost forecasts to the three HAR lag components — the model re-learns HAR-RV with extra estimation variance
- VIX contributes modestly and mainly when above 28, primarily in COVID-19 (2020) and EM selloff (2018-19)
- Machine learning wins on 5-day RV target (R2) and approaches HAR-RV at 5-day horizon (R4)

---

## Models

**Econometric baselines**
- GARCH(1,1), EGARCH(1,1), GJR-GARCH(1,1) — fitted on percentage log returns, Python `arch` library
- HAR-RV — OLS on rv_lag1, rv_5d, rv_22d (Corsi 2009), primary benchmark
- HAR-RV-X — HAR-RV augmented with lagged VIX level

**Machine learning** — all use 19 lagged features, expanding-window walk-forward, refit every 252 days
- Random Forest — 200 trees, max_features=sqrt, min_samples_leaf=5
- XGBoost — 300 trees, depth 4, lr 0.05 + TreeSHAP interpretation
- LSTM — 2 layers × 64 hidden units, dropout 0.2, sequence length 30, Adam lr=1e-3
- TFT — custom PyTorch, d=64, 1 attention head, encoder length 63, QuantileLoss(0.1/0.5/0.9)

---

## Data

Sample: April 2005 – October 2026 (5,199 RBI trading days). Zero nulls.

| Series | Source |
|---|---|
| USD/INR spot rate | RBI DBIE (official reference rate) |
| India 10Y bond yield | Investing.com |
| Call money rate | RBI DBIE (weighted average) |
| Nifty 50 | NSE / Yahoo Finance (^NSEI) |
| VIX | FRED — VIXCLS |
| Brent crude | FRED — DCOILBRENTEU |
| DXY | Yahoo Finance — DX-Y.NYB |
| US 10Y yield | FRED — DGS10 |
| EM proxy (EEM) | Yahoo Finance |
| RBI MPC dates | rbi.org.in (60 meetings, Oct 2016–2026) |
| FOMC dates | federalreserve.gov (175 meetings, 2005–2026) |

Target: 22-day rolling std of log USD/INR returns × √252 (annualised). Train: first 1,500 rows. OOS: 3,699 rows (2011–2026).

---

## Robustness

| Variant | Description | HAR-RV QLIKE | XGBoost QLIKE | LSTM QLIKE |
|---|---|---|---|---|
| Baseline | — | 0.00224 | 0.00532 | 0.01269 |
| R1: no COVID | Exclude 2020-01 to 2020-09 from OOS | 0.00218 | 0.00534 | 0.01273 |
| R2: 5-day RV | Replace target with 5-day RV | 0.1161 | 0.0988 | **0.0868** |
| R3: market only | Drop india10y, us10y, rate_spread, callmoney | — | 0.0080 | 0.0111 |
| R4: 5-day ahead | Forecast horizon = 5 days | 0.02118 | 0.02145 | 0.02551 |

Rankings stable under R1. HAR-RV advantage collapses under R2 (short-memory target favours ML). XGBoost nearly matches HAR-RV under R4.

---

## Project Structure

```
MBFM_project/
├── master_features.csv cleaned feature matrix (5199 × 24)
├── mbfm dataset/ raw downloaded CSVs
├── figures/ all paper figures (fig1–fig14)
├── results/
│ ├── forecasts/ .npy forecast arrays (9 models)
│ ├── shap/ shap_values.npy, shap_feature_names.npy
│ ├── tft/ attention_weights.npy, variable_weights.npy
│ ├── robustness/ r1–r4 CSVs, table7_robustness.csv
│ ├── table4_main_accuracy.csv
│ ├── table5_regime_split.csv
│ └── table6_mz.csv
├── models/
│ ├── Baselines/ garch11.py, egarch.py, gjr_garch.py,
│ │ har_rv.py, har_rv_x.py
│ ├── random_forest.py
│ ├── xgbs.py
│ ├── lstm/ lstm_model.py, lstm_train.py, lstm_run.py
│ └── temporal_fusion_transformer/ tft_model.py, tft_train.py, tft.py
├── robustness/ r1–r4 scripts, summarize.py
├── EDA/ eda.py
├── evaluation.py
└── preprocessing/ data.ipynb, diagnostics.ipynb
```


---

## Setup

```bash
conda create -n myenv python=3.13
conda activate myenv
pip install numpy pandas scikit-learn statsmodels arch xgboost shap torch torchvision tqdm matplotlib scipy
```

GPU required for LSTM and TFT. Tested on RTX 4060 Laptop GPU (8GB VRAM), Windows 11.

---

## Replication

```bash
# baselines
cd models/Baselines
python garch11.py && python egarch.py && python gjr_garch.py && python har_rv.py && python har_rv_x.py

# ML models
cd models
python random_forest.py && python xgbs.py

# LSTM
cd models/lstm && python lstm_run.py

# TFT
cd models/temporal_fusion_transformer && python tft.py

# evaluation (Tables 4–6, Figures 6–7)
cd D:/MBFM_project && python evaluation.py

# robustness (Table 7)
cd robustness
python no_covid.py && python 5day_rv.py && python market_only.py && python 5day_ahead.py && python summarize.py
```

Each model script saves a `.npy` forecast array to `results/forecasts/` and prints RMSE, MAE, QLIKE on completion. LSTM and TFT require CUDA.

---

## References

- Corsi, F. (2009). A simple approximate long-memory model of realized volatility. *Journal of Financial Econometrics*, 7(2), 174–196.
- Lim, B. et al. (2021). Temporal Fusion Transformers for interpretable multi-horizon time series forecasting. *International Journal of Forecasting*, 37(4), 1748–1764.
- Liu, X., Fu, S. and Hong, S. (2025). Forecasting realized volatility in the stock market: a path-dependent perspective. *arXiv:2503.00851*.
- Diebold, F. and Mariano, R. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253–263.
- Lundberg, S. and Lee, S. (2017). A unified approach to interpreting model predictions. *NeurIPS*.
- Patton, A. (2011). Volatility forecast comparison using imperfect volatility proxies. *Journal of Econometrics*, 160(1), 246–256.
- Andersen, T. and Bollerslev, T. (1998). Answering the skeptics. *International Economic Review*, 39(4), 885–905.
