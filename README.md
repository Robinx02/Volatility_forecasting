# INR/USD Realized Volatility Forecasting

Comparing econometric baselines against machine learning models for forecasting INR/USD exchange rate volatility — part of a research project for a Money, Banking & Financial Markets course.

---

## What this project does

The core question is straightforward: can machine learning models forecast INR/USD realized volatility better than the HAR-RV model, which is the standard econometric benchmark for this kind of problem? And if ML does win, what economic channels are actually driving the improvement?

The target variable is 22-day rolling realized volatility of log USD/INR returns, annualized. All models are evaluated on a genuine out-of-sample walk-forward — no look-ahead, no fitting on the test period.

---

## Models

**Econometric baselines**
- GARCH(1,1), EGARCH(1,1), GJR-GARCH(1,1)
- HAR-RV — the primary benchmark (Corsi 2009)
- HAR-RV-X — HAR-RV augmented with VIX

**Machine learning**
- Random Forest
- XGBoost + SHAP interpretation
- LSTM (2-layer, sequence length 30)
- Temporal Fusion Transformer (Lim et al. 2021) — custom PyTorch implementation

---

## Results so far

| Model | RMSE | MAE | QLIKE |
|---|---|---|---|
| GARCH(1,1) | 0.033610 | 0.025026 | 0.154288 |
| EGARCH | 0.031356 | 0.023141 | 0.131604 |
| GJR-GARCH | 0.035170 | 0.026186 | 0.158456 |
| HAR-RV | 0.003470 | 0.001943 | 0.002243 |
| HAR-RV-X | 0.003484 | 0.001982 | 0.002262 |
| Random Forest | 0.009364 | 0.004084 | 0.010747 |
| XGBoost | 0.006806 | 0.002752 | 0.005316 |

QLIKE is the primary metric (standard for realized volatility evaluation). HAR-RV is the benchmark everything else is compared against.

---

## Data

All data is publicly available — no proprietary sources.

| Series | Source |
|---|---|
| USD/INR spot rate | RBI DBIE |
| India 10Y bond yield | Investing.com |
| Call money rate | RBI DBIE |
| Nifty 50 | NSE / Yahoo Finance |
| VIX | FRED (VIXCLS) |
| Brent crude | FRED (DCOILBRENTEU) |
| DXY | Yahoo Finance |
| US 10Y yield | FRED (DGS10) |
| EM proxy (EEM) | Yahoo Finance |
| MPC meeting dates | RBI website |
| FOMC meeting dates | Federal Reserve |

Sample period: April 2005 to October 2026 (~5,200 trading days).
Train/OOS split: first 1,500 rows for training, remaining ~3,700 for out-of-sample evaluation.

---

## Project structure

```
MBFM_project/
├── master_features.csv         cleaned feature matrix (5199 rows, 24 columns)
├── mbfm dataset/               raw downloaded files
├── figures/                    all paper figures
├── results/
│   ├── forecasts/              .npy forecast arrays per model
│   ├── shap/                   SHAP values and feature names
│   └── tft/                    TFT attention weights and variable selection weights
├── models/
│   ├── random_forest.py
│   ├── xgbs.py
│   └── lstm/
│       ├── lstm_model.py       architecture
│       ├── lstm_train.py       training loop
│       └── lstm_run.py          walk-forward evaluation
│   └── tft/
│       ├── tft_model.py        architecture
│       ├── tft_train.py        training loop
│       └── tft.py           walk-forward evaluation
├── EDA/
│   └── eda.py
└── preprocessing/
    └── data.ipynb
```

---

## Setup

```bash
conda create -n myenv python=3.13
conda activate myenv
pip install numpy pandas scikit-learn statsmodels arch xgboost shap torch torchvision tqdm matplotlib
```

GPU required for LSTM and TFT. Tested on RTX 4060 Laptop GPU (8GB VRAM).

---

## Running the models

Run from each model's directory so relative imports resolve correctly.

```bash
# baselines
cd models/Baselines
python garch11.py
python egarch.py
python gjr_garch.py
python har_rv.py
python har_rv_x.py

# ML models
cd models
python random_forest.py
python xgbs.py

# LSTM
cd models/lstm
python lstm_run.py

# TFT
cd models/tft
python tft.py
```

Each script saves a `.npy` forecast array to `results/forecasts/` and prints RMSE, MAE, and QLIKE on completion.

---

## Interpretation

Beyond just comparing forecast accuracy, the project uses two tools to understand what drives the results:

**SHAP (XGBoost)** — which features actually matter for the XGBoost forecasts, and when do they matter most. VIX and crude oil dominate during high volatility periods.

**TFT attention weights** — the temporal self-attention layer shows which past time steps the model focuses on, and the variable selection network gives a built in importance score for each feature without needing post-hoc explanation methods.

---

## References

- Corsi, F. (2009). A Simple Approximate Long-Memory Model of Realized Volatility. *Journal of Financial Econometrics*.
- Lim, B. et al. (2021). Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting. *arXiv:1912.09363*.
- Liu, X. et al. (2025). Forecasting realized volatility in the stock market: a path-dependent perspective. *arXiv:2503.00851*.
- Diebold, F. & Mariano, R. (1995). Comparing Predictive Accuracy. *Journal of Business & Economic Statistics*.
- Lundberg, S. & Lee, S. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS*.

---

## Status

- [x] Data collection and cleaning
- [x] EDA
- [x] GARCH family baselines
- [x] HAR-RV and HAR-RV-X
- [x] Random Forest
- [x] XGBoost + SHAP
- [x] LSTM
- [x] TFT

These below are yet to be done 

- [ ] Full evaluation (DM tests, regime split, Mincer-Zarnowitz)
- [ ] Interpretation figures
- [ ] Robustness checks
- [ ] Paper write-up
