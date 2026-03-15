# 💓 Health Metrics Time Series Forecasting

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

End-to-end ML pipeline that forecasts daily step counts (and other health metrics) using four modelling approaches — from classical statistics to deep learning. Includes a Streamlit dashboard for interactive exploration.

---

## 🎯 Problem Statement

Wearable devices generate rich daily health time series (steps, heart rate, sleep, etc.). This project tackles **next-day step count forecasting**, a representative supervised regression task on temporal health data.

**Why it's interesting:**
- Strong weekly and seasonal patterns require careful feature engineering
- Multiple correlated signals (HR, sleep, calories) can improve predictions
- Comparing classical vs ML vs DL approaches on the same dataset reveals the accuracy/complexity trade-off

---

## 🏗️ Architecture

```
health-timeseries-forecasting/
├── data/
│   ├── generate_data.py        # Synthetic data generator (realistic patterns)
│   └── raw/                    # Place your own Fitbit/Apple Health CSV here
├── src/
│   ├── data_loader.py          # Load, clean, feature-engineer, split
│   ├── evaluate.py             # MAE, RMSE, MAPE, sMAPE, R² + comparison table
│   ├── train.py                # Orchestration script — runs all models
│   └── models/
│       ├── arima_model.py      # AutoARIMA with walk-forward evaluation
│       ├── prophet_model.py    # Facebook Prophet with seasonality tuning
│       ├── xgboost_model.py    # XGBoost + LightGBM with lag/rolling features
│       └── lstm_model.py       # Stacked LSTM with early stopping (PyTorch)
├── dashboard/
│   └── app.py                  # Streamlit dashboard (EDA + results + forecast)
├── results/
│   ├── metrics.json            # Saved model comparison
│   └── predictions.csv         # Test-set predictions per model
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/health-timeseries-forecasting.git
cd health-timeseries-forecasting
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Generate data

```bash
python data/generate_data.py
```

> 🔁 **Or use your own data**: Export from Apple Health / Fitbit and place a CSV with columns `date, steps, heart_rate, sleep_hours, calories` at `data/raw/health_data.csv`.

### 3. Train all models

```bash
# Run the full pipeline (XGB, LightGBM, Prophet, LSTM)
python src/train.py

# Run specific models only
python src/train.py --models xgb lgbm

# Forecast a different target
python src/train.py --models xgb prophet --target heart_rate
```

### 4. Launch the dashboard

```bash
streamlit run dashboard/app.py
```

Open [http://localhost:8501](http://localhost:8501)

---

## 🧠 Models

| Model | Approach | Strengths |
|---|---|---|
| **AutoARIMA** | Statistical, univariate | Interpretable; baseline for comparison |
| **Prophet** | Additive decomposition | Handles holidays, missing data, trend changes |
| **XGBoost** | Gradient boosting + lag features | Fast, accurate, built-in feature importance |
| **LightGBM** | Gradient boosting | Even faster than XGB; leaf-wise growth |
| **LSTM** | Deep learning, sequence modelling | Learns long-range temporal dependencies |

### Naive baseline
Predict tomorrow = same day last week (lag-7). All models should beat this.

---

## 📐 Feature Engineering

For tree-based models, raw time series are converted to a **tabular lag-feature matrix**:

```
lag-1, lag-2, lag-3, lag-7, lag-14          ← autoregressive features
roll7_mean, roll7_std, roll14_mean, ewm7     ← rolling statistics
heart_rate, sleep_hours, calories            ← correlated health signals
dow_sin, dow_cos, month_sin, month_cos       ← cyclical calendar encoding
is_weekend                                   ← binary flag
```

All features are constructed from the **past only** — no data leakage.

---

## 📊 Evaluation

Models are evaluated on a held-out **chronological test set** (last 15% of data). Metrics:

- **MAE** — Mean Absolute Error (steps)
- **RMSE** — Root Mean Squared Error (penalises large errors)
- **MAPE** — Mean Absolute Percentage Error
- **sMAPE** — Symmetric MAPE (robust to near-zero values)
- **R²** — Coefficient of determination

> ⚠️ **Never shuffle time series data** for train/test splitting — this is a common mistake that inflates performance estimates.

---

## 📸 Dashboard Preview

The Streamlit app has three sections:

1. **📊 EDA** — time series charts, weekly heatmaps, correlation matrix, distributions
2. **🤖 Model Results** — metric table with best-model highlighting, prediction overlays
3. **🔮 Forecast** — interactive n-day forecast with adjustable horizon

---

## 🔌 Using Your Own Data

The pipeline expects a CSV at `data/raw/health_data.csv` with at minimum:

| Column | Type | Description |
|---|---|---|
| `date` | YYYY-MM-DD | Daily date |
| `steps` | int | Daily step count |
| `heart_rate` | float | Resting heart rate (bpm) |
| `sleep_hours` | float | Total sleep duration |
| `calories` | int | Calories burned |

Optional but helpful: `active_minutes`, `spo2`

---

## 📚 Key Concepts Demonstrated

- **Time series cross-validation** — walk-forward evaluation to prevent leakage
- **Lag feature engineering** — converting sequential data to ML-friendly tabular format
- **Cyclical feature encoding** — sin/cos for day-of-week, month
- **Hyperparameter tuning** — AutoARIMA order selection, early stopping for LSTM and XGBoost
- **Model comparison** — standardised evaluation across statistical, ML and DL paradigms
- **Scalers** — fit on train, transform test (no leakage)
- **PyTorch training loop** — gradient clipping, LR scheduler, early stopping, best-weight restore

---

## 🔧 Extending the Project

Ideas to go further:
- [ ] Add a **Temporal Fusion Transformer (TFT)** using PyTorch Forecasting
- [ ] **Hyperparameter search** with Optuna for XGBoost/LSTM
- [ ] **Anomaly detection** — flag unusually low activity days
- [ ] **Multi-step forecasting** — predict the next 7 days at once
- [ ] **Real data integration** — connect to Fitbit API or Apple Health XML export
- [ ] **CI/CD** — GitHub Actions to retrain on new data automatically

---

## 📄 License

MIT — feel free to use this for your own learning and portfolio.

---

*Built as a portfolio project demonstrating end-to-end ML engineering on real-world health time series data.*
