# Health Metrics Time Series Forecasting

End-to-end ML pipeline that forecasts daily step counts (and other health metrics) using four modelling approaches — from classical statistics to deep learning. Includes a Streamlit dashboard for interactive exploration.


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


```








---


