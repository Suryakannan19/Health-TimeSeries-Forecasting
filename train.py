"""
Main training script — runs all models and saves a comparison report.

Usage:
    python src/train.py
    python src/train.py --models xgb lgbm   # run subset
    python src/train.py --target heart_rate
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Local imports ──────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data_loader import (
    load_data, clean_data, add_time_features,
    make_lag_features, train_test_split_ts,
)
from src.evaluate import evaluate_all, compare_models, save_results, naive_baseline


def run_arima(train: pd.Series, test: pd.Series) -> np.ndarray:
    from src.models.arima_model import ARIMAForecaster
    print("\n📊 Fitting ARIMA...")
    fc = ARIMAForecaster(seasonal=True, m=7)
    fc.fit(train)
    preds, _ = fc.predict(len(test))
    return preds


def run_prophet(train_df: pd.DataFrame, test_df: pd.DataFrame, target: str) -> np.ndarray:
    from src.models.prophet_model import ProphetForecaster
    print("\n📊 Fitting Prophet...")
    fc = ProphetForecaster()
    fc.fit(train_df[target])
    return fc.predict_on_dates(test_df.index)


def run_xgb(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, target: str) -> np.ndarray:
    from src.models.xgboost_model import XGBoostForecaster
    print("\n📊 Fitting XGBoost...")
    fc = XGBoostForecaster()
    fc.fit(train_df, val_df, target=target)
    preds = fc.predict(test_df)
    print("\n  XGB Feature Importance (top 10):")
    print(fc.feature_importance().head(10).to_string())
    return preds


def run_lgbm(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, target: str) -> np.ndarray:
    from src.models.xgboost_model import LightGBMForecaster
    print("\n📊 Fitting LightGBM...")
    fc = LightGBMForecaster()
    fc.fit(train_df, val_df, target=target)
    return fc.predict(test_df)


def run_lstm(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    from src.models.lstm_model import LSTMForecaster
    print("\n📊 Fitting LSTM...")
    fc = LSTMForecaster(seq_len=14, hidden_size=128, max_epochs=100, patience=15)
    fc.fit(train_df, val_df)
    # LSTM needs seq_len rows of context before predicting test
    context_df = pd.concat([val_df.tail(14), test_df])
    preds_all  = fc.predict(context_df)
    return preds_all[-len(test_df):]


def main(models: list[str], target: str = "steps"):
    # ── Load & preprocess ─────────────────────────────────────────────────────
    print("📂 Loading data...")
    df = load_data()
    df = clean_data(df)
    df = add_time_features(df)
    df = make_lag_features(df, target=target, lags=[1, 2, 3, 7, 14])
    df.dropna(subset=[f"{target}_lag14"], inplace=True)  # ensure all lags available

    # 70/15/15 split
    n = len(df)
    train_df = df.iloc[: int(n * 0.70)]
    val_df   = df.iloc[int(n * 0.70) : int(n * 0.85)]
    test_df  = df.iloc[int(n * 0.85) :]

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    y_test = test_df[target].values

    # ── Naive seasonal baseline (predict = same day last week) ────────────────
    naive_preds = test_df[f"{target}_lag7"].values
    results = {
        "Naive (lag-7)": evaluate_all(y_test, naive_preds)
    }

    # ── Run requested models ──────────────────────────────────────────────────
    model_map = {
        "arima":   lambda: run_arima(train_df[target], test_df[target]),
        "prophet": lambda: run_prophet(train_df, test_df, target),
        "xgb":     lambda: run_xgb(train_df, val_df, test_df, target),
        "lgbm":    lambda: run_lgbm(train_df, val_df, test_df, target),
        "lstm":    lambda: run_lstm(train_df, val_df, test_df),
    }

    predictions = {}
    for name in models:
        if name not in model_map:
            print(f"⚠️  Unknown model '{name}', skipping.")
            continue
        try:
            preds = model_map[name]()
            preds = np.clip(preds, 0, None)  # steps can't be negative
            # Align length (LSTM seq offset may differ)
            min_len = min(len(y_test), len(preds))
            results[name.upper()] = evaluate_all(y_test[:min_len], preds[:min_len])
            predictions[name] = preds
        except Exception as e:
            print(f"❌ {name} failed: {e}")

    # ── Print comparison table ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("📋 MODEL COMPARISON")
    print("=" * 60)
    table = compare_models(results)
    print(table[["MAE", "RMSE", "MAPE", "R2"]].to_string())

    # ── Save results ──────────────────────────────────────────────────────────
    save_results(results)

    # ── Save predictions CSV ──────────────────────────────────────────────────
    pred_df = pd.DataFrame({"date": test_df.index, "actual": y_test})
    for name, preds in predictions.items():
        min_len = min(len(pred_df), len(preds))
        pred_df.loc[pred_df.index[:min_len], f"pred_{name}"] = preds[:min_len]
    pred_df.to_csv("results/predictions.csv", index=False)
    print("✅ Predictions saved → results/predictions.csv")

    return results, predictions, test_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", nargs="+",
        default=["xgb", "lgbm", "prophet", "lstm"],
        help="Models to run: arima prophet xgb lgbm lstm",
    )
    parser.add_argument("--target", default="steps", help="Target column to forecast")
    args = parser.parse_args()
    main(args.models, args.target)
