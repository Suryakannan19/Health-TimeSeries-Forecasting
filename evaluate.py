"""
Evaluation metrics and comparison utilities for time series forecasting.
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric MAPE — less sensitive to near-zero actuals."""
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask  = denom != 0
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100)

def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot)


def evaluate_all(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Return dict of all metrics."""
    return {
        "MAE":   mae(y_true, y_pred),
        "RMSE":  rmse(y_true, y_pred),
        "MAPE":  mape(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
        "R2":    r2(y_true, y_pred),
    }


def compare_models(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Build a formatted comparison table.

    Args:
        results: {model_name: {metric: value}}
    Returns:
        DataFrame with models as rows, metrics as columns.
    """
    df = pd.DataFrame(results).T
    df = df.round(3)

    # Rank each model per metric
    for metric in ["MAE", "RMSE", "MAPE", "sMAPE"]:
        if metric in df.columns:
            df[f"{metric}_rank"] = df[metric].rank().astype(int)
    if "R2" in df.columns:
        df["R2_rank"] = df["R2"].rank(ascending=False).astype(int)

    return df


def save_results(results: Dict[str, Dict[str, float]], path: str = "results/metrics.json"):
    """Persist results to JSON for reproducibility."""
    out = Path(path)
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✅ Metrics saved → {out}")


def naive_baseline(y_true: np.ndarray, y_pred_naive: np.ndarray) -> Dict[str, float]:
    """
    Seasonal naive baseline: predict tomorrow = same day last week.
    Gives context to model improvements.
    """
    return evaluate_all(y_true, y_pred_naive)
