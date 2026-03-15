"""
Data loading, cleaning, and train/test splitting utilities.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple

RAW_PATH = Path(__file__).parent.parent / "data" / "raw" / "health_data.csv"


def load_data(path: Path = RAW_PATH) -> pd.DataFrame:
    """Load raw health CSV and parse dates."""
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    df.sort_index(inplace=True)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values and remove physiologically impossible outliers.
    - Numeric columns: linear interpolation (max 3-day gap), then ffill
    - Outlier clipping based on physiological ranges
    """
    df = df.copy()

    bounds = {
        "steps":          (0,     30_000),
        "heart_rate":     (40,    110),
        "sleep_hours":    (2,     12),
        "calories":       (1000,  5000),
        "active_minutes": (0,     200),
        "spo2":           (88,    100),
    }
    for col, (lo, hi) in bounds.items():
        if col in df.columns:
            df[col] = df[col].clip(lo, hi)

    # Interpolate short gaps, forward-fill any remaining
    df = df.interpolate(method="time", limit=3).ffill().bfill()
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach calendar and cyclical time features useful for tree models."""
    df = df.copy()
    df["day_of_week"]  = df.index.dayofweek
    df["month"]        = df.index.month
    df["day_of_year"]  = df.index.dayofyear
    df["is_weekend"]   = (df.index.dayofweek >= 5).astype(int)
    df["week_of_year"] = df.index.isocalendar().week.astype(int)

    # Cyclical encoding — prevents model treating Mon(0) and Sun(6) as far apart
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def make_lag_features(df: pd.DataFrame, target: str, lags: list[int]) -> pd.DataFrame:
    """Create lag and rolling features for the target column."""
    df = df.copy()
    for lag in lags:
        df[f"{target}_lag{lag}"] = df[target].shift(lag)
    df[f"{target}_roll7_mean"] = df[target].shift(1).rolling(7).mean()
    df[f"{target}_roll7_std"]  = df[target].shift(1).rolling(7).std()
    df[f"{target}_roll14_mean"]= df[target].shift(1).rolling(14).mean()
    df[f"{target}_ewm7"]       = df[target].shift(1).ewm(span=7).mean()
    return df


def train_test_split_ts(
    df: pd.DataFrame,
    test_ratio: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chronological train/test split — never shuffle time series data!
    Returns (train_df, test_df).
    """
    split_idx = int(len(df) * (1 - test_ratio))
    return df.iloc[:split_idx], df.iloc[split_idx:]
