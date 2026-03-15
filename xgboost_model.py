"""
XGBoost / LightGBM gradient-boosted tree models for time series.
Uses lag + rolling + calendar features; no data leakage.
"""

import numpy as np
import pandas as pd
from typing import Optional
import json
from pathlib import Path


FEATURE_COLS = [
    # Lags
    "steps_lag1", "steps_lag2", "steps_lag3", "steps_lag7", "steps_lag14",
    # Rolling stats
    "steps_roll7_mean", "steps_roll7_std", "steps_roll14_mean", "steps_ewm7",
    # Other health metrics (same-day features from prior day via shift)
    "heart_rate", "sleep_hours", "active_minutes", "calories",
    # Calendar
    "dow_sin", "dow_cos", "month_sin", "month_cos", "is_weekend",
]


class XGBoostForecaster:
    """
    XGBoost regressor wrapped for time-series use.
    Features are lag/rolling values — no future info leaked.
    """

    def __init__(
        self,
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        early_stopping_rounds: int = 50,
    ):
        self.params = dict(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            early_stopping_rounds=early_stopping_rounds,
            random_state=42,
        )
        self.model = None
        self.feature_cols: list[str] = []

    def _get_features(self, df: pd.DataFrame) -> list[str]:
        return [c for c in FEATURE_COLS if c in df.columns]

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame, target: str = "steps") -> "XGBoostForecaster":
        try:
            from xgboost import XGBRegressor
        except ImportError:
            raise ImportError("Run: pip install xgboost")

        self.feature_cols = self._get_features(train_df)
        X_train = train_df[self.feature_cols].dropna()
        y_train = train_df.loc[X_train.index, target]
        X_val   = val_df[self.feature_cols].dropna()
        y_val   = val_df.loc[X_val.index, target]

        self.model = XGBRegressor(**self.params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=100,
        )
        print(f"   Best iteration: {self.model.best_iteration}")
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Call .fit() first.")
        X = df[self.feature_cols].fillna(method="ffill")
        return self.model.predict(X)

    def feature_importance(self) -> pd.Series:
        if self.model is None:
            raise RuntimeError("Call .fit() first.")
        return pd.Series(
            self.model.feature_importances_,
            index=self.feature_cols
        ).sort_values(ascending=False)

    def save(self, path: str):
        self.model.save_model(path)

    def load(self, path: str):
        from xgboost import XGBRegressor
        self.model = XGBRegressor()
        self.model.load_model(path)


class LightGBMForecaster:
    """
    LightGBM alternative — typically faster and handles categoricals natively.
    Same feature pipeline as XGBoostForecaster.
    """

    def __init__(
        self,
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        max_depth: int = -1,
        num_leaves: int = 63,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
    ):
        self.params = dict(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=num_leaves,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=42,
            verbose=-1,
        )
        self.model = None
        self.feature_cols: list[str] = []

    def _get_features(self, df: pd.DataFrame) -> list[str]:
        return [c for c in FEATURE_COLS if c in df.columns]

    def fit(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None, target: str = "steps") -> "LightGBMForecaster":
        try:
            from lightgbm import LGBMRegressor
        except ImportError:
            raise ImportError("Run: pip install lightgbm")

        self.feature_cols = self._get_features(train_df)
        X_train = train_df[self.feature_cols].dropna()
        y_train = train_df.loc[X_train.index, target]

        callbacks = []
        eval_set = None
        if val_df is not None:
            from lightgbm import early_stopping, log_evaluation
            X_val = val_df[self.feature_cols].dropna()
            y_val = val_df.loc[X_val.index, target]
            eval_set = [(X_val, y_val)]
            callbacks = [early_stopping(50), log_evaluation(100)]

        self.model = LGBMRegressor(**self.params)
        self.model.fit(X_train, y_train, eval_set=eval_set, callbacks=callbacks)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Call .fit() first.")
        X = df[self.feature_cols].ffill()
        return self.model.predict(X)

    def feature_importance(self) -> pd.Series:
        if self.model is None:
            raise RuntimeError("Call .fit() first.")
        return pd.Series(
            self.model.feature_importances_,
            index=self.feature_cols
        ).sort_values(ascending=False)
