"""
ARIMA / SARIMA model for step-count forecasting.
Uses pmdarima's auto_arima to find optimal (p,d,q)(P,D,Q,m) orders.
"""

import numpy as np
import pandas as pd
from typing import Tuple


class ARIMAForecaster:
    """
    Wrapper around pmdarima's AutoARIMA.
    Handles seasonal weekly patterns (m=7).
    """

    def __init__(self, seasonal: bool = True, m: int = 7):
        self.seasonal = seasonal
        self.m = m
        self.model = None
        self.model_fit = None

    def fit(self, series: pd.Series) -> "ARIMAForecaster":
        try:
            import pmdarima as pm
        except ImportError:
            raise ImportError("Run: pip install pmdarima")

        print("🔍 Running auto_arima (this may take ~30s)...")
        self.model = pm.auto_arima(
            series,
            seasonal=self.seasonal,
            m=self.m,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            information_criterion="aic",
            max_p=3, max_q=3,
            max_P=2, max_Q=2,
        )
        print(f"   Best order: {self.model.order}  seasonal: {self.model.seasonal_order}")
        return self

    def predict(self, n_periods: int) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Call .fit() first.")
        forecast, conf_int = self.model.predict(n_periods=n_periods, return_conf_int=True)
        return forecast, conf_int

    def rolling_forecast(
        self, train: pd.Series, test: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Walk-forward (one-step) forecast over the test period.
        Re-fits the model at each step for realistic evaluation.
        """
        try:
            import pmdarima as pm
        except ImportError:
            raise ImportError("Run: pip install pmdarima")

        history = list(train.values)
        predictions, lower_bounds = [], []

        for i in range(len(test)):
            model = pm.auto_arima(
                history,
                seasonal=self.seasonal,
                m=self.m,
                stepwise=True,
                suppress_warnings=True,
                error_action="ignore",
                information_criterion="aic",
            )
            fc, ci = model.predict(n_periods=1, return_conf_int=True)
            predictions.append(fc[0])
            lower_bounds.append(ci[0][0])
            history.append(test.iloc[i])

            if i % 10 == 0:
                print(f"   ARIMA walk-forward: {i+1}/{len(test)}")

        return np.array(predictions)
