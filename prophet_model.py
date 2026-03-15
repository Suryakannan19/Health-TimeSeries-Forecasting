"""
Facebook Prophet model for health metrics forecasting.
Prophet excels at daily data with strong weekly/yearly seasonality.
"""

import pandas as pd
import numpy as np
from typing import Optional


class ProphetForecaster:
    """
    Wrapper around Prophet with sensible health-data defaults.
    Adds custom seasonalities and holiday effects.
    """

    def __init__(
        self,
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = False,
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
    ):
        self.params = dict(
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality,
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale,
        )
        self.model = None

    def _to_prophet_df(self, series: pd.Series) -> pd.DataFrame:
        """Prophet requires columns named 'ds' and 'y'."""
        return pd.DataFrame({"ds": series.index, "y": series.values})

    def fit(self, series: pd.Series, regressors: Optional[pd.DataFrame] = None) -> "ProphetForecaster":
        try:
            from prophet import Prophet
        except ImportError:
            raise ImportError("Run: pip install prophet")

        df = self._to_prophet_df(series)
        self.model = Prophet(**self.params)

        if regressors is not None:
            for col in regressors.columns:
                self.model.add_regressor(col)
            df = df.merge(regressors.reset_index().rename(columns={"date": "ds"}), on="ds", how="left")

        self.model.fit(df)
        return self

    def predict(self, periods: int, freq: str = "D") -> pd.DataFrame:
        """Forecast 'periods' steps ahead. Returns full Prophet forecast df."""
        if self.model is None:
            raise RuntimeError("Call .fit() first.")
        future = self.model.make_future_dataframe(periods=periods, freq=freq)
        forecast = self.model.predict(future)
        return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]

    def predict_on_dates(self, dates: pd.DatetimeIndex) -> np.ndarray:
        """Predict on specific historical/future dates."""
        if self.model is None:
            raise RuntimeError("Call .fit() first.")
        future = pd.DataFrame({"ds": dates})
        forecast = self.model.predict(future)
        return forecast["yhat"].values

    def plot_components(self):
        """Visualize trend + seasonality decomposition."""
        if self.model is None:
            raise RuntimeError("Call .fit() first.")
        future = self.model.make_future_dataframe(periods=30)
        forecast = self.model.predict(future)
        return self.model.plot_components(forecast)
