"""ARIMA and VAR — time series baseline models for BI-Rate prediction."""

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import VAR
import numpy as np
import warnings
warnings.filterwarnings("ignore")


class ARIMAModel:
    """AutoRegressive Integrated Moving Average model.

    Uses statsmodels ARIMA with configurable order (p, d, q).
    Optimal order is found via pmdarima.auto_arima during tuning.
    """

    def __init__(self, order=(1, 1, 1)):
        self.order = order
        self.fitted_model = None

    def fit(self, y_train):
        model = ARIMA(y_train, order=self.order)
        self.fitted_model = model.fit()
        print(f"ARIMA{self.order} fitted")

    def predict(self, n_periods):
        return self.fitted_model.forecast(steps=n_periods)

    def get_fitted_values(self):
        """Return in-sample predictions for residual computation."""
        if self.fitted_model is None:
            return None
        return self.fitted_model.fittedvalues

    def summary(self):
        if self.fitted_model:
            return self.fitted_model.summary()
        return None

    def save(self, filepath):
        import joblib
        joblib.dump(self, filepath)

    def load(self, filepath):
        import joblib
        loaded = joblib.load(filepath)
        self.order = loaded.order
        self.fitted_model = loaded.fitted_model


class VARModel:
    """Vector AutoRegression — multivariate time series model.

    Simultaneously models relationships between the target variable
    and all macroeconomic features. Lag order selected via AIC/BIC/HQIC.
    """

    def __init__(self, maxlags=3, ic="aic"):
        self.maxlags = maxlags
        self.ic = ic
        self.fitted_model = None
        self.target_idx = None

    def fit(self, data_df, target_col):
        numeric_df = data_df.select_dtypes(include=[np.number]).dropna()
        self.target_idx = list(numeric_df.columns).index(target_col)
        model = VAR(numeric_df)
        self.fitted_model = model.fit(maxlags=self.maxlags, ic=self.ic)
        print(f"VAR fitted, lag order = {self.fitted_model.k_ar}")

    def predict(self, n_periods, last_observations):
        if self.fitted_model.k_ar == 0:
            try:
                val = self.fitted_model.params.iloc[0, self.target_idx]
            except Exception:
                try:
                    val = self.fitted_model.params.values[0, self.target_idx]
                except Exception:
                    val = 0.0
            return np.full(n_periods, val)
        forecast = self.fitted_model.forecast(last_observations, steps=n_periods)
        preds = forecast[:, self.target_idx]
        # Guard: an over-parameterized VAR can have explosive roots, making the
        # multi-step forecast diverge to +/-inf or NaN. Fall back to a naive
        # last-value forecast so the model still emits a finite prediction.
        if not np.all(np.isfinite(preds)):
            last_target = last_observations[-1, self.target_idx]
            if not np.isfinite(last_target):
                last_target = 0.0
            fallback = np.full(n_periods, last_target)
            preds = np.where(np.isfinite(preds), preds, fallback)
        return preds

    def save(self, filepath):
        import joblib
        joblib.dump(self, filepath)

    def load(self, filepath):
        import joblib
        loaded = joblib.load(filepath)
        self.maxlags = loaded.maxlags
        self.ic = loaded.ic
        self.fitted_model = loaded.fitted_model
        self.target_idx = loaded.target_idx