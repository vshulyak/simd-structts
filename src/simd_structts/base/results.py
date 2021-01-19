from dataclasses import dataclass

import numpy as np


@dataclass
class ForecastResult:
    predicted_mean: np.ndarray
    se_mean: np.ndarray


@dataclass
class FilterResult:
    filtered_state: np.ndarray
    filtered_state_cov: np.ndarray
    predicted_state: np.ndarray
    predicted_state_cov: np.ndarray
    forecast: np.ndarray
    forecast_cov: np.ndarray
    forecast_error: np.ndarray
    forecast_error_cov: np.ndarray
    llf: np.float64
    llf_obs: np.ndarray
    model: object

    def get_forecast(self, h, exog=None) -> ForecastResult:
        raise NotImplementedError


@dataclass
class SmootherResult(FilterResult):
    smoothed_state: np.ndarray
    smoothed_state_cov: np.ndarray
    smoothed_forecasts: np.ndarray
    smoothed_forecasts_cov: np.ndarray
