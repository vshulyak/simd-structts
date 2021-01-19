from dataclasses import dataclass

import numpy as np
from simd_structts.base.results import FilterResult
from simd_structts.base.results import ForecastResult
from simd_structts.base.results import SmootherResult


@dataclass
class SMFilterResult(FilterResult):
    def get_forecast(self, h, exog=None):

        forecasts = [
            r.get_forecast(
                h,
                exog=exog[series_idx, ...]
                if exog is not None and exog.ndim == 3
                else exog,
            )
            for series_idx, r in enumerate(self.model)
        ]

        return ForecastResult(
            predicted_mean=np.stack([f.predicted_mean for f in forecasts]),
            se_mean=np.stack([f.se_mean for f in forecasts]),
        )


@dataclass
class SMSmootherResult(SMFilterResult, SmootherResult):
    pass
