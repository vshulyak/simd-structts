from dataclasses import dataclass

import numpy as np
from simd_structts.base.results import FilterResult
from simd_structts.base.results import ForecastResult
from simd_structts.base.results import SmootherResult

from .filter import kalman_filter
from .model_definition import ModelDefinition


@dataclass
class PySSMFilterResult(FilterResult):
    def get_forecast(self, h, exog=None):

        if exog is not None:
            assert exog.shape == (h, self.model.k_exog)
            static_non_exog = self.model.design[0, 0, : -self.model.k_exog]
            static_non_exog_repeated = np.repeat(
                static_non_exog[:, np.newaxis], h, axis=1
            )
            design = np.vstack([static_non_exog_repeated, exog.T])[np.newaxis, :, :]
        else:
            design = (
                self.model.design[:, :, np.newaxis]
                if self.model.design.ndim < 3
                else np.swapaxes(self.model.design.T, 0, 1)
            )

        res = []

        for series_idx in range(self.model.k_series):

            kfilter = self.model.kfilters[series_idx]

            mdef = ModelDefinition(
                obs=np.array([np.nan] * h)[np.newaxis, :],
                nobs=h,
                k_endog=self.model.k_endog,
                k_states=self.model.k_states,
                selection=np.eye(self.model.k_states)[:, :, np.newaxis],
                state_cov=self.model.state_cov[series_idx, :, :][:, :, np.newaxis],
                design=design,
                obs_intercept=self.model.obs_intercept,
                obs_cov=self.model.obs_cov[:, :, np.newaxis],
                transition=self.model.transition[:, :, np.newaxis],
                state_intercept=self.model.state_intercept,
                time_invariant=self.model.time_invariant,
            )
            res += [
                kalman_filter(
                    mdef,
                    initial_state=kfilter.predicted_state[..., -1],
                    initial_state_cov=kfilter.predicted_state_cov[..., -1],
                )
            ]

        return ForecastResult(
            predicted_mean=np.stack([k.forecast[0] for k in res]),
            se_mean=np.stack([np.sqrt(k.forecast_error_cov[0, 0, :]) for k in res]),
        )


@dataclass
class PySSMSmootherResult(PySSMFilterResult, SmootherResult):
    pass
