from dataclasses import dataclass

import numpy as np
import simdkalman
from simd_structts.base.results import FilterResult
from simd_structts.base.results import ForecastResult
from simd_structts.base.results import SmootherResult


@dataclass
class SIMDFilterResult(FilterResult):
    def get_forecast(self, h, exog=None):

        n_vars = self.model.endog.shape[0]
        n_measurements = self.model.endog.shape[1]
        n_states = self.model.transition.shape[0]

        if exog is not None and exog.ndim == 2:
            assert exog.shape == (h, self.model.k_exog)
            static_non_exog = self.model.design[0, 0, : -self.model.k_exog]
            static_non_exog_repeated = np.repeat(
                static_non_exog[np.newaxis, :], h, axis=0
            )
            design = np.hstack([static_non_exog_repeated, exog])[:, np.newaxis, :]
        elif exog is not None and exog.ndim == 3:
            assert exog.shape == (self.model.k_series, h, self.model.k_exog)
            static_non_exog = self.model.design[0, 0, 0, : -self.model.k_exog]
            static_non_exog_repeated = np.repeat(
                static_non_exog[np.newaxis, :], h, axis=0
            )
            static_non_exog_repeated = np.repeat(
                static_non_exog_repeated[np.newaxis, :], self.model.k_series, axis=0
            )
            # design = np.hstack([static_non_exog_repeated, exog])[:, np.newaxis, :]
            design = np.concatenate([static_non_exog_repeated, exog], axis=2)[
                :, :, np.newaxis, :
            ]
        else:
            design = self.model.design

        predicted_mean = np.empty((n_vars, h, 1))
        predicted_cov = np.empty((n_vars, h, n_states, 1))

        m, P = (
            self.predicted_state[:, -1, :, np.newaxis],
            self.predicted_state_cov[:, -1, :],
        )

        for i in range(h):

            H = design

            # handle time-varying matrices (H for now only)
            # separate exog for each series
            if H.ndim == 4:
                H_t = H[:, i, ...]
            # common exog for all series
            elif H.ndim == 3:
                H_t = H[i][np.newaxis, ...]
            # no exog
            elif H.ndim == 2:
                H_t = H[np.newaxis, ...]
            else:
                raise

            obs_mean, obs_cov = simdkalman.primitives.predict_observation(
                m, P, H_t, self.model.obs_cov
            )

            predicted_mean[:, i, :] = obs_mean[..., 0]
            predicted_cov[:, i, :, :] = obs_cov

            m, P = simdkalman.primitives.predict(
                m, P, self.model.transition, self.model.state_cov
            )

        return ForecastResult(
            predicted_mean=predicted_mean[..., 0],
            se_mean=np.sqrt(predicted_cov[:, :, 0, 0]),
        )


@dataclass
class SIMDSmootherResult(SIMDFilterResult, SmootherResult):
    pass
