import numpy as np
from simd_structts.base.model import BaseModel

from .filter import kalman_filter
from .model_definition import ModelDefinition
from .results import PySSMFilterResult
from .results import PySSMSmootherResult
from .smoother import get_kalman_gain
from .smoother import get_smoothed_forecasts
from .smoother import ksmooth_rep


class PySSMStructTS(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kfilters = []

    def filter(self):

        self.kfilters = []

        for series_idx in range(self.k_series):

            mdef = ModelDefinition(
                obs=self.endog[series_idx, ...].T,
                nobs=self.nobs,
                k_endog=self.k_endog,
                k_states=self.k_states,
                selection=self.selection,
                state_cov=self.state_cov[series_idx, :, :][:, :, np.newaxis],
                design=self.design[:, :, np.newaxis]
                if self.design.ndim < 3
                else np.swapaxes(self.design.T, 0, 1),
                obs_intercept=self.obs_intercept,
                obs_cov=self.obs_cov[:, :, np.newaxis],
                transition=self.transition[:, :, np.newaxis],
                state_intercept=self.state_intercept,
                time_invariant=self.time_invariant,
            )

            self.kfilters += [
                kalman_filter(
                    mdef,
                    initial_state=self.initial_value,
                    initial_state_cov=self.initial_covariance,
                )
            ]

        self.filter_result = PySSMFilterResult(
            filtered_state=np.stack([k.filtered_state.T for k in self.kfilters]),
            filtered_state_cov=np.stack(
                [k.filtered_state_cov.T for k in self.kfilters]
            ),
            predicted_state=np.stack([k.predicted_state.T for k in self.kfilters]),
            predicted_state_cov=np.stack(
                [k.predicted_state_cov.T for k in self.kfilters]
            ),
            forecast=np.stack([k.forecast.T for k in self.kfilters]),
            forecast_error=np.stack([k.forecast_error.T for k in self.kfilters]),
            forecast_error_cov=np.stack(
                [k.forecast_error_cov.T for k in self.kfilters]
            ),
            llf_obs=np.stack([k.loglikelihood.T for k in self.kfilters]),
            forecast_cov=None,  # TODO
            llf=None,  # TODO
            model=self,  # TODO
        )
        return self.filter_result

    def smooth(self):
        if not self.kfilters:
            self.filter()

        design = (
            self.design[:, :, np.newaxis]
            if self.design.ndim < 3
            else np.swapaxes(self.design.T, 0, 1)
        )
        transition = self.transition[:, :, np.newaxis]
        obs_cov = self.obs_cov[:, :, np.newaxis]

        ks_r_res = []
        smoothed_forecasts_res = []
        smoothed_forecasts_error_res = []
        smoothed_forecasts_error_cov_res = []

        for series_idx in range(self.k_series):

            endog = self.endog[series_idx, ...].T
            kfilter = self.kfilters[series_idx]

            missing = np.isnan(endog).astype(np.int32)  # same dim as endog
            nmissing = missing.sum(
                axis=0
            )  # (nobs) shape, sum of all missing accross missing axis

            kg = get_kalman_gain(
                k_states=self.k_states,
                k_endog=self.k_endog,
                nobs=self.nobs,
                dtype=np.float64,
                nmissing=nmissing,
                design=design,
                transition=transition,
                predicted_state_cov=kfilter.predicted_state_cov,
                missing=missing,
                forecasts_error_cov=kfilter.forecast_error_cov,
            )

            ks_r = ksmooth_rep(
                k_states=self.k_states,
                k_endog=self.k_endog,
                nobs=self.nobs,
                design_inp=design,
                transition_inp=transition,
                obs_cov_inp=obs_cov,
                kalman_gain_inp=kg,
                predicted_state_inp=kfilter.predicted_state,
                predicted_state_cov_inp=kfilter.predicted_state_cov,
                forecasts_error_inp=kfilter.forecast_error,
                forecasts_error_cov_inp=kfilter.forecast_error_cov,
                nmissing=nmissing,
                missing=missing,
            )

            (
                smoothed_forecasts,
                smoothed_forecasts_error,
                smoothed_forecasts_error_cov,
            ) = get_smoothed_forecasts(
                endog=endog,
                smoothed_state=ks_r.smoothed_state,
                smoothed_state_cov=ks_r.smoothed_state_cov,
                nobs=self.nobs,
                design=design,
                obs_cov=obs_cov,
                obs_intercept=self.obs_intercept,
                missing=missing,
                nmissing=nmissing,
                forecasts=kfilter.forecast,
                forecasts_error=kfilter.forecast_error,
                forecasts_error_cov=kfilter.forecast_error_cov,
                dtype=np.float64,
            )
            ks_r_res += [ks_r]
            smoothed_forecasts_res += [smoothed_forecasts]
            smoothed_forecasts_error_res += [smoothed_forecasts_error]
            smoothed_forecasts_error_cov_res += [smoothed_forecasts_error_cov]

        return PySSMSmootherResult(
            filtered_state=self.filter_result.filtered_state,
            filtered_state_cov=self.filter_result.filtered_state_cov,
            predicted_state=self.filter_result.predicted_state,
            predicted_state_cov=self.filter_result.predicted_state_cov,
            forecast=self.filter_result.forecast,
            forecast_error=self.filter_result.forecast_error,
            forecast_error_cov=self.filter_result.forecast_error_cov,
            llf_obs=self.filter_result.llf_obs,
            smoothed_state=np.stack([k.smoothed_state.T for k in ks_r_res]),
            smoothed_state_cov=np.stack([k.smoothed_state_cov.T for k in ks_r_res]),
            smoothed_forecasts=np.stack(
                [sf.squeeze() for sf in smoothed_forecasts_res]
            ),
            smoothed_forecasts_cov=None,  # TODO
            forecast_cov=None,  # TODO
            llf=None,  # TODO
            model=self,  # TODO
            # TODO
            # smoothed_forecasts_error=np.stack(
            #     [sf.T for sf in smoothed_forecasts_error_res]
            # ),  # smoothed_forecasts_error,
            # TODO
            # smoothed_forecasts_error_cov=np.stack(
            #     [sf.T for sf in smoothed_forecasts_error_cov_res]
            # ),  # smoothed_forecasts_error_cov,
        )
