from dataclasses import dataclass

import numpy as np

from ..base import BaseModel
from .filter import FILTER_CONVENTIONAL
from .filter import INVERT_UNIVARIATE
from .filter import kalman_filter
from .filter import MEMORY_STORE_ALL
from .filter import SOLVE_CHOLESKY
from .filter import STABILITY_FORCE_SYMMETRY
from .smoother import get_kalman_gain
from .smoother import get_smoothed_forecasts
from .smoother import ksmooth_rep


class PySSMPredictionResults:
    def __init__(self, predicted):
        self.predicted = predicted

    @property
    def predicted_mean(self):
        return self.predicted.observations.mean

    @property
    def se_mean(self):
        return self.predicted.observations.cov ** (1 / 2)


class PySSMFilterResults:
    def __init__(
        self, endog, kf, exog=None, initial_value=None, initial_covariance=None
    ):
        self.endog = endog
        self.kf = kf
        self.exog = exog
        self.initial_value = initial_value
        self.initial_covariance = initial_covariance

    def _compute(self, horizon=0, exog=None):
        # TODO: optimize
        if horizon > 0 and exog is not None:

            n_obs, _, k_comp = self.kf.observation_model.shape

            n_exog, k_exog = exog.shape

            # take last row of the design matrix and clone it n_exog times
            design = np.repeat(self.kf.observation_model[-2:-1], n_exog, axis=0)
            design[:, 0, k_comp - k_exog : k_comp] = exog

            observation_model = np.vstack([self.kf.observation_model, design])

            self.kf = kf = EKalmanFilter(
                state_transition=self.kf.state_transition,
                observation_model=observation_model,
                process_noise=self.kf.process_noise,
                observation_noise=self.kf.observation_noise,
            )
        else:
            # FIXME: debug
            # kf = self.kf
            self.kf = kf = EKalmanFilter(
                state_transition=self.kf.state_transition,
                observation_model=self.kf.observation_model,
                process_noise=self.kf.process_noise,
                observation_noise=self.kf.observation_noise,
            )

        return self.kf.compute(
            self.endog,
            horizon,
            filtered=True,
            smoothed=True,
            initial_value=self.initial_value,
            initial_covariance=self.initial_covariance,
        )

    def get_forecast(self, horizon, exog=None):
        """
        TODO: check if params match to sm
        TODO: check uncertainty
        """
        return SIMDKalmanPredictionResults(
            self._compute(horizon=horizon, exog=exog).predicted
        )

    @property
    def filtered_state(self):
        return self._compute().filtered.states.mean

    @property
    def filtered_state_cov(self):
        return self._compute().filtered.states.cov

    @property
    def smoothed_state(self):
        return self._compute().smoothed.states.mean

    @property
    def smoothed_state_cov(self):
        return self._compute().smoothed.states.cov

    @property
    def smoothed_forecasts(self):
        return self._compute().smoothed.observations.mean

    @property
    def predicted_state(self):
        # TODO: compute is broken
        return self.kf.ms

    @property
    def predicted_state_cov(self):
        # TODO: compute is broken
        return self.kf.Ps


@dataclass
class ModelDefinition:

    obs: np.ndarray

    # SS def
    selection: np.ndarray
    state_cov: np.ndarray
    design: np.ndarray
    obs_intercept: np.ndarray
    obs_cov: np.ndarray
    transition: np.ndarray
    state_intercept: np.ndarray
    time_invariant: bool
    k_endog: int

    # dynamic
    nobs: int
    k_states: int

    dtype = np.float64

    # Kalman filter properties
    filter_method = FILTER_CONVENTIONAL
    inversion_method = INVERT_UNIVARIATE | SOLVE_CHOLESKY
    stability_method = STABILITY_FORCE_SYMMETRY
    conserve_memory = MEMORY_STORE_ALL
    tolerance = 0


@dataclass
class FilterResult:
    filtered_state: np.ndarray
    filtered_state_cov: np.ndarray
    predicted_state: np.ndarray
    predicted_state_cov: np.ndarray
    forecast: np.ndarray
    forecast_error: np.ndarray
    forecast_error_cov: np.ndarray
    loglikelihood: np.ndarray


@dataclass
class SmoothResult(FilterResult):
    smoothed_state: np.ndarray
    smoothed_state_cov: np.ndarray
    smoothed_forecasts: np.ndarray
    smoothed_forecasts_error: np.ndarray
    smoothed_forecasts_error_cov: np.ndarray


@dataclass
class ForecastResult:
    predicted_mean: np.ndarray
    se_mean: np.ndarray


class PySSMStructTS(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kfilter = None

    def filter(self, models):
        i = 0

        mdef = ModelDefinition(
            obs=self.endog[i, ...].T,
            nobs=self.nobs,
            k_endog=self.k_endog,
            k_states=self.k_states,
            selection=self.selection,
            state_cov=self.state_cov[i : i + 1, :, :].T,
            design=self.design[:, :, np.newaxis]
            if self.design.ndim < 3
            else np.swapaxes(self.design.T, 0, 1),
            obs_intercept=self.obs_intercept,
            obs_cov=self.obs_cov[:, :, np.newaxis],
            transition=self.transition[:, :, np.newaxis],
            state_intercept=self.state_intercept,
            time_invariant=self.time_invariant,
        )

        self.kfilter = kalman_filter(
            mdef,
            initial_state=self.initial_value,
            initial_state_cov=self.initial_covariance,
        )

        return FilterResult(
            filtered_state=self.kfilter.filtered_state,
            filtered_state_cov=self.kfilter.filtered_state_cov,
            predicted_state=self.kfilter.predicted_state,
            predicted_state_cov=self.kfilter.predicted_state_cov,
            forecast=self.kfilter.forecast,
            forecast_error=self.kfilter.forecast_error,
            forecast_error_cov=self.kfilter.forecast_error_cov,
            loglikelihood=self.kfilter.loglikelihood,
        )

    def forecast(self, h, exog=None):
        if not self.kfilter:
            self.kfilter = self.filter(ssm)

        if exog is not None:
            assert exog.shape == (h, self.k_exog)
            static_non_exog = self.design[0, 0, : -self.k_exog]
            static_non_exog_repeated = np.repeat(
                static_non_exog[:, np.newaxis], h, axis=1
            )
            design = np.vstack([static_non_exog_repeated, exog.T])[np.newaxis, :, :]
        else:
            design = (
                self.design[:, :, np.newaxis]
                if self.design.ndim < 3
                else np.swapaxes(self.design.T, 0, 1)
            )

        mdef = ModelDefinition(
            obs=np.array([np.nan] * h)[np.newaxis, :],
            nobs=h,
            k_endog=self.k_endog,
            k_states=self.k_states,
            selection=np.eye(self.k_states)[:, :, np.newaxis],
            state_cov=self.state_cov[
                0:1, :, :
            ].T,  # TODO: selecting only for the first model
            design=design,
            obs_intercept=self.obs_intercept,
            obs_cov=self.obs_cov[:, :, np.newaxis],
            transition=self.transition[:, :, np.newaxis],
            state_intercept=self.state_intercept,
            time_invariant=self.time_invariant,
        )
        kfilter = kalman_filter(
            mdef,
            initial_state=self.kfilter.predicted_state[..., -1],
            initial_state_cov=self.kfilter.predicted_state_cov[..., -1],
        )

        return ForecastResult(
            predicted_mean=kfilter.forecast[0],
            se_mean=np.sqrt(kfilter.forecast_error_cov[0, 0, :]),
        )

    def smooth(self, ssm, res):
        if not self.kfilter:
            self.kfilter = self.filter(ssm)

        design = (
            self.design[:, :, np.newaxis]
            if self.design.ndim < 3
            else np.swapaxes(self.design.T, 0, 1)
        )
        transition = self.transition[:, :, np.newaxis]
        obs_cov = self.obs_cov[:, :, np.newaxis]

        i = 0

        endog = self.endog[i, ...].T

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
            predicted_state_cov=self.kfilter.predicted_state_cov,
            missing=missing,
            forecasts_error_cov=self.kfilter.forecast_error_cov,
        )

        ks_r = ksmooth_rep(
            k_states=self.k_states,
            k_endog=self.k_endog,
            nobs=self.nobs,
            design_inp=design,
            transition_inp=transition,
            obs_cov_inp=obs_cov,
            kalman_gain_inp=kg,
            predicted_state_inp=self.kfilter.predicted_state,
            predicted_state_cov_inp=self.kfilter.predicted_state_cov,
            forecasts_error_inp=self.kfilter.forecast_error,
            forecasts_error_cov_inp=self.kfilter.forecast_error_cov,
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
            forecasts=self.kfilter.forecast,
            forecasts_error=self.kfilter.forecast_error,
            forecasts_error_cov=self.kfilter.forecast_error_cov,
            dtype=np.float64,
        )

        return SmoothResult(
            filtered_state=self.kfilter.filtered_state,
            filtered_state_cov=self.kfilter.filtered_state_cov,
            predicted_state=self.kfilter.predicted_state,
            predicted_state_cov=self.kfilter.predicted_state_cov,
            forecast=self.kfilter.forecast,
            forecast_error=self.kfilter.forecast_error,
            forecast_error_cov=self.kfilter.forecast_error_cov,
            loglikelihood=self.kfilter.loglikelihood,
            smoothed_state=ks_r.smoothed_state,
            smoothed_state_cov=ks_r.smoothed_state_cov,
            smoothed_forecasts=smoothed_forecasts,
            smoothed_forecasts_error=smoothed_forecasts_error,
            smoothed_forecasts_error_cov=smoothed_forecasts_error_cov,
        )
