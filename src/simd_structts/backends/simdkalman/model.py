import numpy as np

from ..base import BaseModel
from .kalman_filter import EKalmanFilter


class SIMDKalmanPredictionResults:
    def __init__(self, predicted):
        self.predicted = predicted

    @property
    def predicted_mean(self):
        return self.predicted.observations.mean

    @property
    def se_mean(self):
        return self.predicted.observations.cov ** (1 / 2)


class SIMDKalmanFilterResults:
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
            likelihoods=True,  # slower
            log_likelihood=True,  # slower
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


class SIMDStructTS(BaseModel):


    def smooth(self):
        kf = EKalmanFilter(
            state_transition=self.transition,
            observation_model=self.design,
            process_noise=self.state_cov,
            observation_noise=self.obs_cov,
        )

        return SIMDKalmanFilterResults(
            self.endog,
            kf,
            exog=self.exog,
            initial_value=self.initial_value,
            initial_covariance=self.initial_covariance,
        )
