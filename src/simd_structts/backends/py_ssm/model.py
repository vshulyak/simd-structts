import numpy as np

from dataclasses import dataclass

from ..base import BaseModel
from .filter import kalman_filter
from .smoother import get_kalman_gain, ksmooth_rep


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


from .filter import FILTER_CONVENTIONAL, INVERT_UNIVARIATE, SOLVE_CHOLESKY, STABILITY_FORCE_SYMMETRY, MEMORY_STORE_ALL

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

    # dynamic
    nobs: int
    k_states: int
    k_endog: int = 1

    dtype = np.float64

    # Kalman filter properties
    filter_method = FILTER_CONVENTIONAL
    inversion_method = INVERT_UNIVARIATE | SOLVE_CHOLESKY
    stability_method = STABILITY_FORCE_SYMMETRY
    conserve_memory = MEMORY_STORE_ALL
    tolerance = 0




class PySSMStructTS(BaseModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kfilter = None

    def filter(self, ssm):
        mdef = ModelDefinition(
            obs=self.endog[0,...].T,  # TODO use only the first serie for debugging
            nobs=self.nobs,
            # k_endog=self.k_endog,
            k_states=self.k_states,
            selection=np.eye(self.k_states)[:,:,np.newaxis],
            state_cov=self.state_cov[0:1,:,:].T,  # TODO: selecting only for the first model
            design=self.design[:,:,np.newaxis] if self.design.ndim < 3 else np.swapaxes(self.design.T, 0, 1),
            obs_intercept=np.array([[0.]]),
            obs_cov=self.obs_cov[:,:,np.newaxis],
            transition=self.transition[:,:,np.newaxis],
            state_intercept=np.array([[0.]]*self.k_states),
            time_invariant=self.design.ndim < 3  # TODO
        )

        # assert np.allclose(mdef.obs, ssm.obs)
        # assert np.allclose(mdef.nobs, ssm.nobs)
        # assert np.allclose(mdef.k_states, ssm.k_states)
        # # assert np.allclose(mdef.selection, ssm.selection)
        # # assert np.allclose(mdef.state_cov, ssm.state_cov)
        # assert np.allclose(mdef.design, ssm.design)
        # assert np.allclose(mdef.obs_intercept, ssm.obs_intercept)
        # assert np.allclose(mdef.obs_cov, ssm.obs_cov)
        # assert np.allclose(mdef.transition, ssm.transition)
        # assert np.allclose(mdef.state_intercept, ssm.state_intercept)
        # assert np.allclose(mdef.time_invariant, ssm.time_invariant)
        _, self.kfilter = kalman_filter(mdef, initial_value=self.initial_value, initial_covariance=self.initial_covariance)
        return self.kfilter

    def smooth(self, ssm, res):
        if not self.kfilter:
            self.kfilter = self.filter(ssm)

        endog = self.endog[0,...].T

        # forecasts_error = (endog[0,:] - self.kfilter.predicted_state[0,:-1])[np.newaxis,:]
        # # forecasts_error_cov = self.kfilter.predicted_state_cov[0,:-1,0:1,0:1].T
        # forecasts_error_cov = self.kfilter.predicted_state_cov[0:1,0:1,:-1]
        #
        # design_t = 0
        # np.dot(np.dot(ssm.design[:, :, design_t], filter_results.predicted_state_cov), ssm.design[:, :, design_t].T)


        # TODO: move & test these (can stack single call for starters)
        k_endog = 1
        kg = get_kalman_gain(
            k_states=self.k_states,
            k_endog=k_endog,
            nobs=self.nobs,
            dtype=np.float64,
            nmissing=np.array([0]*self.nobs),
            design=self.design[:,:,np.newaxis] if self.design.ndim < 3 else np.swapaxes(self.design.T, 0, 1),
            transition=self.transition[:,:,np.newaxis],
            predicted_state_cov=self.kfilter.predicted_state_cov,
            missing=np.array([0]*self.nobs)[np.newaxis,:],
            forecasts_error_cov=self.kfilter.forecast_error_cov
        )

        assert np.allclose(kg, res.filter_results._kalman_gain)

        # # TODO: stack this for every endog
        ks_r = ksmooth_rep(k_states=self.k_states, #
                        k_endog=k_endog, #
                        nobs=self.nobs, #
                        design_inp=self.design[:,:,np.newaxis] if self.design.ndim < 3 else np.swapaxes(self.design.T, 0, 1), #
                        transition_inp=self.transition[:,:,np.newaxis], #
                        obs_cov_inp=self.obs_cov[:,:,np.newaxis], #np.zeros((1,1,1)), # res_smooth.filter_results.obs_cov,
                        kalman_gain_inp=kg,
                        predicted_state_inp=self.kfilter.predicted_state,
                        predicted_state_cov_inp=self.kfilter.predicted_state_cov,
                        forecasts_error_inp=self.kfilter.forecast_error, #
                        forecasts_error_cov_inp=self.kfilter.forecast_error_cov, #
                        )

        return ks_r
