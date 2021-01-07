from dataclasses import dataclass

import numpy as np
import simdkalman

from ..base import BaseModel


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
    forecast_error: np.ndarray
    forecast_error_cov: np.ndarray
    llf: np.float64
    llf_obs: np.ndarray
    model: object

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


class RawStructTS(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def filter(self):

        data = self.endog
        single_sequence = len(data.shape) == 1
        if single_sequence:
            data = data[np.newaxis, :]

        n_vars = data.shape[0]
        n_measurements = data.shape[1]
        n_states = self.transition.shape[0]
        n_obs = self.design.shape[-2]

        # ---- Define model
        A = self.transition  # state_transition
        Q = self.state_cov  # process_noise

        # ---- Measurement model: different noise level for each sample
        H = self.design  # observation_model
        R = self.obs_cov  # observation_noise

        # initial values
        initial_value = self.initial_value
        initial_covariance = self.initial_covariance

        if len(initial_value.shape) == 1:
            initial_value = initial_value.reshape((n_states, 1))

        assert initial_value.shape[-2:] == (n_states, 1)
        assert initial_covariance.shape[-2:] == (n_states, n_states)

        if len(initial_value.shape) == 2:
            initial_value = np.vstack([initial_value[np.newaxis, ...]] * n_vars)

        if len(initial_covariance.shape) == 2:
            initial_covariance = np.vstack(
                [initial_covariance[np.newaxis, ...]] * n_vars
            )

        m = initial_value
        P = initial_covariance
        # mean, covariance, state_transition, process_noise
        # m, P, A, Q

        # prior_mean, prior_covariance, observation_model, observation_noise, measurement
        # m, P, H, R, y

        filtered_state_mean = np.empty((n_vars, n_measurements, n_states))
        filtered_state_cov = np.empty((n_vars, n_measurements, n_states, n_states))
        forecast_mean = np.empty((n_vars, n_measurements, 1))
        forecast_cov = np.empty((n_vars, n_measurements, n_states, 1))
        predicted_state_mean = np.empty((n_vars, n_measurements + 1, n_states))
        predicted_state_cov = np.empty((n_vars, n_measurements + 1, n_states, n_states))
        llf_obs = np.empty((n_vars, n_measurements))
        llf = 0

        predicted_state_mean[:, 0, :] = m[..., 0]
        predicted_state_cov[:, 0, :, :] = P

        for i in range(self.nobs):
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

            y = data[:, i, ...].reshape((n_vars, n_obs, 1))
            #
            #
            # from simdkalman.primitives import ddot, ddot_t_right, dinv, autoshape
            #
            # # @autoshape
            # def _update(prior_mean, prior_covariance, observation_model, observation_noise, measurement, log_likelihood=False):
            #
            #     n = prior_mean.shape[1]
            #     m = observation_model.shape[1]
            #
            #     assert(measurement.shape[-2:] == (m,1))
            #     assert(prior_covariance.shape[-2:] == (n,n))
            #     assert(observation_model.shape[-2:] == (m,n))
            #     assert(observation_noise.shape[-2:] == (m,m))
            #
            #     # y - H * mp
            #     v = measurement - ddot(observation_model, prior_mean)
            #     # (2, 1, 1) - (1, 1, 4) @ (2, 4, 1)
            #
            #
            #     # H * Pp * H.t + R
            #     S = ddot(observation_model, ddot_t_right(prior_covariance, observation_model)) + observation_noise
            #     invS = dinv(S)
            #
            #     # Kalman gain: Pp * H.t * invS
            #     K = ddot(ddot_t_right(prior_covariance, observation_model), invS)
            #
            #     # K * v + mp
            #     posterior_mean = ddot(K, v) + prior_mean
            #
            #     # Pp - K * H * Pp
            #     posterior_covariance = prior_covariance - ddot(K, ddot(observation_model, prior_covariance))
            #
            #     # inv-chi2 test var
            #     # outlier_test = np.sum(v * ddot(invS, v), axis=0)
            #     if log_likelihood:
            #         l = np.ravel(ddot(v.transpose((0,2,1)), ddot(invS, v)))
            #         l += np.log(np.linalg.det(S))
            #         l *= -0.5
            #         return posterior_mean, posterior_covariance, K, l
            #     else:
            #         return posterior_mean, posterior_covariance, K
            #
            #     return posterior_mean, posterior_covariance
            #
            # mo, Po, Ko = simdkalman.primitives.priv_update_with_nan_check(m, P, H_t, R, y)
            # # m, P, K = _update(m, P, H_t, R, y)
            # m, P, K = _update(m, P, H_t[np.newaxis,...], R[np.newaxis,...], y)

            # assert np.allclose(mo, m)

            # import pdb; pdb.set_trace()

            # forecast of the endog var
            obs_mean, obs_cov = simdkalman.primitives.predict_observation(m, P, H_t, R)
            forecast_mean[:, i, :] = obs_mean[..., 0]
            forecast_cov[:, i, :, :] = obs_cov

            # update
            m, P, K, l = simdkalman.primitives.priv_update_with_nan_check(
                m, P, H_t, R, y, log_likelihood=True
            )

            filtered_state_mean[:, i, :] = m[..., 0]
            filtered_state_cov[:, i, :, :] = P
            llf_obs[:, i] = l
            llf += l

            # predict
            m, P = simdkalman.primitives.predict(m, P, A, Q)

            predicted_state_mean[:, i + 1, :] = m[..., 0]
            predicted_state_cov[:, i + 1, :, :] = P

        return FilterResult(
            filtered_state=filtered_state_mean,
            filtered_state_cov=filtered_state_cov,
            predicted_state=predicted_state_mean,
            predicted_state_cov=predicted_state_cov,
            forecast=forecast_mean,
            forecast_error_cov=forecast_cov[:, :, 0:1, :],
            forecast_error=(data - forecast_mean),
            llf=llf,
            llf_obs=llf_obs,
            model=self,
        )

    def forecast(self):
        raise

    def smooth(self):
        raise
