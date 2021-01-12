import numpy as np
import simdkalman
from simd_structts.base.model import BaseModel

from .kalman import update_with_nan_check
from .results import SIMDFilterResult
from .results import SIMDSmootherResult


class SIMDStructTS(BaseModel):
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

        # ---- Measurement model
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

            # forecast of the endog var
            obs_mean, obs_cov = simdkalman.primitives.predict_observation(m, P, H_t, R)
            forecast_mean[:, i, :] = obs_mean[..., 0]
            forecast_cov[:, i, :, :] = obs_cov

            # update. R matrix is reshaped to be 3d, it's a requirement for the function
            m, P, K, l = update_with_nan_check(
                m, P, H_t, R[np.newaxis, ...], y, univariate=self.k_endog == 1
            )

            filtered_state_mean[:, i, :] = m[..., 0]
            filtered_state_cov[:, i, :, :] = P
            llf_obs[:, i] = l
            llf += l

            # predict
            m, P = simdkalman.primitives.predict(m, P, A, Q)

            predicted_state_mean[:, i + 1, :] = m[..., 0]
            predicted_state_cov[:, i + 1, :, :] = P

        return SIMDFilterResult(
            filtered_state=filtered_state_mean,
            filtered_state_cov=filtered_state_cov,
            predicted_state=predicted_state_mean,
            predicted_state_cov=predicted_state_cov,
            forecast=forecast_mean,
            forecast_cov=forecast_cov,
            forecast_error_cov=forecast_cov[:, :, 0:1, :],
            forecast_error=(data - forecast_mean),
            llf=llf,
            llf_obs=llf_obs,
            model=self,
        )

    def smooth(self):
        filter_result = self.filter()

        data = self.endog
        single_sequence = len(data.shape) == 1
        if single_sequence:
            data = data[np.newaxis, :]

        # ---- Define model
        A = self.transition  # state_transition
        Q = self.state_cov  # process_noise

        # ---- Measurement model
        H = self.design  # observation_model
        R = self.obs_cov  # observation_noise

        n_vars = data.shape[0]
        n_measurements = data.shape[1]
        n_states = self.transition.shape[0]
        n_obs = self.design.shape[-2]

        smoothed_state_mean = 1 * filter_result.filtered_state
        smoothed_state_cov = 1 * filter_result.filtered_state_cov
        smoothed_forecasts_mean = 1 * filter_result.forecast
        smoothed_forecasts_cov = 1 * filter_result.forecast_cov

        ms = filter_result.filtered_state[:, -1, :][..., np.newaxis]
        Ps = filter_result.filtered_state_cov[:, -1, :, :]

        for i in range(n_measurements)[-1::-1]:

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
                ms, Ps, H_t, R
            )
            smoothed_forecasts_mean[:, i, :] = obs_mean[..., 0]
            smoothed_forecasts_cov[:, i, :, :] = obs_cov

            m0 = filter_result.filtered_state[:, i - 1, :][..., np.newaxis]
            P0 = filter_result.filtered_state_cov[:, i - 1, :, :]

            # we skip 0th element as i-1 would otherwise point to the last (-1) element below
            if i > 0:
                ms, Ps, Cs = simdkalman.primitives.priv_smooth(m0, P0, A, Q, ms, Ps)

                smoothed_state_mean[:, i - 1, :] = ms[..., 0]
                smoothed_state_cov[:, i - 1, :, :] = Ps

        return SIMDSmootherResult(
            smoothed_state=smoothed_state_mean,
            smoothed_state_cov=smoothed_state_cov,
            smoothed_forecasts=smoothed_forecasts_mean[..., 0],
            smoothed_forecasts_cov=smoothed_forecasts_cov[..., 0],
            filtered_state=filter_result.filtered_state,
            filtered_state_cov=filter_result.filtered_state_cov,
            predicted_state=filter_result.predicted_state,
            predicted_state_cov=filter_result.predicted_state_cov,
            forecast=filter_result.forecast,
            forecast_cov=filter_result.forecast_cov,
            forecast_error_cov=filter_result.forecast_error_cov,
            forecast_error=filter_result.forecast_error,
            llf=filter_result.llf,
            llf_obs=filter_result.llf_obs,
            model=filter_result.model,
        )
