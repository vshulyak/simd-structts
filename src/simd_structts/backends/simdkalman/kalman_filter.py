import numpy as np

import simdkalman
from simdkalman.kalmanfilter import Gaussian
from simdkalman.primitives import (
    ddot_t_right,
    ensure_matrix,
    predict_observation,
    priv_update_with_nan_check,
)


class EKalmanFilter(simdkalman.KalmanFilter):
    def compute(
        self,
        data,
        n_test,
        initial_value=None,
        initial_covariance=None,
        smoothed=True,
        filtered=False,
        states=True,
        covariances=True,
        observations=True,
        likelihoods=False,
        gains=False,
        log_likelihood=False,
        verbose=False,
    ):

        # pylint: disable=W0201
        result = EKalmanFilter.Result()

        data = ensure_matrix(data)
        single_sequence = len(data.shape) == 1
        if single_sequence:
            data = data[np.newaxis, :]

        n_vars = data.shape[0]
        n_measurements = data.shape[1]
        n_states = self.state_transition.shape[0]
        n_obs = self.observation_model.shape[-2]

        def empty_gaussian(
            n_states=n_states, n_measurements=n_measurements, cov=covariances
        ):
            return Gaussian.empty(n_states, n_vars, n_measurements, cov)

        def auto_flat_observations(obs_gaussian):
            r = obs_gaussian
            if n_obs == 1:
                r = r.unvectorize_state()
            if single_sequence:
                r = r.unvectorize_vars()
            return r

        def auto_flat_states(obs_gaussian):
            if single_sequence:
                return obs_gaussian.unvectorize_vars()
            return obs_gaussian

        if initial_value is None:
            initial_value = np.zeros((n_states, 1))
        initial_value = ensure_matrix(initial_value)
        if len(initial_value.shape) == 1:
            initial_value = initial_value.reshape((n_states, 1))

        if initial_covariance is None:
            initial_covariance = ensure_matrix(
                np.trace(ensure_matrix(self.observation_model)) * (5 ** 2), n_states
            )

        initial_covariance = ensure_matrix(initial_covariance, n_states)
        initial_value = ensure_matrix(initial_value)
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

        keep_filtered = filtered or smoothed
        if filtered or gains:
            result.filtered = EKalmanFilter.Result()

        if log_likelihood:
            result.log_likelihood = np.zeros((n_vars,))
            if likelihoods:
                result.log_likelihoods = np.empty((n_vars, n_measurements))

        if keep_filtered:
            if observations:
                filtered_observations = empty_gaussian(n_states=n_obs)
            filtered_states = empty_gaussian(cov=True)

        if gains:
            result.filtered.gains = np.empty((n_vars, n_measurements, n_states, n_obs))

        for j in range(n_measurements):
            if verbose:
                print("filtering %d/%d" % (j + 1, n_measurements))

            y = data[:, j, ...].reshape((n_vars, n_obs, 1))

            tup = self.update(m, P, y, j, log_likelihood)
            m, P, K = tup[:3]
            if log_likelihood:
                l = tup[-1]
                result.log_likelihood += l
                if likelihoods:
                    result.log_likelihoods[:, j] = l

            if keep_filtered:
                if observations:
                    obs_mean, obs_cov = self.predict_observation(m, P, j)
                    filtered_observations.mean[:, j, :] = obs_mean[..., 0]
                    if covariances:
                        filtered_observations.cov[:, j, :, :] = obs_cov

                filtered_states.mean[:, j, :] = m[..., 0]
                filtered_states.cov[:, j, :, :] = P

            if gains:
                result.filtered.gains[:, j, :, :] = K

            m, P = self.predict_next(m, P)

        if smoothed:
            result.smoothed = EKalmanFilter.Result()
            if states:
                result.smoothed.states = empty_gaussian()

                # lazy trick to keep last filtered = last smoothed
                result.smoothed.states.mean = 1 * filtered_states.mean
                if covariances:
                    result.smoothed.states.cov = 1 * filtered_states.cov

            if observations:
                result.smoothed.observations = empty_gaussian(n_states=n_obs)
                result.smoothed.observations.mean = 1 * filtered_observations.mean
                if covariances:
                    result.smoothed.observations.cov = 1 * filtered_observations.cov

            if gains:
                result.smoothed.gains = np.zeros(
                    (n_vars, n_measurements, n_states, n_states)
                )
                result.pairwise_covariances = np.zeros(
                    (n_vars, n_measurements, n_states, n_states)
                )

            ms = filtered_states.mean[:, -1, :][..., np.newaxis]
            Ps = filtered_states.cov[:, -1, :, :]

            for j in range(n_measurements)[-2::-1]:
                if verbose:
                    print("smoothing %d/%d" % (j + 1, n_measurements))
                m0 = filtered_states.mean[:, j, :][..., np.newaxis]
                P0 = filtered_states.cov[:, j, :, :]

                PsNext = Ps
                ms, Ps, Cs = self.smooth_current(m0, P0, ms, Ps)

                if states:
                    result.smoothed.states.mean[:, j, :] = ms[..., 0]
                    if covariances:
                        result.smoothed.states.cov[:, j, :, :] = Ps

                if observations:
                    obs_mean, obs_cov = self.predict_observation(ms, Ps, j)
                    result.smoothed.observations.mean[:, j, :] = obs_mean[..., 0]
                    if covariances:
                        result.smoothed.observations.cov[:, j, :, :] = obs_cov

                if gains:
                    result.smoothed.gains[:, j, :, :] = Cs
                    result.pairwise_covariances[:, j, :, :] = ddot_t_right(PsNext, Cs)

        if filtered:
            if states:
                result.filtered.states = Gaussian(filtered_states.mean, None)
                if covariances:
                    result.filtered.states.cov = filtered_states.cov
                result.filtered.states = auto_flat_states(result.filtered.states)
            if observations:
                result.filtered.observations = auto_flat_observations(
                    filtered_observations
                )

        if smoothed:
            if observations:
                result.smoothed.observations = auto_flat_observations(
                    result.smoothed.observations
                )
            if states:
                result.smoothed.states = auto_flat_states(result.smoothed.states)

        if n_test > 0:
            result.predicted = EKalmanFilter.Result()
            if observations:
                result.predicted.observations = empty_gaussian(
                    n_measurements=n_test, n_states=n_obs
                )
            if states:
                result.predicted.states = empty_gaussian(n_measurements=n_test)

            for j in range(n_test):
                if verbose:
                    print("predicting %d/%d" % (j + 1, n_test))
                if states:
                    result.predicted.states.mean[:, j, :] = m[..., 0]
                    if covariances:
                        result.predicted.states.cov[:, j, :, :] = P
                if observations:
                    obs_mean, obs_cov = self.predict_observation(
                        m, P, j=n_measurements + j
                    )
                    result.predicted.observations.mean[:, j, :] = obs_mean[..., 0]
                    if covariances:
                        result.predicted.observations.cov[:, j, :, :] = obs_cov

                m, P = self.predict_next(m, P)

            if observations:
                result.predicted.observations = auto_flat_observations(
                    result.predicted.observations
                )
            if states:
                result.predicted.states = auto_flat_states(result.predicted.states)

        return result

    def update(self, m, P, y, j, log_likelihood=False):
        """
        Update KF – with design matrix exog handling for every step
        """
        assert j is not None, "step has to be provided"

        # modify design matrix in case we're dealing with exog variables
        if self.observation_model.ndim == 3:
            observation_model = self.observation_model[j]
        elif self.observation_model.ndim == 2:
            observation_model = self.observation_model
        else:
            raise

        return priv_update_with_nan_check(
            m,
            P,
            observation_model,
            self.observation_noise,
            y,
            log_likelihood=log_likelihood,
        )

    def predict_observation(self, m, P, j):
        """
        Predict KF – with design matrix exog handling for every step
        """
        assert j is not None, "step has to be provided"

        # modify design matrix in case we're dealing with exog variables
        if self.observation_model.ndim == 3:
            observation_model = self.observation_model[j]
        elif self.observation_model.ndim == 2:
            observation_model = self.observation_model
        else:
            raise

        return predict_observation(m, P, observation_model, self.observation_noise)
