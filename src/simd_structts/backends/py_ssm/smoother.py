import numpy as np

from dataclasses import dataclass


@dataclass
class KSResult():
    smoothing_error: np.ndarray
    scaled_smoothed_estimator: np.ndarray
    scaled_smoothed_estimator_cov: np.ndarray
    smoothed_state: np.ndarray
    smoothed_state_cov: np.ndarray


def get_kalman_gain(k_states, k_endog, nobs, dtype, nmissing, design, transition,
                    predicted_state_cov, missing, forecasts_error_cov):
    """
    Kalman gain matrices
    """
    # k x n
    _kalman_gain = np.zeros(
        (k_states, k_endog, nobs), dtype=dtype)

    for t in range(nobs):
        # In the case of entirely missing observations, let the Kalman
        # gain be zeros.
        if nmissing[t] == k_endog:
            continue

        design_t = 0 if design.shape[2] == 1 else t
        transition_t = 0 if transition.shape[2] == 1 else t
        if nmissing[t] == 0:
            _kalman_gain[:, :, t] = np.dot(
                np.dot(
                    transition[:, :, transition_t],
                    predicted_state_cov[:, :, t]
                ),
                np.dot(
                    np.transpose(design[:, :, design_t]),
                    np.linalg.inv(forecasts_error_cov[:, :, t])
                )
            )
        else:
            mask = ~missing[:, t].astype(bool)
            F = forecasts_error_cov[np.ix_(mask, mask, [t])]
            _kalman_gain[:, mask, t] = np.dot(
                np.dot(
                    transition[:, :, transition_t],
                    predicted_state_cov[:, :, t]
                ),
                np.dot(
                    np.transpose(design[mask, :, design_t]),
                    np.linalg.inv(F[:, :, 0])
                )
            )

    return _kalman_gain


def ksmooth_rep(k_states, k_endog, nobs,
            design_inp, transition_inp,
            obs_cov_inp,
            kalman_gain_inp, predicted_state_inp, predicted_state_cov_inp,
            forecasts_error_inp, forecasts_error_cov_inp):

    scaled_smoothed_estimator = (
        np.zeros((k_states, nobs+1))) #   # model.  # , dtype=kfilter.dtype)
    smoothing_error = (
        np.zeros((k_endog, nobs)))  # model.  3 , dtype=kfilter.dtype)
    scaled_smoothed_estimator_cov = (
        np.zeros((k_states, k_states, nobs+1))) # + 1 # model. dtype=kfilter.dtype

    smoothed_state = np.zeros((k_states, nobs))
    smoothed_state_cov = (
        np.zeros((k_states, k_states, nobs)))

    obs_cov_t = 0
    design_t = 0
    transition_t = 0

    for t in range(nobs-1, -1, -1):
#         print(t, t-1)
        predicted_state = predicted_state_inp[:, t] # kfilter.
        predicted_state_cov = predicted_state_cov_inp[:, :, t] # kfilter.

        design = design_inp[:, :, design_t]
        transition = transition_inp[:, :, transition_t]

        obs_cov = obs_cov_inp[:, :, obs_cov_t]  # model.
        kalman_gain = kalman_gain_inp[:, :, t]  # kfilter
        forecasts_error_cov = forecasts_error_cov_inp[:, :, t] # kfilter.
        forecasts_error = forecasts_error_inp[:, t] # kfilter.
#         k_endog = k_endog  # _kfilter.

        L = transition - kalman_gain.dot(design)

        F_inv = np.linalg.inv(forecasts_error_cov)


        # mean
        smoothing_error[:k_endog, t] = (
            F_inv.dot(forecasts_error) -
            kalman_gain.transpose().dot(
                scaled_smoothed_estimator[:, t])
        )

        scaled_smoothed_estimator[:, t - 1] = (
            design.transpose().dot(smoothing_error[:k_endog, t]) +
            transition.transpose().dot(scaled_smoothed_estimator[:, t])
        )

        scaled_smoothed_estimator_cov[:, :, t - 1] = (
            design.transpose().dot(F_inv).dot(design) +
            L.transpose().dot(
                scaled_smoothed_estimator_cov[:, :, t]
            ).dot(L)
        )

        smoothed_state[:, t] = (
            predicted_state +
            predicted_state_cov.dot(scaled_smoothed_estimator[:, t - 1])
        )

        smoothed_state_cov[:, :, t] = (
            predicted_state_cov -
            predicted_state_cov.dot(
                scaled_smoothed_estimator_cov[:, :, t - 1]
            ).dot(predicted_state_cov)
        )


    return KSResult(
        smoothing_error=smoothing_error,
        scaled_smoothed_estimator=scaled_smoothed_estimator[:,:-1],
        scaled_smoothed_estimator_cov=scaled_smoothed_estimator_cov[:,:,:-1],
        smoothed_state=smoothed_state,
        smoothed_state_cov=smoothed_state_cov
    )
