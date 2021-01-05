from dataclasses import dataclass

import numpy as np


@dataclass
class KSResult:
    smoothing_error: np.ndarray
    scaled_smoothed_estimator: np.ndarray
    scaled_smoothed_estimator_cov: np.ndarray
    smoothed_state: np.ndarray
    smoothed_state_cov: np.ndarray


def get_kalman_gain(
    k_states,
    k_endog,
    nobs,
    dtype,
    nmissing,
    design,
    transition,
    predicted_state_cov,
    missing,
    forecasts_error_cov,
):
    """Kalman gain matrices."""
    # k x n
    _kalman_gain = np.zeros((k_states, k_endog, nobs), dtype=dtype)

    for t in range(nobs):
        # In the case of entirely missing observations, let the Kalman
        # gain be zeros.
        if nmissing[t] == k_endog:
            continue

        design_t = 0 if design.shape[2] == 1 else t
        transition_t = 0 if transition.shape[2] == 1 else t
        if nmissing[t] == 0:
            # non-missing
            _kalman_gain[:, :, t] = np.dot(
                np.dot(transition[:, :, transition_t], predicted_state_cov[:, :, t]),
                np.dot(
                    np.transpose(design[:, :, design_t]),
                    np.linalg.inv(forecasts_error_cov[:, :, t]),
                ),
            )
        else:
            # missing
            mask = ~missing[:, t].astype(bool)
            F = forecasts_error_cov[np.ix_(mask, mask, [t])]
            _kalman_gain[:, mask, t] = np.dot(
                np.dot(transition[:, :, transition_t], predicted_state_cov[:, :, t]),
                np.dot(
                    np.transpose(design[mask, :, design_t]), np.linalg.inv(F[:, :, 0])
                ),
            )

    return _kalman_gain


def get_smoothed_forecasts(
    endog,
    smoothed_state,
    smoothed_state_cov,
    nobs,
    design,
    obs_cov,
    obs_intercept,
    missing,
    nmissing,
    forecasts,
    forecasts_error,
    forecasts_error_cov,
    dtype,
):
    # Initialize empty arrays
    smoothed_forecasts = np.zeros(forecasts.shape, dtype)
    smoothed_forecasts_error = np.zeros(forecasts_error.shape, dtype=dtype)
    smoothed_forecasts_error_cov = np.zeros(forecasts_error_cov.shape, dtype=dtype)

    for t in range(nobs):
        design_t = 0 if design.shape[2] == 1 else t
        obs_cov_t = 0 if obs_cov.shape[2] == 1 else t
        obs_intercept_t = 0 if obs_intercept.shape[1] == 1 else t

        mask = ~missing[:, t].astype(bool)
        # We can recover forecasts
        smoothed_forecasts[:, t] = (
            np.dot(design[:, :, design_t], smoothed_state[:, t])
            + obs_intercept[:, obs_intercept_t]
        )
        if nmissing[t] > 0:
            smoothed_forecasts_error[:, t] = np.nan
        smoothed_forecasts_error[mask, t] = endog[mask, t] - smoothed_forecasts[mask, t]
        smoothed_forecasts_error_cov[:, :, t] = (
            np.dot(
                np.dot(design[:, :, design_t], smoothed_state_cov[:, :, t]),
                design[:, :, design_t].T,
            )
            + obs_cov[:, :, obs_cov_t]
        )

    return (smoothed_forecasts, smoothed_forecasts_error, smoothed_forecasts_error_cov)


def ksmooth_rep(
    k_states,
    k_endog,
    nobs,
    design_inp,
    transition_inp,
    obs_cov_inp,
    kalman_gain_inp,
    predicted_state_inp,
    predicted_state_cov_inp,
    forecasts_error_inp,
    forecasts_error_cov_inp,
    nmissing,
    missing,
):

    scaled_smoothed_estimator = np.zeros(
        (k_states, nobs + 1)
    )  #   # model.  # , dtype=kfilter.dtype)
    smoothing_error = np.zeros((k_endog, nobs))  # model.  3 , dtype=kfilter.dtype)
    scaled_smoothed_estimator_cov = np.zeros(
        (k_states, k_states, nobs + 1)
    )  # + 1 # model. dtype=kfilter.dtype

    smoothed_state = np.zeros((k_states, nobs))
    smoothed_state_cov = np.zeros((k_states, k_states, nobs))

    # obs_cov_t = 0
    # design_t = 0
    # transition_t = 0

    # missing = np.isnan(model.obs).astype(np.int32)  # same dim as endog
    # nmissing = missing.sum(axis=0)  # (nobs) shape, sum of all missing accross missing axis

    for t in range(nobs - 1, -1, -1):

        # Get the appropriate (possibly time-varying) indices
        design_t = 0 if design_inp.shape[2] == 1 else t
        obs_cov_t = 0 if obs_cov_inp.shape[2] == 1 else t
        transition_t = 0 if transition_inp.shape[2] == 1 else t
        # selection_t = 0 if selection_inp.shape[2] == 1 else t
        # state_cov_t = 0 if kfilter.state_cov.shape[2] == 1 else t

        predicted_state = predicted_state_inp[:, t]  # kfilter.
        predicted_state_cov = predicted_state_cov_inp[:, :, t]  # kfilter.

        missing_entire_obs = nmissing[t] == k_endog
        missing_partial_obs = not missing_entire_obs and nmissing[t] > 0

        mask = ~missing[:, t].astype(bool)
        if missing_partial_obs:
            raise
            # design = np.array(
            #     _kfilter.selected_design[:k_endog*model.k_states], copy=True
            # ).reshape(k_endog, model.k_states, order='F')
            # obs_cov = np.array(
            #     _kfilter.selected_obs_cov[:k_endog**2], copy=True
            # ).reshape(k_endog, k_endog)
            # kalman_gain = kalman_gain_inp[:, mask, t]
            #
            # forecasts_error_cov = np.array(
            #     _forecasts_error_cov_inp[:, :, t], copy=True
            #     ).ravel(order='F')[:k_endog**2].reshape(k_endog, k_endog)
            # forecasts_error = np.array(
            #     forecasts_error_inp[:k_endog, t], copy=True)
            # F_inv = np.linalg.inv(forecasts_error_cov)
        else:
            if missing_entire_obs:
                design = np.zeros(design_inp.shape[:-1])
            else:
                design = design_inp[:, :, design_t]
            obs_cov = obs_cov_inp[:, :, obs_cov_t]
            kalman_gain = kalman_gain_inp[:, :, t]
            forecasts_error_cov = forecasts_error_cov_inp[:, :, t]
            forecasts_error = forecasts_error_inp[:, t]
            F_inv = np.linalg.inv(forecasts_error_cov)

        transition = transition_inp[:, :, transition_t]
        L = transition - kalman_gain.dot(design)

        if missing_entire_obs:
            # smoothing_error is undefined here, keep it as zeros
            scaled_smoothed_estimator[:, t - 1] = transition.transpose().dot(
                scaled_smoothed_estimator[:, t]
            )
        else:
            smoothing_error[:k_endog, t] = F_inv.dot(
                forecasts_error
            ) - kalman_gain.transpose().dot(scaled_smoothed_estimator[:, t])

            scaled_smoothed_estimator[:, t - 1] = design.transpose().dot(
                smoothing_error[:k_endog, t]
            ) + transition.transpose().dot(scaled_smoothed_estimator[:, t])

        if missing_entire_obs:
            scaled_smoothed_estimator_cov[:, :, t - 1] = (
                L.transpose().dot(scaled_smoothed_estimator_cov[:, :, t]).dot(L)
            )
        else:
            scaled_smoothed_estimator_cov[:, :, t - 1] = design.transpose().dot(
                F_inv
            ).dot(design) + L.transpose().dot(
                scaled_smoothed_estimator_cov[:, :, t]
            ).dot(
                L
            )

        smoothed_state[:, t] = predicted_state + predicted_state_cov.dot(
            scaled_smoothed_estimator[:, t - 1]
        )

        smoothed_state_cov[:, :, t] = predicted_state_cov - predicted_state_cov.dot(
            scaled_smoothed_estimator_cov[:, :, t - 1]
        ).dot(predicted_state_cov)

    return KSResult(
        smoothing_error=smoothing_error,
        scaled_smoothed_estimator=scaled_smoothed_estimator[:, :-1],
        scaled_smoothed_estimator_cov=scaled_smoothed_estimator_cov[:, :, :-1],
        smoothed_state=smoothed_state,
        smoothed_state_cov=smoothed_state_cov,
    )
