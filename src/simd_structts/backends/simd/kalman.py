import numpy as np
from simdkalman.primitives import ddot
from simdkalman.primitives import ddot_t_right
from simdkalman.primitives import dinv


def update_with_nan_check(
    prior_mean,
    prior_covariance,
    observation_model,
    observation_noise,
    measurement,
    univariate=True,
):

    n = prior_mean.shape[1]
    m = observation_model.shape[1]

    assert measurement.shape[-2:] == (m, 1)
    assert prior_covariance.shape[-2:] == (n, n)
    assert observation_model.shape[-2:] == (m, n)
    assert observation_noise.shape[-2:] == (m, m)

    # y - H * mp
    v = measurement - ddot(observation_model, prior_mean)
    # (2, 1, 1) - (1, 1, 4) @ (2, 4, 1)

    # H * Pp * H.t + R
    S = (
        ddot(observation_model, ddot_t_right(prior_covariance, observation_model))
        + observation_noise
    )
    if univariate:
        invS = 1.0 / S
    else:
        invS = dinv(S)

    # Kalman gain: Pp * H.t * invS
    K = ddot(ddot_t_right(prior_covariance, observation_model), invS)

    # K * v + mp
    posterior_mean = ddot(K, v) + prior_mean

    # Pp - K * H * Pp
    posterior_covariance = prior_covariance - ddot(
        K, ddot(observation_model, prior_covariance)
    )

    # inv-chi2 test var
    # outlier_test = np.sum(v * ddot(invS, v), axis=0)
    l = np.ravel(ddot(v.transpose((0, 2, 1)), ddot(invS, v)))
    l += np.log(np.linalg.det(S))
    l *= -0.5

    # nan checks
    is_nan = np.ravel(np.any(np.isnan(posterior_mean), axis=1))

    posterior_mean[is_nan, ...] = prior_mean[is_nan, ...]
    posterior_covariance[is_nan, ...] = prior_covariance[is_nan, ...]
    K[is_nan, ...] = 0
    l[is_nan] = 0

    return posterior_mean, posterior_covariance, K, l
