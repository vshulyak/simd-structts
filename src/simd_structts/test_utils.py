import numpy as np


def assert_filters_equal(m1, m2):
    assert np.allclose(m1.filtered_state, m2.filtered_state)
    assert np.allclose(m1.filtered_state_cov, m2.filtered_state_cov)
    assert np.allclose(m1.predicted_state, m2.predicted_state)
    assert np.allclose(m1.predicted_state_cov, m2.predicted_state_cov)
    # assert np.allclose(m1.forecast_error, m2.forecast_error, equal_nan=True)
    # assert np.allclose(m1.forecast_error_cov, m2.forecast_error_cov)
    # assert np.allclose(m1.llf_obs, m2.llf_obs)
    # TODO: llf


def assert_smoothers_equal(m1, m2):
    assert np.allclose(m1.smoothed_state, m2.smoothed_state)
    assert np.allclose(m1.smoothed_state_cov, m2.smoothed_state_cov)
    assert np.allclose(m1.smoothed_forecasts, m2.smoothed_forecasts)
    # assert np.allclose(m1.smoothed_forecasts_error, m2.smoother_results.smoothed_forecasts_error)
    # assert np.allclose(m1.smoothed_forecasts_error_cov, m2.smoother_results.smoothed_forecasts_error_cov)


def assert_forecasts_equal(m1, m2):
    assert np.allclose(m1.predicted_mean, m2.predicted_mean)
    assert np.allclose(m1.se_mean, m2.se_mean)
