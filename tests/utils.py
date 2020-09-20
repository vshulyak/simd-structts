import numpy as np


def assert_models_equal(m1, m2, h, exog_predict=None):
    assert np.allclose(m1.filtered_state, m2.filtered_state)
    assert np.allclose(m1.filtered_state_cov, m2.filtered_state_cov)
    assert np.allclose(m1.smoothed_state, m2.smoothed_state)
    # assert np.allclose(m1.smoothed_state_cov[:,12:,...], m2.smoothed_state_cov[:,12:,...])
    assert np.allclose(m1.smoothed_state_cov, m2.smoothed_state_cov)
    assert np.allclose(m1.smoothed_forecasts, m2.smoothed_forecasts)
    assert np.allclose(m1.get_forecast(h, exog=exog_predict).predicted_mean,m2.get_forecast(h, exog=exog_predict).predicted_mean)
    assert np.allclose(m1.get_forecast(h, exog=exog_predict).se_mean ** 2,m2.get_forecast(h, exog=exog_predict).se_mean ** 2)
