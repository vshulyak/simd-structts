import numpy as np
import pytest
from simd_structts.backends.py_ssm.model import PySSMStructTS
from simd_structts.backends.statsmodels import MultiUnobservedComponents
from statsmodels.tsa.statespace.kalman_filter import FILTER_CONVENTIONAL
from statsmodels.tsa.statespace.kalman_smoother import SMOOTH_CONVENTIONAL

# from hypothesis.strategies import floats


N = 366
H = 30
EXOG_SINGLE_TUPLE = (np.random.random((N, 1)), np.random.random((H, 1)))
EXOG_DOUBLE_TUPLE = (np.random.random((N, 2)), np.random.random((H, 2)))


# @pytest.mark.parametrize("trend", [True, ]) # False
# @pytest.mark.parametrize("seasonal", [None])  # , 2, 7
# @pytest.mark.parametrize(
#     "freq_seasonal",
#     [
#         None,
#         # [{"period": 365.25 * 2, "harmonics": 1}],
#         # [{"period": 365.25, "harmonics": 3}],
#         # [{"period": 365.25 / 2, "harmonics": 2}, {"period": 365.25, "harmonics": 1}],
#     ],
# )
# @pytest.mark.parametrize(
#     "exog_train,exog_predict", [EXOG_DOUBLE_TUPLE]  # (None, None), EXOG_SINGLE_TUPLE ,EXOG_DOUBLE_TUPLE
# )
@pytest.mark.parametrize("trend", [True, False])  #
@pytest.mark.parametrize("seasonal", [None, 2, 7])  #
@pytest.mark.parametrize(
    "freq_seasonal",
    [
        None,
        [{"period": 365.25 * 2, "harmonics": 1}],
        [{"period": 365.25, "harmonics": 3}],
        [{"period": 365.25 / 2, "harmonics": 2}, {"period": 365.25, "harmonics": 1}],
    ],
)
@pytest.mark.parametrize(
    "exog_train,exog_predict", [(None, None), EXOG_SINGLE_TUPLE, EXOG_DOUBLE_TUPLE]  #
)
def test_permute_params(
    ts1ts2, trend, seasonal, freq_seasonal, exog_train, exog_predict
):
    # NANs
    ts1ts2[:, 10:20] = np.nan

    kwargs = dict(
        level=True,
        stochastic_level=True,
        trend=trend,
        stochastic_trend=trend,
        seasonal=seasonal,
        freq_seasonal=freq_seasonal,
        exog=exog_train,
        mle_regression=False,  # MLE is always false
    )

    sm_m = MultiUnobservedComponents(
        ts1ts2,
        **{
            **kwargs,
            **dict(
                filter_method=FILTER_CONVENTIONAL, smooth_method=SMOOTH_CONVENTIONAL
            ),
        },
    )
    sm_m.initialize_approx_diffuse(obs_cov=1e-1, initial_state_cov=1e3)
    sm_r = sm_m.smooth()

    pyssm_m = PySSMStructTS(ts1ts2, **kwargs)
    pyssm_m.initialize_approx_diffuse(obs_cov=1e-1, initial_state_cov=1e3)
    pyssm_smoother = pyssm_m.smooth()

    pyssm_preds = pyssm_m.forecast(H, exog=exog_predict)
    sm_preds = sm_r.get_forecast(H, exog=exog_predict)

    def assert_filters_equal(m1, m2):
        assert np.allclose(m1.filtered_state, m2.filtered_state)
        assert np.allclose(m1.filtered_state_cov, m2.filtered_state_cov)
        assert np.allclose(m1.predicted_state, m2.predicted_state)
        assert np.allclose(m1.predicted_state_cov, m2.predicted_state_cov)
        assert np.allclose(m1.forecast_error, m2.forecast_error, equal_nan=True)
        assert np.allclose(m1.forecast_error_cov, m2.forecast_error_cov)
        assert np.allclose(m1.llf_obs, m2.llf_obs)
        # TODO: llf

    def assert_smoothers_equal(m1, m2):
        assert np.allclose(m1.smoothed_state, m2.smoothed_state)
        assert np.allclose(m1.smoothed_state_cov, m2.smoothed_state_cov)
        assert np.allclose(
            m1.smoothed_forecasts, m2.smoothed_forecasts
        )
        # assert np.allclose(m1.smoothed_forecasts_error, m2.smoother_results.smoothed_forecasts_error)
        # assert np.allclose(m1.smoothed_forecasts_error_cov, m2.smoother_results.smoothed_forecasts_error_cov)

    def assert_forecasts_equal(m1, m2, h, exog_predict=None):
        assert np.allclose(m1.predicted_mean, m2.predicted_mean)
        assert np.allclose(m1.se_mean, m2.se_mean)

    assert_filters_equal(pyssm_smoother, sm_r)
    assert_smoothers_equal(pyssm_smoother, sm_r)
    assert_forecasts_equal(pyssm_preds, sm_preds, h=H, exog_predict=exog_predict)
