import numpy as np
import pytest
from simd_structts.backends.simd.model import RawStructTS
from simd_structts.backends.statsmodels import MultiUnobservedComponents
from statsmodels.tsa.statespace.kalman_filter import FILTER_CONVENTIONAL
from statsmodels.tsa.statespace.kalman_smoother import SMOOTH_CONVENTIONAL


N = 366
H = 30
N_SERIES = 2

EXOG_SINGLE_TUPLE = (np.random.random((N, 1)), np.random.random((H, 1)))
EXOG_DOUBLE_TUPLE = (np.random.random((N, 2)), np.random.random((H, 2)))
EXOG_INDIVIDUAL_TRIPLE_TUPLE = (
    np.random.random((N_SERIES, N, 3)),
    np.random.random((N_SERIES, H, 3)),
)

OBS_COV = 1e-1
INITIAL_STATE_COV = 1e3
#
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
#     "exog_train,exog_predict", [EXOG_INDIVIDUAL_TRIPLE_TUPLE]  #(None, None),  EXOG_SINGLE_TUPLE ,EXOG_DOUBLE_TUPLE, EXOG_INDIVIDUAL_TRIPLE_TUPLE
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
    "exog_train,exog_predict",
    [
        (None, None),
        EXOG_SINGLE_TUPLE,
        EXOG_DOUBLE_TUPLE,
        EXOG_INDIVIDUAL_TRIPLE_TUPLE,
    ],  #
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
    sm_m.initialize_approx_diffuse(obs_cov=OBS_COV, initial_state_cov=INITIAL_STATE_COV)
    sm_r = sm_m.smooth()
    sm_preds = sm_r.get_forecast(H, exog=exog_predict)

    # simd_m = SIMDStructTS(ts1ts2, **kwargs)
    # simd_m.initialize_approx_diffuse(
    #     obs_cov=OBS_COV, initial_state_cov=INITIAL_STATE_COV
    # )
    # simd_r = simd_m.smooth()
    # simd_preds = simd_r.get_forecast(H, exog=exog_predict)

    raw_m = RawStructTS(ts1ts2, **kwargs)
    raw_m.initialize_approx_diffuse(
        obs_cov=OBS_COV, initial_state_cov=INITIAL_STATE_COV
    )
    raw_smoother = raw_m.filter()
    raw_preds = raw_smoother.get_forecast(H, exog=exog_predict)

    def assert_filters_equal(m1, m2):
        assert np.allclose(m1.filtered_state, m2.filtered_state)
        assert np.allclose(m1.filtered_state_cov, m2.filtered_state_cov)
        assert np.allclose(m1.predicted_state, m2.predicted_state)
        assert np.allclose(m1.predicted_state_cov, m2.predicted_state_cov)
        assert np.allclose(m1.forecast, m2.forecast)
        assert np.allclose(m1.forecast_error, m2.forecast_error, equal_nan=True)
        assert np.allclose(m1.forecast_error_cov, m2.forecast_error_cov)
        # assert np.allclose(m1.llf_obs, m2.llf_obs)
        # TODO: llf

    assert_filters_equal(raw_smoother, sm_r)

    def assert_forecasts_equal(m1, m2):
        assert np.allclose(m1.predicted_mean, m2.predicted_mean)
        assert np.allclose(m1.se_mean, m2.se_mean)

    # assert_filters_equal(raw_smoother, sm_r)
    # assert_smoothers_equal(pyssm_smoother, sm_r)
    assert_forecasts_equal(raw_preds, sm_preds)
