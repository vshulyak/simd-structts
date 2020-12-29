import numpy as np
import pytest

from hypothesis import given
from hypothesis import strategies as st
# from hypothesis.strategies import floats
from hypothesis.extra.numpy import arrays

from simd_structts.backends.py_ssm.model import PySSMStructTS
from simd_structts.backends.statsmodels import MultiUnobservedComponents

from .utils import assert_models_equal

N = 366
H = 30
EXOG_SINGLE_TUPLE = (np.random.random((N, 1)), np.random.random((H, 1)))
EXOG_DOUBLE_TUPLE = (np.random.random((N, 2)), np.random.random((H, 2)))


@pytest.mark.parametrize("trend", [True, ]) # False
@pytest.mark.parametrize("seasonal", [None])  # , 2, 7
@pytest.mark.parametrize(
    "freq_seasonal",
    [
        None,
        # [{"period": 365.25 * 2, "harmonics": 1}],
        # [{"period": 365.25, "harmonics": 3}],
        # [{"period": 365.25 / 2, "harmonics": 2}, {"period": 365.25, "harmonics": 1}],
    ],
)
@pytest.mark.parametrize(
    "exog_train,exog_predict", [(None, None)]  # , EXOG_SINGLE_TUPLE, EXOG_DOUBLE_TUPLE
)
def test_permute_params(
    ts1ts2, trend, seasonal, freq_seasonal, exog_train, exog_predict
):
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

    sm_m = MultiUnobservedComponents(ts1ts2, **kwargs)
    sm_m.initialize_approx_diffuse(obs_cov=1e-1, initial_state_cov=1e3)
    sm_r = sm_m.smooth()

    pyssm_m = PySSMStructTS(ts1ts2, **kwargs)
    pyssm_m.initialize_approx_diffuse(obs_cov=1e-1, initial_state_cov=1e3)
    pyssm_smoother = pyssm_m.smooth(sm_m.models[0].ssm, sm_r.res[0])
    pyssm_filter = pyssm_m.kfilter

    # pyssm_r = pyssm_m.filter(sm_m.models[0].ssm)

    def assert_models_equal(m1, m2, h, exog_predict=None):
        assert np.allclose(m1.filtered_state, m2.filtered_state)
        assert np.allclose(m1.filtered_state_cov, m2.filtered_state_cov)
        assert np.allclose(m1.predicted_state, m2.predicted_state)
        assert np.allclose(m1.predicted_state_cov, m2.predicted_state_cov)
        assert np.allclose(m1.forecast_error, m2.forecasts_error)
        assert np.allclose(m1.forecast_error_cov, m2.forecasts_error_cov)

    assert_models_equal(pyssm_filter, sm_r.res[0], h=H, exog_predict=exog_predict)


    def assert_model_smoother_equal(m1, m2, h, exog_predict=None):
        assert np.allclose(m1.smoothed_state, m2.smoothed_state)
        assert np.allclose(m1.smoothed_state_cov, m2.smoothed_state_cov)
        # assert np.allclose(m1.smoothed_forecasts, m2.smoothed_forecasts)

    assert_model_smoother_equal(pyssm_smoother, sm_r.res[0], h=H, exog_predict=exog_predict)
