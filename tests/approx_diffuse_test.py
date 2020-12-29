import numpy as np
import pytest

from hypothesis import given
from hypothesis import strategies as st
# from hypothesis.strategies import floats
from hypothesis.extra.numpy import arrays

from simd_structts.backends.simdkalman.model import SIMDStructTS
from simd_structts.backends.statsmodels import MultiUnobservedComponents



from .utils import assert_models_equal

N = 366
H = 30
EXOG_SINGLE_TUPLE = (np.random.random((N, 1)), np.random.random((H, 1)))
EXOG_DOUBLE_TUPLE = (np.random.random((N, 2)), np.random.random((H, 2)))


# @given(arrays(np.float, N, elements=st.floats(0, 1)),
#        arrays(np.float, (N+H, 1), elements=st.floats(0, 1)))
# def test_test(ts, exog):
#     import pdb; pdb.set_trace()
#     pass


@pytest.mark.parametrize("trend", [True, False])
@pytest.mark.parametrize("seasonal", [None, 2, 7])
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
    "exog_train,exog_predict", [(None, None), EXOG_SINGLE_TUPLE, EXOG_DOUBLE_TUPLE]
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

    simd_m = SIMDStructTS(ts1ts2, **kwargs)
    simd_m.initialize_approx_diffuse(obs_cov=1e-1, initial_state_cov=1e3)
    simd_r = simd_m.smooth()

    from statsmodels.tsa.statespace.kalman_filter import FILTER_UNIVARIATE, FILTER_CONVENTIONAL
    from statsmodels.tsa.statespace.kalman_smoother import SMOOTH_CONVENTIONAL, SMOOTH_CLASSICAL, SMOOTH_ALTERNATIVE, SMOOTH_UNIVARIATE


    # sm_m = MultiUnobservedComponents(ts1ts2, **{**kwargs, **dict(filter_method=FILTER_CONVENTIONAL)})
    # sm_m = MultiUnobservedComponents(ts1ts2, **{**kwargs, **dict(filter_method=FILTER_UNIVARIATE, smooth_method=SMOOTH_UNIVARIATE)})
    # sm_m = MultiUnobservedComponents(ts1ts2, **{**kwargs, **dict(filter_univariate=True)})
    sm_m = MultiUnobservedComponents(ts1ts2, **kwargs)
    sm_m.initialize_approx_diffuse(obs_cov=1e-1, initial_state_cov=1e3)
    sm_r = sm_m.smooth()

    m1 = simd_r
    m2 = sm_r

    simd_r._compute()

    assert_models_equal(simd_r, sm_r, h=H, exog_predict=exog_predict)
