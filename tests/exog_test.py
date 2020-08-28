import numpy as np
import pytest

from simd_structts.backends.simdkalman.model import SIMDStructTS
from simd_structts.backends.statsmodels import MultiUnobservedComponents

from .utils import assert_models_equal

N = 366
H = 30
EXOG_SINGLE_TUPLE = (np.random.random((N, 1)), np.random.random((H, 1)))
EXOG_DOUBLE_TUPLE = (np.random.random((N, 2)), np.random.random((H, 2)))


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
    simd_m.initialize_fixed()
    simd_r = simd_m.smooth()

    sm_m = MultiUnobservedComponents(ts1ts2, **kwargs)
    sm_m.initialize_fixed()
    sm_r = sm_m.smooth()

    assert_models_equal(simd_r, sm_r, h=H, exog_predict=exog_predict)
