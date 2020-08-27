import numpy as np

from simd_structts.backends.simdkalman.model import SIMDStructTS
from simd_structts.backends.statsmodels import MultiUnobservedComponents

from .utils import assert_models_equal

H = 30


def test_exog(ts1ts2):

    exog_full = np.random.random((ts1ts2.shape[1] + H, 1))
    exog_predict = exog_full[-H:]

    simd_m = SIMDStructTS(
        ts1ts2,
        trend=True,
        stochastic_trend=True,
        level=True,
        stochastic_level=True,
        exog=exog_full[:-H, ...],
        #                     seasonal=4,
        #                     freq_seasonal=[{"period": 365.25, "harmonics": 2}],
        #                     autoregressive=1,
    )
    simd_m.initialize_fixed()
    # with np.printoptions(precision=2, suppress=True):
    #     print(m)
    simd_r = simd_m.smooth()

    simd_r.filtered_state.shape, simd_r.get_forecast(
        horizon=H, exog=exog_predict
    ).predicted_mean.shape

    sm_m = MultiUnobservedComponents(
        ts1ts2,
        trend=True,
        stochastic_trend=True,
        level=True,
        stochastic_level=True,
        exog=exog_full[:-H, ...],
        mle_regression=False,
        #             seasonal=4,
        #             freq_seasonal=[{"period": 365.25, "harmonics": 2}],
        #             autoregressive=1,
    )
    sm_m.initialize_fixed()
    sm_r = sm_m.smooth()

    assert_models_equal(simd_r, sm_r, h=H, exog_predict=exog_predict)
