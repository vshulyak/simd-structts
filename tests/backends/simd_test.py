import numpy as np
import pytest
from simd_structts.backends.simd.model import SIMDStructTS
from simd_structts.backends.statsmodels.model import MultiUnobservedComponents
from statsmodels.tsa.statespace.kalman_filter import FILTER_CONVENTIONAL
from statsmodels.tsa.statespace.kalman_smoother import SMOOTH_CONVENTIONAL
from simd_structts.test_utils import assert_filters_equal
from simd_structts.test_utils import assert_forecasts_equal
from simd_structts.test_utils import assert_smoothers_equal

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

    simd_m = SIMDStructTS(ts1ts2, **kwargs)
    simd_m.initialize_approx_diffuse(
        obs_cov=OBS_COV, initial_state_cov=INITIAL_STATE_COV
    )
    simd_r = simd_m.smooth()
    simd_preds = simd_r.get_forecast(H, exog=exog_predict)

    assert_filters_equal(simd_r, sm_r)
    assert_smoothers_equal(simd_r, sm_r)
    assert_forecasts_equal(simd_preds, sm_preds)
