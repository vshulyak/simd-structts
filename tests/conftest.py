import numpy as np
import pandas as pd
import pytest
np.random.seed(0)

def seasonality(n, s_len=24):
    freq = n / s_len

    t = np.arange(n) / n
    c1 = 1.0 * np.sin(2 * np.pi * t * freq)
    c2 = 0.4 * np.sin(2 * np.pi * 15 * t)

    noise = np.random.rand(n)

    return c1 + c2 + noise


def trend(n, steepness=1.2):
    return np.arange(n) / (n ** steepness) + np.random.rand(n) * 0.1


def create_data(first_date, last_date, level=5, period1=7, period2=365, steepness=0.9):

    index = pd.date_range(first_date, last_date, freq="D")
    ts = (
        level
        + trend(index.shape[0], steepness=steepness)
        + seasonality(index.shape[0], period1)
        + seasonality(index.shape[0], period2)
    )
    return pd.Series(ts, index=index)


# ts1 = create_data("2019-01-01", "2020-01-01", level=50, steepness=0.8)
# ts2 = create_data("2019-01-01", "2020-01-01", level=100, steepness=2)

#
# @pytest.fixture(scope="module")
# def ts1():
#     return create_data("2019-01-01", "2020-01-01", level=50, steepness=0.8)
#
#
# @pytest.fixture(scope="module")
# def ts2():
#     return create_data("2019-01-01", "2020-01-01", level=100, steepness=2)
#


@pytest.fixture(scope="module")
def ts1ts2():
    return np.expand_dims(
        np.stack(
            [
                create_data("2019-01-01", "2020-01-01", level=50, steepness=0.8),
                create_data("2019-01-01", "2020-01-01", level=100, steepness=2),
            ]
        ),
        2,
    )
