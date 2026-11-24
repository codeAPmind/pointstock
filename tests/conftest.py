import numpy as np
import pandas as pd
import pytest


def synthetic_ohlcv(n: int = 600, seed: int = 1, start: str = "2022-01-03") -> pd.DataFrame:
    """Random-walk OHLCV on business days, with realistic high/low/open noise."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n, name="date")
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.02, n)))
    open_ = close * np.exp(rng.normal(0, 0.008, n))
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 0.01, n)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 0.01, n)))
    vol = rng.lognormal(15, 0.4, n)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol}, index=idx)


@pytest.fixture
def ohlcv():
    return synthetic_ohlcv()


@pytest.fixture
def market():
    return {"SPY": synthetic_ohlcv(seed=2), "QQQ": synthetic_ohlcv(seed=3),
            "VIX": synthetic_ohlcv(seed=4), "SOXX": synthetic_ohlcv(seed=5)}
