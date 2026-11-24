"""No look-ahead: features on day t use only data known at the close of t."""
import numpy as np
import pandas as pd
import pytest

from premarket.features import TARGET_COLS, build_features, feature_columns


def test_shift_by_one_day_changes_every_feature_on_last_row(ohlcv, market):
    """DESIGN.md test: dropping the last day must change every feature on the last row.

    If a feature on the last row did not move, it would be ignoring that day's
    data; if it were computed from tomorrow, truncation would break it (see below).
    """
    full = build_features(ohlcv, market, sector="SOXX")
    shifted = build_features(ohlcv.iloc[:-1], market, sector="SOXX")
    cols = feature_columns(full)
    a, b = full[cols].iloc[-1], shifted[cols].iloc[-1]
    assert a.notna().all(), f"NaN features on last row: {list(a[a.isna()].index)}"
    unchanged = [c for c in cols if np.isclose(a[c], b[c])]
    assert not unchanged, f"features ignore the latest day: {unchanged}"


@pytest.mark.parametrize("cut", [250, 400, 599])
def test_truncation_invariance(ohlcv, market, cut):
    """Features for day t are identical whether or not later days exist."""
    full = build_features(ohlcv, market, sector="SOXX")
    trunc = build_features(ohlcv.iloc[:cut], market, sector="SOXX")
    cols = feature_columns(full)
    pd.testing.assert_frame_equal(full[cols].iloc[:cut], trunc[cols], check_exact=False, rtol=1e-10)


def test_future_market_data_is_not_used(ohlcv, market):
    """Market context rows after the stock's last date must not leak in."""
    short = ohlcv.iloc[:300]
    a = build_features(short, {k: v.iloc[:300] for k, v in market.items()}, sector="SOXX")
    b = build_features(short, market, sector="SOXX")
    cols = feature_columns(a)
    pd.testing.assert_frame_equal(a[cols], b[cols])


def test_targets_are_forward_and_last_row_unknown(ohlcv):
    t = build_features(ohlcv)
    assert set(TARGET_COLS) <= set(t.columns)
    assert t[TARGET_COLS].iloc[-1].isna().all()
    logc = np.log(ohlcv["Close"])
    assert np.isclose(t["next_ret"].iloc[10], logc.iloc[11] - logc.iloc[10])
    assert t["target_up"].iloc[10] == float(ohlcv["Close"].iloc[11] > ohlcv["Close"].iloc[10])
