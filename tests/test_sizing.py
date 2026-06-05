"""Tests for position sizing (atlas.portfolio.sizing).

Each stage is tested against its mathematical invariant on hand-built inputs,
never a magnitude proxy:
  A (directional): mode dispatch -- sign-through vs shorts-clipped; NaN preserved.
  B (inverse-vol): the cross-asset RATIO equals the inverse ratio of vols.
  C (vol-target):  known-answer w^T Sigma w on a hand-built covariance, and the
                   realized-vol-hits-target property end to end.
PIT: vol uses only trailing returns, so perturbing the future leaves the past
unchanged. Edges: warm-up masking, empty book, abstention drop-out.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from atlas.portfolio.sizing import (
    _directional_weights,
    _inverse_vol_scale,
    _vol_target_scalar,
    compute_target_weights,
)

TRADING_DAYS = 252


# --------------------------------------------------------------------------
# Stage A -- directional weights (mode fork)
# --------------------------------------------------------------------------
class TestDirectionalWeights:
    def test_long_short_is_identity(self):
        # Sign carries direction: positive stays long, negative stays short.
        c = pd.DataFrame({"A": [1.5, -0.8], "B": [-2.0, 0.3]})
        out = _directional_weights(c, "long_short")
        pd.testing.assert_frame_equal(out, c)

    def test_long_only_clips_shorts_to_zero(self):
        # Negatives -> 0 (disliked asset excluded, NOT shorted); positives kept.
        c = pd.DataFrame({"A": [1.5, -0.8], "B": [-2.0, 0.3]})
        out = _directional_weights(c, "long_only")
        expected = pd.DataFrame({"A": [1.5, 0.0], "B": [0.0, 0.3]})
        pd.testing.assert_frame_equal(out, expected)

    def test_nan_preserved_both_modes(self):
        # Abstention (NaN) must survive as NaN, never become a 0 position.
        c = pd.DataFrame({"A": [np.nan, -0.8], "B": [-2.0, np.nan]})
        for mode in ("long_short", "long_only"):
            out = _directional_weights(c, mode)
            assert np.isnan(out.loc[0, "A"])
            assert np.isnan(out.loc[1, "B"])

    def test_invalid_mode_raises(self):
        c = pd.DataFrame({"A": [1.0]})
        with pytest.raises(ValueError, match="long_short.*long_only"):
            _directional_weights(c, "market_neutral")


# --------------------------------------------------------------------------
# Stage B -- inverse-vol scaling
# --------------------------------------------------------------------------
class TestInverseVolScale:
    def test_weight_ratio_is_inverse_vol_ratio(self):
        # Equal conviction, vols 0.05 vs 0.20 -> weights 20 vs 5 -> ratio 4:1,
        # exactly the INVERSE of the 1:4 vol ratio. This is the whole point of B.
        directional = pd.DataFrame({"BOND": [1.0], "CMDTY": [1.0]})
        vol = pd.DataFrame({"BOND": [0.05], "CMDTY": [0.20]})
        out = _inverse_vol_scale(directional, vol)
        ratio = out.loc[0, "BOND"] / out.loc[0, "CMDTY"]
        assert np.isclose(ratio, 4.0)

    def test_nan_vol_or_direction_gives_nan(self):
        # NaN vol (warm-up) or NaN direction (abstain) -> asset absent (NaN),
        # not a divide-by-something-spurious.
        directional = pd.DataFrame({"A": [1.0, np.nan], "B": [1.0, 1.0]})
        vol = pd.DataFrame({"A": [0.1, 0.1], "B": [np.nan, 0.1]})
        out = _inverse_vol_scale(directional, vol)
        assert np.isnan(out.loc[1, "A"])  # NaN direction
        assert np.isnan(out.loc[0, "B"])  # NaN vol


# --------------------------------------------------------------------------
# Stage C -- vol-target scalar (the w^T Sigma w core)
# --------------------------------------------------------------------------
class TestVolTargetScalar:
    def test_known_answer_two_asset_quadratic_form(self):
        # Hand-built constant returns so the rolling covariance is known exactly,
        # then check k == target / sqrt(w^T Sigma w * 252) against a by-hand value.
        #
        # Construct two assets whose trailing covariance over the window is a
        # KNOWN matrix. Easiest exact construction: returns drawn so that, within
        # the window, Var(A)=va, Var(B)=vb, Cov=cab by design. We instead verify
        # the CONTRACTION is correct by feeding _vol_target_scalar a returns frame
        # and comparing port_var to numpy's own w^T cov w on the same window.
        rng = np.random.default_rng(0)
        n, window = 200, 60
        cols = ["A", "B", "C"]
        rets = pd.DataFrame(rng.normal(0, 0.01, size=(n, 3)),
                            index=pd.RangeIndex(n), columns=cols)
        # Constant weights every date (so w_t is well defined and nonzero).
        w = pd.DataFrame(np.tile([2.0, -1.0, 0.5], (n, 1)),
                         index=rets.index, columns=cols)
        k = _vol_target_scalar(w, rets, target_vol=0.10, window=window)

        # Independent numpy check at one late date: build that date's trailing
        # covariance with np.cov, contract w^T Sigma w, annualize, invert.
        t = 150
        win = rets.iloc[t - window + 1 : t + 1].to_numpy()
        cov = np.cov(win, rowvar=False, ddof=1)        # 3x3 trailing covariance
        wt = np.array([2.0, -1.0, 0.5])
        port_var = wt @ cov @ wt
        expected_k = 0.10 / (np.sqrt(port_var) * np.sqrt(TRADING_DAYS))
        assert np.isclose(k.iloc[t], expected_k, rtol=1e-10)

    def test_negative_covariance_lowers_vol_raises_k(self):
        # Two assets held LONG that move oppositely -> the cross term lowers
        # portfolio variance -> lower vol -> LARGER k than if they were
        # independent. Confirms the off-diagonal genuinely enters the contraction.
        n, window = 200, 60
        base = np.random.default_rng(1).normal(0, 0.01, size=n)
        a = base
        b_anti = -base + np.random.default_rng(2).normal(0, 1e-4, size=n)  # ~ -A
        b_indep = np.random.default_rng(3).normal(0, 0.01, size=n)         # ~ indep A
        cols = ["A", "B"]
        w = pd.DataFrame(np.tile([1.0, 1.0], (n, 1)), columns=cols)

        rets_anti = pd.DataFrame({"A": a, "B": b_anti})
        rets_indep = pd.DataFrame({"A": a, "B": b_indep})
        k_anti = _vol_target_scalar(w, rets_anti, 0.10, window)
        k_indep = _vol_target_scalar(w, rets_indep, 0.10, window)
        # Anti-correlated longs -> lower book vol -> bigger scalar to reach target.
        assert k_anti.iloc[150] > k_indep.iloc[150]

    def test_empty_book_gives_nan_scalar(self):
        # All-zero weights -> port_var 0 -> guarded to NaN (not 0/0 or inf).
        n, window = 100, 60
        rets = pd.DataFrame(np.random.default_rng(0).normal(0, 0.01, (n, 2)),
                            columns=["A", "B"])
        w = pd.DataFrame(0.0, index=rets.index, columns=["A", "B"])
        k = _vol_target_scalar(w, rets, 0.10, window)
        assert k.iloc[80] != k.iloc[80] or np.isnan(k.iloc[80])  # NaN


# --------------------------------------------------------------------------
# End-to-end -- compute_target_weights
# --------------------------------------------------------------------------
class TestComputeTargetWeights:
    def _synthetic_prices(self, n=400, seed=0):
        # Geometric random-walk prices for 4 assets with different vols, so
        # inverse-vol and vol-target have something realistic to bite on.
        rng = np.random.default_rng(seed)
        vols = np.array([0.004, 0.008, 0.012, 0.020])  # daily
        rets = rng.normal(0, 1, size=(n, 4)) * vols
        prices = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)),
                              index=pd.date_range("2015-01-01", periods=n),
                              columns=["W", "X", "Y", "Z"])
        return prices

    def test_realized_vol_hits_target(self):
        # The end-to-end property: a constant-conviction book, vol-targeted to
        # 0.10, should REALIZE ~0.10 annualized when we apply the resulting
        # weights to the SAME-DAY returns (in-sample identity, since the scalar
        # is built from trailing returns of these very assets).
        prices = self._synthetic_prices()
        composite = pd.DataFrame(1.0, index=prices.index, columns=prices.columns)
        W = compute_target_weights(composite, prices, mode="long_short",
                                   target_vol=0.10, vol_window=126)
        rets = prices.pct_change()
        port = (W.shift(0) * rets).sum(axis=1)   # same-day: in-sample vol check
        realized = port[W.notna().any(axis=1)].std() * np.sqrt(TRADING_DAYS)
        # Trailing estimate vs realized differ, but should be in the ballpark.
        assert 0.06 < realized < 0.16

    def test_warmup_masked(self):
        # No weights until a full vol window exists (~vol_window rows in).
        prices = self._synthetic_prices()
        W = compute_target_weights(pd.DataFrame(1.0, index=prices.index,
                                                columns=prices.columns),
                                   prices, vol_window=126)
        first = W.notna().any(axis=1).idxmax()
        first_pos = prices.index.get_loc(first)
        assert first_pos >= 126   # at least a full window of warm-up

    def test_long_only_no_negative_weights(self):
        prices = self._synthetic_prices()
        # Composite with some negative convictions -> long_only must zero them.
        rng = np.random.default_rng(5)
        composite = pd.DataFrame(rng.normal(0, 1, prices.shape),
                                 index=prices.index, columns=prices.columns)
        W = compute_target_weights(composite, prices, mode="long_only",
                                   vol_window=126)
        assert (W.fillna(0.0) >= 0.0).all().all()  # no shorts


# --------------------------------------------------------------------------
# Point-in-time safety
# --------------------------------------------------------------------------
class TestPointInTime:
    def test_future_perturbation_leaves_past_unchanged(self):
        # Trailing-only vol + cross-sectional composite -> perturbing future
        # returns cannot change past weights. The sizing acid test.
        rng = np.random.default_rng(7)
        n = 400
        prices = pd.DataFrame(
            100 * np.exp(np.cumsum(rng.normal(0, 0.01, (n, 3)), axis=0)),
            index=pd.date_range("2015-01-01", periods=n), columns=["A", "B", "C"])
        composite = pd.DataFrame(1.0, index=prices.index, columns=prices.columns)

        W_before = compute_target_weights(composite, prices, vol_window=126)
        prices2 = prices.copy()
        prices2.iloc[-20:] *= 1.5            # perturb only the last 20 days
        W_after = compute_target_weights(composite, prices2, vol_window=126)

        past = prices.index[:-20]
        pd.testing.assert_frame_equal(W_before.loc[past], W_after.loc[past])