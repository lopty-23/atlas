"""Tests for signal blending (atlas.portfolio.blend).

Blend is a pure per-date cross-sectional weighted average over the signals
PRESENT at each cell. The behaviors that can break: (1) renormalization over
present signals (divide by present-weight, not fixed N) -- the .sum()-drops-NaN
trap; (2) the scale divisor (shape-preserving rescale). Tests assert the
mathematical invariant, never a hardcoded proxy. PIT: blend is cross-sectional
per date, so perturbing the future must not change any past cell.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from atlas.portfolio.blend import blend_signals


def _frame(rows: list[list[float]], dates: pd.DatetimeIndex, assets: list[str]) -> pd.DataFrame:
    return pd.DataFrame(rows, index=dates, columns=assets)


DATES = pd.to_datetime(["2020-01-01", "2020-01-02"])
ASSETS = ["X", "Y"]


class TestRenormalization:
    """The core behavior: divide by the weight of PRESENT signals, not fixed N."""

    def test_single_present_signal_equals_that_signal(self):
        # Y on d1 has only A present -> composite must equal A's value exactly,
        # NOT A/2 (which is what dividing by fixed N=2 would give). This is the
        # exact bug class: thin/early cells silently down-weighted.
        a = _frame([[1.0, 2.0], [3.0, 4.0]], DATES, ASSETS)
        b = _frame([[10.0, np.nan], [np.nan, 40.0]], DATES, ASSETS)
        out = blend_signals({"A": a, "B": b})
        assert np.isclose(out.loc[DATES[0], "Y"], 2.0)   # A only, not 2.0/2
        assert np.isclose(out.loc[DATES[1], "X"], 3.0)   # A only, not 3.0/2

    def test_all_present_is_plain_weighted_mean(self):
        # Both present everywhere, equal weight -> simple mean of the two.
        a = _frame([[1.0, 2.0], [3.0, 4.0]], DATES, ASSETS)
        b = _frame([[3.0, 6.0], [5.0, 8.0]], DATES, ASSETS)
        out = blend_signals({"A": a, "B": b})
        assert np.allclose(out.values, [[2.0, 4.0], [4.0, 6.0]])

    def test_absent_signal_excluded_not_treated_as_zero(self):
        # If B (absent on d1,Y) were treated as 0 instead of excluded, the
        # composite would be (2 + 0)/2 = 1.0. Correct exclusion -> 2.0.
        a = _frame([[1.0, 2.0], [3.0, 4.0]], DATES, ASSETS)
        b = _frame([[3.0, np.nan], [5.0, 8.0]], DATES, ASSETS)
        out = blend_signals({"A": a, "B": b})
        assert np.isclose(out.loc[DATES[0], "Y"], 2.0)

    def test_unequal_weights_renormalize_over_present(self):
        # d1,Y: only A present, weight 0.25 -> divide by 0.25, not by 0.25+0.75.
        # Result must still be A's value (1.0), proving present-weight division.
        a = _frame([[1.0, 1.0], [1.0, 1.0]], DATES, ASSETS)
        b = _frame([[2.0, np.nan], [2.0, 2.0]], DATES, ASSETS)
        out = blend_signals({"A": a, "B": b}, weights={"A": 0.25, "B": 0.75})
        # d1,X both present: (0.25*1 + 0.75*2)/(1.0) = 1.75
        assert np.isclose(out.loc[DATES[0], "X"], 1.75)
        # d1,Y only A: (0.25*1)/0.25 = 1.0  (NOT 0.25*1 / 1.0 = 0.25)
        assert np.isclose(out.loc[DATES[0], "Y"], 1.0)


class TestScaling:
    """The divisor: a pure change of units applied before blending."""

    def test_scale_divides_before_blend(self):
        # B scaled by 2: 10->5, 40->20. d1,X = mean(1, 5) = 3.0.
        a = _frame([[1.0, 2.0], [3.0, 4.0]], DATES, ASSETS)
        b = _frame([[10.0, 20.0], [30.0, 40.0]], DATES, ASSETS)
        out = blend_signals({"A": a, "B": b}, scales={"B": 2.0})
        assert np.isclose(out.loc[DATES[0], "X"], 3.0)   # mean(1, 10/2)

    def test_scale_preserves_shape_ratio(self):
        # Dividing by a constant preserves the ratio between two dates of the
        # same signal -- the shape-preservation property (regime info kept).
        a = _frame([[2.0, 2.0], [4.0, 4.0]], DATES, ASSETS)  # d2 is 2x d1
        out_raw = blend_signals({"A": a})
        out_scaled = blend_signals({"A": a}, scales={"A": 2.0})
        r_raw = out_raw.loc[DATES[1], "X"] / out_raw.loc[DATES[0], "X"]
        r_scaled = out_scaled.loc[DATES[1], "X"] / out_scaled.loc[DATES[0], "X"]
        assert np.isclose(r_raw, r_scaled)   # ratio unchanged by scaling

    def test_scale_does_not_revive_nan(self):
        # NaN / scale stays NaN -- rescaling never turns abstention into a value.
        a = _frame([[1.0, 2.0], [3.0, 4.0]], DATES, ASSETS)
        b = _frame([[np.nan, 20.0], [30.0, 40.0]], DATES, ASSETS)
        out = blend_signals({"A": a, "B": b}, scales={"B": 2.0})
        assert np.isclose(out.loc[DATES[0], "X"], 1.0)   # B absent -> A only


class TestInvariant:
    """Composite must lie within [min, max] of the present scaled components."""

    def test_composite_bounded_by_components(self):
        rng = np.random.default_rng(0)
        dates = pd.date_range("2020-01-01", periods=50)
        assets = ["A1", "A2", "A3", "A4"]
        a = pd.DataFrame(rng.normal(size=(50, 4)), index=dates, columns=assets)
        b = pd.DataFrame(rng.normal(size=(50, 4)), index=dates, columns=assets)
        b.iloc[::3] = np.nan          # B abstains on some dates
        scales = {"B": 2.0}
        out = blend_signals({"A": a, "B": b}, scales=scales)
        lo = pd.concat([a, b / 2.0]).groupby(level=0).min()
        hi = pd.concat([a, b / 2.0]).groupby(level=0).max()
        valid = out.notna()
        # Every composite cell within [min, max] of its present scaled components.
        assert (out[valid] >= lo[valid] - 1e-9).all().all()
        assert (out[valid] <= hi[valid] + 1e-9).all().all()


class TestPointInTime:
    """Cross-sectional per date -> perturbing the future cannot change the past."""

    def test_future_perturbation_leaves_past_unchanged(self):
        rng = np.random.default_rng(1)
        dates = pd.date_range("2020-01-01", periods=30)
        assets = ["A1", "A2", "A3"]
        a = pd.DataFrame(rng.normal(size=(30, 3)), index=dates, columns=assets)
        b = pd.DataFrame(rng.normal(size=(30, 3)), index=dates, columns=assets)
        out_before = blend_signals({"A": a, "B": b})
        a2 = a.copy()
        a2.iloc[-5:] += 100.0          # perturb only the last 5 dates
        out_after = blend_signals({"A": a2, "B": b})
        # All dates BEFORE the perturbation must be identical.
        past = dates[:-5]
        pd.testing.assert_frame_equal(out_before.loc[past], out_after.loc[past])


class TestEdgeCases:
    def test_single_signal_passthrough_when_unscaled(self):
        a = _frame([[1.0, 2.0], [3.0, 4.0]], DATES, ASSETS)
        out = blend_signals({"A": a})
        pd.testing.assert_frame_equal(out, a)

    def test_all_nan_cell_is_nan(self):
        a = _frame([[np.nan, 2.0], [3.0, 4.0]], DATES, ASSETS)
        b = _frame([[np.nan, 6.0], [5.0, 8.0]], DATES, ASSETS)
        out = blend_signals({"A": a, "B": b})
        assert np.isnan(out.loc[DATES[0], "X"])   # no signal present -> NaN

    def test_disjoint_assets_union_grid(self):
        # A covers X, B covers Y -> union grid has both; each cell has one signal.
        a = _frame([[1.0], [2.0]], DATES, ["X"])
        b = _frame([[3.0], [4.0]], DATES, ["Y"])
        out = blend_signals({"A": a, "B": b})
        assert list(out.columns) == ["X", "Y"]
        assert np.isclose(out.loc[DATES[0], "X"], 1.0)   # A only
        assert np.isclose(out.loc[DATES[0], "Y"], 3.0)   # B only

    def test_default_weights_are_equal(self):
        # No weights passed -> equal weight -> plain mean where both present.
        a = _frame([[1.0, 2.0], [3.0, 4.0]], DATES, ASSETS)
        b = _frame([[3.0, 4.0], [5.0, 6.0]], DATES, ASSETS)
        out = blend_signals({"A": a, "B": b})
        assert np.allclose(out.values, [[2.0, 3.0], [4.0, 5.0]])