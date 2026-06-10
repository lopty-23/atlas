"""Tests for risk limits (atlas.portfolio.risk).

Caps are ABSOLUTE thresholds (max_X * max_gross): always feasible, independent
of book concentration. Invariants checked: each cap holds exactly; caps only
reduce |w|; within-bucket and whole-book ratios preserved by proportional
scaling; coverage guard fires; NaN preserved; PIT (per-date, no cross-date leak).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from atlas.portfolio.risk import (
    _verify_bucket_coverage,
    _cap_positions,
    _cap_buckets,
    _cap_leverage,
    apply_risk_limits,
)

BUCKETS = {"eq": ["A", "B"], "rates": ["C", "D"]}


class TestBucketCoverage:
    def test_passes_when_all_assets_bucketed(self):
        _verify_bucket_coverage(pd.Index(["A", "B", "C", "D"]), BUCKETS)  # no raise

    def test_raises_on_unbucketed_asset(self):
        with pytest.raises(ValueError, match="not assigned"):
            _verify_bucket_coverage(pd.Index(["A", "B", "C", "D", "E"]), BUCKETS)

    def test_raises_on_double_bucketed_asset(self):
        bad = {"eq": ["A", "B"], "rates": ["B", "C", "D"]}  # B in two buckets
        with pytest.raises(ValueError, match="more than one"):
            _verify_bucket_coverage(pd.Index(["A", "B", "C", "D"]), bad)

    def test_extra_config_ticker_ignored(self):
        extra = {"eq": ["A", "B", "Z"], "rates": ["C", "D"]}  # Z not traded
        _verify_bucket_coverage(pd.Index(["A", "B", "C", "D"]), extra)  # no raise


class TestCapPositions:
    def test_trims_to_absolute_limit_both_signs(self):
        w = pd.DataFrame({"A": [0.9, -0.9], "B": [0.3, -0.1]})
        out = _cap_positions(w, position_limit=0.6)
        assert out.loc[0, "A"] == 0.6   # long trimmed
        assert out.loc[1, "A"] == -0.6  # short trimmed
        assert out.loc[0, "B"] == 0.3   # under cap, untouched

    def test_only_reduces(self):
        w = pd.DataFrame({"A": [0.9], "B": [-0.7]})
        out = _cap_positions(w, 0.6)
        assert (out.abs() <= w.abs() + 1e-12).all().all()

    def test_single_name_book_is_capped(self):
        # The long-only degeneracy: one name = whole book. Absolute cap still bites.
        w = pd.DataFrame({"A": [1.0], "B": [0.0]})
        out = _cap_positions(w, 0.6)
        assert out.loc[0, "A"] == 0.6

    def test_nan_preserved(self):
        w = pd.DataFrame({"A": [np.nan, 0.9], "B": [0.3, np.nan]})
        out = _cap_positions(w, 0.6)
        assert np.isnan(out.loc[0, "A"]) and np.isnan(out.loc[1, "B"])


class TestCapBuckets:
    def test_scales_overcap_bucket_to_limit(self):
        # eq = A+B = 1.0 + 0.5 = 1.5 abs; cap 1.0 -> factor 1/1.5, eq sums to 1.0.
        w = pd.DataFrame({"A": [1.0], "B": [0.5], "C": [0.2], "D": [0.1]})
        out = _cap_buckets(w, BUCKETS, bucket_limit=1.0)
        assert np.isclose(out.loc[0, ["A", "B"]].abs().sum(), 1.0)

    def test_preserves_within_bucket_ratio(self):
        # A:B was 2:1 -> must stay 2:1 after scaling (size shrinks, ranking kept).
        w = pd.DataFrame({"A": [1.0], "B": [0.5], "C": [0.2], "D": [0.1]})
        out = _cap_buckets(w, BUCKETS, bucket_limit=1.0)
        assert np.isclose(out.loc[0, "A"] / out.loc[0, "B"], 2.0)

    def test_undercap_bucket_untouched(self):
        # rates = C+D = 0.3 < cap 1.0 -> unchanged.
        w = pd.DataFrame({"A": [1.0], "B": [0.5], "C": [0.2], "D": [0.1]})
        out = _cap_buckets(w, BUCKETS, bucket_limit=1.0)
        pd.testing.assert_series_equal(out.loc[:, ["C", "D"]].iloc[0],
                                       w.loc[:, ["C", "D"]].iloc[0])

    def test_handles_shorts_via_absolute_sum(self):
        # eq = |+0.8| + |-0.8| = 1.6 abs; cap 1.0 -> both scale by 1/1.6.
        w = pd.DataFrame({"A": [0.8], "B": [-0.8], "C": [0.0], "D": [0.0]})
        out = _cap_buckets(w, BUCKETS, bucket_limit=1.0)
        assert np.isclose(out.loc[0, ["A", "B"]].abs().sum(), 1.0)
        assert out.loc[0, "A"] > 0 and out.loc[0, "B"] < 0  # signs preserved


class TestCapLeverage:
    def test_scales_book_to_max_gross(self):
        w = pd.DataFrame({"A": [2.0], "B": [-1.5], "C": [1.0], "D": [0.5]})  # gross 5.0
        out = _cap_leverage(w, max_gross=3.0)
        assert np.isclose(out.abs().sum(axis=1).iloc[0], 3.0)

    def test_preserves_all_ratios(self):
        w = pd.DataFrame({"A": [2.0], "B": [-1.5], "C": [1.0], "D": [0.5]})
        out = _cap_leverage(w, 3.0)
        # uniform scaling -> every pairwise ratio unchanged.
        assert np.isclose(out.loc[0, "A"] / out.loc[0, "C"], 2.0 / 1.0)

    def test_undercap_book_untouched(self):
        w = pd.DataFrame({"A": [1.0], "B": [-0.5], "C": [0.3], "D": [0.1]})  # gross 1.9
        out = _cap_leverage(w, 3.0)
        pd.testing.assert_frame_equal(out, w)


class TestApplyRiskLimits:
    def _book(self):
        # A is huge (hits per-asset), eq bucket also over, gross also over -> all
        # three caps engage in one book.
        return pd.DataFrame({
            "A": [2.0], "B": [1.0], "C": [0.5], "D": [-0.3],
        })

    def test_all_caps_satisfied_simultaneously(self):
        w = self._book()
        out = apply_risk_limits(w, BUCKETS, max_position=0.20,
                                max_bucket=0.50, max_gross=3.0)
        pos_cap, buc_cap = 0.20 * 3.0, 0.50 * 3.0     # 0.60, 1.50
        assert (out.abs() <= pos_cap + 1e-9).all().all()          # per-asset
        for cols in BUCKETS.values():
            assert (out[cols].abs().sum(axis=1) <= buc_cap + 1e-9).all()  # bucket
        assert (out.abs().sum(axis=1) <= 3.0 + 1e-9).all()        # leverage

    def test_only_reduces(self):
        w = self._book()
        out = apply_risk_limits(w, BUCKETS)
        assert (out.abs() <= w.abs() + 1e-9).all().all()

    def test_coverage_guard_runs_first(self):
        w = pd.DataFrame({"A": [1.0], "X": [1.0]})  # X unbucketed
        with pytest.raises(ValueError, match="not assigned"):
            apply_risk_limits(w, BUCKETS)


class TestPointInTime:
    def test_each_date_independent(self):
        # Caps are per-date; row 0 must be unaffected by row 1's values.
        rng = np.random.default_rng(0)
        w = pd.DataFrame(rng.normal(0, 1, (50, 4)), columns=["A", "B", "C", "D"])
        out_full = apply_risk_limits(w, BUCKETS)
        w2 = w.copy()
        w2.iloc[25:] *= 5.0                  # perturb only the second half
        out_pert = apply_risk_limits(w2, BUCKETS)
        pd.testing.assert_frame_equal(out_full.iloc[:25], out_pert.iloc[:25])