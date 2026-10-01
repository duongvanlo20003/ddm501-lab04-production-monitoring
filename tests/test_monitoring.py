"""
Tests for the drift and fairness computation.

PSI is a formula, so it can be tested against cases where the answer is known
in advance — which is the only way to be sure a drift monitor that reports
0.0 is reporting "no drift" rather than "not working".
"""

import numpy as np
import pytest

from app.monitoring import (
    EPSILON,
    MONITORED_FEATURES,
    MonitoringWindow,
    ReferenceDistribution,
    drift_status,
    population_stability_index,
)

EDGES = [-np.inf, 0.25, 0.5, 0.75, np.inf]
UNIFORM = [0.25, 0.25, 0.25, 0.25]


# =============================================================================
class TestPSIProperties:
    """Properties that must hold for any correct implementation."""

    def test_identical_distribution_gives_zero(self):
        values = np.array([0.1] * 25 + [0.4] * 25 + [0.6] * 25 + [0.9] * 25)
        assert population_stability_index(values, EDGES, UNIFORM) == pytest.approx(0.0, abs=1e-9)

    def test_psi_is_never_negative(self):
        rng = np.random.default_rng(7)
        for _ in range(50):
            values = rng.random(300)
            expected = rng.dirichlet(np.ones(4))
            assert population_stability_index(values, EDGES, list(expected)) >= 0.0

    def test_psi_grows_with_the_size_of_the_shift(self):
        rng = np.random.default_rng(11)
        base = rng.random(2000)
        scores = [
            population_stability_index(np.clip(base + shift, 0, 1), EDGES, UNIFORM)
            for shift in (0.0, 0.05, 0.15, 0.35)
        ]
        assert scores == sorted(scores)

    def test_psi_is_symmetric_in_the_two_distributions(self):
        """PSI(a,e) == PSI(e,a). This is what makes it a distance-like measure
        rather than a directional one — and why it cannot tell you which way
        the population moved, only how far."""
        a = np.array([0.1] * 10 + [0.4] * 40 + [0.6] * 30 + [0.9] * 20)
        e = [0.25, 0.25, 0.25, 0.25]
        forward = population_stability_index(a, EDGES, e)
        counts_a = np.histogram(a, bins=EDGES)[0] / len(a)
        reverse = float(np.sum((np.asarray(e) - counts_a) * np.log(np.asarray(e) / counts_a)))
        assert forward == pytest.approx(reverse, abs=1e-9)

    def test_empty_bucket_does_not_produce_infinity(self):
        """log(0) is the bug that makes a drift monitor report catastrophe on a
        small sample. EPSILON is the floor that prevents it."""
        values = np.array([0.1] * 100)  # everything in one bucket
        psi = population_stability_index(values, EDGES, UNIFORM)
        assert np.isfinite(psi) and psi > 0

    def test_empty_input_is_zero_not_an_error(self):
        assert population_stability_index(np.array([]), EDGES, UNIFORM) == 0.0

    def test_values_outside_the_reference_range_land_in_the_end_buckets(self):
        """The outer edges are -inf/+inf on purpose. A value beyond the
        training range is exactly the drift worth seeing; dropping it would
        make the monitor blindest precisely when it matters."""
        values = np.array([-500.0] * 50 + [500.0] * 50)
        assert population_stability_index(values, EDGES, UNIFORM) > 0.25


class TestDriftStatus:
    def test_the_conventional_bands(self):
        assert drift_status(0.05) == "stable"
        assert drift_status(0.10) == "moderate"
        assert drift_status(0.24) == "moderate"
        assert drift_status(0.25) == "significant"
        assert drift_status(3.0) == "significant"


# =============================================================================
class TestMonitoringWindow:
    def _reference(self):
        return ReferenceDistribution(
            bins={f: list(EDGES) for f in MONITORED_FEATURES},
            expected={f: list(UNIFORM) for f in MONITORED_FEATURES},
        )

    def _row(self, value=0.4):
        return {f: value for f in MONITORED_FEATURES}

    def test_window_is_bounded(self):
        """The process must not grow without limit. maxlen drops the oldest
        row for free."""
        window = MonitoringWindow(max_size=100)
        for _ in range(500):
            window.record(self._row(), 0.3, 1)
        assert len(window) == 100

    def test_small_window_reports_insufficient_rather_than_zero(self):
        """A drift score of 0.0 because nothing is configured looks exactly
        like a drift score of 0.0 because nothing has drifted."""
        window = MonitoringWindow(reference=self._reference(), min_size=200)
        for _ in range(50):
            window.record(self._row(), 0.3, 1)
        state = window.publish()
        assert state["sufficient_data"] is False
        assert state["feature_psi"] == {}

    def test_no_reference_means_no_drift_reported(self):
        window = MonitoringWindow(reference=None, min_size=10)
        for _ in range(50):
            window.record(self._row(), 0.3, 1)
        assert window.compute_drift() == {}

    def test_drift_is_detected_when_the_window_shifts(self):
        window = MonitoringWindow(reference=self._reference(), min_size=100)
        for _ in range(400):
            window.record(self._row(0.9), 0.3, 1)   # all mass in the last bucket
        drift = window.compute_drift()
        assert all(v > 0.25 for v in drift.values()), drift

    def test_selection_rate_uses_the_review_threshold(self):
        window = MonitoringWindow(min_size=10, threshold=0.5)
        for _ in range(60):
            window.record(self._row(), 0.9, 1)      # all above threshold
        for _ in range(60):
            window.record(self._row(), 0.1, 2)      # all below
        rates = window.compute_fairness()
        assert rates["1"] == pytest.approx(1.0)
        assert rates["2"] == pytest.approx(0.0)

    def test_tiny_groups_are_not_reported(self):
        """A rate computed on five requests swings wildly; publishing it as a
        fairness signal is worse than silence."""
        window = MonitoringWindow(min_size=10, threshold=0.5)
        for _ in range(100):
            window.record(self._row(), 0.9, 1)
        for _ in range(5):
            window.record(self._row(), 0.1, 2)
        assert "2" not in window.compute_fairness()

    def test_fairness_gap_needs_two_groups(self):
        window = MonitoringWindow(min_size=10, threshold=0.5)
        for _ in range(100):
            window.record(self._row(), 0.9, 1)
        assert window.publish()["fairness_gap"] == 0.0

    def test_window_is_thread_safe(self):
        """uvicorn serves requests concurrently. A half-appended row would
        corrupt every statistic computed from the window."""
        import threading

        window = MonitoringWindow(max_size=5000)

        def writer():
            for _ in range(500):
                window.record(self._row(), 0.4, 1)

        threads = [threading.Thread(target=writer) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(window) == 4000


# =============================================================================
class TestReference:
    """The reference file itself — the thing every PSI is measured against."""

    def test_reference_covers_the_monitored_features(self, reference_payload):
        for feature in MONITORED_FEATURES:
            assert feature in reference_payload["bins"], feature

    def test_bin_edges_are_increasing(self, reference_payload):
        for feature, edges in reference_payload["bins"].items():
            assert all(a < b for a, b in zip(edges, edges[1:])), feature

    def test_outer_edges_are_open(self, reference_payload):
        for feature, edges in reference_payload["bins"].items():
            assert edges[0] == float("-inf") and edges[-1] == float("inf"), feature

    def test_expected_proportions_sum_to_one(self, reference_payload):
        for feature, expected in reference_payload["expected"].items():
            assert sum(expected) == pytest.approx(1.0, abs=1e-6), feature

    def test_bins_start_out_roughly_equally_populated(self, reference_payload):
        """Quantile bins, not equal-width. Equal-width bins on a skewed feature
        like LIMIT_BAL put most of the mass in one bucket, and the PSI then
        barely moves however much the distribution shifts."""
        for feature, expected in reference_payload["expected"].items():
            if len(expected) >= 10:
                assert max(expected) < 3 * min(expected), feature

    def test_training_data_scores_near_zero_against_its_own_reference(
        self, reference_payload, training_frame
    ):
        """The sanity check that catches a mis-built reference: the data the
        reference was computed FROM must not look drifted."""
        for feature, edges in reference_payload["bins"].items():
            values = training_frame[feature].dropna().to_numpy()
            psi = population_stability_index(
                values, edges, reference_payload["expected"][feature]
            )
            assert psi < 0.01, f"{feature} scores {psi:.4f} against its own reference"
