"""
End-to-end tests: traffic in, monitoring signal out.

These are the tests that would have caught the bug this lab's solution
actually had — derived features arriving as None, so their PSI was pinned at
0.0 forever and the monitor looked healthy while measuring nothing.
"""

import pytest

from app.monitoring import MONITORED_FEATURES
from scripts.load_test import apply_drift, apply_group_bias, row_to_payload


def _feed(client, payloads):
    for payload in payloads:
        response = client.post("/predict", json=payload)
        assert response.status_code == 200


# =============================================================================
class TestMonitoringEndpoint:
    def test_monitoring_returns_the_expected_shape(self, client):
        body = client.get("/monitoring").json()
        for key in ("window_size", "sufficient_data", "feature_psi",
                    "drift_score", "drift_status", "selection_rate", "fairness_gap"):
            assert key in body

    def test_window_grows_with_traffic(self, client, application):
        before = client.get("/monitoring").json()["window_size"]
        for _ in range(10):
            client.post("/predict", json=application)
        after = client.get("/monitoring").json()["window_size"]
        assert after == before + 10

    def test_batch_requests_are_recorded_too(self, client, application):
        before = client.get("/monitoring").json()["window_size"]
        client.post("/predict/batch", json={"applications": [application] * 7})
        after = client.get("/monitoring").json()["window_size"]
        assert after == before + 7


class TestDerivedFeaturesReachTheMonitor:
    """The regression test for a real bug in this codebase.

    /predict receives the 23 raw columns. Three of the six monitored features
    are derived. If the frame is recorded before derivation, those three are
    None, dropna() empties them, and their PSI is 0.0 forever — a monitor that
    reports perfect stability precisely because it is measuring nothing.
    """

    def test_every_monitored_feature_reports_a_psi(self, client, sample_rows):
        _feed(client, sample_rows * 4)
        state = client.get("/monitoring").json()
        assert state["sufficient_data"], "need a bigger sample for this test"
        for feature in MONITORED_FEATURES:
            assert feature in state["feature_psi"], f"{feature} is not being measured"

    def test_derived_features_are_not_pinned_at_zero(self, client, sample_rows):
        _feed(client, sample_rows * 4)
        psi = client.get("/monitoring").json()["feature_psi"]
        derived = ["utilisation_ratio", "payment_ratio", "max_delay"]
        assert any(psi[f] > 0 for f in derived), (
            "all derived features report exactly 0.0, which is what a broken "
            "monitor looks like"
        )


class TestDetection:
    """The lab's central claim: drift and unfairness are visible in the metrics."""

    def test_drifted_traffic_raises_the_drift_score(self, client, raw):
        import random

        rng = random.Random(9)
        frame = raw.sample(n=260, random_state=9)
        payloads = [apply_drift(row_to_payload(row), rng) for _, row in frame.iterrows()]
        _feed(client, payloads * 4)          # flush the window with drifted rows
        state = client.get("/monitoring").json()
        assert state["drift_score"] > 0.25
        assert state["drift_status"] == "significant"

    def test_biased_traffic_widens_the_selection_rate_gap(self, client, raw):
        import random

        rng = random.Random(13)
        frame = raw.sample(n=260, random_state=13)
        payloads = [
            apply_group_bias(row_to_payload(row), rng) for _, row in frame.iterrows()
        ]
        _feed(client, payloads * 4)
        state = client.get("/monitoring").json()
        assert state["fairness_gap"] > 0.10
        assert len(state["selection_rate"]) == 2


# =============================================================================
class TestExplain:
    def test_explanation_has_contributions(self, client, risky_application):
        body = client.post("/explain", json=risky_application).json()
        assert body["contributions"]
        assert body["decision"] in {"APPROVE", "REVIEW", "DECLINE"}

    def test_contributions_are_ordered_by_magnitude(self, client, risky_application):
        body = client.post("/explain", json=risky_application).json()
        magnitudes = [abs(c["contribution"]) for c in body["contributions"]]
        assert magnitudes == sorted(magnitudes, reverse=True)

    def test_contributions_are_in_the_applicants_own_units(self, client, risky_application):
        """An adverse action notice quoting the scaler's view of a feature
        cannot be reconciled with the application form by anyone outside the
        ML team."""
        body = client.post("/explain", json=risky_application).json()
        values = {c["feature"]: c["value"] for c in body["contributions"]}
        if "PAY_0" in values:
            assert values["PAY_0"] == pytest.approx(risky_application["pay_status"][0])
        if "utilisation_ratio" in values:
            assert 0.0 <= values["utilisation_ratio"] <= 3.0

    def test_a_risky_applicant_gets_risk_increasing_factors(self, client, risky_application):
        body = client.post("/explain", json=risky_application).json()
        assert any(c["direction"] == "increases risk" for c in body["contributions"])

    def test_explanation_agrees_with_the_prediction(self, client, risky_application):
        predicted = client.post("/predict", json=risky_application).json()
        explained = client.post("/explain", json=risky_application).json()
        assert explained["default_probability"] == pytest.approx(
            predicted["default_probability"]
        )
        assert explained["decision"] == predicted["decision"]

    def test_explain_is_measured_separately_from_predict(self, client, risky_application):
        client.post("/explain", json=risky_application)
        body = client.get("/metrics").text
        assert "ml_explain_duration_seconds" in body
        assert "ml_explanations_total" in body


class TestModelInfo:
    def test_model_info_reports_the_reference_state(self, client):
        body = client.get("/model/info").json()
        assert body["drift_reference_loaded"] is True
        assert body["is_loaded"] is True
