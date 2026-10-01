"""
Tests for the instrumentation itself.

A monitoring system is code, and untested monitoring code fails in the worst
possible way: silently, during the incident it was built for. Every test here
asks a question you would otherwise only find the answer to at 3am.
"""

import re

import pytest
from prometheus_client import REGISTRY

from app import metrics as M


def _families():
    return {f.name: f for f in REGISTRY.collect()}


def _value(name: str, **labels) -> float:
    """Read one sample out of the default registry."""
    for family in REGISTRY.collect():
        for sample in family.samples:
            if sample.name == name and all(
                sample.labels.get(k) == v for k, v in labels.items()
            ):
                return sample.value
    return 0.0


# =============================================================================
class TestExposition:
    """The /metrics endpoint, as Prometheus sees it."""

    def test_metrics_endpoint_returns_200(self, client):
        assert client.get("/metrics").status_code == 200

    def test_content_type_is_the_prometheus_format(self, client):
        content_type = client.get("/metrics").headers["content-type"]
        # Prometheus decides how to parse the body from this header. A JSON
        # content type here means a target that scrapes to nothing.
        assert "text/plain" in content_type

    def test_exposition_parses(self, client):
        body = client.get("/metrics").text
        for line in body.splitlines():
            if line and not line.startswith("#"):
                assert re.match(r"^[a-zA-Z_:][a-zA-Z0-9_:]*(\{.*\})? .+$", line), line

    def test_every_metric_has_a_help_string(self, client):
        """`# HELP <name> <text>` — a metric without one is a metric whose
        meaning lives only in the head of whoever added it."""
        body = client.get("/metrics").text
        names = {
            line.split()[2] for line in body.splitlines() if line.startswith("# TYPE")
        }
        helps = {
            line.split()[2] for line in body.splitlines() if line.startswith("# HELP")
        }
        assert names <= helps


class TestNamingConventions:
    """Prometheus conventions are not decoration; tooling depends on them."""

    def test_counters_are_declared_with_the_total_suffix(self):
        """Checked against the SOURCE, not the registry.

        prometheus_client normalises the name on the way out — declare
        `ml_predictions` and it still exposes `ml_predictions_total` — so the
        registry cannot tell you whether the convention was followed. The
        declaration is where the mistake is visible, and where a reviewer
        would see it.
        """
        import ast
        import pathlib

        source = pathlib.Path(M.__file__).read_text()
        for node in ast.walk(ast.parse(source)):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "Counter"
            ):
                name = node.args[0].value
                assert name.endswith("_total"), name

    def test_durations_are_in_seconds(self):
        names = [f.name for f in REGISTRY.collect()]
        assert "http_request_duration_seconds" in names
        assert "ml_prediction_duration_seconds" in names
        # A metric in milliseconds forces every dashboard and every alert to
        # convert, and one of them eventually forgets.
        assert not any(n.endswith("_ms") or n.endswith("_milliseconds") for n in names)

    def test_latency_buckets_bracket_the_slo(self):
        # The SLO is 100ms, so there must be a bucket boundary there — a
        # histogram_quantile can only interpolate between boundaries it has.
        assert 0.1 in M.REQUEST_LATENCY._upper_bounds


class TestMetricTypes:
    """The right instrument for the right measurement."""

    def test_request_count_is_a_counter(self):
        assert _families()["http_requests"].type == "counter"

    def test_drift_score_is_a_gauge(self):
        # It goes down as well as up. Recorded as a counter, Prometheus would
        # read every decrease as a process restart and rate() would be garbage.
        assert _families()["ml_drift_score"].type == "gauge"

    def test_score_is_a_histogram(self):
        assert _families()["ml_prediction_score"].type == "histogram"

    def test_score_buckets_cover_the_probability_range(self):
        bounds = M.PREDICTION_SCORE._upper_bounds
        assert min(bounds) <= 0.05 and 1.0 in bounds


# =============================================================================
class TestRecording:
    """The metrics actually move when the service does something."""

    def test_prediction_increments_the_counter(self, client, application):
        before = _value("ml_predictions_total", model_version="1.0.0")
        client.post("/predict", json=application)
        after = _value("ml_predictions_total", model_version="1.0.0")
        assert after == pytest.approx(before + 1)

    def test_batch_increments_by_the_batch_size(self, client, application):
        before = _value("ml_predictions_total", model_version="1.0.0")
        client.post("/predict/batch", json={"applications": [application] * 5})
        after = _value("ml_predictions_total", model_version="1.0.0")
        assert after == pytest.approx(before + 5)

    def test_decision_counter_carries_the_decision_label(self, client, application):
        client.post("/predict", json=application)
        body = client.get("/metrics").text
        assert 'ml_decisions_total{decision=' in body

    def test_http_metrics_record_the_status(self, client, application):
        client.post("/predict", json=application)
        assert _value("http_requests_total", endpoint="/predict", status="200") > 0

    def test_validation_failures_are_counted_too(self, client):
        """A 422 is a request. Middleware that only counts successes reports a
        healthy service while every client is being rejected."""
        before = _value("http_requests_total", endpoint="/predict", status="422")
        client.post("/predict", json={"limit_bal": "not a number"})
        after = _value("http_requests_total", endpoint="/predict", status="422")
        assert after > before

    def test_model_loaded_gauge_is_one(self, client):
        client.get("/health")
        assert _value("ml_model_loaded") == 1.0

    def test_model_info_is_exposed(self, client):
        body = client.get("/metrics").text
        assert "ml_model_info" in body


class TestCardinality:
    """The failure that takes down the monitoring stack, not the service."""

    def test_endpoint_label_is_the_route_template(self, client, application):
        """Unbounded label values are the classic Prometheus outage. This test
        exists so that adding a path-parameter route later cannot silently
        start minting one time series per id."""
        client.post("/predict", json=application)
        client.get("/health")
        endpoints = set()
        for family in REGISTRY.collect():
            for sample in family.samples:
                if "endpoint" in sample.labels:
                    endpoints.add(sample.labels["endpoint"])
        assert endpoints <= {"/predict", "/predict/batch", "/health", "/explain",
                             "/monitoring", "/", "/model/info", "/docs", "/openapi.json"}

    def test_metrics_endpoint_excludes_itself(self, client):
        """Prometheus scrapes every 10s. Counting those would dominate the
        traffic panel on a quiet service."""
        client.get("/metrics")
        assert _value("http_requests_total", endpoint="/metrics", status="200") == 0.0
