"""
Tests for the configuration that is not Python.

Prometheus rules and Grafana dashboards are code — they break in production
like code, and a typo in a PromQL expression is discovered during an incident
unless something checks it first. These tests are cheap and they catch the
mistakes that are otherwise invisible until they matter.
"""

from pathlib import Path

import pytest
import yaml

BASE = Path(__file__).resolve().parents[1]
PROM = BASE / "monitoring" / "prometheus"
GRAFANA = BASE / "monitoring" / "grafana"


def _rule_files():
    return sorted((PROM / "alerts").glob("*.yml"))


def _all_rules():
    rules = []
    for path in _rule_files():
        for group in yaml.safe_load(path.read_text())["groups"]:
            rules.extend(group.get("rules") or [])
    return rules


# =============================================================================
class TestPrometheusConfig:
    def test_config_is_valid_yaml(self):
        assert yaml.safe_load((PROM / "prometheus.yml").read_text())

    def test_the_api_is_scraped(self):
        config = yaml.safe_load((PROM / "prometheus.yml").read_text())
        jobs = {job["job_name"] for job in config["scrape_configs"]}
        assert "credit-risk-api" in jobs

    def test_scrape_target_matches_the_compose_service_name(self):
        """Docker's embedded DNS resolves service names. Rename the service in
        docker-compose.yml and this scrape silently stops working."""
        config = yaml.safe_load((PROM / "prometheus.yml").read_text())
        compose = yaml.safe_load((BASE / "docker-compose.yml").read_text())
        targets = [
            t
            for job in config["scrape_configs"]
            for sc in job.get("static_configs", [])
            for t in sc["targets"]
        ]
        for target in targets:
            host = target.split(":")[0]
            if host != "localhost":
                assert host in compose["services"], f"{host} is not a compose service"

    def test_rule_files_exist_where_the_config_says(self):
        config = yaml.safe_load((PROM / "prometheus.yml").read_text())
        for entry in config["rule_files"]:
            assert (PROM / "alerts" / Path(entry).name).exists(), entry


class TestAlertRules:
    def test_rule_files_are_valid_yaml(self):
        for path in _rule_files():
            assert yaml.safe_load(path.read_text())["groups"]

    def test_the_expected_ml_alerts_are_defined(self):
        """The rule set is not complete until these exist.

        Without this test an empty `rules: []` passes every structural check
        above vacuously — a green suite for a monitoring system that watches
        nothing.
        """
        names = {rule["alert"] for rule in _all_rules()}
        expected = {
            "ModerateFeatureDrift",
            "SignificantFeatureDrift",
            "DriftWindowTooSmall",
            "DecisionMixShift",
            "FairnessGapWidened",
            "PredictionErrorsRising",
            "ExplanationLatencyHigh",
        }
        assert expected <= names, f"missing: {sorted(expected - names)}"

    def test_every_alert_has_a_severity(self):
        for rule in _all_rules():
            assert rule.get("labels", {}).get("severity"), rule["alert"]

    def test_every_alert_says_what_to_do(self):
        """An alert whose recipient has to reverse-engineer the intent at 3am
        is an alert that will be ignored by the second week."""
        for rule in _all_rules():
            annotations = rule.get("annotations", {})
            assert annotations.get("summary"), rule["alert"]
            assert len(annotations.get("description", "")) > 40, rule["alert"]

    def test_every_alert_waits_longer_than_one_scrape(self):
        """Without `for:`, a single unlucky sample pages someone."""
        for rule in _all_rules():
            assert "for" in rule, rule["alert"]

    def test_drift_thresholds_are_the_conventional_bands(self):
        expressions = {r["alert"]: r["expr"] for r in _all_rules()}
        for name, threshold in (("ModerateFeatureDrift", "0.10"),
                                ("SignificantFeatureDrift", "0.25")):
            assert name in expressions, f"{name} is not defined"
            assert threshold in expressions[name], (
                f"{name} should use the conventional band {threshold}"
            )

    def test_alerts_only_reference_metrics_the_service_exports(self, client):
        """The test that catches a renamed metric. An alert on a metric that
        no longer exists never fires, and never tells you it is not firing."""
        exported = {
            line.split()[2]
            for line in client.get("/metrics").text.splitlines()
            if line.startswith("# TYPE")
        }
        # Histograms expose _bucket/_sum/_count; counters expose _total.
        exported |= {f"{n}_bucket" for n in exported} | {f"{n}_total" for n in exported}
        exported |= {"up", "node_cpu_seconds_total", "node_memory_MemAvailable_bytes"}

        import re

        for rule in _all_rules():
            for name in re.findall(r"\b([a-z_][a-z0-9_]{3,})\b(?=[\s{(\[])", rule["expr"]):
                if name in {"rate", "sum", "by", "histogram_quantile", "clamp_min", "job"}:
                    continue
                if name.startswith(("ml_", "http_", "node_")) or name == "up":
                    assert name in exported, f"{rule['alert']} references {name}"


class TestGrafana:
    def test_datasource_is_provisioned(self):
        path = GRAFANA / "provisioning" / "datasources" / "prometheus.yml"
        config = yaml.safe_load(path.read_text())
        assert config["datasources"][0]["type"] == "prometheus"

    def test_datasource_url_matches_the_compose_service(self):
        path = GRAFANA / "provisioning" / "datasources" / "prometheus.yml"
        url = yaml.safe_load(path.read_text())["datasources"][0]["url"]
        compose = yaml.safe_load((BASE / "docker-compose.yml").read_text())
        assert url.split("//")[1].split(":")[0] in compose["services"]

    def test_dashboards_are_valid_json(self):
        import json

        for path in (GRAFANA / "dashboards").glob("*.json"):
            assert json.loads(path.read_text())["panels"]

    def test_dashboard_panels_all_have_a_datasource(self):
        """A provisioned dashboard whose panels have no datasource comes up
        empty and asks the viewer to pick one — which defeats the point of
        provisioning it."""
        import json

        uid = yaml.safe_load(
            (GRAFANA / "provisioning" / "datasources" / "prometheus.yml").read_text()
        )["datasources"][0]["uid"]
        for path in (GRAFANA / "dashboards").glob("*.json"):
            for panel in json.loads(path.read_text())["panels"]:
                if panel["type"] == "row":
                    continue
                assert panel["datasource"]["uid"] == uid, (path.name, panel["title"])

    def test_dashboard_panels_have_targets(self):
        import json

        for path in (GRAFANA / "dashboards").glob("*.json"):
            for panel in json.loads(path.read_text())["panels"]:
                if panel["type"] == "row":
                    continue
                assert panel["targets"], (path.name, panel["title"])

    def test_dashboards_have_unique_uids(self):
        import json

        uids = [
            json.loads(p.read_text())["uid"] for p in (GRAFANA / "dashboards").glob("*.json")
        ]
        assert len(uids) == len(set(uids))


class TestCompose:
    def test_compose_is_valid_yaml(self):
        assert yaml.safe_load((BASE / "docker-compose.yml").read_text())["services"]

    def test_the_stack_is_complete(self):
        services = yaml.safe_load((BASE / "docker-compose.yml").read_text())["services"]
        assert {"api", "prometheus", "grafana"} <= set(services)

    def test_everything_shares_one_network(self):
        """Service-name DNS only resolves inside a shared network."""
        compose = yaml.safe_load((BASE / "docker-compose.yml").read_text())
        for name, service in compose["services"].items():
            assert "monitoring" in service.get("networks", []), name

    def test_the_model_is_mounted_read_only(self):
        compose = yaml.safe_load((BASE / "docker-compose.yml").read_text())
        mounts = compose["services"]["api"]["volumes"]
        assert any(m.startswith("./models") and m.endswith(":ro") for m in mounts)

    def test_the_image_runs_as_a_non_root_user(self):
        dockerfile = (BASE / "Dockerfile").read_text()
        assert "USER appuser" in dockerfile

    def test_the_healthcheck_probes_the_health_endpoint(self):
        """A check that only proves the process is listening passes happily
        while every prediction returns 503."""
        assert "/health" in (BASE / "Dockerfile").read_text()
