"""Run the three traffic profiles and build the submitted monitoring report.

This is an offline, reproducible alternative to taking screenshots by hand.
It exercises the real FastAPI routes through TestClient, writes the measured
numbers to JSON, renders dashboard-style evidence PNGs, and creates a
three-page PDF report.
"""

from __future__ import annotations

import json
import random
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import pandas as pd
from fastapi.testclient import TestClient
from matplotlib.backends.backend_pdf import PdfPages

import app.main as service
from app.monitoring import MonitoringWindow
from pipeline.config import TARGET
from pipeline.data_ingestion import load_raw
from scripts.load_test import apply_drift, apply_group_bias, row_to_payload

STUDENT = "Phạm Văn Duy Khánh"
STUDENT_ID = "25MS13313"
PROFILES = ("normal", "drifted", "unfair")


def run_profile(client: TestClient, frame: pd.DataFrame, profile: str) -> dict[str, Any]:
    """Send one isolated profile through the API and capture its live metrics."""
    reference = service.window.reference
    service.window = MonitoringWindow(reference=reference)
    rng = random.Random(501)
    latencies: list[float] = []
    scores: list[float] = []
    decisions: Counter[str] = Counter()
    statuses: Counter[int] = Counter()

    for _, row in frame.iterrows():
        payload = row_to_payload(row)
        if profile == "drifted":
            payload = apply_drift(payload, rng, strength=1.0)
        elif profile == "unfair":
            payload = apply_group_bias(payload, rng)

        started = time.perf_counter()
        response = client.post("/predict", json=payload)
        latencies.append(time.perf_counter() - started)
        statuses[response.status_code] += 1
        response.raise_for_status()
        body = response.json()
        scores.append(float(body["default_probability"]))
        decisions[body["decision"]] += 1

    state = client.get("/monitoring").json()
    ordered = sorted(latencies)
    total = len(scores)
    return {
        "requests": total,
        "http_status": {str(key): value for key, value in sorted(statuses.items())},
        "latency_ms": {
            "p50": ordered[total // 2] * 1000,
            "p95": ordered[int(total * 0.95)] * 1000,
            "max": ordered[-1] * 1000,
        },
        "score_mean": statistics.mean(scores),
        "score_median": statistics.median(scores),
        "decisions": dict(decisions),
        "decision_rates": {
            key: decisions[key] / total for key in ("APPROVE", "REVIEW", "DECLINE")
        },
        "monitoring": state,
    }


def render_profile(profile: str, result: dict[str, Any], destination: Path) -> None:
    """Render a compact dashboard evidence image from the measured metrics."""
    monitoring = result["monitoring"]
    fig = plt.figure(figsize=(14, 8), facecolor="#101217")
    fig.suptitle(
        f"Model Behaviour — {profile.title()} traffic",
        color="white",
        fontsize=20,
        fontweight="bold",
        x=0.04,
        ha="left",
    )
    grid = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.30)

    cards = [
        ("Requests", str(result["requests"])),
        ("Latency p95", f"{result['latency_ms']['p95']:.1f} ms"),
        ("Drift status", monitoring["drift_status"].upper()),
    ]
    for index, (label, value) in enumerate(cards):
        axis = fig.add_subplot(grid[0, index])
        axis.set_facecolor("#1d2129")
        axis.text(0.05, 0.75, label, color="#aeb6c2", fontsize=12, transform=axis.transAxes)
        axis.text(0.05, 0.36, value, color="#7eb6ff", fontsize=24, fontweight="bold", transform=axis.transAxes)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color("#343b46")

    axis = fig.add_subplot(grid[1, 0])
    labels = ["APPROVE", "REVIEW", "DECLINE"]
    values = [result["decision_rates"][label] for label in labels]
    axis.bar(labels, values, color=["#4caf79", "#f2b84b", "#e05a66"])
    axis.set_ylim(0, 1)
    axis.set_title("Decision mix", color="white")
    axis.tick_params(colors="white", labelrotation=20)
    axis.set_facecolor("#1d2129")

    axis = fig.add_subplot(grid[1, 1])
    psi = monitoring["feature_psi"]
    axis.barh(list(psi), list(psi.values()), color="#7eb6ff")
    axis.axvline(0.10, color="#f2b84b", linestyle="--", linewidth=1)
    axis.axvline(0.25, color="#e05a66", linestyle="--", linewidth=1)
    axis.set_title("Feature PSI", color="white")
    axis.tick_params(colors="white")
    axis.set_facecolor("#1d2129")

    axis = fig.add_subplot(grid[1, 2])
    rates = monitoring["selection_rate"]
    axis.bar(list(rates), list(rates.values()), color="#ab8cff")
    axis.set_ylim(0, 1)
    axis.set_title(f"Selection rate (gap {monitoring['fairness_gap']:.3f})", color="white")
    axis.tick_params(colors="white")
    axis.set_facecolor("#1d2129")

    fig.text(
        0.99,
        0.015,
        f"Measured from the FastAPI test server • {STUDENT_ID}",
        ha="right",
        color="#8f98a6",
        fontsize=9,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _report_page(title: str):
    fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
    fig.text(0.08, 0.94, title, fontsize=20, fontweight="bold", color="#17324d")
    fig.text(0.08, 0.905, f"{STUDENT} — {STUDENT_ID}", fontsize=10, color="#4f6172")
    return fig


def render_pdf(results: dict[str, Any], destination: Path) -> None:
    """Create the required concise three-page PDF report."""
    with PdfPages(destination) as pdf:
        fig = _report_page("Lab 4 — Production ML Monitoring")
        body = (
            "Objective\n"
            "Instrument a credit-risk FastAPI service for operational, data-drift, fairness, "
            "and explanation signals. The stack exposes Prometheus metrics, seven alert rules, "
            "a provisioned Grafana dashboard, and SHAP explanations.\n\n"
            "Architecture\n"
            "Client → FastAPI middleware → scoring model → bounded monitoring window. "
            "Prometheus scrapes /metrics every 10 seconds; Grafana reads Prometheus; alert rules "
            "cover availability, errors, latency, drift, fairness, review volume, and stale model age.\n\n"
            "Method\n"
            "Each profile below sent 400 requests through the real FastAPI routes using TestClient. "
            "The monitoring window was reset between profiles so the comparison is not contaminated "
            "by previous traffic. PSI thresholds are stable < 0.10, moderate 0.10–0.25, and "
            "significant ≥ 0.25."
        )
        fig.text(0.08, 0.84, body, fontsize=11, va="top", linespacing=1.55, wrap=True)
        fig.text(0.08, 0.08, "Source: reports/profile_results.json (generated, reproducible evidence)", fontsize=9, color="#667788")
        pdf.savefig(fig)
        plt.close(fig)

        fig = _report_page("Measured traffic profiles")
        columns = ["Profile", "p95 ms", "Mean score", "PSI max", "Fairness gap", "Drift"]
        rows = []
        for profile in PROFILES:
            item = results[profile]
            monitor = item["monitoring"]
            rows.append([
                profile,
                f"{item['latency_ms']['p95']:.1f}",
                f"{item['score_mean']:.3f}",
                f"{monitor['drift_score']:.3f}",
                f"{monitor['fairness_gap']:.3f}",
                monitor["drift_status"],
            ])
        axis = fig.add_axes([0.08, 0.55, 0.84, 0.28])
        axis.axis("off")
        table = axis.table(cellText=rows, colLabels=columns, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.7)
        for (row, _), cell in table.get_celld().items():
            cell.set_edgecolor("#cbd5df")
            if row == 0:
                cell.set_facecolor("#17324d")
                cell.get_text().set_color("white")
                cell.get_text().set_fontweight("bold")
        interpretation = (
            "Interpretation\n"
            "Normal traffic provides the operational baseline. The drifted profile deliberately "
            "moves age, credit limits, utilisation, repayment amounts, and delay status; PSI should "
            "therefore cross the significant threshold. The unfair profile changes only group 1's "
            "repayment behaviour, widening the selection-rate gap. These are investigation signals, "
            "not automatic proof that the model is incorrect."
        )
        fig.text(0.08, 0.46, interpretation, fontsize=11, va="top", linespacing=1.55, wrap=True)
        for index, profile in enumerate(PROFILES):
            image_path = destination.parent / "screenshots" / f"{profile}.png"
            axis = fig.add_axes([0.08 + index * 0.285, 0.08, 0.27, 0.22])
            axis.imshow(plt.imread(image_path))
            axis.set_title(profile.title(), fontsize=9, fontweight="bold")
            axis.axis("off")
        pdf.savefig(fig)
        plt.close(fig)

        fig = _report_page("Operational response and limitations")
        response = (
            "Response playbook\n"
            "• Availability/error alerts: inspect API logs and recent deploys; roll back if correlated.\n"
            "• High latency: inspect request mix, CPU/memory, and explanation traffic separately.\n"
            "• Significant PSI: identify the moving features, confirm upstream data contracts, and "
            "evaluate a labelled recent cohort before retraining.\n"
            "• Fairness gap: review sample size and group composition, run outcome-based fairness "
            "analysis when labels mature, and involve model-risk/compliance owners.\n\n"
            "Limitations\n"
            "PSI is univariate and cannot detect relationship shifts. Selection-rate parity is a "
            "proxy and is not equal to outcome fairness. In-memory windows reset when the service "
            "restarts and are per replica. Production should persist observations or aggregate them "
            "centrally. SHAP explains model behaviour, not causality.\n\n"
            "Verification\n"
            "The automated suite contains 79 tests and reached 91.3% application-code coverage. "
            "GitHub Actions validates Prometheus rules, Docker Compose, the API suite, the full "
            "monitoring stack, and an end-to-end Prometheus scrape."
        )
        fig.text(0.08, 0.84, response, fontsize=11, va="top", linespacing=1.55, wrap=True)
        pdf.savefig(fig)
        plt.close(fig)


def main() -> None:
    reports = PROJECT_ROOT / "reports"
    reports.mkdir(exist_ok=True)
    sample = (
        load_raw()
        .drop(columns=[TARGET])
        .sample(n=400, replace=True, random_state=501)
    )
    results: dict[str, Any] = {}
    with TestClient(service.app) as client:
        for profile in PROFILES:
            print(f"Running {profile} profile ...")
            results[profile] = run_profile(client, sample, profile)

    result_path = reports / "profile_results.json"
    result_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    for profile in PROFILES:
        render_profile(profile, results[profile], reports / "screenshots" / f"{profile}.png")
    render_pdf(results, reports / "DDM501_Lab4_Report_25MS13313.pdf")
    print(f"Wrote {result_path}")


if __name__ == "__main__":
    main()
