"""
Production monitoring: drift and fairness, computed from live traffic.

The problem this module exists to solve
---------------------------------------
You cannot monitor accuracy in production. Whether an applicant defaults is
known next month at the earliest, and for a rejected applicant it is never
known at all. By the time the accuracy number arrives, the model has been
wrong for weeks.

So production monitoring for ML is the practice of watching PROXIES that move
before the accuracy does:

    input drift     the applicants arriving no longer look like the ones the
                    model was fitted on
    output drift    the distribution of scores has shifted
    decision mix    more applications are being sent to manual review
    fairness gap    one group is being flagged at a different rate than before

None of these prove the model is wrong. Each of them is a reason to look.

Why PSI
-------
Population Stability Index is the standard drift measure in consumer credit,
which is where this problem comes from. For a feature binned into k buckets,
with expected proportion e_i from the reference and actual proportion a_i from
the live window.

The conventional reading, and the one the alerts use:

    PSI < 0.10          no meaningful shift
    0.10 <= PSI < 0.25  moderate shift, worth investigating
    PSI >= 0.25         significant shift, act

It is symmetric, bounded below at zero, and cheap enough to recompute on every
request. Its weakness is that it is univariate: it cannot see a shift in the
RELATIONSHIP between two features, only in each one separately.
"""

import json
import logging
import threading
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

import numpy as np
import pandas as pd

from app.config import REVIEW_THRESHOLD
from app.metrics import (
    DRIFT_SCORE,
    DRIFT_WINDOW_SIZE,
    FAIRNESS_GAP,
    FEATURE_DRIFT_PSI,
    SELECTION_RATE,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Features watched for drift. Not every column: a dashboard with 23 drift
# lines is a dashboard nobody reads. These are the ones whose movement would
# actually change the model's behaviour.
MONITORED_FEATURES = [
    "LIMIT_BAL",
    "AGE",
    "PAY_0",
    "utilisation_ratio",
    "payment_ratio",
    "max_delay",
]

# Smallest proportion used in the PSI formula. Without it an empty bucket gives
# log(0) and the whole score becomes infinite — which looks like catastrophic
# drift and is really just a small sample.
EPSILON = 1e-4


class ReferenceDistribution:
    """The training-time distribution every live window is compared against.

    Frozen at build time by scripts/make_reference.py and shipped with the
    model. Recomputing it from recent traffic would defeat the purpose: drift
    measured against yesterday's data can never detect a slow trend, because
    the baseline moves with it.
    """

    def __init__(self, bins: Dict[str, List[float]], expected: Dict[str, List[float]]):
        self.bins = bins
        self.expected = expected

    @classmethod
    def load(cls, path: Path) -> "ReferenceDistribution":
        payload = json.loads(Path(path).read_text())
        return cls(bins=payload["bins"], expected=payload["expected"])

    @property
    def features(self) -> List[str]:
        return list(self.bins)


def population_stability_index(
    actual: np.ndarray, bin_edges: List[float], expected: List[float]
) -> float:
    """PSI of one feature's live values against its reference proportions.

    TASK 1 — the formula:

        PSI = sum over bins of (a_i - e_i) * ln(a_i / e_i)

    where e_i is the reference proportion of bin i and a_i is the live one.

    Steps:
      1. If `actual` is empty, return 0.0 — an empty window is not drift.
      2. np.histogram(actual, bins=bin_edges) to get the live counts.
      3. Turn the counts into proportions.
      4. Floor BOTH proportion arrays at EPSILON before the log. An empty
         bucket gives log(0) = -inf, and a monitor that reports infinite
         drift on a small sample is a monitor nobody believes.
      5. Return the sum as a float.

    Check yourself: PSI of a distribution against itself must be 0, and PSI
    must never be negative. tests/test_monitoring.py asserts both.
    """
    values = np.asarray(actual, dtype=float)
    if values.size == 0:
        return 0.0
    counts, _ = np.histogram(values, bins=np.asarray(bin_edges, dtype=float))
    actual_proportions = counts / counts.sum()
    actual_safe = np.maximum(actual_proportions, EPSILON)
    expected_safe = np.maximum(np.asarray(expected, dtype=float), EPSILON)
    return float(np.sum((actual_safe - expected_safe) * np.log(actual_safe / expected_safe)))


class MonitoringWindow:
    """A bounded, thread-safe window of recent scored applications.

    Bounded because the process must not grow without limit; a deque with
    maxlen drops the oldest row for free. Thread-safe because uvicorn serves
    concurrent requests and a half-appended row would corrupt the statistics.
    """

    def __init__(
        self,
        reference: Optional[ReferenceDistribution] = None,
        max_size: int = 2000,
        min_size: int = 200,
        threshold: float = REVIEW_THRESHOLD,
    ):
        self.reference = reference
        self.max_size = max_size
        # Below this, the statistics are noise. Reporting a PSI computed on
        # 12 requests is how a monitoring system loses its audience.
        self.min_size = min_size
        self.threshold = threshold
        self._rows: Deque[Dict[str, Any]] = deque(maxlen=max_size)
        self._lock = threading.Lock()

    # -------------------------------------------------------------------------
    def record(self, features: Dict[str, Any], score: float, group: Any) -> None:
        """Add one scored application to the window."""
        row = {k: features.get(k) for k in MONITORED_FEATURES}
        row["_score"] = float(score)
        row["_group"] = str(group)
        with self._lock:
            self._rows.append(row)
        DRIFT_WINDOW_SIZE.set(len(self._rows))

    def record_frame(self, frame: pd.DataFrame, scores: np.ndarray, group_column: str) -> None:
        """Add a whole batch at once."""
        for (_, row), score in zip(frame.iterrows(), scores):
            self.record(row.to_dict(), float(score), row.get(group_column))

    # -------------------------------------------------------------------------
    def snapshot(self) -> pd.DataFrame:
        with self._lock:
            return pd.DataFrame(list(self._rows))

    def __len__(self) -> int:
        return len(self._rows)

    # -------------------------------------------------------------------------
    def compute_drift(self) -> Dict[str, float]:
        """PSI per monitored feature, and the maximum across them.

        TASK 2:
          - Return {} when there is no reference, or when the window holds
            fewer than self.min_size rows. Returning zeros instead would make
            "not measuring" indistinguishable from "nothing has drifted".
          - Otherwise snapshot the window and compute the PSI of each feature
            in self.reference.features that is present in the frame.
          - pd.to_numeric(..., errors="coerce").dropna() before binning: a
            column that arrived as None must not silently become 0.
        """
        if self.reference is None or len(self) < self.min_size:
            return {}
        frame = self.snapshot()
        drift: Dict[str, float] = {}
        for feature in self.reference.features:
            if feature not in frame:
                continue
            values = pd.to_numeric(frame[feature], errors="coerce").dropna().to_numpy()
            if values.size == 0:
                continue
            drift[feature] = population_stability_index(
                values,
                self.reference.bins[feature],
                self.reference.expected[feature],
            )
        return drift

    def compute_fairness(self) -> Dict[str, float]:
        """Selection rate per group: the share sent to review or decline.

        TASK 3:
          - Return {} below self.min_size, as above.
          - Group the snapshot by "_group" and, for each group, compute the
            share of rows whose "_score" is >= self.threshold.
          - SKIP any group with fewer than 30 rows. A rate computed on a
            handful of requests swings wildly, and publishing it as a
            fairness signal is worse than saying nothing.
        """
        if len(self) < self.min_size:
            return {}
        frame = self.snapshot()
        rates: Dict[str, float] = {}
        for group, subset in frame.groupby("_group"):
            if len(subset) < 30:
                continue
            rates[str(group)] = float((subset["_score"] >= self.threshold).mean())
        return rates

    # -------------------------------------------------------------------------
    def publish(self) -> Dict[str, Any]:
        """Recompute everything and push it into the Prometheus gauges.

        TASK 4:
          - compute_drift(), then set FEATURE_DRIFT_PSI per feature and
            DRIFT_SCORE to the maximum (0.0 when there is nothing to report).
          - compute_fairness(), then set SELECTION_RATE per group and
            FAIRNESS_GAP to max - min. A gap needs two groups; with fewer,
            report 0.0.
          - Return the dict that /monitoring serves. MonitoringResponse in
            app/schemas.py is the exact shape, and tests/test_api_monitoring.py
            asserts every key.
        """
        drift = self.compute_drift()
        for feature, value in drift.items():
            FEATURE_DRIFT_PSI.labels(feature=feature).set(value)
        drift_score = max(drift.values(), default=0.0)
        DRIFT_SCORE.set(drift_score)

        fairness = self.compute_fairness()
        for group, value in fairness.items():
            SELECTION_RATE.labels(group=group).set(value)
        fairness_gap = (
            float(max(fairness.values()) - min(fairness.values()))
            if len(fairness) >= 2 else 0.0
        )
        FAIRNESS_GAP.set(fairness_gap)
        return {
            "window_size": len(self),
            "min_window_size": self.min_size,
            "sufficient_data": len(self) >= self.min_size,
            "feature_psi": drift,
            "drift_score": drift_score,
            "drift_status": drift_status(drift_score),
            "selection_rate": fairness,
            "fairness_gap": fairness_gap,
        }


def drift_status(psi: float) -> str:
    """The conventional PSI reading, as a word rather than a number."""
    if psi >= 0.25:
        return "significant"
    if psi >= 0.10:
        return "moderate"
    return "stable"
