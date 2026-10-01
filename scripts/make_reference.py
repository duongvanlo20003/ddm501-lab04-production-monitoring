"""
Freeze the training-time distribution that drift is measured against.

Writes models/reference.json: for each monitored feature, the bin edges and
the proportion of training rows that fell in each bin.

Two decisions in here are worth understanding, because getting either wrong
gives you a drift monitor that cannot detect drift.

Quantile bins, not equal-width. Equal-width bins on a skewed feature like
LIMIT_BAL put 95% of the mass in the first bucket, and the PSI then barely
moves however much the distribution shifts. Quantile bins start out equally
populated, so any shift redistributes mass visibly.

Computed ONCE, from the training data, and shipped with the model. It is
tempting to recompute the reference from recent traffic — it is fresher, after
all. That would make slow drift invisible: the baseline would move with the
data, and the PSI would sit near zero while the population walked away from
what the model was fitted on. The reference is a property of the MODEL, not of
the traffic, and it changes only when the model is retrained.

Usage:
    python scripts/make_reference.py
    python scripts/make_reference.py --bins 10
"""

import argparse
import json
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd

from app.monitoring import MONITORED_FEATURES
from pipeline.data_ingestion import load_raw, split_data
from pipeline.preprocessing import add_derived_features

OUT_PATH = BASE_DIR / "models" / "reference.json"


def build_reference(frame: pd.DataFrame, n_bins: int = 10) -> dict:
    """Quantile bin edges and expected proportions per monitored feature.

    TASK 11. For each feature in MONITORED_FEATURES that is in the frame:
      - Coerce to numeric and drop NaN.
      - Edges from np.quantile over np.linspace(0, 1, n_bins + 1), passed
        through np.unique (a feature with ties produces duplicate edges).
      - Skip the feature if fewer than 3 distinct edges survive: a PSI
        computed on two buckets is noise.
      - Replace the OUTER edges with -inf and +inf. A live value beyond the
        training range is exactly the drift worth seeing; with closed edges
        np.histogram drops it and the monitor goes blindest precisely when
        it matters.
      - Expected proportions = counts / counts.sum().

    Return {"bins": ..., "expected": ..., "n_bins": n_bins}.

    Why quantiles and not equal width: equal-width bins on a skewed feature
    like LIMIT_BAL put most of the mass in the first bucket, and the PSI then
    barely moves however far the distribution shifts.
    """
    bins = {}
    expected = {}
    quantiles = np.linspace(0, 1, n_bins + 1)
    for feature in MONITORED_FEATURES:
        if feature not in frame:
            continue
        values = pd.to_numeric(frame[feature], errors="coerce").dropna().to_numpy()
        if values.size == 0:
            continue
        edges = np.unique(np.quantile(values, quantiles))
        if len(edges) < 3:
            continue
        edges = edges.astype(float)
        edges[0], edges[-1] = -np.inf, np.inf
        counts, _ = np.histogram(values, bins=edges)
        bins[feature] = edges.tolist()
        expected[feature] = (counts / counts.sum()).astype(float).tolist()
    return {"bins": bins, "expected": expected, "n_bins": n_bins}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    out = Path(args.out) if args.out else OUT_PATH

    print("Building the drift reference from the TRAINING split")
    raw = load_raw()
    X_train, _, _, _ = split_data(raw)
    # Derived features are monitored too, so they have to exist here.
    frame = add_derived_features(X_train)
    print(f"  {len(frame):,} training rows\n")

    reference = build_reference(frame, args.bins)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(reference, indent=2))
    print(f"\nWrote {len(reference['bins'])} features -> {out}")
    print("Restart the API to pick it up.")


if __name__ == "__main__":
    main()
