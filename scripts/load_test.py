"""
Generate traffic against the running service.

Three profiles, and the second and third are the point of the whole lab.

    --profile normal    applicants drawn from the training distribution.
                        Establishes what "healthy" looks like on the dashboard.

    --profile drifted   the population has shifted: younger applicants, lower
                        credit limits, higher utilisation. Nothing is broken.
                        The model is unchanged and still returns well-formed
                        probabilities. Only the INPUTS have moved — which is
                        exactly the failure that no test in Lab 3 can catch,
                        because there is no bug to catch.

    --profile unfair    one demographic group's applications are made
                        systematically riskier, so the selection-rate gap
                        widens while the aggregate score distribution barely
                        moves.

Usage:
    python scripts/load_test.py --profile normal  --requests 500
    python scripts/load_test.py --profile drifted --requests 500
    python scripts/load_test.py --profile drifted --strength 0.3   # a milder shift
    watch -n2 'curl -s localhost:8000/monitoring | python -m json.tool'
"""

import argparse
import json
import random
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from pipeline.config import TARGET
from pipeline.data_ingestion import load_raw

PAY_COLUMNS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]


def row_to_payload(row: pd.Series) -> Dict[str, Any]:
    """Turn a dataset row into the API's request shape."""
    return {
        "limit_bal": float(row["LIMIT_BAL"]),
        "sex": int(row["SEX"]),
        "education": int(row["EDUCATION"]),
        "marriage": int(row["MARRIAGE"]),
        "age": int(row["AGE"]),
        "pay_status": [int(row[c]) for c in PAY_COLUMNS],
        "bill_amt": [float(row[f"BILL_AMT{i}"]) for i in range(1, 7)],
        "pay_amt": [float(row[f"PAY_AMT{i}"]) for i in range(1, 7)],
    }


def apply_drift(
    payload: Dict[str, Any], rng: random.Random, strength: float = 1.0
) -> Dict[str, Any]:
    """Shift the applicant towards a younger, more stretched population.

    Deliberately plausible. A drift that produced impossible values would be
    caught by the schema; this one produces perfectly valid applications that
    simply are not the population the model was fitted on.

    `strength` scales the whole shift between 0.0 (no change) and 1.0 (the
    full shift). It exists so you can watch the PSI climb through the three
    conventional bands instead of only seeing it saturate: run the profile at
    0.15, 0.35 and 1.0 and compare /monitoring each time.
    """

    def lerp(full: float) -> float:
        """Interpolate a multiplier between 1.0 (no drift) and `full`."""
        return 1.0 + strength * (full - 1.0)

    payload = dict(payload)
    payload["age"] = max(21, int(payload["age"] * lerp(rng.uniform(0.62, 0.80))))
    payload["limit_bal"] = max(
        10_000, payload["limit_bal"] * lerp(rng.uniform(0.35, 0.55))
    )
    inflate = lerp(rng.uniform(1.6, 2.4))
    payload["bill_amt"] = [
        min(b * inflate, payload["limit_bal"] * 1.2) for b in payload["bill_amt"]
    ]
    payload["pay_amt"] = [p * lerp(rng.uniform(0.20, 0.45)) for p in payload["pay_amt"]]
    bump = rng.choice([0, 1, 1, 2])
    payload["pay_status"] = [
        min(8, s + int(round(bump * strength))) for s in payload["pay_status"]
    ]
    return payload


def apply_group_bias(payload: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    """Make one group's applications riskier, leaving the other alone."""
    if payload["sex"] != 1:
        return payload
    payload = dict(payload)
    payload["pay_status"] = [min(8, s + rng.choice([1, 2, 2, 3])) for s in payload["pay_status"]]
    payload["pay_amt"] = [p * 0.25 for p in payload["pay_amt"]]
    return payload


def post(url: str, payload: Dict[str, Any], timeout: float = 10.0):
    """POST one JSON body, returning (status, parsed body, seconds)."""
    body = json.dumps(payload).encode()
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.load(response), time.perf_counter() - start
    except urllib.error.HTTPError as exc:
        return exc.code, None, time.perf_counter() - start
    except Exception:  # noqa: BLE001
        return 0, None, time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--profile", default="normal",
                        choices=["normal", "drifted", "unfair"])
    parser.add_argument("--requests", type=int, default=300)
    parser.add_argument("--delay", type=float, default=0.02,
                        help="seconds between requests")
    parser.add_argument("--seed", type=int, default=501)
    parser.add_argument("--strength", type=float, default=1.0,
                        help="drift magnitude, 0.0-1.0 (drifted profile only)")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    frame = load_raw().drop(columns=[TARGET]).sample(
        n=args.requests, replace=True, random_state=args.seed
    )

    label = args.profile
    if args.profile == "drifted":
        label += f" (strength {args.strength:g})"
    print(f"{label} traffic: {args.requests} requests -> {args.url}/predict\n")

    latencies: List[float] = []
    decisions: Counter = Counter()
    statuses: Counter = Counter()
    scores: List[float] = []

    for i, (_, row) in enumerate(frame.iterrows(), 1):
        payload = row_to_payload(row)
        if args.profile == "drifted":
            payload = apply_drift(payload, rng, args.strength)
        elif args.profile == "unfair":
            payload = apply_group_bias(payload, rng)

        status, body, elapsed = post(f"{args.url}/predict", payload)
        statuses[status] += 1
        latencies.append(elapsed)
        if body:
            decisions[body["decision"]] += 1
            scores.append(body["default_probability"])

        if i % 50 == 0:
            print(f"  {i}/{args.requests}")
        time.sleep(args.delay)

    if not latencies:
        print("no responses — is the service running?")
        sys.exit(1)

    latencies.sort()
    print("\n" + "=" * 58)
    print(f"requests      {len(latencies)}")
    print(f"status        {dict(statuses)}")
    print(f"latency p50   {latencies[len(latencies)//2]*1000:.1f} ms")
    print(f"latency p95   {latencies[int(len(latencies)*0.95)]*1000:.1f} ms")
    print(f"latency max   {latencies[-1]*1000:.1f} ms")
    if scores:
        print(f"score  mean   {statistics.mean(scores):.4f}")
        print(f"score  median {statistics.median(scores):.4f}")
    total = sum(decisions.values()) or 1
    for decision in ("APPROVE", "REVIEW", "DECLINE"):
        print(f"{decision:<13} {decisions[decision]:>5}  ({decisions[decision]/total:.1%})")
    print("=" * 58)

    status, monitoring, _ = post(f"{args.url}/predict", {})  # deliberate 422, ignored
    try:
        with urllib.request.urlopen(f"{args.url}/monitoring", timeout=10) as response:
            state = json.load(response)
        print("\nmonitoring after this run")
        print(f"  window        {state['window_size']}")
        print(f"  drift score   {state['drift_score']}  ({state['drift_status']})")
        print(f"  per feature   {state['feature_psi']}")
        print(f"  selection     {state['selection_rate']}")
        print(f"  fairness gap  {state['fairness_gap']}")
    except Exception as exc:  # noqa: BLE001
        print(f"\ncould not read /monitoring: {exc}")


if __name__ == "__main__":
    main()
