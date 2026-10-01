"""Shared fixtures.

The important one is `client`. It builds a TestClient with a context manager,
which is what runs FastAPI's lifespan — without it the model is never loaded
and every test hits a 503 that has nothing to do with what it was testing.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from pipeline.data_ingestion import load_raw, split_data
from pipeline.preprocessing import add_derived_features

BASE_DIR = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def client():
    """A test client with the application lifespan actually run."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def raw():
    return load_raw()


@pytest.fixture(scope="session")
def training_frame(raw):
    """The training split, with derived features — what the reference is built from."""
    X_train, _, _, _ = split_data(raw)
    return add_derived_features(X_train)


@pytest.fixture(scope="session")
def reference_payload():
    path = BASE_DIR / "models" / "reference.json"
    if not path.exists():
        pytest.skip("no reference.json; run scripts/make_reference.py")
    return json.loads(path.read_text())


@pytest.fixture
def application():
    """A well-formed, unremarkable application."""
    return {
        "limit_bal": 200000.0,
        "sex": 2,
        "education": 2,
        "marriage": 1,
        "age": 35,
        "pay_status": [0, 0, 0, 0, 0, 0],
        "bill_amt": [50000.0, 48000.0, 46000.0, 44000.0, 42000.0, 40000.0],
        "pay_amt": [5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0],
    }


@pytest.fixture
def risky_application(application):
    """Late on everything, nearly at the limit, paying almost nothing."""
    risky = dict(application)
    risky["limit_bal"] = 30000.0
    risky["pay_status"] = [2, 2, 2, 2, 2, 2]
    risky["bill_amt"] = [29000.0] * 6
    risky["pay_amt"] = [200.0] * 6
    return risky


@pytest.fixture
def sample_rows(raw):
    """A deterministic handful of real rows, as API payloads."""
    from scripts.load_test import row_to_payload

    frame = raw.sample(n=60, random_state=404)
    return [row_to_payload(row) for _, row in frame.iterrows()]
