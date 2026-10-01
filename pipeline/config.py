"""
Configuration for the credit default ML pipeline.

"""

import os
from pathlib import Path
from typing import Any, Dict, List

# =============================================================================
# Paths
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

for _d in (DATA_DIR, MODELS_DIR, ARTIFACTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

DATA_PATH = Path(os.getenv("DATA_PATH", str(DATA_DIR / "credit_default.csv")))

# =============================================================================
# Data
# =============================================================================
TARGET = "default_payment_next_month"
CATEGORICAL_FEATURES = ["SEX", "EDUCATION", "MARRIAGE"]
PAY_FEATURES = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_FEATURES = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT_FEATURES = [f"PAY_AMT{i}" for i in range(1, 7)]

RAW_FEATURES = (
    ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"]
    + PAY_FEATURES
    + BILL_FEATURES
    + PAY_AMT_FEATURES
)

# Features created in preprocessing.add_derived_features
DERIVED_FEATURES = [
    "utilisation_ratio",
    "payment_ratio",
    "max_delay",
    "n_months_delayed",
    "avg_bill_amt",
    "avg_pay_amt",
]

TEST_SIZE = float(os.getenv("TEST_SIZE", 0.2))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", 501))

# Attribute used for the fairness slice in evaluation. Session 6 explains why
# this belongs in the pipeline and not in a notebook someone runs once.
SENSITIVE_ATTRIBUTE = "SEX"

# =============================================================================
# Data validation thresholds
# =============================================================================
MAX_MISSING_FRACTION = 0.02  # per column
MIN_ROWS = 5_000
MIN_POSITIVE_RATE = 0.05  # a degenerate target means something upstream broke
MAX_POSITIVE_RATE = 0.60

# =============================================================================
# Models
# =============================================================================
DEFAULT_MODEL_TYPE = "hgb"

MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "logreg": {"C": 1.0, "max_iter": 1000},
    "rf": {"n_estimators": 300, "max_depth": 12, "min_samples_leaf": 20},
    "hgb": {"max_iter": 300, "learning_rate": 0.06, "max_depth": 6, "l2_regularization": 1.0},
}

# The sweep run by experiments/run_experiments.py
EXPERIMENT_GRID: List[Dict[str, Any]] = [
    {"model_type": "logreg", "C": 0.1, "max_iter": 1000},
    {"model_type": "logreg", "C": 1.0, "max_iter": 1000},
    {"model_type": "rf", "n_estimators": 200, "max_depth": 8, "min_samples_leaf": 20},
    {"model_type": "rf", "n_estimators": 300, "max_depth": 12, "min_samples_leaf": 20},
    {
        "model_type": "hgb",
        "max_iter": 200,
        "learning_rate": 0.10,
        "max_depth": 4,
        "l2_regularization": 1.0,
    },
    {
        "model_type": "hgb",
        "max_iter": 300,
        "learning_rate": 0.06,
        "max_depth": 6,
        "l2_regularization": 1.0,
    },
    {
        "model_type": "hgb",
        "max_iter": 500,
        "learning_rate": 0.03,
        "max_depth": 8,
        "l2_regularization": 2.0,
    },
]

# =============================================================================
# Quality bar
# =============================================================================
# The minimum a model must clear to be shippable. tests/model/ asserts these, so
# a retrain that falls below the bar fails CI instead of reaching production.
PRIMARY_METRIC = "roc_auc"
MIN_ROC_AUC = float(os.getenv("MIN_ROC_AUC", 0.70))
MIN_PR_AUC = float(os.getenv("MIN_PR_AUC", 0.45))
MAX_FAIRNESS_GAP = float(os.getenv("MAX_FAIRNESS_GAP", 0.10))

# Decision thresholds, identical to Lab 1 so the two agree.
REVIEW_THRESHOLD = float(os.getenv("REVIEW_THRESHOLD", 0.30))
DECLINE_THRESHOLD = float(os.getenv("DECLINE_THRESHOLD", 0.60))
