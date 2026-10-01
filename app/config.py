"""
Configuration settings for the Credit Risk Scoring API.

"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Model settings
MODEL_PATH = os.getenv("MODEL_PATH", str(BASE_DIR / "models" / "credit_model.joblib"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0.0")

# Business decision thresholds.
# A score above DECLINE_THRESHOLD is rejected outright; between the two it goes
# to a human underwriter. These are BUSINESS decisions, not model properties —
# see Lesson 02, "Setting the right threshold".
REVIEW_THRESHOLD = float(os.getenv("REVIEW_THRESHOLD", 0.30))
DECLINE_THRESHOLD = float(os.getenv("DECLINE_THRESHOLD", 0.60))

# API settings
API_TITLE = "Credit Default Risk Scoring API"
API_DESCRIPTION = (
    "Scores a credit card applicant's probability of defaulting on the next "
    "payment, and turns that score into an underwriting decision."
)
API_VERSION = "1.0.0"

# Server settings
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
