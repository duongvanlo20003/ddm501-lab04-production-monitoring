"""
Model construction and training.

"""

import logging
from typing import Any, Dict, List

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from pipeline.config import MODEL_CONFIGS, RANDOM_STATE
from pipeline.preprocessing import build_feature_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "logreg": LogisticRegression,
    "rf": RandomForestClassifier,
    "hgb": HistGradientBoostingClassifier,
}


def build_model(model_type: str, **params: Any) -> Any:
    """Instantiate an estimator by name."""
    cls = MODEL_CLASSES.get(model_type)
    if cls is None:
        raise ValueError(f"Unknown model_type '{model_type}'. Available: {sorted(MODEL_CLASSES)}")
    merged: Dict[str, Any] = {**MODEL_CONFIGS.get(model_type, {}), **params}
    merged.setdefault("random_state", RANDOM_STATE)
    if model_type == "rf":
        merged.setdefault("n_jobs", -1)
    return cls(**merged)


def build_pipeline(model_type: str, raw_columns: List[str], **params: Any) -> Pipeline:
    """Derivation + preprocessing + estimator, fitted together.

    Takes the RAW column list. Feature derivation happens inside the pipeline,
    so the fitted artifact accepts exactly the frame the API builds — no caller
    has to remember a preprocessing step, and training-serving skew of that kind
    cannot happen.
    """
    return Pipeline(
        [
            ("features", build_feature_pipeline(raw_columns)),
            ("classifier", build_model(model_type, **params)),
        ]
    )


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = "hgb",
    **params: Any,
) -> Pipeline:
    """Fit and return a pipeline."""
    pipeline = build_pipeline(model_type, list(X_train.columns), **params)
    logger.info("Training %s on %s rows", model_type, len(X_train))
    pipeline.fit(X_train, y_train)
    return pipeline
