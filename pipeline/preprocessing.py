"""
Preprocessing and feature engineering.

Two things happen here, and keeping them straight matters:

  add_derived_features  works on the DataFrame and encodes DOMAIN knowledge —
                        ratios and counts a credit analyst would compute by hand.
  build_preprocessor    returns an unfitted sklearn transformer that is part of
                        the model Pipeline, so scaling and encoding are FITTED ON
                        TRAINING DATA ONLY and travel with the model.

"""

import logging
from typing import List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from pipeline.config import BILL_FEATURES, CATEGORICAL_FEATURES, PAY_AMT_FEATURES, PAY_FEATURES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the six engineered features listed in config.DERIVED_FEATURES.

    Returns a copy — mutating the caller's frame is how a pipeline stage ends up
    depending on the order it was called in.
    """
    out = df.copy()

    avg_bill = out[BILL_FEATURES].mean(axis=1)
    avg_pay = out[PAY_AMT_FEATURES].mean(axis=1)

    # How much of the credit line is being used. The single most informative
    # ratio in consumer credit risk.
    out["utilisation_ratio"] = (avg_bill / out["LIMIT_BAL"].replace(0, np.nan)).clip(0, 5)

    # What share of the statement the customer actually pays.
    out["payment_ratio"] = (out["PAY_AMT1"] / out["BILL_AMT1"].replace(0, np.nan)).clip(0, 5)

    # Worst and how-many-months of arrears across the six-month window.
    out["max_delay"] = out[PAY_FEATURES].max(axis=1)
    out["n_months_delayed"] = (out[PAY_FEATURES] > 0).sum(axis=1)

    out["avg_bill_amt"] = avg_bill
    out["avg_pay_amt"] = avg_pay

    # The ratios are NaN where the denominator was zero — a real "not applicable",
    # which the imputer in the preprocessor handles explicitly.
    return out


def build_preprocessor(feature_columns: List[str]) -> ColumnTransformer:
    """Unfitted transformer: one-hot the categoricals, impute and scale the rest."""
    categorical = [c for c in CATEGORICAL_FEATURES if c in feature_columns]
    numeric = [c for c in feature_columns if c not in categorical]

    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """The full feature step: derive, then drop nothing and let the model decide."""
    return add_derived_features(df)


def build_feature_pipeline(raw_columns: List[str]) -> Pipeline:
    """Derivation + transformation as one step, taking RAW columns in.

    This is the fix for the most expensive bug in ML serving. If callers must
    remember to run add_derived_features before predict, one day one of them
    will not, and the model will score a frame that is silently missing six
    columns. Folding the derivation into the fitted pipeline makes that
    impossible: the artifact accepts exactly what the API produces.
    """
    from pipeline.config import DERIVED_FEATURES

    derived_columns = list(raw_columns) + [c for c in DERIVED_FEATURES if c not in raw_columns]
    return Pipeline(
        [
            ("derive", FunctionTransformer(add_derived_features, validate=False)),
            ("preprocess", build_preprocessor(derived_columns)),
        ]
    )
