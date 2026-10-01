"""
Train and save the credit default risk model.

Steps:
    1. Load data/credit_default.csv
    2. Split into train/test, stratified on the target
    3. Fit a gradient boosting pipeline
    4. Evaluate: ROC AUC, PR AUC, and the confusion matrix at the review threshold
    5. Save the pipeline plus its metadata to models/credit_model.joblib

Usage:
    python scripts/train_model.py
"""

from datetime import datetime, timezone
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

import joblib
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from pipeline.config import RANDOM_STATE, REVIEW_THRESHOLD, TARGET
from pipeline.training import build_pipeline

DATA_PATH = BASE_DIR / "data" / "credit_default.csv"
MODEL_PATH = BASE_DIR / "models" / "credit_model.joblib"


def main() -> None:
    print("=" * 62)
    print("Credit Default Risk — model training")
    print("=" * 62)

    # -------------------------------------------------------------------------
    print(f"\n[1/5] Loading {DATA_PATH.relative_to(BASE_DIR)} ...")
    if not DATA_PATH.exists():
        raise SystemExit("Dataset not found. Run 'python scripts/make_dataset.py' first.")
    df = pd.read_csv(DATA_PATH)
    print(f"      {len(df):,} rows x {df.shape[1]} columns")
    print(f"      Default rate: {df[TARGET].mean():.1%}")

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # -------------------------------------------------------------------------
    print("\n[2/5] Splitting train / test (80 / 20, stratified) ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"      train={len(X_train):,}  test={len(X_test):,}")

    # -------------------------------------------------------------------------
    print("\n[3/5] Fitting the pipeline ...")
    # build_pipeline folds feature derivation INTO the artifact, so the saved
    # model accepts exactly the raw frame the API produces.
    pipeline = build_pipeline("hgb", list(X_train.columns))
    pipeline.fit(X_train, y_train)
    print("      done")

    # -------------------------------------------------------------------------
    print("\n[4/5] Evaluating on the held-out test set ...")
    proba = pipeline.predict_proba(X_test)[:, 1]
    predicted = (proba >= REVIEW_THRESHOLD).astype(int)

    roc_auc = roc_auc_score(y_test, proba)
    pr_auc = average_precision_score(y_test, proba)
    tn, fp, fn, tp = confusion_matrix(y_test, predicted).ravel()

    print(f"      ROC AUC : {roc_auc:.4f}")
    print(f"      PR  AUC : {pr_auc:.4f}   (baseline = {y_test.mean():.4f})")
    print(f"\n      At threshold {REVIEW_THRESHOLD}:")
    print(f"        true negatives  {tn:>6,}   false positives {fp:>6,}")
    print(f"        false negatives {fn:>6,}   true positives  {tp:>6,}")
    print()
    print(classification_report(y_test, predicted, target_names=["no default", "default"]))

    # -------------------------------------------------------------------------
    print(f"[5/5] Saving to {MODEL_PATH.relative_to(BASE_DIR)} ...")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model_type": "HistGradientBoostingClassifier",
        "feature_engineering": "in-pipeline (derive -> preprocess -> classifier)",
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "features": list(X.columns),
        "metrics": {
            "roc_auc": round(float(roc_auc), 4),
            "pr_auc": round(float(pr_auc), 4),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_negatives": int(tn),
            "threshold": REVIEW_THRESHOLD,
        },
    }
    joblib.dump({"pipeline": pipeline, "metadata": metadata}, MODEL_PATH)
    print(f"      saved ({MODEL_PATH.stat().st_size / 1024:.0f} KB)")

    print("\n" + "=" * 62)
    print("Next steps")
    print("=" * 62)
    print("  1. Start the API : uvicorn app.main:app --reload")
    print("  2. Open the docs : http://localhost:8000/docs")
    print("  3. Score someone : POST /predict")


if __name__ == "__main__":
    main()
