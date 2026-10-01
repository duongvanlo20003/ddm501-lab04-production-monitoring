"""
Prometheus metrics for the credit risk service.

Metric types, and when each is right:

    Counter    only ever goes up. Totals: requests, predictions, errors.
               Never a value that can decrease — Prometheus assumes a drop
               means a process restart and compensates, which silently
               corrupts every rate() over that window.
    Gauge      goes up and down. A current value: model loaded, drift score.
    Histogram  a distribution in buckets. Latency, and prediction scores.
    Info       static key-value labels. Model version and type.

"""

from prometheus_client import Counter, Gauge, Histogram, Info

# =============================================================================
# HTTP — the first three of the four golden signals
# =============================================================================
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency",
    ["method", "endpoint"],
    # Buckets are chosen around the SLO, not spread evenly. The interesting
    # question is "how many requests missed 100 ms", so there is a bucket
    # boundary exactly there.
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

REQUESTS_IN_PROGRESS = Gauge(
    "http_requests_in_progress",
    "Requests currently being served",
)

# =============================================================================
# Prediction volume and speed
# =============================================================================
PREDICTION_COUNT = Counter(
    "ml_predictions_total",
    "Total predictions served",
    ["model_version"],
)

PREDICTION_LATENCY = Histogram(
    "ml_prediction_duration_seconds",
    "Time spent inside the model, excluding HTTP overhead",
    ["model_version"],
    buckets=[0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25],
)

PREDICTION_ERRORS = Counter(
    "ml_prediction_errors_total",
    "Predictions that failed",
    ["error_type", "model_version"],
)

BATCH_SIZE = Histogram(
    "ml_batch_size",
    "Applications per batch request",
    buckets=[1, 2, 5, 10, 25, 50, 100, 250, 500],
)

# =============================================================================
# What the model is actually saying — the ML-specific signal
# =============================================================================
# The score histogram is the single most useful ML metric in production. The
# true accuracy cannot be measured today: whether an applicant defaults is
# known next month at the earliest. The distribution of scores is observable
# immediately, and it moves before the accuracy does.
PREDICTION_SCORE = Histogram(
    "ml_prediction_score",
    "Distribution of predicted default probabilities",
    ["model_version"],
    buckets=[0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

# The business-visible consequence of those scores. When this mix shifts, the
# volume of manual underwriting work shifts with it — which is what makes the
# metric legible to people outside the ML team.
DECISION_COUNT = Counter(
    "ml_decisions_total",
    "Underwriting decisions issued",
    ["decision", "model_version"],
)

# =============================================================================
# Drift — the leading indicator
# =============================================================================
FEATURE_DRIFT_PSI = Gauge(
    "ml_feature_drift_psi",
    "Population Stability Index per feature, live window vs training reference",
    ["feature"],
)

DRIFT_SCORE = Gauge(
    "ml_drift_score",
    "Largest PSI across all monitored features",
)

DRIFT_WINDOW_SIZE = Gauge(
    "ml_drift_window_size",
    "Requests currently in the drift window",
)

# =============================================================================
# Fairness, observed in production rather than at evaluation time
# =============================================================================
SELECTION_RATE = Gauge(
    "ml_selection_rate",
    "Share of applicants in a group sent to review or decline",
    ["group"],
)

FAIRNESS_GAP = Gauge(
    "ml_fairness_gap",
    "Largest selection-rate difference between any two groups",
)

# =============================================================================
# Model identity and freshness
# =============================================================================
MODEL_LOADED = Gauge(
    "ml_model_loaded",
    "1 when the model is loaded and servable, 0 otherwise",
)

MODEL_INFO = Info(
    "ml_model",
    "Static information about the loaded model",
)

MODEL_LAST_RELOAD = Gauge(
    "ml_model_last_reload_timestamp_seconds",
    "Unix timestamp of the last model load",
)

# =============================================================================
# Explanation cost
# =============================================================================
# SHAP is far more expensive than a prediction. Measuring it separately means
# a slow /explain cannot hide inside the prediction latency percentile, and an
# alert on one does not fire because of the other.
EXPLAIN_LATENCY = Histogram(
    "ml_explain_duration_seconds",
    "Time to compute a SHAP explanation",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

EXPLAIN_COUNT = Counter(
    "ml_explanations_total",
    "Explanations served",
)
