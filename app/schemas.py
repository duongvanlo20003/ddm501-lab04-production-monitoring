"""
Pydantic schemas for request/response validation.

"""

from typing import Dict, List, Literal

from pydantic import BaseModel, Field, field_validator


# =============================================================================
# Request
# =============================================================================
class CreditApplication(BaseModel):
    """One applicant, as the core banking system sends it."""

    limit_bal: float = Field(
        ..., gt=0, le=2_000_000, description="Credit limit in NT dollars", examples=[120000]
    )
    sex: Literal[1, 2] = Field(..., description="1 = male, 2 = female", examples=[2])
    education: Literal[1, 2, 3, 4] = Field(
        ...,
        description="1 = graduate school, 2 = university, 3 = high school, 4 = others",
        examples=[2],
    )
    marriage: Literal[1, 2, 3] = Field(
        ..., description="1 = married, 2 = single, 3 = others", examples=[2]
    )
    age: int = Field(..., ge=18, le=100, description="Age in years", examples=[34])

    pay_status: List[int] = Field(
        ...,
        min_length=6,
        max_length=6,
        description=(
            "Repayment status for months t-1 .. t-6. "
            "-2 = no consumption, -1 = paid in full, 0 = revolving credit, "
            "1..8 = months of payment delay."
        ),
        examples=[[0, 0, 0, 0, 0, 0]],
    )
    bill_amt: List[float] = Field(
        ...,
        min_length=6,
        max_length=6,
        description="Bill statement amount for months t-1 .. t-6",
        examples=[[45000, 43000, 41000, 39000, 37000, 35000]],
    )
    pay_amt: List[float] = Field(
        ...,
        min_length=6,
        max_length=6,
        description="Amount paid for months t-1 .. t-6",
        examples=[[2000, 1800, 1700, 1600, 1500, 1400]],
    )

    @field_validator("pay_status")
    @classmethod
    def _check_pay_status(cls, v: List[int]) -> List[int]:
        if any(x < -2 or x > 8 for x in v):
            raise ValueError("pay_status values must be between -2 and 8")
        return v

    @field_validator("pay_amt")
    @classmethod
    def _check_pay_amt(cls, v: List[float]) -> List[float]:
        if any(x < 0 for x in v):
            raise ValueError("pay_amt values cannot be negative")
        return v


# =============================================================================
# Responses
# =============================================================================
class PredictionResponse(BaseModel):
    """Scoring result plus the decision derived from it."""

    default_probability: float = Field(
        ..., ge=0.0, le=1.0, description="Probability the applicant defaults next month"
    )
    risk_band: Literal["LOW", "MEDIUM", "HIGH"] = Field(
        ..., description="Coarse risk bucket, derived from the thresholds"
    )
    decision: Literal["APPROVE", "REVIEW", "DECLINE"] = Field(
        ..., description="Underwriting action implied by the score"
    )
    review_threshold: float = Field(..., description="Score above which a human reviews")
    decline_threshold: float = Field(
        ..., description="Score above which the application is declined"
    )
    model_version: str = Field(..., description="Version of the model that produced the score")


class HealthResponse(BaseModel):
    """Liveness and readiness of the service."""

    status: Literal["healthy", "unhealthy"]
    model_loaded: bool
    model_version: str


class BatchPredictionRequest(BaseModel):
    """Up to 500 applications scored in one call."""

    applications: List[CreditApplication] = Field(..., min_length=1, max_length=500)


class BatchPredictionResponse(BaseModel):
    """Results in the same order as the request."""

    predictions: List[PredictionResponse]
    total_count: int


# =============================================================================
# Monitoring and explanation (Lab 4)
# =============================================================================
class FeatureContribution(BaseModel):
    """One feature's effect on one decision."""

    feature: str
    value: float
    contribution: float = Field(..., description="SHAP value, in log-odds")
    direction: Literal["increases risk", "reduces risk"]


class ExplanationResponse(BaseModel):
    """Why this applicant got this score."""

    default_probability: float = Field(..., ge=0.0, le=1.0)
    decision: Literal["APPROVE", "REVIEW", "DECLINE"]
    base_value: float = Field(..., description="Model output before any feature moves it")
    contributions: List[FeatureContribution]
    model_version: str
    note: str


class MonitoringResponse(BaseModel):
    """Drift and fairness over the live window."""

    window_size: int
    min_window_size: int
    sufficient_data: bool = Field(
        ..., description="False while the window is too small for the numbers to mean anything"
    )
    feature_psi: Dict[str, float]
    drift_score: float
    drift_status: Literal["stable", "moderate", "significant"]
    selection_rate: Dict[str, float]
    fairness_gap: float
