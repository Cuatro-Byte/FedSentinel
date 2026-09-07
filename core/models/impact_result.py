"""
Canonical ImpactResult contract (Engineering Contract §10).

Impact estimation is SEPARATE from threat detection (Contract §10, Important distinction).
- Threat asks: "How suspicious is this update?"
- Impact asks: "How much influence/damage could this update have?"
They must not be treated as the same variable.

Produced by Person 3's impact estimation pipeline.
Impact score is normalized 0.00–1.00 (Contract §17.3).
"""

from datetime import datetime

from pydantic import BaseModel, Field

from core.models.enums import ImpactLevel


class ImpactResult(BaseModel):
    """Canonical impact result structure per Engineering Contract §10."""

    update_id: str
    client_id: str
    round_id: int
    impact_score: float = Field(ge=0.0, le=1.0)
    influence_estimate: float
    parameter_displacement: float
    aggregation_weight: float
    estimated_accuracy_change: float | None = None
    estimated_loss_change: float | None = None
    impact_level: ImpactLevel
    explanation_codes: list[str] = Field(default_factory=list)
    impact_version: str = "impact-v1"
    created_at: datetime = Field(default_factory=datetime.utcnow)
