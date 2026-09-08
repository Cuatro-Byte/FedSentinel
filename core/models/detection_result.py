"""
Canonical DetectionResult contract (Engineering Contract §9).

Produced by Person 3's detection pipeline.
Threat score is normalized 0.00–1.00 (Contract §15).
"""

from datetime import datetime

from pydantic import BaseModel, Field

from core.models.enums import ThreatLevel, ResponseAction


class DetectionResult(BaseModel):
    """Canonical detection result structure per Engineering Contract §9."""

    update_id: str
    client_id: str
    round_id: int
    threat_score: float = Field(ge=0.0, le=1.0)
    threat_level: ThreatLevel
    action: ResponseAction
    feature_summary: dict[str, float] = Field(default_factory=dict)
    anomaly_score: float = Field(ge=0.0, le=1.0)
    similarity_score: float = Field(ge=0.0, le=1.0)
    reputation_score: float = Field(ge=0.0, le=1.0)
    explanation_codes: list[str] = Field(default_factory=list)
    detector_version: str = "detector-v1"
    created_at: datetime = Field(default_factory=datetime.utcnow)
