"""
Canonical SimulationMetrics contract (Engineering Contract §12).

Assembled by Person 4 from data produced by P1, P2, P3.
"""

from datetime import datetime

from pydantic import BaseModel, Field


class SimulationMetrics(BaseModel):
    """Canonical simulation metrics structure per Engineering Contract §12."""

    run_id: str
    round_id: int
    model_version: str
    accuracy: float
    loss: float
    attack_success_rate: float | None = None
    malicious_updates: int = 0
    suspicious_updates: int = 0
    quarantined_updates: int = 0
    accepted_updates: int = 0
    downweighted_updates: int = 0
    recovery_triggered: bool = False
    recovery_count: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
