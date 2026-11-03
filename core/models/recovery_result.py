"""
Canonical RecoveryResult contract (Engineering Contract §11).

Recovery is selective — it identifies the smallest reasonable set of
suspicious/high-impact updates, not a blanket discard (Contract §20.3).

Recovery trigger intelligence is owned by Person 3.
Checkpoint handling and selective re-aggregation are owned by Person 1.
Person 4 orchestrates and persists the result.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from core.models.enums import RecoveryStatus


class RecoveryResult(BaseModel):
    """Canonical recovery result structure per Engineering Contract §11."""

    recovery_id: str
    run_id: str
    round_id: int
    trigger: str
    affected_update_ids: list[str] = Field(default_factory=list)
    excluded_client_ids: list[str] = Field(default_factory=list)
    previous_model_version: str
    recovered_model_version: str
    before_accuracy: float | None = None
    after_accuracy: float | None = None
    before_loss: float | None = None
    after_loss: float | None = None
    recovery_status: RecoveryStatus
    recovery_version: str = "recovery-v1"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    selected_action: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
