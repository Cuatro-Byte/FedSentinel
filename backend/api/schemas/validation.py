"""
Validation telemetry schemas.

API response models for server validation gate telemetry.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class ValidationRecordResponse(BaseModel):
    """Schema for an individual server validation gate evaluation record."""
    run_id: str
    round_id: int
    model_version: Optional[str] = None
    validation_loss: float
    validation_accuracy: float
    loss_spiked: bool
    baseline_loss: Optional[float] = None
    baseline_accuracy: Optional[float] = None
    loss_delta: float
    accuracy_delta: float
    validation_status: str  # "HEALTHY" or "ANOMALOUS"
    created_at: Optional[str] = None
