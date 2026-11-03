"""
Canonical ModelUpdate contract (Engineering Contract §8).

Represents a single client's model update sent to the server.

Rules:
- update_id uniquely identifies one update.
- client_id identifies the simulated client.
- round_id identifies the FL round.
- parameters contain the model delta/update.
- Raw client training data is NOT included.
- Attack labels are NOT included in this object.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ModelUpdate(BaseModel):
    """Canonical model update structure per Engineering Contract §8."""

    update_id: str
    run_id: str
    round_id: int
    client_id: str
    model_version: str
    base_model_version: str
    parameters: dict[str, Any]
    sample_count: int
    local_loss: float | None = None
    local_accuracy: float | None = None
    training_epochs: int
    learning_rate: float | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": False}
