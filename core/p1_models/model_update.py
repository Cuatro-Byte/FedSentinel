from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

@dataclass
class ModelUpdate:
    """
    Canonical structure representing a single client's model update in FedSentinel.
    This is a neutral, shared FL data structure and does not contain threat detection
    or attack logic results.
    """
    update_id: str
    run_id: str
    round_id: int
    client_id: str
    model_version: str
    base_model_version: str
    parameters: dict[str, Any]
    sample_count: int
    local_loss: float | None
    local_accuracy: float | None
    training_epochs: int
    learning_rate: float | None
    created_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)
