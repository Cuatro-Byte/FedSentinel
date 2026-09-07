from dataclasses import dataclass
from datetime import datetime

@dataclass
class SimulationMetrics:
    """
    Canonical structure representing metrics for a simulation round in FedSentinel.
    Tracks both model performance and security outcomes.
    """
    run_id: str
    round_id: int
    model_version: str
    accuracy: float
    loss: float
    attack_success_rate: float | None
    malicious_updates: int
    suspicious_updates: int
    quarantined_updates: int
    accepted_updates: int
    downweighted_updates: int
    recovery_triggered: bool
    recovery_count: int
    created_at: datetime
