"""
recovery_models.py
==================
FedSentinel — Person B (Person 3): Sentinel Recovery Module

Defines the data models used by the ImpactEstimator and the recovery
decision engine.  These models are intentionally lightweight — they carry
only the fields that the Sentinel side needs to communicate its decision
to the rest of the pipeline.

Allowed imports (Engineering Contract v2.0):
    typing, dataclasses, enum, datetime, collections
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# RecoveryState
# ---------------------------------------------------------------------------

class RecoveryState(Enum):
    """
    Represents the outcome of a single recovery-evaluation pass.

    Attributes
    ----------
    NOT_REQUIRED:
        Anomaly scores and validation metrics do not warrant recovery.
        The global model is considered clean for this round.
    TRIGGERED:
        Either a validation loss spike was detected (VALIDATION_LOSS_SPIKE)
        or anomaly scores for one or more clients persistently exceeded the
        configured threshold across the look-back window.  The caller should
        proceed with selective re-aggregation using the quarantined client
        list returned in RecoveryDecision.
    """

    NOT_REQUIRED = "NOT_REQUIRED"
    TRIGGERED = "TRIGGERED"


# ---------------------------------------------------------------------------
# RecoveryDecision
# ---------------------------------------------------------------------------

@dataclass
class RecoveryDecision:
    """
    The complete output of one call to ``ImpactEstimator.evaluate_recovery_need``.

    Fields
    ------
    state : RecoveryState
        Whether recovery was triggered or not.
    recovery_round : Optional[int]
        The FL round number to which the system should recover.  This is
        the earliest recent round that contained clients now quarantined.
        ``None`` when state is NOT_REQUIRED or history is empty.
    quarantined_clients : List[str]
        Client IDs that exceeded the anomaly threshold across the look-back
        window, or whose presence in round history makes them suspect.
        Empty list when state is NOT_REQUIRED.
    trigger_reason : str
        Human-readable reason code for the decision, e.g.:
            "NOT_REQUIRED"
            "ANOMALY_THRESHOLD_EXCEEDED"
            "VALIDATION_LOSS_SPIKE"
    validation_metrics : Optional[Dict[str, float]]
        Pass-through of the validation metrics provided to
        ``evaluate_recovery_need``.  None when no metrics were supplied.
        Keys: "val_loss", "val_acc", "loss_delta", "acc_delta".
    evaluated_at : datetime
        UTC timestamp recorded when the decision was produced.

    Notes
    -----
    This dataclass does NOT trigger any recovery.  It is a pure data
    carrier.  Checkpoint loading, re-aggregation, and model restoration
    are the responsibility of Person 1 (FL Core).
    """

    state: RecoveryState
    recovery_round: Optional[int]
    quarantined_clients: List[str] = field(default_factory=list)
    trigger_reason: str = "NOT_REQUIRED"
    validation_metrics: Optional[Dict[str, float]] = None
    evaluated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def is_triggered(self) -> bool:
        """Return True when the state is TRIGGERED."""
        return self.state is RecoveryState.TRIGGERED

    def summary(self) -> str:
        """
        Return a single-line human-readable summary of the decision.

        Useful for logging without importing the full dataclass repr.
        """
        if self.is_triggered:
            clients = ", ".join(self.quarantined_clients) if self.quarantined_clients else "none"
            return (
                f"[TRIGGERED] reason={self.trigger_reason} | "
                f"recovery_round={self.recovery_round} | "
                f"quarantined=[{clients}]"
            )
        return f"[NOT_REQUIRED] reason={self.trigger_reason}"

    def __repr__(self) -> str:  # noqa: D401
        return (
            f"RecoveryDecision("
            f"state={self.state.value!r}, "
            f"recovery_round={self.recovery_round!r}, "
            f"quarantined_clients={self.quarantined_clients!r}, "
            f"trigger_reason={self.trigger_reason!r}, "
            f"validation_metrics={self.validation_metrics!r}, "
            f"evaluated_at={self.evaluated_at.isoformat()!r}"
            f")"
        )
