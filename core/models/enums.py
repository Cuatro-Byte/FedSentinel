"""
Frozen enums from the FedSentinel Engineering Contract v2.0.

These enums are shared across all team members.
Do NOT modify without following the Shared Schema Change Rule (Contract §39).
"""

from enum import Enum


class ThreatLevel(str, Enum):
    """Threat classification levels (Contract §16)."""
    SAFE = "SAFE"
    SUSPICIOUS = "SUSPICIOUS"
    MALICIOUS = "MALICIOUS"


class ResponseAction(str, Enum):
    """Response actions for detected updates (Contract §9).

    Values: ACCEPT, DOWN_WEIGHT, QUARANTINE.
    There is NO REJECT action.
    """
    ACCEPT = "ACCEPT"
    DOWN_WEIGHT = "DOWN_WEIGHT"
    QUARANTINE = "QUARANTINE"


class ImpactLevel(str, Enum):
    """Impact classification levels (Contract §10, §17)."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class RecoveryStatus(str, Enum):
    """Recovery operation status (Contract §11)."""
    NOT_REQUIRED = "NOT_REQUIRED"
    TRIGGERED = "TRIGGERED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
