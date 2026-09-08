"""
core/models/detection_result.py

Canonical shared DetectionResult model for FedSentinel.
Owner: P1 — FL/ML Infrastructure Engineer (shared schema).

This is a pure data structure. It does NOT contain:
- detection algorithms
- threat score calculations
- anomaly/similarity/reputation logic
- attack logic
- aggregation logic

Contract reference: FEDSENTINEL_NEW_TEAM_ENGINEERING_CONTRACT_v2.md §9
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List


class ThreatLevel(str, Enum):
    """Canonical threat classification levels per contract §9."""
    SAFE = "SAFE"
    SUSPICIOUS = "SUSPICIOUS"
    MALICIOUS = "MALICIOUS"


class ResponseAction(str, Enum):
    """
    Canonical response actions per contract §9.
    P1 uses these to determine aggregation contribution:
      ACCEPT      → full sample-count weight
      DOWN_WEIGHT → reduced weight (configurable)
      QUARANTINE  → zero weight (excluded)
    """
    ACCEPT = "ACCEPT"
    DOWN_WEIGHT = "DOWN_WEIGHT"
    QUARANTINE = "QUARANTINE"


@dataclass(frozen=True)
class DetectionResult:
    """
    Canonical typed structure representing a single client's security assessment.

    This is produced by P3 Sentinel and consumed by P1 for trust-aware aggregation.
    It is frozen (immutable) to prevent post-construction mutation by any layer.

    Fields match the contract exactly (§9). All identifiers are required and
    non-empty. Numeric scores must be finite floats in [0.0, 1.0].
    """
    update_id: str
    client_id: str
    round_id: int
    threat_score: float
    threat_level: ThreatLevel
    action: ResponseAction
    feature_summary: Dict[str, float]
    anomaly_score: float
    similarity_score: float
    reputation_score: float
    explanation_codes: List[str]
    detector_version: str
    created_at: datetime

    def __post_init__(self) -> None:
        """Validate required fields and value constraints on construction."""
        import math

        if not self.update_id or not isinstance(self.update_id, str):
            raise ValueError("DetectionResult.update_id must be a non-empty string.")
        if not self.client_id or not isinstance(self.client_id, str):
            raise ValueError("DetectionResult.client_id must be a non-empty string.")
        if not isinstance(self.round_id, int) or self.round_id < 0:
            raise ValueError("DetectionResult.round_id must be a non-negative integer.")

        if not isinstance(self.threat_level, ThreatLevel):
            raise ValueError(f"DetectionResult.threat_level must be a ThreatLevel enum, got {self.threat_level!r}.")
        if not isinstance(self.action, ResponseAction):
            raise ValueError(f"DetectionResult.action must be a ResponseAction enum, got {self.action!r}.")

        for score_name, score_val in [
            ("threat_score", self.threat_score),
            ("anomaly_score", self.anomaly_score),
            ("similarity_score", self.similarity_score),
            ("reputation_score", self.reputation_score),
        ]:
            if not isinstance(score_val, (int, float)) or not math.isfinite(score_val):
                raise ValueError(f"DetectionResult.{score_name} must be a finite float.")

        if not isinstance(self.feature_summary, dict):
            raise ValueError("DetectionResult.feature_summary must be a dict.")
        if not isinstance(self.explanation_codes, list):
            raise ValueError("DetectionResult.explanation_codes must be a list.")
        if not self.detector_version or not isinstance(self.detector_version, str):
            raise ValueError("DetectionResult.detector_version must be a non-empty string.")
        if not isinstance(self.created_at, datetime):
            raise ValueError("DetectionResult.created_at must be a datetime object.")

    @staticmethod
    def from_dict(data: dict) -> "DetectionResult":
        """
        Construct a canonical DetectionResult from a raw P3 pipeline dict.

        This is the P1-side adapter. It converts:
          - ISO string → datetime for created_at
          - str → ThreatLevel enum
          - str → ResponseAction enum

        Does NOT recalculate any field values.

        Args:
            data: The raw dict from P3's build_detection_result() output.

        Returns:
            A validated, immutable DetectionResult instance.

        Raises:
            ValueError: If any required field is missing or invalid.
            KeyError: If a required field is absent in the data dict.
        """
        from datetime import timezone

        required_keys = [
            "update_id", "client_id", "round_id", "threat_score",
            "threat_level", "action", "feature_summary", "anomaly_score",
            "similarity_score", "reputation_score", "explanation_codes",
            "detector_version", "created_at",
        ]
        for key in required_keys:
            if key not in data:
                raise KeyError(f"DetectionResult.from_dict: missing required key '{key}'.")

        # Parse created_at: accept either ISO string or existing datetime
        raw_ts = data["created_at"]
        if isinstance(raw_ts, datetime):
            created_at = raw_ts
        elif isinstance(raw_ts, str):
            # Handle both offset-aware and naive ISO strings
            try:
                created_at = datetime.fromisoformat(raw_ts)
            except ValueError as e:
                raise ValueError(f"DetectionResult.from_dict: cannot parse created_at '{raw_ts}': {e}") from e
        else:
            raise ValueError(f"DetectionResult.from_dict: created_at must be a str or datetime, got {type(raw_ts).__name__}.")

        # Parse enums — raise clearly on invalid values
        raw_action = data["action"]
        try:
            action = ResponseAction(raw_action)
        except ValueError:
            raise ValueError(f"DetectionResult.from_dict: invalid action '{raw_action}'. Must be one of {[a.value for a in ResponseAction]}.")

        raw_level = data["threat_level"]
        try:
            threat_level = ThreatLevel(raw_level)
        except ValueError:
            raise ValueError(f"DetectionResult.from_dict: invalid threat_level '{raw_level}'. Must be one of {[l.value for l in ThreatLevel]}.")

        return DetectionResult(
            update_id=data["update_id"],
            client_id=data["client_id"],
            round_id=int(data["round_id"]),
            threat_score=float(data["threat_score"]),
            threat_level=threat_level,
            action=action,
            feature_summary=dict(data["feature_summary"]),
            anomaly_score=float(data["anomaly_score"]),
            similarity_score=float(data["similarity_score"]),
            reputation_score=float(data["reputation_score"]),
            explanation_codes=list(data["explanation_codes"]),
            detector_version=str(data["detector_version"]),
            created_at=created_at,
        )
