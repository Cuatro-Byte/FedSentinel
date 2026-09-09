"""Client-update validation firewall implementation for FedSentinel.

Enforces structural and numerical integrity of client model updates BEFORE
they reach the P3 Sentinel intelligence pipeline or aggregation.

Contract & Security invariants:
- Rejects/quarantines updates containing NaN, Inf, empty/missing parameters,
  malformed values, or incompatible shapes/keys.
- Does NOT access or depend on attack ground truth.
- Preserves client_id and update_id for the audit trail.
- Returns explicit quarantine DetectionResult and ImpactResult objects so
  invalid updates are safely recorded and never fall through to ACCEPT.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import torch

from core.models.model_update import ModelUpdate
from core.models.detection_result import DetectionResult
from core.models.impact_result import ImpactResult
from core.models.enums import ThreatLevel, ResponseAction, ImpactLevel


class ClientUpdateFirewall:
    """Validation firewall for client-side model updates prior to Sentinel processing."""

    def __init__(self, firewall_version: str = "firewall-v1") -> None:
        self.firewall_version = firewall_version

    def validate_update(
        self,
        update: Any,
        expected_keys: Optional[Set[str]] = None,
        expected_shapes: Optional[Dict[str, Tuple[int, ...]]] = None,
    ) -> Tuple[bool, List[str]]:
        """Validate a single client update.

        Args:
            update: The ModelUpdate object to validate.
            expected_keys: Optional set of parameter tensor keys that must be present.
            expected_shapes: Optional dictionary mapping parameter names to expected shapes.

        Returns:
            Tuple of (is_valid, list_of_rejection_reasons).
        """
        reasons: List[str] = []

        if not isinstance(update, ModelUpdate):
            return False, ["INVALID_OBJECT_TYPE"]

        # Validate basic identifiers
        if not getattr(update, "update_id", None) or not isinstance(update.update_id, str):
            reasons.append("INVALID_UPDATE_ID")
        if not getattr(update, "client_id", None) or not isinstance(update.client_id, str):
            reasons.append("INVALID_CLIENT_ID")

        # Validate sample count
        sample_count = getattr(update, "sample_count", None)
        if sample_count is None or not isinstance(sample_count, (int, np.integer)) or sample_count <= 0:
            reasons.append("INVALID_SAMPLE_COUNT")

        # Validate parameters dictionary
        params = getattr(update, "parameters", None)
        if params is None or not isinstance(params, dict) or len(params) == 0:
            reasons.append("MISSING_OR_EMPTY_PARAMETERS")
            return False, reasons

        # Validate expected keys if reference keys are provided
        param_keys = set(params.keys())
        if expected_keys is not None:
            if param_keys != expected_keys:
                missing_keys = expected_keys - param_keys
                extra_keys = param_keys - expected_keys
                if missing_keys:
                    reasons.append(f"MISSING_PARAMETER_KEYS:{sorted(list(missing_keys))[:3]}")
                if extra_keys:
                    reasons.append(f"EXTRA_PARAMETER_KEYS:{sorted(list(extra_keys))[:3]}")

        # Validate each parameter tensor
        for key, value in params.items():
            if not isinstance(key, str) or not key.strip():
                reasons.append("INVALID_PARAMETER_KEY")
                continue

            # Check parameter type
            if isinstance(value, torch.Tensor):
                tensor = value
            elif isinstance(value, np.ndarray):
                try:
                    tensor = torch.as_tensor(value)
                except Exception:
                    reasons.append(f"NON_NUMERIC_PARAMETER:{key}")
                    continue
            else:
                reasons.append(f"NON_NUMERIC_PARAMETER:{key}")
                continue

            # Verify tensor data type is floating point or integer
            if not (tensor.is_floating_point() or tensor.dtype in (torch.int32, torch.int64, torch.int16, torch.int8)):
                reasons.append(f"INCOMPATIBLE_DATA_TYPE:{key}")
                continue

            # Check for NaN values
            if torch.isnan(tensor).any():
                reasons.append(f"PARAMETER_NAN_DETECTED:{key}")

            # Check for Inf / -Inf values
            if torch.isinf(tensor).any():
                reasons.append(f"PARAMETER_INF_DETECTED:{key}")

            # Check expected tensor shapes if reference shapes provided
            if expected_shapes is not None and key in expected_shapes:
                expected_shape = tuple(expected_shapes[key])
                actual_shape = tuple(tensor.shape)
                if actual_shape != expected_shape:
                    reasons.append(f"PARAMETER_SHAPE_MISMATCH:{key}:expected_{expected_shape}_got_{actual_shape}")

        is_valid = len(reasons) == 0
        return is_valid, reasons

    def create_quarantine_records(
        self,
        update: ModelUpdate,
        round_id: int,
        reasons: List[str],
    ) -> Tuple[DetectionResult, ImpactResult]:
        """Create explicit quarantine DetectionResult and ImpactResult for an invalid update."""
        now = datetime.now(timezone.utc)

        detection = DetectionResult(
            update_id=update.update_id if hasattr(update, "update_id") else "unknown",
            client_id=update.client_id if hasattr(update, "client_id") else "unknown",
            round_id=round_id,
            threat_score=1.0,
            threat_level=ThreatLevel.MALICIOUS,
            action=ResponseAction.QUARANTINE,
            feature_summary={
                "firewall_rejected": 1.0,
                "reason_count": float(len(reasons)),
            },
            anomaly_score=1.0,
            similarity_score=0.0,
            reputation_score=0.0,
            explanation_codes=["FIREWALL_REJECTED"] + reasons,
            detector_version=self.firewall_version,
            created_at=now,
        )

        impact = ImpactResult(
            impact_id=f"imp-firewall-{update.update_id if hasattr(update, 'update_id') else 'unknown'}",
            update_id=update.update_id if hasattr(update, "update_id") else "unknown",
            client_id=update.client_id if hasattr(update, "client_id") else "unknown",
            round_id=round_id,
            impact_score=1.0,
            influence_estimate=0.0,
            parameter_displacement=0.0,
            aggregation_weight=0.0,
            impact_level=ImpactLevel.CRITICAL,
            explanation_codes=["FIREWALL_REJECTED"] + reasons,
            impact_version=self.firewall_version,
            created_at=now,
        )

        return detection, impact

    def validate_updates(
        self,
        updates: List[ModelUpdate],
        round_id: int,
        expected_keys: Optional[Set[str]] = None,
        expected_shapes: Optional[Dict[str, Tuple[int, ...]]] = None,
    ) -> Tuple[List[ModelUpdate], List[DetectionResult], List[ImpactResult]]:
        """Filter updates through the firewall.

        Returns:
            Tuple of:
            - valid_updates: List of updates that passed all checks.
            - quarantine_detections: DetectionResult objects with QUARANTINE action for invalid updates.
            - quarantine_impacts: ImpactResult objects for invalid updates.
        """
        valid_updates: List[ModelUpdate] = []
        quarantine_detections: List[DetectionResult] = []
        quarantine_impacts: List[ImpactResult] = []

        for update in updates:
            is_valid, reasons = self.validate_update(
                update,
                expected_keys=expected_keys,
                expected_shapes=expected_shapes,
            )
            if is_valid:
                valid_updates.append(update)
            else:
                det, imp = self.create_quarantine_records(update, round_id, reasons)
                quarantine_detections.append(det)
                quarantine_impacts.append(imp)

        return valid_updates, quarantine_detections, quarantine_impacts
