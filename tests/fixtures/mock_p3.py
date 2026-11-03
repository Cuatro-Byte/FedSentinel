"""
Minimal deterministic P3 (Sentinel) test stub.

This is a TEST FIXTURE ONLY — not a competing security implementation.
It exists solely to exercise Person 4's orchestration, persistence, and API.

It will be replaced by Person 3's real module during integration.

The stub uses simple parameter-norm thresholds to produce deterministic
outputs. It does NOT implement real feature extraction, anomaly detection,
similarity analysis, reputation tracking, impact estimation, or recovery
intelligence.
"""

import math
import uuid
from datetime import datetime

from backend.adapters.interfaces import SentinelInterface
from core.models import ModelUpdate, DetectionResult, ImpactResult, RecoveryResult
from core.models.enums import (
    ThreatLevel, ResponseAction, ImpactLevel, RecoveryStatus,
)


def _compute_param_norm(update: ModelUpdate) -> float:
    """Compute a simple L2-like norm of parameter values for test purposes."""
    total = 0.0
    for key, values in update.parameters.items():
        if isinstance(values, list):
            total += sum(v * v for v in values)
        elif isinstance(values, (int, float)):
            total += values * values
    return math.sqrt(total)


class MockP3Sentinel(SentinelInterface):
    """Minimal deterministic Sentinel stub for testing P4 infrastructure.

    Uses simple norm thresholds — NOT real detection/impact/recovery algorithms.
    """

    NORM_SUSPICIOUS_THRESHOLD = 1.0
    NORM_MALICIOUS_THRESHOLD = 5.0

    def reset(self) -> None:
        """Reset internal state (no-op for mock)."""
        pass

    def detect(self, updates: list[ModelUpdate],
               round_id: int) -> list[DetectionResult]:
        """Produce deterministic DetectionResults based on parameter norm."""
        results = []
        for update in updates:
            norm = _compute_param_norm(update)

            if norm > self.NORM_MALICIOUS_THRESHOLD:
                threat_score = min(0.80 + (norm - self.NORM_MALICIOUS_THRESHOLD) * 0.02, 1.0)
                threat_level = ThreatLevel.MALICIOUS
                action = ResponseAction.QUARANTINE
                codes = ["HIGH_UPDATE_NORM", "OUTLIER_UPDATE"]
            elif norm > self.NORM_SUSPICIOUS_THRESHOLD:
                threat_score = 0.50 + (norm - self.NORM_SUSPICIOUS_THRESHOLD) * 0.075
                threat_score = min(threat_score, 0.79)
                threat_level = ThreatLevel.SUSPICIOUS
                action = ResponseAction.DOWN_WEIGHT
                codes = ["ELEVATED_UPDATE_NORM"]
            else:
                threat_score = norm * 0.4
                threat_level = ThreatLevel.SAFE
                action = ResponseAction.ACCEPT
                codes = []

            results.append(DetectionResult(
                update_id=update.update_id,
                client_id=update.client_id,
                round_id=round_id,
                threat_score=round(threat_score, 4),
                threat_level=threat_level,
                action=action,
                feature_summary={"param_norm": round(norm, 4)},
                anomaly_score=round(min(norm / 10.0, 1.0), 4),
                similarity_score=round(max(1.0 - norm / 10.0, 0.0), 4),
                reputation_score=round(max(1.0 - threat_score * 0.5, 0.3), 4),
                explanation_codes=codes,
                detector_version="detector-v1",
                created_at=datetime.utcnow(),
            ))
        return results

    def estimate_impact(self, updates: list[ModelUpdate],
                        detections: list[DetectionResult],
                        round_id: int) -> list[ImpactResult]:
        """Produce deterministic ImpactResults. Impact is SEPARATE from threat."""
        detection_map = {d.update_id: d for d in detections}
        results = []
        for update in updates:
            norm = _compute_param_norm(update)
            det = detection_map.get(update.update_id)

            # Impact based on parameter magnitude (separate dimension from threat)
            impact_score = min(norm / 10.0, 1.0)
            if impact_score >= 0.75:
                impact_level = ImpactLevel.CRITICAL
            elif impact_score >= 0.50:
                impact_level = ImpactLevel.HIGH
            elif impact_score >= 0.25:
                impact_level = ImpactLevel.MEDIUM
            else:
                impact_level = ImpactLevel.LOW

            codes = []
            if impact_score >= 0.50:
                codes.append("HIGH_ESTIMATED_INFLUENCE")
            if norm > self.NORM_MALICIOUS_THRESHOLD:
                codes.append("LARGE_PARAMETER_DISPLACEMENT")

            results.append(ImpactResult(
                update_id=update.update_id,
                client_id=update.client_id,
                round_id=round_id,
                impact_score=round(impact_score, 4),
                influence_estimate=round(norm * 0.1, 4),
                parameter_displacement=round(norm, 4),
                aggregation_weight=1.0 / max(len(updates), 1),
                estimated_accuracy_change=round(-impact_score * 0.05, 4) if impact_score > 0.5 else None,
                estimated_loss_change=round(impact_score * 0.1, 4) if impact_score > 0.5 else None,
                impact_level=impact_level,
                explanation_codes=codes,
                impact_version="impact-v1",
                created_at=datetime.utcnow(),
            ))
        return results

    def decide_response(self, detections: list[DetectionResult],
                        impacts: list[ImpactResult]) -> dict[str, ResponseAction]:
        """Return actions map. Uses actions already determined in detect()."""
        return {d.update_id: d.action for d in detections}

    def check_recovery(self, evaluation: dict,
                       detections: list[DetectionResult],
                       impacts: list[ImpactResult],
                       run_id: str, round_id: int,
                       model_version: str,
                       config: dict,
                       val_metrics: dict | None = None,
                       loss_spiked: bool = False) -> RecoveryResult:
        """Deterministic recovery check stub.

        Triggers recovery if any MALICIOUS detections exist AND
        accuracy dropped below threshold. NOT real recovery intelligence.
        """
        accuracy = evaluation.get("accuracy", 1.0)
        loss = evaluation.get("loss", 0.0)
        threshold = config.get("accuracy_drop_threshold", 0.05)

        malicious_detections = [
            d for d in detections
            if d.threat_level == ThreatLevel.MALICIOUS
        ]
        high_impact = [
            imp for imp in impacts
            if imp.impact_level in (ImpactLevel.HIGH, ImpactLevel.CRITICAL)
        ]

        # Simple stub: trigger if malicious + high impact exist
        if malicious_detections and high_impact:
            affected_ids = [d.update_id for d in malicious_detections]
            excluded_clients = list(set(d.client_id for d in malicious_detections))
            return RecoveryResult(
                recovery_id=f"recovery-{run_id}-r{round_id}",
                run_id=run_id,
                round_id=round_id,
                trigger="MALICIOUS_HIGH_IMPACT_DETECTED",
                affected_update_ids=affected_ids,
                excluded_client_ids=excluded_clients,
                previous_model_version=model_version,
                recovered_model_version="",  # P1 fills this after re-aggregation
                before_accuracy=accuracy,
                after_accuracy=None,
                before_loss=loss,
                after_loss=None,
                recovery_status=RecoveryStatus.TRIGGERED,
                recovery_version="recovery-v1",
                created_at=datetime.utcnow(),
            )

        return RecoveryResult(
            recovery_id=f"recovery-{run_id}-r{round_id}",
            run_id=run_id,
            round_id=round_id,
            trigger="NONE",
            affected_update_ids=[],
            excluded_client_ids=[],
            previous_model_version=model_version,
            recovered_model_version=model_version,
            before_accuracy=accuracy,
            after_accuracy=accuracy,
            before_loss=loss,
            after_loss=loss,
            recovery_status=RecoveryStatus.NOT_REQUIRED,
            recovery_version="recovery-v1",
            created_at=datetime.utcnow(),
        )
