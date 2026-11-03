"""
Test: Canonical contract models validate correctly.

Verifies that the frozen Pydantic models from core/models/ enforce
the correct field types, enum values, and validation rules.
"""

import pytest
from datetime import datetime

from core.models import (
    ModelUpdate, DetectionResult, ImpactResult, RecoveryResult, SimulationMetrics,
)
from core.models.enums import (
    ThreatLevel, ResponseAction, ImpactLevel, RecoveryStatus,
)


class TestEnums:
    """Verify frozen enum values."""

    def test_threat_level_values(self):
        assert ThreatLevel.SAFE == "SAFE"
        assert ThreatLevel.SUSPICIOUS == "SUSPICIOUS"
        assert ThreatLevel.MALICIOUS == "MALICIOUS"
        assert len(ThreatLevel) == 3

    def test_response_action_values(self):
        """No REJECT — only ACCEPT, DOWN_WEIGHT, QUARANTINE."""
        assert ResponseAction.ACCEPT == "ACCEPT"
        assert ResponseAction.DOWN_WEIGHT == "DOWN_WEIGHT"
        assert ResponseAction.QUARANTINE == "QUARANTINE"
        assert len(ResponseAction) == 3
        # Verify REJECT does not exist
        with pytest.raises(ValueError):
            ResponseAction("REJECT")

    def test_impact_level_values(self):
        assert ImpactLevel.LOW == "LOW"
        assert ImpactLevel.MEDIUM == "MEDIUM"
        assert ImpactLevel.HIGH == "HIGH"
        assert ImpactLevel.CRITICAL == "CRITICAL"
        assert len(ImpactLevel) == 4

    def test_recovery_status_values(self):
        assert RecoveryStatus.NOT_REQUIRED == "NOT_REQUIRED"
        assert RecoveryStatus.TRIGGERED == "TRIGGERED"
        assert RecoveryStatus.COMPLETED == "COMPLETED"
        assert RecoveryStatus.FAILED == "FAILED"
        assert len(RecoveryStatus) == 4


class TestModelUpdate:
    """Verify ModelUpdate contract (§8)."""

    def test_valid_model_update(self):
        update = ModelUpdate(
            update_id="U001",
            run_id="RUN-001",
            round_id=1,
            client_id="client-0",
            model_version="model-v1",
            base_model_version="model-v0",
            parameters={"layer1.weight": [0.1, 0.2]},
            sample_count=100,
            local_loss=0.3,
            local_accuracy=0.87,
            training_epochs=2,
            learning_rate=0.01,
        )
        assert update.update_id == "U001"
        assert update.round_id == 1
        assert update.sample_count == 100
        assert isinstance(update.created_at, datetime)

    def test_optional_fields(self):
        update = ModelUpdate(
            update_id="U002",
            run_id="RUN-001",
            round_id=1,
            client_id="client-1",
            model_version="model-v0",
            base_model_version="model-v0",
            parameters={},
            sample_count=50,
            training_epochs=1,
        )
        assert update.local_loss is None
        assert update.local_accuracy is None
        assert update.learning_rate is None
        assert update.metadata == {}


class TestDetectionResult:
    """Verify DetectionResult contract (§9)."""

    def test_valid_detection_result(self):
        result = DetectionResult(
            update_id="U001",
            client_id="client-0",
            round_id=1,
            threat_score=0.91,
            threat_level=ThreatLevel.MALICIOUS,
            action=ResponseAction.QUARANTINE,
            feature_summary={"param_norm": 5.2},
            anomaly_score=0.88,
            similarity_score=0.15,
            reputation_score=0.31,
            explanation_codes=["HIGH_UPDATE_NORM", "LOW_PEER_SIMILARITY"],
            detector_version="detector-v1",
        )
        assert result.threat_score == 0.91
        assert result.threat_level == ThreatLevel.MALICIOUS
        assert result.action == ResponseAction.QUARANTINE

    def test_threat_score_range(self):
        """Threat score must be 0.00–1.00."""
        with pytest.raises(Exception):
            DetectionResult(
                update_id="U001", client_id="c0", round_id=1,
                threat_score=1.5,  # Invalid: > 1.0
                threat_level=ThreatLevel.SAFE,
                action=ResponseAction.ACCEPT,
                anomaly_score=0.1, similarity_score=0.9, reputation_score=0.8,
            )

    def test_no_reject_action(self):
        """REJECT is NOT a valid ResponseAction."""
        with pytest.raises(ValueError):
            DetectionResult(
                update_id="U001", client_id="c0", round_id=1,
                threat_score=0.5,
                threat_level=ThreatLevel.SUSPICIOUS,
                action="REJECT",  # Invalid
                anomaly_score=0.5, similarity_score=0.5, reputation_score=0.5,
            )


class TestImpactResult:
    """Verify ImpactResult contract (§10)."""

    def test_valid_impact_result(self):
        result = ImpactResult(
            update_id="U001",
            client_id="client-0",
            round_id=1,
            impact_score=0.86,
            influence_estimate=0.72,
            parameter_displacement=3.5,
            aggregation_weight=0.05,
            estimated_accuracy_change=-0.03,
            estimated_loss_change=0.08,
            impact_level=ImpactLevel.CRITICAL,
            explanation_codes=["HIGH_ESTIMATED_INFLUENCE"],
        )
        assert result.impact_score == 0.86
        assert result.impact_level == ImpactLevel.CRITICAL
        # Impact is separate from threat
        assert hasattr(result, "impact_score")
        assert hasattr(result, "influence_estimate")

    def test_impact_score_range(self):
        """Impact score must be 0.00–1.00."""
        with pytest.raises(Exception):
            ImpactResult(
                update_id="U001", client_id="c0", round_id=1,
                impact_score=-0.1,  # Invalid
                influence_estimate=0.0,
                parameter_displacement=0.0,
                aggregation_weight=0.0,
                impact_level=ImpactLevel.LOW,
            )


class TestRecoveryResult:
    """Verify RecoveryResult contract (§11)."""

    def test_valid_recovery_result(self):
        result = RecoveryResult(
            recovery_id="REC-001",
            run_id="RUN-001",
            round_id=5,
            trigger="MALICIOUS_HIGH_IMPACT_DETECTED",
            affected_update_ids=["U001", "U002"],
            excluded_client_ids=["client-17"],
            previous_model_version="model-v5",
            recovered_model_version="model-v5-recovered-1",
            before_accuracy=0.72,
            after_accuracy=0.89,
            before_loss=0.55,
            after_loss=0.25,
            recovery_status=RecoveryStatus.COMPLETED,
        )
        assert result.recovery_status == RecoveryStatus.COMPLETED
        assert len(result.affected_update_ids) == 2
        assert "client-17" in result.excluded_client_ids


class TestSimulationMetrics:
    """Verify SimulationMetrics contract (§12)."""

    def test_valid_metrics(self):
        metrics = SimulationMetrics(
            run_id="RUN-001",
            round_id=5,
            model_version="model-v5",
            accuracy=0.89,
            loss=0.25,
            malicious_updates=2,
            suspicious_updates=1,
            quarantined_updates=2,
            accepted_updates=17,
            downweighted_updates=1,
            recovery_triggered=True,
            recovery_count=1,
        )
        assert metrics.accuracy == 0.89
        assert metrics.recovery_triggered is True
        assert metrics.accepted_updates == 17
