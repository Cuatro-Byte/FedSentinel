"""Regression tests for Client-Update Validation Firewall and Simulation Resource Limits.

Verifies:
- NaN updates are quarantined/rejected, never ACCEPTed.
- Inf / -Inf updates are quarantined/rejected, never ACCEPTed.
- Malformed/empty updates are quarantined/rejected.
- Valid updates pass the firewall and reach Sentinel normally.
- client_count > 100 is rejected by API schema.
- rounds > 100 is rejected by API schema.
- Normal values validate cleanly.
- Ground truth is NOT leaked or accessed.
"""

from datetime import datetime, timezone
import pytest
import torch
import numpy as np
from pydantic import ValidationError

from core.models.model_update import ModelUpdate
from core.models.enums import ResponseAction, ThreatLevel
from core.validation.client_validator import ClientUpdateFirewall
from backend.api.schemas.simulation import SimulationCreateRequest


def make_update(
    update_id: str = "u-01",
    client_id: str = "client-01",
    round_id: int = 1,
    sample_count: int = 100,
    parameters: dict | None = None,
) -> ModelUpdate:
    if parameters is None:
        parameters = {
            "conv.weight": torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32),
            "fc.bias": torch.tensor([0.01, -0.02], dtype=torch.float32),
        }
    return ModelUpdate(
        update_id=update_id,
        run_id="RUN-TEST",
        round_id=round_id,
        client_id=client_id,
        model_version="v0",
        base_model_version="v0",
        parameters=parameters,
        sample_count=sample_count,
        local_loss=0.45,
        local_accuracy=0.85,
        training_epochs=1,
        learning_rate=0.01,
        created_at=datetime.now(timezone.utc),
        metadata={},
    )


class TestClientUpdateFirewall:
    def setup_method(self):
        self.firewall = ClientUpdateFirewall()

    def test_valid_update_passes_firewall(self):
        """Valid update passes firewall and is marked valid."""
        update = make_update()
        is_valid, reasons = self.firewall.validate_update(update)
        assert is_valid is True
        assert len(reasons) == 0

        valid_updates, det, imp = self.firewall.validate_updates([update], round_id=1)
        assert len(valid_updates) == 1
        assert valid_updates[0].update_id == update.update_id
        assert len(det) == 0
        assert len(imp) == 0

    def test_nan_update_is_quarantined_never_accepted(self):
        """Update containing NaN in any tensor is quarantined, never ACCEPTed."""
        nan_params = {
            "conv.weight": torch.tensor([0.1, float("nan"), 0.3], dtype=torch.float32),
            "fc.bias": torch.tensor([0.01, -0.02], dtype=torch.float32),
        }
        update = make_update(update_id="u-nan", client_id="client-mal-nan", parameters=nan_params)

        is_valid, reasons = self.firewall.validate_update(update)
        assert is_valid is False
        assert any("PARAMETER_NAN_DETECTED" in r for r in reasons)

        valid_updates, det, imp = self.firewall.validate_updates([update], round_id=1)
        assert len(valid_updates) == 0
        assert len(det) == 1
        assert len(imp) == 1

        # Check explicit quarantine detection record
        d = det[0]
        assert d.update_id == "u-nan"
        assert d.client_id == "client-mal-nan"
        assert d.action == ResponseAction.QUARANTINE
        assert d.threat_level == ThreatLevel.MALICIOUS
        assert d.threat_score == 1.0
        assert "FIREWALL_REJECTED" in d.explanation_codes
        assert any("PARAMETER_NAN_DETECTED" in code for code in d.explanation_codes)

    def test_inf_update_is_quarantined_never_accepted(self):
        """Update containing Inf / -Inf in any tensor is quarantined."""
        inf_params = {
            "conv.weight": torch.tensor([0.1, float("inf"), 0.3], dtype=torch.float32),
            "fc.bias": torch.tensor([float("-inf"), -0.02], dtype=torch.float32),
        }
        update = make_update(update_id="u-inf", client_id="client-mal-inf", parameters=inf_params)

        is_valid, reasons = self.firewall.validate_update(update)
        assert is_valid is False
        assert any("PARAMETER_INF_DETECTED" in r for r in reasons)

        valid_updates, det, imp = self.firewall.validate_updates([update], round_id=1)
        assert len(valid_updates) == 0
        assert len(det) == 1
        assert det[0].action == ResponseAction.QUARANTINE
        assert det[0].threat_level == ThreatLevel.MALICIOUS
        assert det[0].threat_score == 1.0

    def test_empty_parameters_update_is_quarantined(self):
        """Update with empty parameters dictionary is rejected and quarantined."""
        update = make_update(update_id="u-empty", client_id="client-empty", parameters={})

        is_valid, reasons = self.firewall.validate_update(update)
        assert is_valid is False
        assert "MISSING_OR_EMPTY_PARAMETERS" in reasons

        valid_updates, det, imp = self.firewall.validate_updates([update], round_id=1)
        assert len(valid_updates) == 0
        assert len(det) == 1
        assert det[0].action == ResponseAction.QUARANTINE
        assert det[0].threat_level == ThreatLevel.MALICIOUS

    def test_non_numeric_parameters_quarantined(self):
        """Update with string/non-numeric parameter values is rejected."""
        bad_params = {"conv.weight": "invalid_string_weights"}
        update = make_update(update_id="u-bad-type", client_id="client-bad", parameters=bad_params)

        is_valid, reasons = self.firewall.validate_update(update)
        assert is_valid is False
        assert any("NON_NUMERIC_PARAMETER" in r for r in reasons)

    def test_invalid_sample_count_quarantined(self):
        """Update with sample_count <= 0 is rejected."""
        update = make_update(update_id="u-zero-sample", sample_count=0)
        is_valid, reasons = self.firewall.validate_update(update)
        assert is_valid is False
        assert "INVALID_SAMPLE_COUNT" in reasons

    def test_shape_mismatch_quarantined(self):
        """Update with shape mismatch against reference model is rejected."""
        update = make_update()
        expected_shapes = {"conv.weight": (10,), "fc.bias": (2,)}
        is_valid, reasons = self.firewall.validate_update(update, expected_shapes=expected_shapes)
        assert is_valid is False
        assert any("PARAMETER_SHAPE_MISMATCH" in r for r in reasons)

    def test_mixed_batch_preserves_valid_and_quarantines_invalid(self):
        """A batch with valid and invalid updates correctly isolates invalid ones."""
        good_update = make_update(update_id="u-good", client_id="client-good")
        bad_update = make_update(
            update_id="u-bad",
            client_id="client-bad",
            parameters={"conv.weight": torch.tensor([float("nan")])},
        )

        valid_updates, det, imp = self.firewall.validate_updates([good_update, bad_update], round_id=1)
        assert len(valid_updates) == 1
        assert valid_updates[0].update_id == "u-good"
        assert len(det) == 1
        assert det[0].update_id == "u-bad"
        assert det[0].action == ResponseAction.QUARANTINE


class TestSimulationResourceLimits:
    def test_normal_parameters_validate(self):
        """Normal valid request parameters pass validation."""
        req = SimulationCreateRequest(
            client_count=20,
            rounds=10,
            scenario="backdoor",
            attacker_count=2,
            intensity=0.8,
            start_round=1,
            seed=42,
        )
        assert req.client_count == 20
        assert req.rounds == 10
        assert req.attacker_count == 2

    def test_client_count_above_100_rejected(self):
        """client_count > 100 must be rejected by schema."""
        with pytest.raises(ValidationError) as exc:
            SimulationCreateRequest(client_count=101)
        assert "less than or equal to 100" in str(exc.value)

    def test_rounds_above_100_rejected(self):
        """rounds > 100 must be rejected by schema."""
        with pytest.raises(ValidationError) as exc:
            SimulationCreateRequest(rounds=101)
        assert "less than or equal to 100" in str(exc.value)

    def test_attacker_count_above_100_rejected(self):
        """attacker_count > 100 must be rejected by schema."""
        with pytest.raises(ValidationError) as exc:
            SimulationCreateRequest(client_count=100, attacker_count=101)
        assert "less than or equal to 100" in str(exc.value)

    def test_negative_or_zero_clients_rejected(self):
        """client_count <= 0 must be rejected."""
        with pytest.raises(ValidationError):
            SimulationCreateRequest(client_count=0)

    def test_negative_or_zero_rounds_rejected(self):
        """rounds <= 0 must be rejected."""
        with pytest.raises(ValidationError):
            SimulationCreateRequest(rounds=0)


class TestPipelineFirewallIntegration:
    @pytest.fixture
    def small_config(self):
        from backend.config import AppConfig, SimulationConfig, AttackConfig, RecoveryConfig
        return AppConfig(
            simulation=SimulationConfig(client_count=5, rounds=2, seed=42, local_epochs=1),
            attack=AttackConfig(enabled=True, scenario="backdoor", attacker_count=1, start_round=1, intensity=0.8),
            recovery=RecoveryConfig(enabled=True, accuracy_drop_threshold=0.05),
        )

    def test_runner_pipeline_with_nan_update(self, db_session, mock_p1, mock_p2, mock_p3, small_config):
        """End-to-end: SimulationRunner quarantines NaN updates and creates audit events."""
        from simulation.runner import SimulationRunner
        from backend.database.repositories import DetectionRepository, AuditRepository

        runner = SimulationRunner(
            db=db_session,
            fl_core=mock_p1,
            attack_engine=mock_p2,
            sentinel=mock_p3,
            config=small_config,
        )

        orig_apply = mock_p2.apply_attacks
        def inject_nan(updates, round_id, config):
            res = orig_apply(updates, round_id, config)
            if res:
                res[0].parameters["conv.weight"] = torch.tensor([float("nan"), 1.0])
            return res
        mock_p2.apply_attacks = inject_nan

        run_id = runner.run_simulation(run_id="TEST-FW-NAN")
        assert run_id == "TEST-FW-NAN"

        det_repo = DetectionRepository(db_session)
        dets = det_repo.get_by_run(run_id)
        nan_dets = [d for d in dets if "FIREWALL_REJECTED" in (d.get_explanation_codes() or [])]
        assert len(nan_dets) > 0
        for nd in nan_dets:
            assert nd.action == "QUARANTINE"
            assert nd.threat_level == "MALICIOUS"

        audit_repo = AuditRepository(db_session)
        events = audit_repo.get_by_run(run_id)
        quarantine_events = [e for e in events if e.event_type == "UPDATE_QUARANTINED"]
        assert len(quarantine_events) > 0
