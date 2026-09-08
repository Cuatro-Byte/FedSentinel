"""
Test: Mock adapters comply with interfaces and produce valid contract models.
"""

import pytest

from core.models import ModelUpdate, DetectionResult, ImpactResult, RecoveryResult
from core.models.enums import ThreatLevel, ResponseAction, ImpactLevel, RecoveryStatus


class TestMockP1:
    def test_initialize(self, mock_p1):
        version = mock_p1.initialize_global_model({})
        assert version == "model-v0"

    def test_train_clients(self, mock_p1):
        mock_p1.initialize_global_model({})
        updates = mock_p1.train_clients(
            "RUN-001", 1, "model-v0",
            ["c0", "c1", "c2"], {"local_epochs": 2}
        )
        assert len(updates) == 3
        for u in updates:
            assert isinstance(u, ModelUpdate)
            assert u.run_id == "RUN-001"
            assert u.round_id == 1

    def test_aggregate(self, mock_p1):
        mock_p1.initialize_global_model({})
        updates = mock_p1.train_clients(
            "RUN-001", 1, "model-v0", ["c0"], {"local_epochs": 2}
        )
        actions = {u.update_id: ResponseAction.ACCEPT for u in updates}
        new_version = mock_p1.aggregate(updates, actions, "model-v0")
        assert new_version.startswith("model-v")

    def test_evaluate(self, mock_p1):
        mock_p1.initialize_global_model({})
        result = mock_p1.evaluate("model-v0")
        assert "accuracy" in result
        assert "loss" in result
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_re_aggregate(self, mock_p1):
        mock_p1.initialize_global_model({})
        updates = mock_p1.train_clients(
            "RUN-001", 1, "model-v0", ["c0", "c1"], {"local_epochs": 2}
        )
        recovered = mock_p1.re_aggregate(updates, ["c1"], "model-v1")
        assert "recovered" in recovered


class TestMockP2:
    def test_no_attack_before_start_round(self, mock_p2):
        from tests.fixtures.mock_p1 import MockP1FL
        p1 = MockP1FL()
        p1.initialize_global_model({})
        updates = p1.train_clients(
            "RUN-001", 1, "model-v0", ["c0", "c1"], {"local_epochs": 2}
        )
        config = {"enabled": True, "start_round": 5, "attacker_count": 1, "client_count": 2}
        result = mock_p2.apply_attacks(updates, 1, config)
        # Before start_round, updates should be unchanged
        for orig, mod in zip(updates, result):
            assert orig.parameters == mod.parameters

    def test_attack_after_start_round(self, mock_p2):
        from tests.fixtures.mock_p1 import MockP1FL
        p1 = MockP1FL()
        p1.initialize_global_model({})
        updates = p1.train_clients(
            "RUN-001", 5, "model-v0",
            ["client-0", "client-1"], {"local_epochs": 2}
        )
        config = {
            "enabled": True, "start_round": 5, "intensity": 0.8,
            "attacker_count": 1, "client_count": 2
        }
        result = mock_p2.apply_attacks(updates, 5, config)
        # Last client should have modified (scaled) parameters
        attacker_ids = mock_p2.get_attacker_ids(5, config)
        assert len(attacker_ids) == 1

    def test_get_attacker_ids(self, mock_p2):
        ids = mock_p2.get_attacker_ids(5, {"attacker_count": 2, "client_count": 20})
        assert len(ids) == 2


class TestMockP3:
    def test_detect_produces_valid_results(self, mock_p3):
        from tests.fixtures.mock_p1 import MockP1FL
        p1 = MockP1FL()
        p1.initialize_global_model({})
        updates = p1.train_clients("RUN-001", 1, "model-v0", ["c0"], {"local_epochs": 2})
        detections = mock_p3.detect(updates, 1)
        assert len(detections) == 1
        d = detections[0]
        assert isinstance(d, DetectionResult)
        assert 0.0 <= d.threat_score <= 1.0
        assert d.threat_level in list(ThreatLevel)
        assert d.action in list(ResponseAction)

    def test_estimate_impact_produces_valid_results(self, mock_p3):
        from tests.fixtures.mock_p1 import MockP1FL
        p1 = MockP1FL()
        p1.initialize_global_model({})
        updates = p1.train_clients("RUN-001", 1, "model-v0", ["c0"], {"local_epochs": 2})
        detections = mock_p3.detect(updates, 1)
        impacts = mock_p3.estimate_impact(updates, detections, 1)
        assert len(impacts) == 1
        imp = impacts[0]
        assert isinstance(imp, ImpactResult)
        assert 0.0 <= imp.impact_score <= 1.0
        assert imp.impact_level in list(ImpactLevel)

    def test_decide_response(self, mock_p3):
        from tests.fixtures.mock_p1 import MockP1FL
        p1 = MockP1FL()
        p1.initialize_global_model({})
        updates = p1.train_clients("RUN-001", 1, "model-v0", ["c0", "c1"], {"local_epochs": 2})
        detections = mock_p3.detect(updates, 1)
        impacts = mock_p3.estimate_impact(updates, detections, 1)
        actions = mock_p3.decide_response(detections, impacts)
        assert isinstance(actions, dict)
        for uid, action in actions.items():
            assert action in list(ResponseAction)

    def test_check_recovery_not_required(self, mock_p3):
        evaluation = {"accuracy": 0.90, "loss": 0.20}
        # No detections → no recovery
        result = mock_p3.check_recovery(
            evaluation, [], [], "RUN-001", 1, "model-v1", {}
        )
        assert isinstance(result, RecoveryResult)
        assert result.recovery_status == RecoveryStatus.NOT_REQUIRED
