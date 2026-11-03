"""
Test: Full simulation pipeline — end-to-end orchestrator test.

Runs a complete multi-round simulation with mock P1/P2/P3 and verifies
that all results are persisted correctly.
"""

import pytest

from backend.config import AppConfig, SimulationConfig, AttackConfig, RecoveryConfig
from backend.database.repositories import (
    SimulationRepository, ClientRepository, RoundRepository,
    DetectionRepository, ImpactRepository, RecoveryRepository,
    MetricRepository, AuditRepository, ModelUpdateRepository,
)
from simulation.runner import SimulationRunner


@pytest.fixture
def small_config():
    """Small simulation config for testing."""
    return AppConfig(
        simulation=SimulationConfig(client_count=5, rounds=3, seed=42, local_epochs=2),
        attack=AttackConfig(enabled=True, scenario="backdoor",
                            attacker_count=1, start_round=2, intensity=0.8),
        recovery=RecoveryConfig(enabled=True, accuracy_drop_threshold=0.05),
    )


class TestSimulationRunner:
    def test_full_simulation_executes(self, db_session, mock_p1, mock_p2, mock_p3, small_config):
        """Verify that a complete multi-round simulation runs without error."""
        runner = SimulationRunner(
            db=db_session,
            fl_core=mock_p1,
            attack_engine=mock_p2,
            sentinel=mock_p3,
            config=small_config,
        )
        run_id = runner.run_simulation(run_id="TEST-001", scenario="backdoor")
        assert run_id == "TEST-001"

    def test_simulation_persists_run(self, db_session, mock_p1, mock_p2, mock_p3, small_config):
        """Verify SimulationRun is persisted with correct status."""
        runner = SimulationRunner(
            db=db_session, fl_core=mock_p1, attack_engine=mock_p2,
            sentinel=mock_p3, config=small_config,
        )
        runner.run_simulation(run_id="TEST-002")
        sim_repo = SimulationRepository(db_session)
        run = sim_repo.get("TEST-002")
        assert run is not None
        assert run.status == "COMPLETED"
        assert run.round_count == 3
        assert run.client_count == 5

    def test_rounds_persisted(self, db_session, mock_p1, mock_p2, mock_p3, small_config):
        """Verify all rounds are persisted."""
        runner = SimulationRunner(
            db=db_session, fl_core=mock_p1, attack_engine=mock_p2,
            sentinel=mock_p3, config=small_config,
        )
        runner.run_simulation(run_id="TEST-003")
        round_repo = RoundRepository(db_session)
        rounds = round_repo.get_by_run("TEST-003")
        assert len(rounds) == 3
        for r in rounds:
            assert r.status == "COMPLETED"
            assert r.update_count == 5

    def test_clients_persisted(self, db_session, mock_p1, mock_p2, mock_p3, small_config):
        """Verify clients are created and updated."""
        runner = SimulationRunner(
            db=db_session, fl_core=mock_p1, attack_engine=mock_p2,
            sentinel=mock_p3, config=small_config,
        )
        runner.run_simulation(run_id="TEST-004")
        client_repo = ClientRepository(db_session)
        clients = client_repo.get_by_run("TEST-004")
        assert len(clients) == 5

    def test_model_updates_persisted(self, db_session, mock_p1, mock_p2, mock_p3, small_config):
        """Verify model updates are persisted for each round."""
        runner = SimulationRunner(
            db=db_session, fl_core=mock_p1, attack_engine=mock_p2,
            sentinel=mock_p3, config=small_config,
        )
        runner.run_simulation(run_id="TEST-005")
        update_repo = ModelUpdateRepository(db_session)
        # 3 rounds × 5 clients = 15 updates
        for r in range(1, 4):
            updates = update_repo.get_by_round("TEST-005", r)
            assert len(updates) == 5

    def test_detection_results_persisted(self, db_session, mock_p1, mock_p2, mock_p3, small_config):
        """Verify detection results are persisted."""
        runner = SimulationRunner(
            db=db_session, fl_core=mock_p1, attack_engine=mock_p2,
            sentinel=mock_p3, config=small_config,
        )
        runner.run_simulation(run_id="TEST-006")
        det_repo = DetectionRepository(db_session)
        detections = det_repo.get_by_run("TEST-006")
        # 3 rounds × 5 clients = 15 detection results
        assert len(detections) == 15
        # Verify valid enum values
        for d in detections:
            assert d.threat_level in ("SAFE", "SUSPICIOUS", "MALICIOUS")
            assert d.action in ("ACCEPT", "DOWN_WEIGHT", "QUARANTINE")

    def test_impact_results_persisted(self, db_session, mock_p1, mock_p2, mock_p3, small_config):
        """Verify impact results are persisted."""
        runner = SimulationRunner(
            db=db_session, fl_core=mock_p1, attack_engine=mock_p2,
            sentinel=mock_p3, config=small_config,
        )
        runner.run_simulation(run_id="TEST-007")
        impact_repo = ImpactRepository(db_session)
        impacts = impact_repo.get_by_run("TEST-007")
        assert len(impacts) == 15
        for imp in impacts:
            assert imp.impact_level in ("LOW", "MEDIUM", "HIGH", "CRITICAL")

    def test_recovery_events_persisted(self, db_session, mock_p1, mock_p2, mock_p3, small_config):
        """Verify recovery events are recorded for every round."""
        runner = SimulationRunner(
            db=db_session, fl_core=mock_p1, attack_engine=mock_p2,
            sentinel=mock_p3, config=small_config,
        )
        runner.run_simulation(run_id="TEST-008")
        rec_repo = RecoveryRepository(db_session)
        recoveries = rec_repo.get_by_run("TEST-008")
        # One recovery result per round
        assert len(recoveries) == 3
        statuses = [r.recovery_status for r in recoveries]
        # At least some should be NOT_REQUIRED (before attack starts)
        assert "NOT_REQUIRED" in statuses or "COMPLETED" in statuses

    def test_metrics_persisted(self, db_session, mock_p1, mock_p2, mock_p3, small_config):
        """Verify metrics are recorded for every round."""
        runner = SimulationRunner(
            db=db_session, fl_core=mock_p1, attack_engine=mock_p2,
            sentinel=mock_p3, config=small_config,
        )
        runner.run_simulation(run_id="TEST-009")
        metric_repo = MetricRepository(db_session)
        metrics = metric_repo.get_by_run("TEST-009")
        assert len(metrics) == 3
        for m in metrics:
            assert 0.0 <= m.accuracy <= 1.0
            assert m.loss >= 0.0

    def test_audit_events_persisted(self, db_session, mock_p1, mock_p2, mock_p3, small_config):
        """Verify audit events are recorded."""
        runner = SimulationRunner(
            db=db_session, fl_core=mock_p1, attack_engine=mock_p2,
            sentinel=mock_p3, config=small_config,
        )
        runner.run_simulation(run_id="TEST-010")
        audit_repo = AuditRepository(db_session)
        events = audit_repo.get_by_run("TEST-010")
        assert len(events) > 0
        event_types = [e.event_type for e in events]
        assert "SIMULATION_STARTED" in event_types
        assert "SIMULATION_COMPLETED" in event_types
        assert "ROUND_STARTED" in event_types
        assert "ROUND_COMPLETED" in event_types

    def test_orchestrator_runs_without_real_p1_p2_p3(self, db_session, mock_p1, mock_p2, mock_p3, small_config):
        """Verify the orchestrator can run entirely with mock adapters."""
        runner = SimulationRunner(
            db=db_session, fl_core=mock_p1, attack_engine=mock_p2,
            sentinel=mock_p3, config=small_config,
        )
        # This is the key integration test — the whole pipeline works
        # without any real P1/P2/P3 modules
        run_id = runner.run_simulation(run_id="TEST-STANDALONE")
        sim_repo = SimulationRepository(db_session)
        run = sim_repo.get(run_id)
        assert run.status == "COMPLETED"
