"""
Test: Database initialization, ORM models, and repository CRUD.

Verifies that the database layer correctly persists and retrieves
data matching the canonical contract models.
"""

import pytest
from datetime import datetime

from backend.database.repositories import (
    SimulationRepository, ClientRepository, RoundRepository,
    ModelUpdateRepository, DetectionRepository, ImpactRepository,
    RecoveryRepository, MetricRepository, AuditRepository,
)
from core.models import (
    ModelUpdate, DetectionResult, ImpactResult, RecoveryResult, SimulationMetrics,
)
from core.models.enums import (
    ThreatLevel, ResponseAction, ImpactLevel, RecoveryStatus,
)


class TestSimulationRepository:
    def test_create_and_get(self, db_session):
        repo = SimulationRepository(db_session)
        run = repo.create("RUN-001", "backdoor", 20, 10)
        assert run.run_id == "RUN-001"
        assert run.status == "PENDING"

        fetched = repo.get("RUN-001")
        assert fetched is not None
        assert fetched.scenario == "backdoor"

    def test_update_status(self, db_session):
        repo = SimulationRepository(db_session)
        repo.create("RUN-002", "normal", 10, 5)
        repo.update_status("RUN-002", "RUNNING", current_round=3)
        run = repo.get("RUN-002")
        assert run.status == "RUNNING"
        assert run.current_round == 3

    def test_list_all(self, db_session):
        repo = SimulationRepository(db_session)
        repo.create("RUN-A", "normal", 10, 5)
        repo.create("RUN-B", "backdoor", 20, 10)
        runs = repo.list_all()
        assert len(runs) == 2


class TestClientRepository:
    def test_create_batch_and_get(self, db_session):
        sim_repo = SimulationRepository(db_session)
        sim_repo.create("RUN-001", "normal", 3, 5)

        client_repo = ClientRepository(db_session)
        clients = client_repo.create_batch("RUN-001", ["c0", "c1", "c2"])
        assert len(clients) == 3

        fetched = client_repo.get_by_run("RUN-001")
        assert len(fetched) == 3

    def test_update_from_detection(self, db_session):
        sim_repo = SimulationRepository(db_session)
        sim_repo.create("RUN-001", "normal", 2, 5)
        client_repo = ClientRepository(db_session)
        client_repo.create("RUN-001", "c0")

        detection = DetectionResult(
            update_id="U001", client_id="c0", round_id=1,
            threat_score=0.85, threat_level=ThreatLevel.MALICIOUS,
            action=ResponseAction.QUARANTINE,
            anomaly_score=0.9, similarity_score=0.1, reputation_score=0.3,
            explanation_codes=["HIGH_UPDATE_NORM"],
        )
        impact = ImpactResult(
            update_id="U001", client_id="c0", round_id=1,
            impact_score=0.78, influence_estimate=0.6,
            parameter_displacement=4.5, aggregation_weight=0.05,
            impact_level=ImpactLevel.CRITICAL,
        )
        client_repo.update_from_detection("RUN-001", detection, impact)

        client = client_repo.get("RUN-001", "c0")
        assert client.last_threat_score == 0.85
        assert client.last_impact_score == 0.78
        assert client.malicious_count == 1
        assert client.quarantine_count == 1


class TestRoundRepository:
    def test_create_and_get(self, db_session):
        sim_repo = SimulationRepository(db_session)
        sim_repo.create("RUN-001", "normal", 10, 5)

        round_repo = RoundRepository(db_session)
        fl_round = round_repo.create("RUN-001", 1)
        assert fl_round.round_id == 1
        assert fl_round.status == "PENDING"

        fetched = round_repo.get("RUN-001", 1)
        assert fetched is not None


class TestModelUpdateRepository:
    def test_create_from_canonical(self, db_session):
        sim_repo = SimulationRepository(db_session)
        sim_repo.create("RUN-001", "normal", 10, 5)
        round_repo = RoundRepository(db_session)
        fl_round = round_repo.create("RUN-001", 1)

        update = ModelUpdate(
            update_id="U001", run_id="RUN-001", round_id=1,
            client_id="c0", model_version="model-v0",
            base_model_version="model-v0",
            parameters={"w": [0.1]}, sample_count=100, training_epochs=2,
        )
        repo = ModelUpdateRepository(db_session)
        record = repo.create_from_canonical(update, fl_round.id)
        assert record.update_id == "U001"
        assert record.sample_count == 100


class TestDetectionRepository:
    def test_create_and_query(self, db_session):
        sim_repo = SimulationRepository(db_session)
        sim_repo.create("RUN-001", "normal", 10, 5)
        round_repo = RoundRepository(db_session)
        fl_round = round_repo.create("RUN-001", 1)

        detection = DetectionResult(
            update_id="U001", client_id="c0", round_id=1,
            threat_score=0.91, threat_level=ThreatLevel.MALICIOUS,
            action=ResponseAction.QUARANTINE,
            anomaly_score=0.88, similarity_score=0.15, reputation_score=0.31,
            explanation_codes=["HIGH_UPDATE_NORM"],
        )
        repo = DetectionRepository(db_session)
        repo.create_from_canonical(detection, "RUN-001", fl_round.id)

        results = repo.get_by_run("RUN-001")
        assert len(results) == 1
        assert results[0].threat_level == "MALICIOUS"
        assert results[0].action == "QUARANTINE"


class TestImpactRepository:
    def test_create_and_query(self, db_session):
        sim_repo = SimulationRepository(db_session)
        sim_repo.create("RUN-001", "normal", 10, 5)
        round_repo = RoundRepository(db_session)
        fl_round = round_repo.create("RUN-001", 1)

        impact = ImpactResult(
            update_id="U001", client_id="c0", round_id=1,
            impact_score=0.86, influence_estimate=0.72,
            parameter_displacement=3.5, aggregation_weight=0.05,
            impact_level=ImpactLevel.CRITICAL,
        )
        repo = ImpactRepository(db_session)
        repo.create_from_canonical(impact, "RUN-001", fl_round.id)

        results = repo.get_by_run("RUN-001")
        assert len(results) == 1
        assert results[0].impact_level == "CRITICAL"


class TestRecoveryRepository:
    def test_create_and_query(self, db_session):
        sim_repo = SimulationRepository(db_session)
        sim_repo.create("RUN-001", "normal", 10, 5)

        recovery = RecoveryResult(
            recovery_id="REC-001", run_id="RUN-001", round_id=5,
            trigger="MALICIOUS_DETECTED",
            affected_update_ids=["U001"],
            excluded_client_ids=["c17"],
            previous_model_version="model-v5",
            recovered_model_version="model-v5-recovered-1",
            before_accuracy=0.72, after_accuracy=0.89,
            recovery_status=RecoveryStatus.COMPLETED,
        )
        repo = RecoveryRepository(db_session)
        repo.create_from_canonical(recovery)

        results = repo.get_by_run("RUN-001")
        assert len(results) == 1
        assert results[0].recovery_status == "COMPLETED"


class TestMetricRepository:
    def test_create_and_query(self, db_session):
        sim_repo = SimulationRepository(db_session)
        sim_repo.create("RUN-001", "normal", 10, 5)

        metrics = SimulationMetrics(
            run_id="RUN-001", round_id=1, model_version="model-v1",
            accuracy=0.87, loss=0.30,
            accepted_updates=18, quarantined_updates=2,
        )
        repo = MetricRepository(db_session)
        repo.create_from_canonical(metrics)

        results = repo.get_by_run("RUN-001")
        assert len(results) == 1
        assert results[0].accuracy == 0.87


class TestAuditRepository:
    def test_create_and_query(self, db_session):
        sim_repo = SimulationRepository(db_session)
        sim_repo.create("RUN-001", "normal", 10, 5)

        repo = AuditRepository(db_session)
        repo.create("RUN-001", "SIMULATION_STARTED", "Simulation started", severity="INFO")
        repo.create("RUN-001", "UPDATE_QUARANTINED", "Client c17 quarantined",
                     severity="WARN", round_id=5, client_id="c17")

        events = repo.get_by_run("RUN-001")
        assert len(events) == 2
        assert events[0].event_type == "SIMULATION_STARTED"
