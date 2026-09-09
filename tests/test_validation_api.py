"""
Test: Server Validation Gate API and telemetry verification.

Verifies:
1. Validation endpoint returns real validation records.
2. Validation endpoint returns empty list appropriately when no records exist.
3. Validation telemetry contains no ground-truth fields.
4. Validation telemetry contains no raw tensors/data.
5. Validation anomaly/loss-spike is represented correctly.
6. Existing validation gate behavior remains unchanged.
"""

import pytest
import torch
import torch.nn as nn
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.database.database import Base, get_db
import backend.database.models
from backend.database.repositories import ValidationRepository
from backend.main import app
from core.validation.server_validator import ServerValidationGate


_test_engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
_TestSession = sessionmaker(autocommit=False, autoflush=False, bind=_test_engine)


def _override_get_db():
    db = _TestSession()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(autouse=True)
def setup_test_db():
    Base.metadata.create_all(bind=_test_engine)
    app.dependency_overrides[get_db] = _override_get_db
    yield
    app.dependency_overrides.clear()
    Base.metadata.drop_all(bind=_test_engine)


@pytest.fixture
def client():
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


class TestValidationAPI:
    def test_validation_endpoint_empty_when_no_records(self, client):
        response = client.get("/api/v1/validation/RUN-EMPTY")
        assert response.status_code == 200
        assert response.json() == []

    def test_validation_endpoint_returns_real_records(self, client):
        db = _TestSession()
        try:
            repo = ValidationRepository(db)
            repo.create(
                run_id="RUN-REAL",
                round_id=1,
                model_version="model_v1",
                validation_loss=0.34,
                validation_accuracy=0.92,
                loss_spiked=False,
                baseline_loss=0.34,
                baseline_accuracy=0.92,
                loss_delta=0.0,
                accuracy_delta=0.0,
                validation_status="HEALTHY",
            )
            repo.create(
                run_id="RUN-REAL",
                round_id=2,
                model_version="model_v2",
                validation_loss=0.78,
                validation_accuracy=0.68,
                loss_spiked=True,
                baseline_loss=0.34,
                baseline_accuracy=0.92,
                loss_delta=0.44,
                accuracy_delta=0.24,
                validation_status="ANOMALOUS",
            )
        finally:
            db.close()

        response = client.get("/api/v1/validation/RUN-REAL")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2

        # Verify round 1
        r1 = data[0]
        assert r1["round_id"] == 1
        assert r1["validation_loss"] == 0.34
        assert r1["validation_accuracy"] == 0.92
        assert r1["loss_spiked"] is False
        assert r1["validation_status"] == "HEALTHY"

        # Verify round 2 (loss spike anomaly)
        r2 = data[1]
        assert r2["round_id"] == 2
        assert r2["validation_loss"] == 0.78
        assert r2["validation_accuracy"] == 0.68
        assert r2["loss_spiked"] is True
        assert r2["validation_status"] == "ANOMALOUS"
        assert r2["loss_delta"] == 0.44
        assert r2["accuracy_delta"] == 0.24

    def test_validation_telemetry_contains_no_ground_truth_or_raw_tensors(self, client):
        db = _TestSession()
        try:
            repo = ValidationRepository(db)
            repo.create(
                run_id="RUN-SAFE",
                round_id=1,
                model_version="model_v1",
                validation_loss=0.25,
                validation_accuracy=0.95,
                loss_spiked=False,
                baseline_loss=0.25,
                baseline_accuracy=0.95,
                loss_delta=0.0,
                accuracy_delta=0.0,
                validation_status="HEALTHY",
            )
        finally:
            db.close()

        response = client.get("/api/v1/validation/RUN-SAFE")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        record = data[0]

        # Invariant checks: No ground-truth fields
        forbidden_keys = [
            "is_malicious", "attacker_id", "attacker_ids", "attack_target",
            "ground_truth", "dataset", "tensors", "weights", "gradients",
        ]
        for key in forbidden_keys:
            assert key not in record

    def test_server_validation_gate_behavior_preserved(self):
        gate = ServerValidationGate(val_dataset=None)
        metrics, loss_spiked = gate.evaluate_checkpoint(nn.Linear(10, 2))
        assert loss_spiked is False
        assert metrics["val_loss"] == 0.0
        assert metrics["val_acc"] == 1.0
