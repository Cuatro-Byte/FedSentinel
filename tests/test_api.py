"""
Test: REST API endpoint contract verification.

Verifies that API endpoints return correct response shapes and status codes.
Uses FastAPI TestClient with an in-memory database.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session

from backend.database.database import Base, get_db
import backend.database.models  # Required to register tables with Base.metadata
from backend.main import app


from sqlalchemy.pool import StaticPool

# Create a module-level test engine and session factory
_test_engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
_TestSession = sessionmaker(autocommit=False, autoflush=False, bind=_test_engine)


def _override_get_db():
    """Override get_db to use the test in-memory database."""
    db = _TestSession()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(autouse=True)
def setup_test_db():
    """Create tables before each test, drop after."""
    print("TABLES BEFORE:", Base.metadata.tables.keys())
    Base.metadata.create_all(bind=_test_engine)
    app.dependency_overrides[get_db] = _override_get_db
    yield
    app.dependency_overrides.clear()
    Base.metadata.drop_all(bind=_test_engine)


@pytest.fixture
def client():
    """Provide a TestClient."""
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


class TestHealthEndpoint:
    def test_health_check(self, client):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "fedsentinel-backend"


class TestSimulationEndpoints:
    def test_create_simulation(self, client):
        """POST /api/v1/simulations — start a simulation."""
        response = client.post("/api/v1/simulations", json={
            "client_count": 5,
            "rounds": 3,
            "scenario": "backdoor",
            "attack_enabled": True,
        })
        assert response.status_code == 200, f"Failed: {response.text[:300]}"
        data = response.json()
        assert "run_id" in data
        assert data["status"] == "COMPLETED"

    def test_get_simulation(self, client):
        """GET /api/v1/simulations/{run_id}."""
        create_resp = client.post("/api/v1/simulations", json={
            "client_count": 5, "rounds": 2, "scenario": "normal",
            "attack_enabled": False,
        })
        assert create_resp.status_code == 200, f"Create failed: {create_resp.text[:300]}"
        run_id = create_resp.json()["run_id"]

        response = client.get(f"/api/v1/simulations/{run_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["run_id"] == run_id
        assert data["status"] == "COMPLETED"

    def test_get_simulation_not_found(self, client):
        response = client.get("/api/v1/simulations/NONEXISTENT")
        assert response.status_code == 404

    def test_list_simulations(self, client):
        client.post("/api/v1/simulations", json={
            "client_count": 3, "rounds": 1, "scenario": "normal",
            "attack_enabled": False,
        })
        response = client.get("/api/v1/simulations")
        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 1

    def test_create_simulation_background(self, client):
        """POST /api/v1/simulations with background=True."""
        response = client.post("/api/v1/simulations", json={
            "client_count": 3,
            "rounds": 1,
            "scenario": "normal",
            "attack_enabled": False,
            "background": True,
        })
        assert response.status_code == 200, f"Failed: {response.text[:300]}"
        data = response.json()
        assert "run_id" in data
        assert "simulation_id" in data
        assert data["run_id"] == data["simulation_id"]
        assert data["status"] in ["PENDING", "RUNNING", "COMPLETED"]

    def test_simulation_validation_errors(self, client):
        """Verify input validation rejects invalid simulation configs."""
        # 1. Invalid scenario
        resp = client.post("/api/v1/simulations", json={
            "client_count": 5, "rounds": 3, "scenario": "totally_invalid_scenario"
        })
        assert resp.status_code == 422

        # 2. Non-positive rounds
        resp = client.post("/api/v1/simulations", json={
            "client_count": 5, "rounds": 0, "scenario": "normal"
        })
        assert resp.status_code == 422

        # 3. Non-positive client_count
        resp = client.post("/api/v1/simulations", json={
            "client_count": 0, "rounds": 3, "scenario": "normal"
        })
        assert resp.status_code == 422

        # 4. start_round > rounds
        resp = client.post("/api/v1/simulations", json={
            "client_count": 5, "rounds": 2, "start_round": 5, "scenario": "model_poisoning"
        })
        assert resp.status_code == 422

        # 5. attacker_count > client_count
        resp = client.post("/api/v1/simulations", json={
            "client_count": 3, "rounds": 2, "attacker_count": 10, "scenario": "model_poisoning"
        })
        assert resp.status_code == 422

    def test_cors_headers(self, client):
        """Verify strict non-wildcard CORS headers for frontend origin."""
        response = client.get(
            "/api/v1/health",
            headers={"Origin": "http://localhost:5173"}
        )
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"
        assert response.headers.get("access-control-allow-credentials") == "true"


class TestDataEndpoints:
    """Test data retrieval endpoints after running a simulation."""

    @pytest.fixture(autouse=True)
    def _run_simulation(self, client):
        """Run a simulation to populate data."""
        resp = client.post("/api/v1/simulations", json={
            "client_count": 5, "rounds": 3, "scenario": "backdoor",
            "attack_enabled": True,
        })
        assert resp.status_code == 200, f"Sim create failed: {resp.text[:300]}"
        self.run_id = resp.json()["run_id"]
        self.client = client

    def test_get_clients(self):
        response = self.client.get(f"/api/v1/clients/{self.run_id}")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 5
        for c in data:
            assert "client_id" in c
            assert "reputation_score" in c

    def test_get_rounds(self):
        response = self.client.get(f"/api/v1/rounds/{self.run_id}")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 3

    def test_get_round_detail(self):
        response = self.client.get(f"/api/v1/rounds/{self.run_id}/1")
        assert response.status_code == 200
        data = response.json()
        assert data["round_id"] == 1
        assert "update_count" in data
        assert "accepted_count" in data
        assert "quarantined_count" in data

    def test_get_threats(self):
        response = self.client.get(f"/api/v1/threats/{self.run_id}")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 15  # 3 rounds × 5 clients
        for d in data:
            assert "threat_score" in d
            assert "threat_level" in d
            assert d["action"] in ("ACCEPT", "DOWN_WEIGHT", "QUARANTINE")

    def test_get_impacts(self):
        response = self.client.get(f"/api/v1/impact/{self.run_id}")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 15
        for imp in data:
            assert "impact_score" in imp
            assert "impact_level" in imp

    def test_get_metrics(self):
        response = self.client.get(f"/api/v1/metrics/{self.run_id}")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 3
        for m in data:
            assert "accuracy" in m
            assert "loss" in m
            assert "quarantined_updates" in m
            assert "accepted_updates" in m

    def test_get_recovery(self):
        response = self.client.get(f"/api/v1/recovery/{self.run_id}")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 3  # One per round
        for r in data:
            assert "recovery_status" in r
            assert r["recovery_status"] in (
                "NOT_REQUIRED", "TRIGGERED", "COMPLETED", "FAILED"
            )

    def test_get_audit(self):
        response = self.client.get(f"/api/v1/audit/{self.run_id}")
        assert response.status_code == 200
        data = response.json()
        assert len(data) > 0
        event_types = [e["event_type"] for e in data]
        assert "SIMULATION_STARTED" in event_types
        assert "SIMULATION_COMPLETED" in event_types
