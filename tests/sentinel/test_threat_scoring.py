"""
tests/sentinel/test_threat_scoring.py

Unit tests for core/sentinel/threat_scoring.py
Owner: Person 3
"""

from __future__ import annotations

from typing import Any
import pytest
import numpy as np
from core.sentinel.threat_scoring import ThreatScoringEngine, ThreatScoringError, SCHEMA_VERSION

@pytest.fixture
def engine() -> ThreatScoringEngine:
    return ThreatScoringEngine()

def dummy_inputs(anomaly_score=0.0, rep_score=1.0, rounds_seen=10, 
                 sim_anomaly=0.0, consistency=1.0,
                 susp=0, mal=0, quar=0) -> dict[str, Any]:
    f = {"update_id": "u1", "client_id": "c1", "round_id": 1}
    s = {"dummy": 1}
    sim = {"dummy": 1}
    a = {"anomaly_score": anomaly_score, "signal_breakdown": {"similarity_anomaly": sim_anomaly}}
    r = {
        "reputation_score": rep_score,
        "reputation_metrics": {"rounds_seen": rounds_seen, "consistency_score": consistency},
        "behavior_counts": {"suspicious_count": susp, "malicious_count": mal, "quarantine_count": quar}
    }
    return f, s, sim, a, r

class TestThreatScoringLogic:
    def test_safe_client(self, engine: ThreatScoringEngine):
        f, s, sim, a, r = dummy_inputs(anomaly_score=0.0, rep_score=1.0, sim_anomaly=0.0)
        res = engine.compute_threat(f, s, sim, a, r)
        assert res["threat_level"] == "SAFE"
        assert res["threat_score"] == 0.0
        assert res["confidence_score"] == 1.0

    def test_critical_client(self, engine: ThreatScoringEngine):
        f, s, sim, a, r = dummy_inputs(anomaly_score=1.0, rep_score=0.0, sim_anomaly=1.0, susp=10, mal=10, quar=10)
        res = engine.compute_threat(f, s, sim, a, r)
        assert res["threat_level"] == "CRITICAL"
        assert res["threat_score"] >= 0.99

    def test_missing_inputs(self, engine: ThreatScoringEngine):
        with pytest.raises(ThreatScoringError):
            engine.compute_threat({}, {}, {}, {}, {})

    def test_nan_values(self, engine: ThreatScoringEngine):
        f, s, sim, a, r = dummy_inputs(anomaly_score=float("nan"))
        res = engine.compute_threat(f, s, sim, a, r)
        assert res["threat_score"] > 0.0

    def test_inf_values(self, engine: ThreatScoringEngine):
        f, s, sim, a, r = dummy_inputs(anomaly_score=float("inf"))
        res = engine.compute_threat(f, s, sim, a, r)
        assert res["threat_score"] > 0.0

    def test_explanations(self, engine: ThreatScoringEngine):
        f, s, sim, a, r = dummy_inputs(anomaly_score=1.0, rep_score=0.0, sim_anomaly=1.0)
        res = engine.compute_threat(f, s, sim, a, r)
        assert len(res["threat_summary"]) >= 3
        
    def test_confidence_score(self, engine: ThreatScoringEngine):
        f, s, sim, a, r = dummy_inputs(rounds_seen=5, consistency=0.5)
        res = engine.compute_threat(f, s, sim, a, r)
        assert abs(res["confidence_score"] - 0.5) < 1e-9

    def test_boundary_mapping(self, engine: ThreatScoringEngine):
        assert engine._map_threat_level(0.20)[0] == "SAFE"
        assert engine._map_threat_level(0.21)[0] == "LOW"
        assert engine._map_threat_level(0.40)[0] == "LOW"
        assert engine._map_threat_level(0.41)[0] == "MEDIUM"
        assert engine._map_threat_level(0.60)[0] == "MEDIUM"
        assert engine._map_threat_level(0.61)[0] == "HIGH"
        assert engine._map_threat_level(0.80)[0] == "HIGH"
        assert engine._map_threat_level(0.81)[0] == "CRITICAL"
