"""
tests/sentinel/test_decision_engine.py

Unit tests for core/sentinel/decision_engine.py
Owner: Person 3
"""

from __future__ import annotations

import pytest
import numpy as np
from core.sentinel.decision_engine import (
    DecisionEngine, DecisionEngineError, SCHEMA_VERSION,
    DECISION_ENGINE_VERSION
)

@pytest.fixture
def engine() -> DecisionEngine:
    return DecisionEngine()

def dummy_inputs(t_score=0.0, i_score=0.0, c_score=1.0, 
                 hist_damage=0.0, rep_score=1.0, consistency=1.0) -> tuple:
    f = {"update_id": "u1", "client_id": "c1", "round_id": 1}
    t = {"threat_score": t_score, "confidence_score": c_score}
    i = {"impact_score": i_score, "historical_damage_risk": hist_damage}
    r = {
        "reputation_score": rep_score,
        "reputation_metrics": {"consistency_score": consistency}
    }
    a = {"dummy": 1}
    return f, t, i, r, a

class TestDecisionEngineLogic:
    def test_safe_client(self, engine: DecisionEngine):
        f, t, i, r, a = dummy_inputs()
        res = engine.compute_decision(f, t, i, r, a)
        assert res["recommended_action"] == "ACCEPT"
        assert res["escalation_level"] == "NONE"
        assert res["risk_matrix"] == "LOW_RISK"
        assert "NOMINAL_BEHAVIOR" in res["decision_explanation"]["reason_codes"]

    def test_critical_client_rollback(self, engine: DecisionEngine):
        f, t, i, r, a = dummy_inputs(t_score=0.9, i_score=0.9)
        res = engine.compute_decision(f, t, i, r, a)
        assert res["recommended_action"] == "ROLLBACK_RECOMMENDED"
        assert res["escalation_level"] == "CRITICAL"
        assert res["risk_matrix"] == "CRITICAL"

    def test_quarantine_action(self, engine: DecisionEngine):
        # High impact + history but low threat
        f, t, i, r, a = dummy_inputs(t_score=0.4, i_score=0.8, hist_damage=0.6)
        res = engine.compute_decision(f, t, i, r, a)
        assert res["recommended_action"] == "QUARANTINE"
        assert res["escalation_level"] == "HIGH"

    def test_flag_action(self, engine: DecisionEngine):
        f, t, i, r, a = dummy_inputs(t_score=0.6, i_score=0.1)
        res = engine.compute_decision(f, t, i, r, a)
        assert res["recommended_action"] == "FLAG"
        assert res["escalation_level"] == "MEDIUM"

    def test_monitor_action(self, engine: DecisionEngine):
        f, t, i, r, a = dummy_inputs(t_score=0.35, i_score=0.2)
        res = engine.compute_decision(f, t, i, r, a)
        assert res["recommended_action"] == "MONITOR"
        assert res["escalation_level"] == "LOW"

    def test_missing_inputs(self, engine: DecisionEngine):
        with pytest.raises(DecisionEngineError):
            engine.compute_decision(None, None, None, None, None)

    def test_nan_values_handled(self, engine: DecisionEngine):
        f, t, i, r, a = dummy_inputs(t_score=float('nan'))
        res = engine.compute_decision(f, t, i, r, a)
        # NaN is clipped to 1.0 -> ROLLBACK
        assert res["recommended_action"] in ("QUARANTINE", "ROLLBACK_RECOMMENDED", "FLAG")
        
    def test_inf_values_handled(self, engine: DecisionEngine):
        f, t, i, r, a = dummy_inputs(t_score=float('inf'))
        res = engine.compute_decision(f, t, i, r, a)
        assert res["recommended_action"] in ("QUARANTINE", "ROLLBACK_RECOMMENDED", "FLAG")

    def test_simulation_mode(self, engine: DecisionEngine):
        f, t, i, r, a = dummy_inputs()
        res1 = engine.compute_decision(f, t, i, r, a)
        res2 = engine.simulate_decision(f, t, i, r, a)
        assert res1 == res2

    def test_confidence_aggregation(self, engine: DecisionEngine):
        f, t, i, r, a = dummy_inputs(t_score=1.0, c_score=1.0, consistency=1.0, hist_damage=0.0)
        res = engine.compute_decision(f, t, i, r, a)
        assert res["decision_confidence"] > 0.9

    def test_risk_matrix(self, engine: DecisionEngine):
        assert engine._map_risk_matrix(0.9, 0.9) == "CRITICAL"
        assert engine._map_risk_matrix(0.9, 0.1) == "ELEVATED"
        assert engine._map_risk_matrix(0.1, 0.9) == "WATCHLIST"
        assert engine._map_risk_matrix(0.1, 0.1) == "LOW_RISK"
