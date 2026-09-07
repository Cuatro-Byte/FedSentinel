"""
tests/sentinel/test_recovery_engine.py

Unit tests for core/sentinel/recovery_engine.py
Owner: Person 3
"""

from __future__ import annotations

import pytest
import numpy as np
from core.sentinel.recovery_engine import (
    RecoveryEngine, RecoveryEngineError, SCHEMA_VERSION,
    RECOVERY_ENGINE_VERSION
)

@pytest.fixture
def engine() -> RecoveryEngine:
    return RecoveryEngine()

def dummy_inputs(action="ACCEPT", d_conf=1.0, t_score=0.0, c_score=1.0, 
                 i_score=0.0, hist_damage=0.0, consistency=1.0) -> tuple:
    d = {"update_id": "u1", "client_id": "c1", "round_id": 1, 
         "recommended_action": action, "decision_confidence": d_conf}
    t = {"threat_score": t_score, "confidence_score": c_score}
    i = {"impact_score": i_score, "historical_damage_risk": hist_damage}
    r = {"reputation_metrics": {"consistency_score": consistency}}
    return d, t, i, r

class TestRecoveryEngineLogic:
    def test_safe_client(self, engine: RecoveryEngine):
        d, t, i, r = dummy_inputs()
        res = engine.compute_recovery(d, t, i, r)
        assert res["primary_action"] == "ACCEPT"
        assert res["recovery_confidence"] == 1.0

    def test_critical_client_rollback(self, engine: RecoveryEngine):
        d, t, i, r = dummy_inputs(action="ROLLBACK_RECOMMENDED", t_score=0.9, i_score=0.9, hist_damage=0.6)
        res = engine.compute_recovery(d, t, i, r)
        assert res["primary_action"] == "ROLLBACK"
        assert res["rollback_recommended"] is True
        assert "QUARANTINE_CLIENT" in res["secondary_actions"]
        assert res["monitoring_plan"]["monitoring_frequency"] == "continuous"

    def test_quarantine_action(self, engine: RecoveryEngine):
        d, t, i, r = dummy_inputs(action="QUARANTINE", t_score=0.7, i_score=0.7)
        res = engine.compute_recovery(d, t, i, r)
        assert res["primary_action"] == "QUARANTINE"
        assert res["quarantine_recommended"] is True
        assert res["monitoring_plan"]["monitoring_frequency"] == "high"

    def test_isolate_client(self, engine: RecoveryEngine):
        d, t, i, r = dummy_inputs(action="FLAG", t_score=0.5, i_score=0.5)
        res = engine.compute_recovery(d, t, i, r)
        assert res["primary_action"] == "ISOLATE_CLIENT"
        assert res["isolation_recommended"] is True
        assert res["monitoring_plan"]["monitoring_frequency"] == "elevated"

    def test_monitor_action(self, engine: RecoveryEngine):
        d, t, i, r = dummy_inputs(action="MONITOR", t_score=0.3, i_score=0.3)
        res = engine.compute_recovery(d, t, i, r)
        assert res["primary_action"] == "MONITOR"

    def test_missing_inputs(self, engine: RecoveryEngine):
        with pytest.raises(RecoveryEngineError):
            engine.compute_recovery(None, None, None, None)

    def test_nan_values_handled(self, engine: RecoveryEngine):
        d, t, i, r = dummy_inputs(t_score=float('nan'))
        res = engine.compute_recovery(d, t, i, r)
        # NaN is clipped to 1.0
        assert res["primary_action"] in ("ROLLBACK", "QUARANTINE", "ISOLATE_CLIENT")
        
    def test_inf_values_handled(self, engine: RecoveryEngine):
        d, t, i, r = dummy_inputs(t_score=float('inf'))
        res = engine.compute_recovery(d, t, i, r)
        assert res["primary_action"] in ("ROLLBACK", "QUARANTINE", "ISOLATE_CLIENT")

    def test_simulation_mode(self, engine: RecoveryEngine):
        d, t, i, r = dummy_inputs()
        res1 = engine.compute_recovery(d, t, i, r)
        res2 = engine.simulate_recovery(d, t, i, r)
        assert res1 == res2

    def test_idempotency(self, engine: RecoveryEngine):
        d, t, i, r = dummy_inputs(action="ROLLBACK_RECOMMENDED")
        res1 = engine.compute_recovery(d, t, i, r)
        res2 = engine.compute_recovery(d, t, i, r)
        assert res1 == res2

    def test_recovery_confidence_bounds(self, engine: RecoveryEngine):
        d, t, i, r = dummy_inputs(d_conf=1.0, c_score=1.0, consistency=1.0, hist_damage=0.0)
        res = engine.compute_recovery(d, t, i, r)
        assert res["recovery_confidence"] == 1.0

        d, t, i, r = dummy_inputs(d_conf=0.0, c_score=0.0, consistency=0.0, hist_damage=1.0)
        res = engine.compute_recovery(d, t, i, r)
        assert res["recovery_confidence"] == 0.0
