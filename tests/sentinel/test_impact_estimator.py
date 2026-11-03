"""
tests/sentinel/test_impact_estimator.py

Unit tests for core/sentinel/impact_estimator.py
Owner: Person 3
"""

from __future__ import annotations

import pytest
import numpy as np
from core.sentinel.impact_estimator import (
    ImpactEstimator, ImpactEstimatorError, SCHEMA_VERSION,
    MODEL_DRIFT_WEIGHT, LAYER_IMPACT_WEIGHT, THREAT_AMPLIFICATION_WEIGHT,
    HISTORICAL_DAMAGE_WEIGHT, CONFIDENCE_WEIGHT
)

@pytest.fixture
def engine() -> ImpactEstimator:
    return ImpactEstimator()

def dummy_inputs(l2=0.0, l_rms=0.0, t_score=0.0, c_score=1.0, 
                 susp=0, mal=0, quar=0, 
                 layer_keys=["layer_1"]) -> tuple:
    f = {"update_id": "u1", "client_id": "c1", "round_id": 1, "layer_shapes": {k: (1,) for k in layer_keys}}
    s = {
        "global_statistics": {"l2_norm": l2, "rms": 0.0, "energy": 0.0},
        "layer_statistics": {k: {"rms": l_rms, "variance": 0.0, "max_abs": 0.0} for k in layer_keys}
    }
    sim = {"dummy": 1}
    a = {"dummy": 1}
    r = {
        "reputation_metrics": {"rounds_seen": 10, "consistency_score": 1.0},
        "behavior_counts": {"suspicious_count": susp, "malicious_count": mal, "quarantine_count": quar}
    }
    t = {"threat_score": t_score, "confidence_score": c_score}
    return f, s, sim, a, r, t

class TestImpactEstimatorLogic:
    def test_weights_sum_to_one(self):
        total = MODEL_DRIFT_WEIGHT + LAYER_IMPACT_WEIGHT + THREAT_AMPLIFICATION_WEIGHT + HISTORICAL_DAMAGE_WEIGHT + CONFIDENCE_WEIGHT
        assert abs(total - 1.0) < 1e-9

    def test_zero_impact(self, engine: ImpactEstimator):
        f, s, sim, a, r, t = dummy_inputs()
        res = engine.compute_impact(f, s, sim, a, r, t)
        assert res["impact_severity"] == "MINIMAL"
        
        breakdown = res["impact_breakdown"]
        total_p = sum(breakdown.values())
        assert abs(total_p - 100.0) < 1e-9

    def test_severe_impact(self, engine: ImpactEstimator):
        f, s, sim, a, r, t = dummy_inputs(l2=100.0, l_rms=100.0, t_score=1.0, mal=5, quar=5)
        res = engine.compute_impact(f, s, sim, a, r, t)
        assert res["impact_score"] > 0.8
        assert res["impact_severity"] in ("HIGH", "SEVERE")
        
        breakdown = res["impact_breakdown"]
        total_p = sum(breakdown.values())
        assert abs(total_p - 100.0) < 1e-9

    def test_missing_inputs(self, engine: ImpactEstimator):
        with pytest.raises(ImpactEstimatorError):
            engine.compute_impact(None, None, None, None, None, None)

    def test_missing_layer_shapes(self, engine: ImpactEstimator):
        f, s, sim, a, r, t = dummy_inputs()
        f["layer_shapes"] = {}
        with pytest.raises(ImpactEstimatorError):
            engine.compute_impact(f, s, sim, a, r, t)

    def test_nan_values_handled(self, engine: ImpactEstimator):
        f, s, sim, a, r, t = dummy_inputs(l2=float('nan'))
        res = engine.compute_impact(f, s, sim, a, r, t)
        assert res["impact_score"] > 0.0
        
    def test_inf_values_handled(self, engine: ImpactEstimator):
        f, s, sim, a, r, t = dummy_inputs(l2=float('inf'))
        res = engine.compute_impact(f, s, sim, a, r, t)
        assert res["impact_score"] > 0.0

    def test_layer_ranking_stable(self, engine: ImpactEstimator):
        f, s, sim, a, r, t = dummy_inputs(layer_keys=["layer_C", "layer_A", "layer_B"])
        # Give them identical scores by leaving l_rms=0.0
        res = engine.compute_impact(f, s, sim, a, r, t)
        layers = res["top_impacted_layers"]
        
        # Should sort by (-score, name), since scores are identical, sorts by name: A, B, C
        assert layers[0]["layer_name"] == "layer_A"
        assert layers[1]["layer_name"] == "layer_B"
        assert layers[2]["layer_name"] == "layer_C"
        
        # Contribution percentages must sum to 100
        p_sum = sum(x["contribution_percentage"] for x in layers)
        assert abs(p_sum - 100.0) < 1e-9

    def test_historical_damage_risk(self, engine: ImpactEstimator):
        f, s, sim, a, r, t = dummy_inputs(susp=10)
        res = engine.compute_impact(f, s, sim, a, r, t)
        assert res["historical_damage_risk"] == 1.0

    def test_threat_amplification(self, engine: ImpactEstimator):
        f, s, sim, a, r, t = dummy_inputs(t_score=0.5, c_score=0.0)
        # stability = 0.0, threat = 0.5 -> amp = 0.5 * (1 + 1) = 1.0
        res = engine.compute_impact(f, s, sim, a, r, t)
        assert res["threat_amplification"] == 1.0

    def test_severity_boundaries(self, engine: ImpactEstimator):
        assert engine._map_severity(0.20) == "MINIMAL"
        assert engine._map_severity(0.21) == "LOW"
        assert engine._map_severity(0.40) == "LOW"
        assert engine._map_severity(0.41) == "MODERATE"
        assert engine._map_severity(0.60) == "MODERATE"
        assert engine._map_severity(0.61) == "HIGH"
        assert engine._map_severity(0.80) == "HIGH"
        assert engine._map_severity(0.81) == "SEVERE"
