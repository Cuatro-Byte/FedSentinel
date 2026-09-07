"""
tests/sentinel/test_reputation.py

Unit tests for core/sentinel/reputation.py
Owner: Person 3 — FedSentinel Detection, Impact & Recovery Intelligence
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from core.sentinel.reputation import (
    REPUTATION_ENGINE_VERSION,
    SCHEMA_VERSION,
    DECAY_COEF,
    RECOVERY_COEF,
    ANOMALY_THRESHOLD,
    ReputationEngine,
    ReputationEngineError,
)

@pytest.fixture
def engine() -> ReputationEngine:
    return ReputationEngine()

def make_dummy_anomaly(round_id: int, anomaly_score: float) -> dict[str, Any]:
    return {
        "round_id": round_id,
        "anomaly_score": anomaly_score,
    }

class TestReputationEngineInitialization:
    """Initialization tests."""

    def test_new_client_initialization(self, engine: ReputationEngine):
        res = engine.update_reputation("client_1", make_dummy_anomaly(1, 0.0))
        assert res["reputation_metrics"]["rounds_seen"] == 1
        assert "NEW_CLIENT" in res["reputation_flags"]
        assert res["reputation_score"] == 1.0  # (1.0 + 0.05 * 1.0) clipped to 1.0

    def test_multiple_new_clients(self, engine: ReputationEngine):
        res1 = engine.update_reputation("client_1", make_dummy_anomaly(1, 0.0))
        res2 = engine.update_reputation("client_2", make_dummy_anomaly(1, 1.0))
        assert res1["reputation_metrics"]["rounds_seen"] == 1
        assert res2["reputation_metrics"]["rounds_seen"] == 1
        assert res1["reputation_score"] == 1.0
        assert res2["reputation_score"] < 1.0

class TestReputationEngineUpdates:
    """Update tests."""

    def test_one_round_update(self, engine: ReputationEngine):
        res = engine.update_reputation("client_1", make_dummy_anomaly(1, 1.0))
        # 1.0 - (1.0 * 0.2) = 0.8
        assert abs(res["reputation_score"] - (1.0 - DECAY_COEF)) < 1e-6

    def test_multiple_rounds_update(self, engine: ReputationEngine):
        engine.update_reputation("client_1", make_dummy_anomaly(1, 1.0))
        res = engine.update_reputation("client_1", make_dummy_anomaly(2, 1.0))
        # 0.8 - (1.0 * 0.2) = 0.6
        assert abs(res["reputation_score"] - (1.0 - 2 * DECAY_COEF)) < 1e-6
        assert res["reputation_metrics"]["rounds_seen"] == 2

    def test_duplicate_round_rejection(self, engine: ReputationEngine):
        engine.update_reputation("client_1", make_dummy_anomaly(1, 0.0))
        with pytest.raises(ReputationEngineError):
            engine.update_reputation("client_1", make_dummy_anomaly(1, 0.0))

    def test_out_of_order_round_handling(self, engine: ReputationEngine):
        engine.update_reputation("client_1", make_dummy_anomaly(2, 0.0))
        with pytest.raises(ReputationEngineError):
            engine.update_reputation("client_1", make_dummy_anomaly(1, 0.0))

class TestReputationEngineDecay:
    """Decay tests."""

    def test_increasing_anomaly_history(self, engine: ReputationEngine):
        res1 = engine.update_reputation("client_1", make_dummy_anomaly(1, 0.6))
        res2 = engine.update_reputation("client_1", make_dummy_anomaly(2, 0.8))
        res3 = engine.update_reputation("client_1", make_dummy_anomaly(3, 1.0))
        
        assert res1["reputation_score"] > res2["reputation_score"] > res3["reputation_score"]
        assert "DECLINING_REPUTATION" in res3["reputation_flags"]

    def test_constant_high_anomaly_history(self, engine: ReputationEngine):
        for i in range(1, 6):
            res = engine.update_reputation("client_1", make_dummy_anomaly(i, 1.0))
        
        # 1.0 - 0.2 * 5 = 0.0
        assert abs(res["reputation_score"] - 0.0) < 1e-9
        assert "DECLINING_REPUTATION" in res["reputation_flags"]
        assert "HIGH_ANOMALY_HISTORY" in res["reputation_flags"]

    def test_reputation_lower_bound(self, engine: ReputationEngine):
        for i in range(1, 10):
            res = engine.update_reputation("client_1", make_dummy_anomaly(i, 1.0))
        assert res["reputation_score"] == 0.0

class TestReputationEngineRecovery:
    """Recovery tests."""

    def test_decreasing_anomaly_history(self, engine: ReputationEngine):
        # Push reputation down first
        for i in range(1, 4):
            engine.update_reputation("client_1", make_dummy_anomaly(i, 1.0))
            
        # Then recover enough times to make the trend positive
        for i in range(4, 10):
            res = engine.update_reputation("client_1", make_dummy_anomaly(i, 0.1))
        
        assert "RECOVERING_REPUTATION" in res["reputation_flags"]

    def test_constant_low_anomaly_history(self, engine: ReputationEngine):
        # Push down
        engine.update_reputation("client_1", make_dummy_anomaly(1, 1.0))
        
        # Recover slowly
        for i in range(2, 6):
            res = engine.update_reputation("client_1", make_dummy_anomaly(i, 0.0))
        
        assert "RECOVERING_REPUTATION" in res["reputation_flags"]

    def test_reputation_upper_bound(self, engine: ReputationEngine):
        for i in range(1, 10):
            res = engine.update_reputation("client_1", make_dummy_anomaly(i, 0.0))
        assert res["reputation_score"] == 1.0

class TestReputationEngineTrend:
    """Trend tests."""

    def test_positive_anomaly_trend(self, engine: ReputationEngine):
        for i, score in enumerate([0.1, 0.2, 0.3, 0.4, 0.5]):
            res = engine.update_reputation("client_1", make_dummy_anomaly(i+1, score))
        assert res["reputation_metrics"]["anomaly_trend"] > 0.0

    def test_negative_anomaly_trend(self, engine: ReputationEngine):
        for i, score in enumerate([0.9, 0.8, 0.7, 0.6, 0.5]):
            res = engine.update_reputation("client_1", make_dummy_anomaly(i+1, score))
        assert res["reputation_metrics"]["anomaly_trend"] < 0.0

    def test_stable_trend(self, engine: ReputationEngine):
        for i in range(1, 6):
            res = engine.update_reputation("client_1", make_dummy_anomaly(i, 0.5))
        assert abs(res["reputation_metrics"]["anomaly_trend"]) < 1e-9

class TestReputationEngineConsistency:
    """Consistency tests."""

    def test_stable_client(self, engine: ReputationEngine):
        for i in range(1, 6):
            res = engine.update_reputation("client_1", make_dummy_anomaly(i, 0.0))
        assert "STABLE_CLIENT" in res["reputation_flags"]
        assert res["reputation_metrics"]["consistency_score"] == 1.0

    def test_oscillating_client(self, engine: ReputationEngine):
        for i in range(1, 6):
            score = 1.0 if i % 2 == 0 else 0.0
            res = engine.update_reputation("client_1", make_dummy_anomaly(i, score))
        assert res["reputation_metrics"]["consistency_score"] < 1.0
        assert "STABLE_CLIENT" not in res["reputation_flags"]

class TestReputationEngineStability:
    """Stability edge cases."""

    def test_nan_anomaly_rejection(self, engine: ReputationEngine):
        with pytest.raises(ReputationEngineError):
            engine.update_reputation("client_1", make_dummy_anomaly(1, float("nan")))

    def test_inf_anomaly_rejection(self, engine: ReputationEngine):
        with pytest.raises(ReputationEngineError):
            engine.update_reputation("client_1", make_dummy_anomaly(1, float("inf")))

    def test_empty_history(self, engine: ReputationEngine):
        state = engine._client_states.get("client_1")
        assert state is None
        
    def test_window_overflow(self, engine: ReputationEngine):
        for i in range(1, 20):
            res = engine.update_reputation("client_1", make_dummy_anomaly(i, 0.0))
        
        # History in memory continues to grow (by design for full audit trailing),
        # but the rolling averages only consider the last HISTORY_WINDOW items.
        # We check that it didn't crash and output is valid.
        assert res["reputation_metrics"]["rounds_seen"] == 19

    def test_json_serialization(self, engine: ReputationEngine):
        import json
        res = engine.update_reputation("client_1", make_dummy_anomaly(1, 0.0))
        # Will raise error if not serializable
        json.dumps(res)

class TestReputationEnginePerformance:
    """Performance test with many clients."""

    def test_many_clients(self, engine: ReputationEngine):
        # 1000 clients, 5 rounds each
        for round_id in range(1, 6):
            for client_id in range(1000):
                engine.update_reputation(f"c_{client_id}", make_dummy_anomaly(round_id, 0.1))
        
        assert len(engine._client_states) == 1000
        assert engine._client_states["c_999"].rounds_seen == 5
