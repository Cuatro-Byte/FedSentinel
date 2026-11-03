"""
tests/sentinel/test_detection_result_adapter.py

Tests for the P1 FL Integration Contract Adapter.
"""

from __future__ import annotations

import pytest
import numpy as np
import json
from typing import Any
from dataclasses import dataclass

from core.sentinel.sentinel import Sentinel

@dataclass
class MockUpdate:
    update_id: str
    client_id: str
    round_id: int
    parameters: dict[str, np.ndarray]
    sample_count: int = 100
    training_epochs: int = 5
    learning_rate: float = 0.01
    local_loss: float = 0.5
    local_accuracy: float = 0.9

def generate_update(c_id: str, u_id: str, r_id: int, scale: float = 1.0) -> MockUpdate:
    return MockUpdate(
        update_id=u_id,
        client_id=c_id,
        round_id=r_id,
        parameters={"layer_0": np.ones((10, 10)) * scale, "layer_1": np.zeros(5)}
    )

@pytest.fixture
def orchestrator() -> Sentinel:
    return Sentinel()

class TestDetectionResultAdapter:
    def test_adapter_present_in_output(self, orchestrator: Sentinel):
        update = generate_update("c1", "u1", 1)
        res = orchestrator.run_round([update], 1)[0]
        assert "detection_result" in res["client_result"]
        
        det_res = res["client_result"]["detection_result"]
        assert det_res["update_id"] == "u1"
        assert det_res["client_id"] == "c1"
        assert det_res["round_id"] == 1
        assert "threat_score" in det_res
        assert "action" in det_res

    def test_action_mapping_accept(self, orchestrator: Sentinel):
        res = {
            "client_result": {"decision_output": {"recommended_action": "ACCEPT"}},
            "pipeline_metadata": {"client_id": "c1", "round_id": 1}
        }
        update = generate_update("c1", "u1", 1)
        det_res = orchestrator.build_detection_result(update, res)
        assert det_res["action"] == "ACCEPT"

    def test_action_mapping_monitor(self, orchestrator: Sentinel):
        res = {
            "client_result": {"decision_output": {"recommended_action": "MONITOR"}},
            "pipeline_metadata": {"client_id": "c1", "round_id": 1}
        }
        update = generate_update("c1", "u1", 1)
        det_res = orchestrator.build_detection_result(update, res)
        assert det_res["action"] == "DOWN_WEIGHT"

    def test_action_mapping_flag(self, orchestrator: Sentinel):
        res = {
            "client_result": {"decision_output": {"recommended_action": "FLAG"}},
            "pipeline_metadata": {"client_id": "c1", "round_id": 1}
        }
        update = generate_update("c1", "u1", 1)
        det_res = orchestrator.build_detection_result(update, res)
        assert det_res["action"] == "DOWN_WEIGHT"

    def test_action_mapping_quarantine(self, orchestrator: Sentinel):
        res = {
            "client_result": {"decision_output": {"recommended_action": "QUARANTINE"}},
            "pipeline_metadata": {"client_id": "c1", "round_id": 1}
        }
        update = generate_update("c1", "u1", 1)
        det_res = orchestrator.build_detection_result(update, res)
        assert det_res["action"] == "QUARANTINE"

    def test_action_mapping_rollback_recommended(self, orchestrator: Sentinel):
        res = {
            "client_result": {"decision_output": {"recommended_action": "ROLLBACK_RECOMMENDED"}},
            "pipeline_metadata": {"client_id": "c1", "round_id": 1}
        }
        update = generate_update("c1", "u1", 1)
        det_res = orchestrator.build_detection_result(update, res)
        assert det_res["action"] == "QUARANTINE"

    def test_serialization(self, orchestrator: Sentinel):
        update = generate_update("c1", "u1", 1)
        res = orchestrator.run_round([update], 1)[0]
        # The whole structure should still be serializable
        try:
            exported = orchestrator.export_pipeline_result(res)
            json_str = json.dumps(exported)
            assert isinstance(json_str, str)
        except Exception as e:
            pytest.fail(f"Serialization failed: {e}")
            

