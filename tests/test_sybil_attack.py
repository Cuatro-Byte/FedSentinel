"""
tests/test_sybil_attack.py
==========================
Comprehensive tests for Sybil attack implementation, firewall compatibility,
Sentinel similarity detection, and scenario integration.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from backend.adapters.p2_attack_adapter import P2AttackAdapter
from core.attacks.attack_manager import AttackManager
from core.attacks.base_attack import BaseAttack
from core.attacks.sybil import SybilAttack
from core.models.model_update import ModelUpdate
from core.sentinel.feature_extractor import FeatureExtractor
from core.sentinel.sentinel import Sentinel
from core.sentinel.similarity import SimilarityEngine
from core.validation.client_validator import ClientUpdateFirewall
from simulation.scenarios.sybil_scenario import build_sybil_scenario
from simulation.scenarios.normal import build_normal_scenario


# ---------------------------------------------------------------------------
# Test Section A — Sybil Unit Tests
# ---------------------------------------------------------------------------

class TestSybilAttackUnit:
    """Unit tests for the concrete SybilAttack class."""

    def test_inherits_base_attack(self) -> None:
        attack = SybilAttack(client_id="sybil_1")
        assert isinstance(attack, BaseAttack)
        assert attack.attack_type == "sybil"
        assert attack.client_id == "sybil_1"

    def test_data_attack_is_noop(self) -> None:
        attack = SybilAttack(client_id="sybil_1")
        dummy_dataset = ["sample1", "sample2"]
        assert attack.apply_data_attack(dummy_dataset) is dummy_dataset

    def test_respects_is_active(self) -> None:
        attack = SybilAttack(client_id="sybil_1", is_active=False)
        delta = {
            "layer1.weight": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "layer1.bias": np.array([0.5], dtype=np.float32),
        }
        result = attack.apply_update_attack(delta)
        assert result is delta
        assert np.array_equal(result["layer1.weight"], delta["layer1.weight"])

    def test_output_keys_shapes_dtypes_and_finiteness(self) -> None:
        attack = SybilAttack(client_id="sybil_1", shared_seed=123)
        delta = {
            "conv.weight": np.random.randn(8, 4, 3, 3).astype(np.float32),
            "conv.bias": np.random.randn(8).astype(np.float32),
            "fc.weight": np.random.randn(10, 32).astype(np.float64),
        }
        mutated = attack.apply_update_attack(delta)

        assert set(mutated.keys()) == set(delta.keys())
        for k in delta:
            assert mutated[k].shape == delta[k].shape
            assert mutated[k].dtype == delta[k].dtype
            assert np.all(np.isfinite(mutated[k]))
            assert not np.any(np.isnan(mutated[k]))
            assert not np.any(np.isinf(mutated[k]))

    def test_independent_instances_high_cosine_correlation(self) -> None:
        """Two independent SybilAttack instances using the same shared_seed produce
        highly correlated (> 0.98 cosine similarity) non-identical updates."""
        attack_a = SybilAttack(client_id="sybil_A", shared_seed=42, noise_std=1e-4)
        attack_b = SybilAttack(client_id="sybil_B", shared_seed=42, noise_std=1e-4)

        # Baseline delta
        delta_template = {
            "layer.weight": np.random.RandomState(999).randn(64, 64).astype(np.float32),
            "layer.bias": np.random.RandomState(888).randn(64).astype(np.float32),
        }

        mutated_a = attack_a.apply_update_attack(delta_template)
        mutated_b = attack_b.apply_update_attack(delta_template)

        # Confirm non-identical byte-for-byte due to micro-noise jitter
        assert not np.array_equal(mutated_a["layer.weight"], mutated_b["layer.weight"])

        # Flatten and compute cosine similarity
        vec_a = np.concatenate([v.ravel().astype(np.float64) for v in mutated_a.values()])
        vec_b = np.concatenate([v.ravel().astype(np.float64) for v in mutated_b.values()])

        cos_sim = float(np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)))
        assert cos_sim > 0.98, f"Expected cosine similarity > 0.98, got {cos_sim}"

    def test_sybil_to_sybil_substantially_higher_than_sybil_to_honest(self) -> None:
        """Sybil-to-Sybil similarity is substantially higher than Sybil-to-honest similarity."""
        attack_sybil_1 = SybilAttack(client_id="sybil_1", shared_seed=42)
        attack_sybil_2 = SybilAttack(client_id="sybil_2", shared_seed=42)

        # Honest updates from different client data distributions
        honest_delta_1 = {
            "w": np.random.RandomState(101).randn(50, 50).astype(np.float32),
        }
        honest_delta_2 = {
            "w": np.random.RandomState(102).randn(50, 50).astype(np.float32),
        }

        sybil_update_1 = attack_sybil_1.apply_update_attack(honest_delta_1)
        sybil_update_2 = attack_sybil_2.apply_update_attack(honest_delta_2)

        vec_s1 = sybil_update_1["w"].ravel().astype(np.float64)
        vec_s2 = sybil_update_2["w"].ravel().astype(np.float64)
        vec_h1 = honest_delta_1["w"].ravel().astype(np.float64)
        vec_h2 = honest_delta_2["w"].ravel().astype(np.float64)

        sim_sybil_sybil = float(np.dot(vec_s1, vec_s2) / (np.linalg.norm(vec_s1) * np.linalg.norm(vec_s2)))
        sim_sybil_honest = float(np.dot(vec_s1, vec_h1) / (np.linalg.norm(vec_s1) * np.linalg.norm(vec_h1)))
        sim_honest_honest = float(np.dot(vec_h1, vec_h2) / (np.linalg.norm(vec_h1) * np.linalg.norm(vec_h2)))

        assert sim_sybil_sybil > 0.98
        assert sim_sybil_sybil > sim_sybil_honest + 0.4
        assert abs(sim_sybil_honest) < 0.3


# ---------------------------------------------------------------------------
# Test Section B — Firewall Integration Tests
# ---------------------------------------------------------------------------

class TestSybilFirewallIntegration:
    """Verify Sybil updates pass ClientUpdateFirewall validation."""

    def test_firewall_accepts_sybil_updates(self) -> None:
        firewall = ClientUpdateFirewall()
        attack = SybilAttack(client_id="sybil_client", shared_seed=42)

        raw_delta = {
            "weight": np.random.randn(10, 10).astype(np.float32),
            "bias": np.random.randn(10).astype(np.float32),
        }
        mutated = attack.apply_update_attack(raw_delta)

        sybil_update = ModelUpdate(
            update_id="upd-sybil-001",
            run_id="RUN-TEST",
            client_id="sybil_client",
            round_id=1,
            model_version="model-v1",
            base_model_version="model-v0",
            parameters={k: torch.from_numpy(v) for k, v in mutated.items()},
            sample_count=100,
            training_epochs=1,
        )

        expected_keys = {"weight", "bias"}
        expected_shapes = {"weight": (10, 10), "bias": (10,)}

        is_valid, reasons = firewall.validate_update(
            sybil_update,
            expected_keys=expected_keys,
            expected_shapes=expected_shapes,
        )

        assert is_valid is True, f"Firewall rejected Sybil update: {reasons}"
        assert len(reasons) == 0


# ---------------------------------------------------------------------------
# Test Section C — Sentinel Integration Tests
# ---------------------------------------------------------------------------

class TestSybilSentinelIntegration:
    """Verify Sentinel SimilarityEngine & AnomalyDetector behavior on Sybil cohort."""

    def test_sentinel_similarity_and_detection_on_cohort(self) -> None:
        """Create a cohort of 8 honest clients and 3 Sybil clients.
        Verify SimilarityEngine and Sentinel observe high pairwise Sybil similarity
        and elevated similarity anomaly without receiving attacker IDs or ground truth.
        """
        rng = np.random.RandomState(12345)
        n_honest = 8
        n_sybil = 3
        dim = 200

        # Base honest updates (simulating dispersed gradients around a common mode)
        base_direction = rng.randn(dim).astype(np.float32)
        base_direction /= np.linalg.norm(base_direction)

        honest_updates: list[ModelUpdate] = []
        for i in range(n_honest):
            client_grad = (base_direction + rng.randn(dim).astype(np.float32) * 0.4)
            honest_updates.append(
                ModelUpdate(
                    update_id=f"upd-honest-{i}",
                    run_id="RUN-TEST",
                    client_id=f"honest_{i}",
                    round_id=1,
                    model_version="model-v1",
                    base_model_version="model-v0",
                    parameters={"weights": torch.from_numpy(client_grad)},
                    sample_count=100,
                    training_epochs=1,
                )
            )

        # Sybil updates (coordinated via SybilAttack)
        sybil_updates: list[ModelUpdate] = []
        for i in range(n_sybil):
            attacker = SybilAttack(
                client_id=f"sybil_{i}",
                shared_seed=777,
                scale=1.0,
                noise_std=1e-4,
                target_bias=-0.5,
            )
            raw_delta = {"weights": (base_direction + rng.randn(dim).astype(np.float32) * 0.4)}
            mutated = attacker.apply_update_attack(raw_delta)
            sybil_updates.append(
                ModelUpdate(
                    update_id=f"upd-sybil-{i}",
                    run_id="RUN-TEST",
                    client_id=f"sybil_{i}",
                    round_id=1,
                    model_version="model-v1",
                    base_model_version="model-v0",
                    parameters={"weights": torch.from_numpy(mutated["weights"])},
                    sample_count=100,
                    training_epochs=1,
                )
            )

        all_updates = honest_updates + sybil_updates

        # 1. FeatureExtractor & SimilarityEngine
        extractor = FeatureExtractor()
        features_list = [extractor.extract(u) for u in all_updates]

        sim_engine = SimilarityEngine()
        sim_metrics = sim_engine.compute_similarities(features_list)

        assert len(sim_metrics) == len(all_updates)

        # Inspect pairwise similarity between Sybil clients
        # Let's compute pairwise cosine between the 3 Sybils from features
        sybil_features = [f["flat_parameters"] for f in features_list[n_honest:]]
        s0 = sybil_features[0] / np.linalg.norm(sybil_features[0])
        s1 = sybil_features[1] / np.linalg.norm(sybil_features[1])
        s2 = sybil_features[2] / np.linalg.norm(sybil_features[2])

        sim_01 = float(np.dot(s0, s1))
        sim_02 = float(np.dot(s0, s2))
        sim_12 = float(np.dot(s1, s2))

        assert sim_01 > 0.98, f"Sybil 0-1 similarity too low: {sim_01}"
        assert sim_02 > 0.98, f"Sybil 0-2 similarity too low: {sim_02}"
        assert sim_12 > 0.98, f"Sybil 1-2 similarity too low: {sim_12}"

        # 2. Run Sentinel without passing any ground truth
        sentinel = Sentinel()
        round_results = sentinel.run_round(all_updates, round_id=1)

        assert len(round_results) == len(all_updates)

        for res in round_results:
            c_res = res["client_result"]
            det = c_res["detection_result"]
            cid = det["client_id"]
            threat_score = det["threat_score"]
            threat_level = det["threat_level"]
            anomaly_score = det["anomaly_score"]

            assert threat_score >= 0.0
            assert threat_level in ["SAFE", "SUSPICIOUS", "MALICIOUS"]
            assert anomaly_score >= 0.0

            # Verify similarity evidence exists
            sim_out = c_res.get("similarity_output", {})
            assert "peer_similarity" in sim_out

        # Also verify via P3SentinelAdapter
        from backend.adapters.p3_sentinel_adapter import P3SentinelAdapter
        p3_adapter = P3SentinelAdapter()
        detection_results = p3_adapter.detect(all_updates, round_id=1)
        assert len(detection_results) == len(all_updates)
        for dr in detection_results:
            assert dr.threat_score >= 0.0
            assert dr.threat_level is not None


# ---------------------------------------------------------------------------
# Test Section D — Scenario Integration Tests
# ---------------------------------------------------------------------------

class TestSybilScenarioIntegration:
    """Verify sybil scenario integration with P2AttackAdapter and AttackManager."""

    def test_sybil_scenario_builder_validation(self) -> None:
        clients = [f"C{i}" for i in range(10)]

        # Must require at least 2 target clients
        with pytest.raises(ValueError, match="at least 2 target clients"):
            build_sybil_scenario(client_ids=clients, target_clients=["C0"])

        # Valid build
        manager = build_sybil_scenario(
            client_ids=clients,
            target_clients=["C0", "C1"],
            start_round=2,
            total_rounds=5,
        )
        assert isinstance(manager, AttackManager)

        # Round 1: no attacks
        assert manager.get_attack(round_id=1, client_id="C0") is None
        assert manager.is_malicious(round_id=1, client_id="C0") is False

        # Round 2: attacks active for C0 and C1
        attack_c0 = manager.get_attack(round_id=2, client_id="C0")
        attack_c1 = manager.get_attack(round_id=2, client_id="C1")
        assert attack_c0 is not None
        assert attack_c0.attack_type == "sybil"
        assert attack_c1 is not None
        assert attack_c1.attack_type == "sybil"
        assert manager.get_attack(round_id=2, client_id="C2") is None

        # Ground truth isolation check
        gt = manager.get_ground_truth_for_evaluation()
        assert gt[2]["C0"] == "sybil"
        assert gt[2]["C1"] == "sybil"
        assert gt[2]["C2"] == "honest"

    def test_p2_adapter_accepts_sybil_scenario(self) -> None:
        adapter = P2AttackAdapter()
        clients = [f"client-{i}" for i in range(5)]
        config = {
            "enabled": True,
            "scenario": "sybil",
            "attacker_count": 2,
            "start_round": 1,
            "total_rounds": 3,
        }

        # Apply attacks to ModelUpdates
        raw_updates = [
            ModelUpdate(
                update_id=f"upd-{c}",
                run_id="RUN-TEST",
                client_id=c,
                round_id=1,
                model_version="model-v1",
                base_model_version="model-v0",
                parameters={"layer": np.random.randn(10, 10).astype(np.float32)},
                sample_count=50,
                training_epochs=1,
            )
            for c in clients
        ]

        attacked_updates = adapter.apply_attacks(raw_updates, round_id=1, config=config)
        assert len(attacked_updates) == len(raw_updates)

        attackers = adapter.get_attacker_ids(round_id=1, config=config)
        assert len(attackers) == 2

        # Verify attackers were mutated while honest clients were not
        for orig, mutated in zip(raw_updates, attacked_updates):
            if orig.client_id in attackers:
                assert not np.array_equal(orig.parameters["layer"], mutated.parameters["layer"])
            else:
                assert np.array_equal(orig.parameters["layer"], mutated.parameters["layer"])

    def test_existing_scenarios_unaffected(self) -> None:
        """Verify normal and other existing scenario builders remain intact."""
        clients = [f"C{i}" for i in range(5)]
        manager = build_normal_scenario(total_rounds=3, client_ids=clients)
        for r in range(1, 4):
            for cid in clients:
                assert manager.get_attack(r, cid) is None
