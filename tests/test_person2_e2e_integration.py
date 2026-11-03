"""
tests/test_person2_e2e_integration.py
======================================
Comprehensive End-to-End Multi-Phase Integration Test Suite for Person 2
(Adversarial ML / Attack Engineer) in FedSentinel.

Validates the full interaction surface across all 7 development phases:
- Phase 1: Base Attack Infrastructure (BaseAttack, AttackManager, Ground Truth Isolation)
- Phase 2: Model-Poisoning Attacks (SignFlip, Scaling, AdaptiveStealth)
- Phase 3: Data-Plane Attacks (LabelFlipDatasetWrapper, LabelFlipAttack)
- Phase 4: Backdoor Watermarking (BackdoorDatasetWrapper, BackdoorAttack)
- Phase 5: Byzantine Fault Attacks (GaussianNoise, ZeroUpdate, ExtremeValue)
- Phase 6: Scenario Builders (Normal, Model Poisoning, Label Poisoning, Backdoor, Mixed)
- Phase 7: The Sleeper Scenario, Config Binding, and Contractual Isolation Boundaries

Contract Reference:
    FedSentinel Team Engineering Contract v2.0 — Person 2
    Sections 3.2, 4.2, 5, 20, 27, 28, 32

Run with:
    pytest tests/test_person2_e2e_integration.py -v
"""

from __future__ import annotations

import copy
from typing import Dict

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset, TensorDataset

from core.attacks.attack_manager import AttackManager
from core.attacks.backdoor import BackdoorAttack, BackdoorDatasetWrapper
from core.attacks.base_attack import BaseAttack
from core.attacks.byzantine import (
    ExtremeValueAttack,
    GaussianNoiseAttack,
    ZeroUpdateAttack,
)
from core.attacks.label_poisoning import LabelFlipAttack
from core.attacks.model_poisoning import (
    AdaptiveStealthAttack,
    ScalingAttack,
    SignFlipAttack,
)
from simulation.scenarios.backdoor import build_backdoor_scenario
from simulation.scenarios.mixed_attack import build_mixed_attack_scenario
from simulation.scenarios.sleeper_scenario import build_sleeper_scenario


# ---------------------------------------------------------------------------
# Test Case 1: Mock FL Simulation Pipeline Integration (Phases 1, 2, 3, 4, 5)
# ---------------------------------------------------------------------------

class TestMockFLSimulationPipeline:
    """Simulate a realistic multi-round, multi-client federated training loop."""

    def test_mock_fl_simulation_lifecycle(self) -> None:
        client_ids = ["C1", "C2", "C3", "C4", "C5"]
        rounds = [1, 2, 3]

        manager = AttackManager()

        # Round 1: All clients honest (no registrations)

        # Round 2: C1 -> SignFlipAttack, C2 -> BackdoorAttack
        manager.register_attack(
            round_id=2,
            client_id="C1",
            attack=SignFlipAttack(client_id="C1", intensity=1.0),
        )
        manager.register_attack(
            round_id=2,
            client_id="C2",
            attack=BackdoorAttack(
                client_id="C2",
                target_label=0,
                poison_rate=1.0,
                trigger_size=3,
                trigger_value=1.0,
            ),
        )

        # Round 3: C3 -> GaussianNoiseAttack, C4 -> ScalingAttack
        manager.register_attack(
            round_id=3,
            client_id="C3",
            attack=GaussianNoiseAttack(client_id="C3", mean=0.0, std=1.0, seed=42),
        )
        manager.register_attack(
            round_id=3,
            client_id="C4",
            attack=ScalingAttack(client_id="C4", scale_factor=10.0),
        )

        # Execute multi-round simulation loop
        for r in rounds:
            for cid in client_ids:
                attack = manager.get_attack(round_id=r, client_id=cid)

                # a) Create a mock PyTorch tensor dataset (10 samples, shape (3, 32, 32))
                x_data = torch.zeros((10, 3, 32, 32), dtype=torch.float32)
                y_data = torch.ones(10, dtype=torch.long) * 5  # class label 5
                mock_dataset: Dataset = TensorDataset(x_data, y_data)

                # b) Intercept data via attack.apply_data_attack(dataset) if attack exists
                if attack is not None:
                    effective_dataset = attack.apply_data_attack(mock_dataset)
                else:
                    effective_dataset = mock_dataset

                # c) Verify that C2's poisoned samples have the 3x3 trigger stamped in round 2
                if r == 2 and cid == "C2":
                    assert isinstance(attack, BackdoorAttack)
                    assert isinstance(effective_dataset, BackdoorDatasetWrapper)
                    assert len(effective_dataset) == 10
                    for sample_idx in range(len(effective_dataset)):
                        sample_x, sample_y = effective_dataset[sample_idx]
                        # Verify top-left 3x3 patch is filled with trigger_value (1.0)
                        patch = sample_x[:, :3, :3]
                        assert torch.all(patch == 1.0)
                        # Verify non-trigger pixels remain intact (0.0)
                        assert torch.all(sample_x[:, 3:, 3:] == 0.0)
                        # Verify label is flipped to target_label 0
                        assert int(sample_y) == 0

                # d) Generate a mock parameter delta dictionary
                mock_delta: Dict[str, np.ndarray] = {
                    "conv.weight": np.ones((16, 3, 3, 3), dtype=np.float32),
                    "fc.bias": np.zeros(10, dtype=np.float32),
                }

                # e) Intercept weights via attack.apply_update_attack(delta) if attack exists
                if attack is not None:
                    mutated_delta = attack.apply_update_attack(mock_delta)
                else:
                    mutated_delta = mock_delta

                # f) Verify weight mutations
                if r == 2 and cid == "C1":
                    # SignFlipAttack inverts direction
                    np.testing.assert_allclose(
                        mutated_delta["conv.weight"],
                        -1.0 * np.ones((16, 3, 3, 3), dtype=np.float32),
                    )
                    np.testing.assert_allclose(
                        mutated_delta["fc.bias"],
                        np.zeros(10, dtype=np.float32),
                    )
                elif r == 3 and cid == "C3":
                    # GaussianNoiseAttack adds noise
                    assert not np.allclose(
                        mutated_delta["conv.weight"],
                        np.ones((16, 3, 3, 3), dtype=np.float32),
                    )
                    # Finite values
                    assert np.all(np.isfinite(mutated_delta["conv.weight"]))
                elif r == 3 and cid == "C4":
                    # ScalingAttack scales by 10.0
                    np.testing.assert_allclose(
                        mutated_delta["conv.weight"],
                        10.0 * np.ones((16, 3, 3, 3), dtype=np.float32),
                    )
                    np.testing.assert_allclose(
                        mutated_delta["fc.bias"],
                        np.zeros(10, dtype=np.float32),
                    )
                elif attack is None or (r == 2 and cid == "C2"):
                    # Honest or data-plane only attack leaves delta unperturbed
                    np.testing.assert_allclose(
                        mutated_delta["conv.weight"],
                        mock_delta["conv.weight"],
                    )
                    np.testing.assert_allclose(
                        mutated_delta["fc.bias"],
                        mock_delta["fc.bias"],
                    )

                # g) Assert that resulting updates retain exact keys, shapes, and float32 dtypes
                assert set(mutated_delta.keys()) == {"conv.weight", "fc.bias"}
                assert mutated_delta["conv.weight"].shape == (16, 3, 3, 3)
                assert mutated_delta["fc.bias"].shape == (10,)
                assert mutated_delta["conv.weight"].dtype == np.float32
                assert mutated_delta["fc.bias"].dtype == np.float32

                # h) Verify that NO is_malicious key or attribute exists anywhere
                assert not hasattr(effective_dataset, "is_malicious")
                assert "is_malicious" not in mutated_delta
                if attack is not None:
                    assert not hasattr(attack, "is_malicious")
                    assert "is_malicious" not in attack.to_dict()


# ---------------------------------------------------------------------------
# Test Case 2: Full Canonical Backdoor Scenario Run (Phases 4, 5, 6 — Sec 28)
# ---------------------------------------------------------------------------

class TestCanonicalBackdoorScenario:
    """Verify Section 28 Backdoor demo scenario lifecycle and ground truth integrity."""

    def test_canonical_backdoor_run(self) -> None:
        client_ids = [f"C{i:02d}" for i in range(1, 21)]  # 20 clients
        start_round = 5
        total_rounds = 10

        manager = build_backdoor_scenario(
            client_ids=client_ids,
            start_round=start_round,
            total_rounds=total_rounds,
        )

        assert isinstance(manager, AttackManager)

        # Loop from round 1 to 10 for all 20 clients
        for r in range(1, total_rounds + 1):
            for cid in client_ids:
                attack = manager.get_attack(round_id=r, client_id=cid)

                # Rounds 1-4: All 20 clients return None
                if r < start_round:
                    assert attack is None
                    assert manager.is_malicious(round_id=r, client_id=cid) is False
                else:
                    # Rounds 5-10: Only C17 returns BackdoorAttack
                    if cid == "C17":
                        assert isinstance(attack, BackdoorAttack)
                        assert attack.client_id == "C17"
                        assert attack.target_label == 0
                        assert attack.poison_rate == pytest.approx(0.3)
                        assert manager.is_malicious(round_id=r, client_id=cid) is True
                    else:
                        assert attack is None
                        assert manager.is_malicious(round_id=r, client_id=cid) is False

        # Retrieve ground truth for evaluation
        gt = manager.get_ground_truth_for_evaluation()

        # Verify ground truth logs contain "honest" for all in rounds 1-4
        for r in range(1, start_round):
            for cid in client_ids:
                assert gt[r][cid] == "honest"

        # Verify round 5+ records "backdoor" for C17 and "honest" for all others
        for r in range(start_round, total_rounds + 1):
            for cid in client_ids:
                if cid == "C17":
                    assert gt[r][cid] == "backdoor"
                else:
                    assert gt[r][cid] == "honest"

        # Verify mutating ground_truth externally does not affect internal registry
        gt[5]["C17"] = "corrupted_attacker"
        gt[1]["C01"] = "corrupted_honest"
        fresh_gt = manager.get_ground_truth_for_evaluation()
        assert fresh_gt[5]["C17"] == "backdoor"
        assert fresh_gt[1]["C01"] == "honest"


# ---------------------------------------------------------------------------
# Test Case 3: Sleeper Attack Lifecycle & Recovery Trigger (Phases 2, 4, 5, 7)
# ---------------------------------------------------------------------------

class TestSleeperLifecycleAndRecoveryTrigger:
    """Verify reputation acquisition, stealth infiltration, and escalation triggering."""

    def test_sleeper_lifecycle(self) -> None:
        client_ids = [f"C{i:02d}" for i in range(1, 11)]  # 10 clients
        target = "C07"
        stealth_round = 3
        escalation_round = 4
        total_rounds = 5

        manager = build_sleeper_scenario(
            client_ids=client_ids,
            target_client=target,
            stealth_round=stealth_round,
            escalation_round=escalation_round,
            total_rounds=total_rounds,
            target_label=0,
            stealth_poison_rate=0.2,
            escalation_scale=15.0,
        )

        test_delta: Dict[str, np.ndarray] = {
            "layer1.weight": np.ones((8, 8), dtype=np.float32),
        }
        baseline_l2_norm = float(np.linalg.norm(test_delta["layer1.weight"]))

        # Rounds 1-2: Assert C07 receives None, delta unperturbed (reputation building)
        for r in [1, 2]:
            attack = manager.get_attack(round_id=r, client_id=target)
            assert attack is None
            assert manager.is_malicious(round_id=r, client_id=target) is False
            # Delta passes through unchanged
            res = attack.apply_update_attack(test_delta) if attack else test_delta
            np.testing.assert_allclose(res["layer1.weight"], test_delta["layer1.weight"])

        # Round 3 (Stealth): Assert C07 receives BackdoorAttack
        attack_r3 = manager.get_attack(round_id=stealth_round, client_id=target)
        assert isinstance(attack_r3, BackdoorAttack)
        assert attack_r3.poison_rate == pytest.approx(0.2)

        # Apply to dataset and assert top-left trigger is present with label flipped
        mock_ds = TensorDataset(
            torch.zeros((50, 3, 16, 16), dtype=torch.float32),
            torch.ones(50, dtype=torch.long) * 9,
        )
        wrapped_ds = attack_r3.apply_data_attack(mock_ds)
        assert isinstance(wrapped_ds, BackdoorDatasetWrapper)

        # Confirm poisoned samples exist in wrapped dataset
        poisoned_count = 0
        for idx in range(len(wrapped_ds)):
            x_item, y_item = wrapped_ds[idx]
            if int(y_item) == 0:
                assert torch.all(x_item[:, :3, :3] == 1.0)
                poisoned_count += 1
        assert poisoned_count > 0, "Stealth round must inject poisoned samples"

        # Delta remains unscaled in stealth round
        stealth_delta = attack_r3.apply_update_attack(test_delta)
        np.testing.assert_allclose(stealth_delta["layer1.weight"], test_delta["layer1.weight"])
        assert float(np.linalg.norm(stealth_delta["layer1.weight"])) == pytest.approx(baseline_l2_norm)

        # Round 4 (Escalation): Assert C07 receives ScalingAttack(scale_factor=15.0)
        attack_r4 = manager.get_attack(round_id=escalation_round, client_id=target)
        assert isinstance(attack_r4, ScalingAttack)
        assert attack_r4.scale_factor == pytest.approx(15.0)

        # Apply to delta and assert L2 norm explodes by exactly 15x
        escalated_delta = attack_r4.apply_update_attack(test_delta)
        escalated_l2_norm = float(np.linalg.norm(escalated_delta["layer1.weight"]))
        assert escalated_l2_norm == pytest.approx(15.0 * baseline_l2_norm)

        # Round 5: Assert C07 remains in attack state
        attack_r5 = manager.get_attack(round_id=5, client_id=target)
        assert isinstance(attack_r5, ScalingAttack)
        assert attack_r5.scale_factor == pytest.approx(15.0)

        # Confirm ground truth transitions cleanly: "honest" -> "backdoor" -> "scaling"
        gt = manager.get_ground_truth_for_evaluation()
        assert gt[1][target] == "honest"
        assert gt[2][target] == "honest"
        assert gt[3][target] == "backdoor"
        assert gt[4][target] == "scaling"
        assert gt[5][target] == "scaling"


# ---------------------------------------------------------------------------
# Test Case 4: Multi-Vector Mixed Scenario Stress Test (Phases 2, 3, 4, 6)
# ---------------------------------------------------------------------------

class TestMultiVectorMixedScenario:
    """Stress test concurrent Byzantine sign-flip and Backdoor watermark attacks."""

    def test_mixed_scenario_concurrent_execution(self) -> None:
        client_ids = [f"C{i:02d}" for i in range(1, 11)]
        byzantine_clients = ["C01", "C02"]
        backdoor_clients = ["C08", "C09"]
        honest_clients = ["C03", "C04", "C05", "C06", "C07", "C10"]
        start_round = 2
        total_rounds = 4

        manager = build_mixed_attack_scenario(
            client_ids=client_ids,
            byzantine_clients=byzantine_clients,
            backdoor_clients=backdoor_clients,
            start_round=start_round,
            total_rounds=total_rounds,
        )

        for r in range(1, total_rounds + 1):
            for cid in client_ids:
                attack = manager.get_attack(round_id=r, client_id=cid)

                if r < start_round:
                    assert attack is None
                else:
                    if cid in byzantine_clients:
                        assert isinstance(attack, SignFlipAttack)
                        assert attack.client_id == cid
                    elif cid in backdoor_clients:
                        assert isinstance(attack, BackdoorAttack)
                        assert attack.client_id == cid
                    else:
                        assert attack is None
                        assert cid in honest_clients

                # Apply both data and update attacks across all clients
                mock_ds = TensorDataset(
                    torch.ones((4, 3, 8, 8), dtype=torch.float32),
                    torch.zeros(4, dtype=torch.long),
                )
                mock_delta = {
                    "weight": np.ones((5, 5), dtype=np.float32),
                }

                # Verify no crashes, shape mismatches, or cross-contamination
                if attack is not None:
                    out_ds = attack.apply_data_attack(mock_ds)
                    out_delta = attack.apply_update_attack(mock_delta)
                else:
                    out_ds = mock_ds
                    out_delta = mock_delta

                assert len(out_ds) == 4
                assert out_delta["weight"].shape == (5, 5)
                assert out_delta["weight"].dtype == np.float32


# ---------------------------------------------------------------------------
# Test Case 5: End-to-End Contract & Security Boundary Invariants Check
# ---------------------------------------------------------------------------

class TestContractAndSecurityBoundaryInvariants:
    """Ensure strict isolation, no schema leakage, and non-destructive hooks."""

    def test_all_attacks_contract_compliance(self) -> None:
        attacks: list[BaseAttack] = [
            SignFlipAttack(client_id="C1", intensity=1.5),
            ScalingAttack(client_id="C2", scale_factor=10.0),
            AdaptiveStealthAttack(client_id="C3", max_norm_ratio=1.1, intensity=2.0),
            LabelFlipAttack(client_id="C4", source_class=1, target_class=7, poison_rate=1.0),
            BackdoorAttack(client_id="C5", target_label=0, poison_rate=0.3, trigger_size=3, trigger_value=1.0),
            GaussianNoiseAttack(client_id="C6", mean=0.0, std=1.0, seed=42),
            ZeroUpdateAttack(client_id="C7"),
            ExtremeValueAttack(client_id="C8", extreme_val=1e6),
        ]

        for attack in attacks:
            # Assert hasattr(attack, "is_malicious") is False
            assert hasattr(attack, "is_malicious") is False
            # Assert to_dict() outputs only {"client_id", "attack_type", "intensity"}
            d = attack.to_dict()
            assert set(d.keys()) == {"client_id", "attack_type", "intensity"}
            assert d["client_id"] == attack.client_id
            assert d["attack_type"] == attack.attack_type
            assert isinstance(d["intensity"], (int, float))

    def test_ground_truth_docstring_isolation_warning(self) -> None:
        doc = AttackManager.get_ground_truth_for_evaluation.__doc__
        assert doc is not None
        assert "PERSON 3" in doc or "Person 3" in doc
        assert "MUST NEVER BE CALLED BY PERSON 3" in doc

    def test_apply_update_attack_does_not_mutate_in_place(self) -> None:
        attacks: list[BaseAttack] = [
            SignFlipAttack(client_id="C1"),
            ScalingAttack(client_id="C2", scale_factor=5.0),
            AdaptiveStealthAttack(client_id="C3"),
            GaussianNoiseAttack(client_id="C4", seed=123),
            ZeroUpdateAttack(client_id="C5"),
            ExtremeValueAttack(client_id="C6"),
        ]

        for attack in attacks:
            original_delta = {
                "layer.weight": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
                "layer.bias": np.array([0.5, -0.5], dtype=np.float32),
            }
            weight_copy = original_delta["layer.weight"].copy()
            bias_copy = original_delta["layer.bias"].copy()

            result = attack.apply_update_attack(original_delta)

            # Returned dict must be a distinct dictionary
            assert result is not original_delta
            # Input dictionary arrays must remain completely unmutated
            np.testing.assert_array_equal(original_delta["layer.weight"], weight_copy)
            np.testing.assert_array_equal(original_delta["layer.bias"], bias_copy)
