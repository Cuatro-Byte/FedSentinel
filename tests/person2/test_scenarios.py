"""
tests/test_scenarios.py
=======================
Unit tests for simulation scenario builders in ``simulation.scenarios``:
    * :func:`~simulation.scenarios.normal.build_normal_scenario`
    * :func:`~simulation.scenarios.model_poisoning.build_model_poisoning_scenario`
    * :func:`~simulation.scenarios.label_poisoning.build_label_poisoning_scenario`
    * :func:`~simulation.scenarios.backdoor.build_backdoor_scenario`
    * :func:`~simulation.scenarios.mixed_attack.build_mixed_attack_scenario`

Run with::

    pytest tests/test_scenarios.py -v

Contract reference: FedSentinel Team Engineering Contract v2.0 — Person 2
"""

from __future__ import annotations

import pytest

from core.attacks.attack_manager import AttackManager
from core.attacks.backdoor import BackdoorAttack
from core.attacks.label_poisoning import LabelFlipAttack
from core.attacks.model_poisoning import ScalingAttack, SignFlipAttack
from simulation.scenarios import (
    build_backdoor_scenario,
    build_label_poisoning_scenario,
    build_mixed_attack_scenario,
    build_model_poisoning_scenario,
    build_normal_scenario,
)


@pytest.fixture
def standard_clients() -> list[str]:
    return [f"C{i:02d}" for i in range(1, 21)]  # C01 .. C20


# ---------------------------------------------------------------------------
# Section A — Normal Scenario
# ---------------------------------------------------------------------------

class TestNormalScenario:
    """Verify 100% honest baseline scenario."""

    def test_normal_scenario_all_honest(self, standard_clients: list[str]) -> None:
        total_rounds = 5
        manager = build_normal_scenario(total_rounds=total_rounds, client_ids=standard_clients)

        assert isinstance(manager, AttackManager)

        for r in range(1, total_rounds + 1):
            for cid in standard_clients:
                assert manager.get_attack(round_id=r, client_id=cid) is None
                assert manager.is_malicious(round_id=r, client_id=cid) is False

        gt = manager.get_ground_truth_for_evaluation()
        for r in range(1, total_rounds + 1):
            assert r in gt
            for cid in standard_clients:
                assert gt[r][cid] == "honest"


# ---------------------------------------------------------------------------
# Section B — Model Poisoning Scenario
# ---------------------------------------------------------------------------

class TestModelPoisoningScenario:
    """Verify ScalingAttack and SignFlipAttack scheduling."""

    def test_scaling_attack_schedule(self, standard_clients: list[str]) -> None:
        targets = ["C03", "C07"]
        start_round = 3
        total_rounds = 6
        scale = 15.0

        manager = build_model_poisoning_scenario(
            client_ids=standard_clients,
            target_clients=targets,
            start_round=start_round,
            total_rounds=total_rounds,
            scale_factor=scale,
            use_sign_flip=False,
        )

        # Before start_round: all clients honest
        for r in range(1, start_round):
            for cid in standard_clients:
                assert manager.get_attack(r, cid) is None
                assert manager.is_malicious(r, cid) is False

        # From start_round: targets malicious, others honest
        for r in range(start_round, total_rounds + 1):
            for cid in standard_clients:
                attack = manager.get_attack(r, cid)
                if cid in targets:
                    assert isinstance(attack, ScalingAttack)
                    assert attack.scale_factor == scale
                    assert manager.is_malicious(r, cid) is True
                else:
                    assert attack is None
                    assert manager.is_malicious(r, cid) is False

        gt = manager.get_ground_truth_for_evaluation()
        assert gt[start_round]["C03"] == "scaling"
        assert gt[start_round]["C01"] == "honest"

    def test_sign_flip_mode(self, standard_clients: list[str]) -> None:
        targets = ["C02"]
        start_round = 2
        total_rounds = 4

        manager = build_model_poisoning_scenario(
            client_ids=standard_clients,
            target_clients=targets,
            start_round=start_round,
            total_rounds=total_rounds,
            use_sign_flip=True,
        )

        attack = manager.get_attack(round_id=start_round, client_id="C02")
        assert isinstance(attack, SignFlipAttack)
        assert manager.is_malicious(start_round, "C02") is True


# ---------------------------------------------------------------------------
# Section C — Label Poisoning Scenario
# ---------------------------------------------------------------------------

class TestLabelPoisoningScenario:
    """Verify LabelFlipAttack scheduling."""

    def test_label_poisoning_schedule(self, standard_clients: list[str]) -> None:
        targets = ["C05", "C06"]
        start_round = 4
        total_rounds = 7

        manager = build_label_poisoning_scenario(
            client_ids=standard_clients,
            target_clients=targets,
            source_class=2,
            target_class=8,
            poison_rate=0.75,
            start_round=start_round,
            total_rounds=total_rounds,
        )

        # Before start_round
        for r in range(1, start_round):
            for cid in targets:
                assert manager.get_attack(r, cid) is None

        # From start_round
        for r in range(start_round, total_rounds + 1):
            for cid in targets:
                attack = manager.get_attack(r, cid)
                assert isinstance(attack, LabelFlipAttack)
                assert attack.source_class == 2
                assert attack.target_class == 8
                assert attack.poison_rate == 0.75

        gt = manager.get_ground_truth_for_evaluation()
        assert gt[start_round]["C05"] == "label_flip"
        assert gt[start_round]["C01"] == "honest"


# ---------------------------------------------------------------------------
# Section D — Backdoor Scenario (Contract Sec 28)
# ---------------------------------------------------------------------------

class TestBackdoorScenario:
    """Verify BackdoorAttack scheduling according to Contract Section 28."""

    def test_backdoor_default_c17_target(self, standard_clients: list[str]) -> None:
        # Standard Contract Section 28: 20 clients, C17 becomes malicious at round 5
        start_round = 5
        total_rounds = 10

        manager = build_backdoor_scenario(
            client_ids=standard_clients,
            target_clients=None,  # Defaults to C17
            start_round=start_round,
            total_rounds=total_rounds,
            target_label=0,
            poison_rate=0.3,
            intensity=0.8,
        )

        # Rounds 1-4: C17 is clean
        for r in range(1, start_round):
            assert manager.get_attack(r, "C17") is None
            assert manager.is_malicious(r, "C17") is False

        # Rounds 5-10: C17 is backdoor attacker
        for r in range(start_round, total_rounds + 1):
            attack = manager.get_attack(r, "C17")
            assert isinstance(attack, BackdoorAttack)
            assert attack.target_label == 0
            assert attack.poison_rate == 0.3
            assert attack.trigger_value == 0.8
            assert manager.is_malicious(r, "C17") is True

        # Other clients remain clean across all rounds
        for r in range(1, total_rounds + 1):
            assert manager.get_attack(r, "C01") is None

        gt = manager.get_ground_truth_for_evaluation()
        assert gt[4]["C17"] == "honest"
        assert gt[5]["C17"] == "backdoor"

    def test_backdoor_fallback_first_client(self) -> None:
        # If C17 is not present in client list, use first client
        clients = ["client_A", "client_B"]
        manager = build_backdoor_scenario(client_ids=clients, target_clients=None, start_round=2, total_rounds=3)

        assert manager.is_malicious(1, "client_A") is False
        assert manager.is_malicious(2, "client_A") is True
        assert manager.is_malicious(2, "client_B") is False


# ---------------------------------------------------------------------------
# Section E — Mixed Attack Scenario
# ---------------------------------------------------------------------------

class TestMixedAttackScenario:
    """Verify concurrent Byzantine and Backdoor attacks across client pools."""

    def test_mixed_attack_partitioning(self, standard_clients: list[str]) -> None:
        byzantine = ["C02", "C03"]
        backdoor = ["C08", "C09"]
        start_round = 3
        total_rounds = 5

        manager = build_mixed_attack_scenario(
            client_ids=standard_clients,
            byzantine_clients=byzantine,
            backdoor_clients=backdoor,
            start_round=start_round,
            total_rounds=total_rounds,
        )

        # Before start_round: all honest
        for r in range(1, start_round):
            for cid in byzantine + backdoor:
                assert manager.get_attack(r, cid) is None

        # From start_round: concurrent attacks
        for r in range(start_round, total_rounds + 1):
            for cid in byzantine:
                attack = manager.get_attack(r, cid)
                assert isinstance(attack, SignFlipAttack)
            for cid in backdoor:
                attack = manager.get_attack(r, cid)
                assert isinstance(attack, BackdoorAttack)
            for cid in ["C01", "C04", "C05"]:
                assert manager.get_attack(r, cid) is None

        gt = manager.get_ground_truth_for_evaluation()
        assert gt[start_round]["C02"] == "sign_flip"
        assert gt[start_round]["C08"] == "backdoor"
        assert gt[start_round]["C01"] == "honest"


# ---------------------------------------------------------------------------
# Section F — Input Validation & Error Handling
# ---------------------------------------------------------------------------

class TestScenarioValidation:
    """Verify input validation across all builders."""

    def test_invalid_rounds_raise(self, standard_clients: list[str]) -> None:
        with pytest.raises(ValueError, match="start_round .* cannot exceed total_rounds"):
            build_model_poisoning_scenario(standard_clients, ["C01"], start_round=11, total_rounds=10)

        with pytest.raises(ValueError, match="start_round must be an integer >= 1"):
            build_label_poisoning_scenario(standard_clients, ["C01"], start_round=0, total_rounds=5)

        with pytest.raises(ValueError, match="total_rounds must be an integer >= 1"):
            build_normal_scenario(total_rounds=0, client_ids=standard_clients)

    def test_missing_client_in_client_ids_raises(self, standard_clients: list[str]) -> None:
        with pytest.raises(ValueError, match="Target clients not present in client_ids"):
            build_model_poisoning_scenario(standard_clients, ["NON_EXISTENT"])

        with pytest.raises(ValueError, match="Target clients not present in client_ids"):
            build_backdoor_scenario(standard_clients, target_clients=["NON_EXISTENT"])

    def test_empty_clients_raise(self) -> None:
        with pytest.raises(ValueError, match="client_ids must be.*non-empty"):
            build_normal_scenario(total_rounds=5, client_ids=[])

    def test_mixed_overlapping_clients_raises(self, standard_clients: list[str]) -> None:
        with pytest.raises(ValueError, match="Clients cannot be assigned both"):
            build_mixed_attack_scenario(
                standard_clients,
                byzantine_clients=["C02"],
                backdoor_clients=["C02"],
            )


# ---------------------------------------------------------------------------
# Section G — Contract Invariants & Boundaries
# ---------------------------------------------------------------------------

class TestContractInvariants:
    """Ensure strict isolation and adherence to Contract v2.0."""

    def test_no_is_malicious_in_scheduled_attacks(self, standard_clients: list[str]) -> None:
        manager = build_mixed_attack_scenario(
            standard_clients,
            byzantine_clients=["C01"],
            backdoor_clients=["C02"],
            start_round=1,
            total_rounds=2,
        )

        for r in [1, 2]:
            for cid in standard_clients:
                att = manager.get_attack(r, cid)
                if att is not None:
                    assert not hasattr(att, "is_malicious")
                    assert "is_malicious" not in att.to_dict()

    def test_ground_truth_pure_strings(self, standard_clients: list[str]) -> None:
        manager = build_mixed_attack_scenario(
            standard_clients,
            byzantine_clients=["C01"],
            backdoor_clients=["C02"],
            start_round=1,
            total_rounds=2,
        )

        gt = manager.get_ground_truth_for_evaluation()
        for r, client_map in gt.items():
            assert isinstance(r, int)
            for cid, label in client_map.items():
                assert isinstance(cid, str)
                assert isinstance(label, str)
                assert label in {"honest", "sign_flip", "backdoor"}
