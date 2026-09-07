"""
tests/test_sleeper_scenario.py
==============================
Unit tests for ``simulation.scenarios.sleeper_scenario.build_sleeper_scenario``:
    * Lifecycle trajectory (honest -> covert backdoor -> overt scaling)
    * Non-target client isolation (strictly honest)
    * Ground truth evaluation log isolation
    * Input validation and round boundary constraints
    * Contract compliance (no is_malicious attribute or leak)

Run with::

    pytest tests/test_sleeper_scenario.py -v

Contract reference: FedSentinel Team Engineering Contract v2.0 — Person 2
"""

from __future__ import annotations

import pytest

from core.attacks.attack_manager import AttackManager
from core.attacks.backdoor import BackdoorAttack
from core.attacks.model_poisoning import ScalingAttack
from simulation.scenarios.sleeper_scenario import build_sleeper_scenario
from simulation.scenarios import build_sleeper_scenario as build_sleeper_from_init


@pytest.fixture
def sample_clients() -> list[str]:
    return [f"C{i:02d}" for i in range(1, 21)]  # C01 .. C20


# ---------------------------------------------------------------------------
# Section A — Package Export
# ---------------------------------------------------------------------------

class TestImports:
    """Verify package re-export of build_sleeper_scenario."""

    def test_import_from_init(self) -> None:
        assert build_sleeper_from_init is build_sleeper_scenario


# ---------------------------------------------------------------------------
# Section B — Lifecycle Trajectory
# ---------------------------------------------------------------------------

class TestSleeperLifecycle:
    """Verify the 3-phase sleeper lifecycle on the target client."""

    def test_default_lifecycle_trajectory(self, sample_clients: list[str]) -> None:
        total_rounds = 10
        stealth_round = 4
        escalation_round = 5
        target = "C17"

        manager = build_sleeper_scenario(
            client_ids=sample_clients,
            target_client=target,
            stealth_round=stealth_round,
            escalation_round=escalation_round,
            total_rounds=total_rounds,
            target_label=0,
            stealth_poison_rate=0.2,
            escalation_scale=15.0,
        )

        assert isinstance(manager, AttackManager)

        # Phase 1 (Rounds 1-3): Target client is honest (reputation building)
        for r in range(1, stealth_round):
            attack = manager.get_attack(round_id=r, client_id=target)
            assert attack is None
            assert manager.is_malicious(round_id=r, client_id=target) is False

        # Phase 2 (Round 4): Stealth covert backdoor
        attack_r4 = manager.get_attack(round_id=stealth_round, client_id=target)
        assert isinstance(attack_r4, BackdoorAttack)
        assert attack_r4.target_label == 0
        assert attack_r4.poison_rate == pytest.approx(0.2)
        assert manager.is_malicious(round_id=stealth_round, client_id=target) is True

        # Phase 3 (Rounds 5-10): Escalation scaling attack
        for r in range(escalation_round, total_rounds + 1):
            attack = manager.get_attack(round_id=r, client_id=target)
            assert isinstance(attack, ScalingAttack)
            assert attack.scale_factor == pytest.approx(15.0)
            assert manager.is_malicious(round_id=r, client_id=target) is True

    def test_non_target_clients_remain_honest(self, sample_clients: list[str]) -> None:
        manager = build_sleeper_scenario(
            client_ids=sample_clients,
            target_client="C17",
            stealth_round=4,
            escalation_round=5,
            total_rounds=8,
        )

        for r in range(1, 9):
            for cid in sample_clients:
                if cid != "C17":
                    assert manager.get_attack(round_id=r, client_id=cid) is None
                    assert manager.is_malicious(round_id=r, client_id=cid) is False


# ---------------------------------------------------------------------------
# Section C — Ground Truth Logging
# ---------------------------------------------------------------------------

class TestGroundTruthLogging:
    """Verify evaluation ground truth matches the lifecycle phases."""

    def test_ground_truth_records(self, sample_clients: list[str]) -> None:
        manager = build_sleeper_scenario(
            client_ids=sample_clients,
            target_client="C17",
            stealth_round=4,
            escalation_round=5,
            total_rounds=7,
        )

        gt = manager.get_ground_truth_for_evaluation()

        # Rounds 1-3: Target recorded as honest
        for r in [1, 2, 3]:
            assert gt[r]["C17"] == "honest"

        # Round 4: Target recorded as backdoor
        assert gt[4]["C17"] == "backdoor"

        # Rounds 5-7: Target recorded as scaling
        for r in [5, 6, 7]:
            assert gt[r]["C17"] == "scaling"

        # Other clients recorded as honest across all rounds
        for r in range(1, 8):
            assert gt[r]["C01"] == "honest"
            assert gt[r]["C02"] == "honest"


# ---------------------------------------------------------------------------
# Section D — Input Validation & Error Handling
# ---------------------------------------------------------------------------

class TestInputValidation:
    """Verify robust parameter validation and boundary conditions."""

    def test_stealth_round_not_greater_than_one(self, sample_clients: list[str]) -> None:
        with pytest.raises(ValueError, match="Expected 1 < stealth_round"):
            build_sleeper_scenario(
                client_ids=sample_clients,
                target_client="C17",
                stealth_round=1,
                escalation_round=5,
            )

    def test_stealth_round_equal_to_escalation_round(self, sample_clients: list[str]) -> None:
        with pytest.raises(ValueError, match="Expected 1 < stealth_round"):
            build_sleeper_scenario(
                client_ids=sample_clients,
                target_client="C17",
                stealth_round=5,
                escalation_round=5,
            )

    def test_stealth_round_greater_than_escalation_round(self, sample_clients: list[str]) -> None:
        with pytest.raises(ValueError, match="Expected 1 < stealth_round"):
            build_sleeper_scenario(
                client_ids=sample_clients,
                target_client="C17",
                stealth_round=6,
                escalation_round=5,
            )

    def test_escalation_round_greater_than_total_rounds(self, sample_clients: list[str]) -> None:
        with pytest.raises(ValueError, match="Expected 1 < stealth_round"):
            build_sleeper_scenario(
                client_ids=sample_clients,
                target_client="C17",
                stealth_round=4,
                escalation_round=12,
                total_rounds=10,
            )

    def test_target_client_not_in_client_ids(self, sample_clients: list[str]) -> None:
        with pytest.raises(ValueError, match="target_client 'UNKNOWN' must be in client_ids"):
            build_sleeper_scenario(
                client_ids=sample_clients,
                target_client="UNKNOWN",
            )

    def test_empty_client_ids_raises(self) -> None:
        with pytest.raises(ValueError, match="client_ids must be a non-empty list"):
            build_sleeper_scenario(client_ids=[], target_client="C17")

    def test_zero_total_rounds_raises(self, sample_clients: list[str]) -> None:
        with pytest.raises(ValueError, match="total_rounds must be an integer >= 1"):
            build_sleeper_scenario(client_ids=sample_clients, total_rounds=0)


# ---------------------------------------------------------------------------
# Section E — Contract Invariants & Boundaries
# ---------------------------------------------------------------------------

class TestContractInvariants:
    """Ensure strict isolation and zero ground-truth leakage."""

    def test_no_is_malicious_leakage_in_attacks(self, sample_clients: list[str]) -> None:
        manager = build_sleeper_scenario(
            client_ids=sample_clients,
            target_client="C17",
            stealth_round=2,
            escalation_round=3,
            total_rounds=4,
        )

        for r in range(1, 5):
            for cid in sample_clients:
                attack = manager.get_attack(r, cid)
                if attack is not None:
                    assert not hasattr(attack, "is_malicious")
                    assert "is_malicious" not in attack.to_dict()

    def test_ground_truth_contains_only_clean_strings(self, sample_clients: list[str]) -> None:
        manager = build_sleeper_scenario(
            client_ids=sample_clients,
            target_client="C17",
            stealth_round=2,
            escalation_round=3,
            total_rounds=4,
        )

        gt = manager.get_ground_truth_for_evaluation()
        for r, client_dict in gt.items():
            for cid, label in client_dict.items():
                assert isinstance(label, str)
                assert label in {"honest", "backdoor", "scaling"}
