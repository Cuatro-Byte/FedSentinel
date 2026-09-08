"""
tests/test_attack_manager.py
============================
Unit tests for ``core.attacks.attack_manager.AttackManager``:
    * Centralized attack scheduling
    * Ground-truth recording and isolation
    * Post-hoc evaluation copy immutability
    * Multi-round, multi-client schedules
    * Contract boundary enforcement

Run with::

    pytest tests/test_attack_manager.py -v

Contract reference: FedSentinel Team Engineering Contract v2.0 — Person 2
"""

from __future__ import annotations

import pytest

from core.attacks.attack_manager import AttackManager
from core.attacks.backdoor import BackdoorAttack
from core.attacks.base_attack import BaseAttack
from core.attacks.byzantine import GaussianNoiseAttack, ZeroUpdateAttack
from core.attacks.model_poisoning import SignFlipAttack
from core.attacks import AttackManager as AttackManagerFromInit


# ---------------------------------------------------------------------------
# Section A — Package Export & Initialization
# ---------------------------------------------------------------------------

class TestImportsAndInit:
    """Verify package import and initial clean state."""

    def test_package_export(self) -> None:
        assert AttackManagerFromInit is AttackManager

    def test_initial_state_empty(self) -> None:
        manager = AttackManager()
        assert manager._schedule == {}
        assert manager._ground_truth == {}
        assert manager.get_ground_truth_for_evaluation() == {}


# ---------------------------------------------------------------------------
# Section B — Registration & Retrieval
# ---------------------------------------------------------------------------

class TestRegistrationAndRetrieval:
    """Verify attack registration, retrieval, and status queries."""

    def test_register_and_get_attack(self) -> None:
        manager = AttackManager()
        attack = SignFlipAttack(client_id="client_0", intensity=1.5)

        manager.register_attack(round_id=1, client_id="client_0", attack=attack)

        retrieved = manager.get_attack(round_id=1, client_id="client_0")
        assert retrieved is attack
        assert isinstance(retrieved, BaseAttack)

    def test_unscheduled_client_returns_none(self) -> None:
        manager = AttackManager()
        retrieved = manager.get_attack(round_id=1, client_id="client_honest")
        assert retrieved is None

    def test_is_malicious_query(self) -> None:
        manager = AttackManager()
        attack = ZeroUpdateAttack(client_id="client_bad")

        manager.register_attack(round_id=2, client_id="client_bad", attack=attack)

        assert manager.is_malicious(round_id=2, client_id="client_bad") is True
        assert manager.is_malicious(round_id=2, client_id="client_good") is False
        assert manager.is_malicious(round_id=99, client_id="client_bad") is False

    def test_overwrite_existing_registration(self) -> None:
        manager = AttackManager()
        attack1 = SignFlipAttack(client_id="client_0")
        attack2 = GaussianNoiseAttack(client_id="client_0", std=2.0)

        manager.register_attack(round_id=1, client_id="client_0", attack=attack1)
        assert manager.get_attack(round_id=1, client_id="client_0") is attack1

        # Overwrite with attack2
        manager.register_attack(round_id=1, client_id="client_0", attack=attack2)
        assert manager.get_attack(round_id=1, client_id="client_0") is attack2

        gt = manager.get_ground_truth_for_evaluation()
        assert gt[1]["client_0"] == "byzantine_gaussian"


# ---------------------------------------------------------------------------
# Section C — Ground-Truth Isolation & Recording
# ---------------------------------------------------------------------------

class TestGroundTruthIsolation:
    """Verify ground-truth tracking, unassigned client logging, and deep copy isolation."""

    def test_registered_attacks_log_attack_type(self) -> None:
        manager = AttackManager()
        manager.register_attack(
            round_id=0,
            client_id="c_flip",
            attack=SignFlipAttack(client_id="c_flip"),
        )
        manager.register_attack(
            round_id=0,
            client_id="c_backdoor",
            attack=BackdoorAttack(client_id="c_backdoor"),
        )

        gt = manager.get_ground_truth_for_evaluation()
        assert gt[0]["c_flip"] == "sign_flip"
        assert gt[0]["c_backdoor"] == "backdoor"

    def test_querying_unassigned_client_logs_honest(self) -> None:
        manager = AttackManager()
        # Querying an unassigned client should record "honest" for evaluation
        res = manager.get_attack(round_id=3, client_id="c_benign")
        assert res is None

        gt = manager.get_ground_truth_for_evaluation()
        assert 3 in gt
        assert gt[3]["c_benign"] == "honest"

    def test_deep_copy_prevents_external_tampering(self) -> None:
        manager = AttackManager()
        manager.register_attack(
            round_id=1,
            client_id="c_bad",
            attack=ZeroUpdateAttack(client_id="c_bad"),
        )

        gt_export = manager.get_ground_truth_for_evaluation()
        # Attempt to tamper with exported ground truth
        gt_export[1]["c_bad"] = "tampered_label"
        gt_export[1]["injected_client"] = "fake"

        # Internal state must remain untouched
        gt_internal = manager.get_ground_truth_for_evaluation()
        assert gt_internal[1]["c_bad"] == "byzantine_zero"
        assert "injected_client" not in gt_internal[1]

    def test_critical_docstring_present_on_evaluation_export(self) -> None:
        doc = AttackManager.get_ground_truth_for_evaluation.__doc__
        assert doc is not None
        assert "CRITICAL INTEGRATION CONTRACT NOTICE" in doc
        assert "MUST NEVER BE CALLED BY PERSON 3" in doc


# ---------------------------------------------------------------------------
# Section D — Multi-Round, Multi-Client Scheduling & Clearing
# ---------------------------------------------------------------------------

class TestMultiRoundSchedulingAndClear:
    """Verify complex multi-round scenarios and clean clearing."""

    def test_multi_round_scenario(self) -> None:
        manager = AttackManager()

        # Round 0: all honest
        _ = manager.get_attack(round_id=0, client_id="c0")
        _ = manager.get_attack(round_id=0, client_id="c1")

        # Round 1: c0 Byzantine noise, c1 honest
        manager.register_attack(
            round_id=1,
            client_id="c0",
            attack=GaussianNoiseAttack(client_id="c0", std=1.0),
        )
        _ = manager.get_attack(round_id=1, client_id="c1")

        # Round 2: c1 Backdoor, c0 honest
        manager.register_attack(
            round_id=2,
            client_id="c1",
            attack=BackdoorAttack(client_id="c1", target_label=0),
        )
        _ = manager.get_attack(round_id=2, client_id="c0")

        gt = manager.get_ground_truth_for_evaluation()
        assert gt[0] == {"c0": "honest", "c1": "honest"}
        assert gt[1] == {"c0": "byzantine_gaussian", "c1": "honest"}
        assert gt[2] == {"c0": "honest", "c1": "backdoor"}

    def test_clear_empties_all_state(self) -> None:
        manager = AttackManager()
        manager.register_attack(
            round_id=1,
            client_id="c0",
            attack=SignFlipAttack(client_id="c0"),
        )
        _ = manager.get_attack(round_id=1, client_id="c1")

        assert len(manager._schedule) > 0
        assert len(manager._ground_truth) > 0

        manager.clear()

        assert manager._schedule == {}
        assert manager._ground_truth == {}
        assert manager.get_ground_truth_for_evaluation() == {}


# ---------------------------------------------------------------------------
# Section E — Input Validation & Contract Invariants
# ---------------------------------------------------------------------------

class TestValidationAndContractInvariants:
    """Input validation and contract guarantees."""

    def test_negative_round_id_raises(self) -> None:
        manager = AttackManager()
        with pytest.raises(ValueError, match="round_id must be a non-negative integer"):
            manager.register_attack(-1, "c0", SignFlipAttack(client_id="c0"))

    def test_invalid_round_type_raises(self) -> None:
        manager = AttackManager()
        with pytest.raises(ValueError, match="round_id must be a non-negative integer"):
            manager.register_attack("round1", "c0", SignFlipAttack(client_id="c0"))  # type: ignore

    def test_empty_client_id_raises(self) -> None:
        manager = AttackManager()
        with pytest.raises(ValueError, match="client_id must be a non-empty string"):
            manager.register_attack(0, "", SignFlipAttack(client_id="c0"))

    def test_non_base_attack_raises(self) -> None:
        manager = AttackManager()
        with pytest.raises(TypeError, match="attack must be an instance of BaseAttack"):
            manager.register_attack(0, "c0", "not_an_attack")  # type: ignore

    def test_get_attack_does_not_inject_is_malicious(self) -> None:
        manager = AttackManager()
        attack = SignFlipAttack(client_id="c0")
        manager.register_attack(0, "c0", attack)

        retrieved = manager.get_attack(0, "c0")
        assert retrieved is not None
        assert not hasattr(retrieved, "is_malicious")
        assert "is_malicious" not in retrieved.to_dict()

    def test_ground_truth_values_are_pure_strings(self) -> None:
        manager = AttackManager()
        manager.register_attack(1, "c_flip", SignFlipAttack(client_id="c_flip"))
        _ = manager.get_attack(1, "c_honest")

        gt = manager.get_ground_truth_for_evaluation()
        for round_id, client_map in gt.items():
            assert isinstance(round_id, int)
            for client_id, label in client_map.items():
                assert isinstance(client_id, str)
                assert isinstance(label, str)
                assert label in {"sign_flip", "honest"}
