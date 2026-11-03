"""
core/attacks/attack_manager.py
==============================
Centralized attack scheduling and ground-truth isolation manager for FedSentinel.

Contract reference: FedSentinel Team Engineering Contract v2.0 — Person 2
(Adversarial ML / Attack Engineer)

Critical Security & Contract Invariants
---------------------------------------
1. Ground-truth isolation:
   Person 3 (Sentinel / Detection Engine) MUST NEVER call
   ``get_ground_truth_for_evaluation()`` during inference. The detection pipeline
   receives only ``ModelUpdate[]`` and must infer threat levels autonomously.
2. Evaluation reservation:
   Ground-truth attack labels are strictly reserved for post-hoc evaluation by
   Person 4 (backend evaluation metrics, dashboard ROC/confusion matrices).
3. No schema leakage:
   No `is_malicious` field or ground-truth flag is ever attached to ``BaseAttack``
   instances, ``ModelUpdate``, or return dictionaries.
"""

from __future__ import annotations

import copy
from typing import Dict, Optional

from core.attacks.base_attack import BaseAttack


class AttackManager:
    """Centralized schedule coordinator and isolated ground-truth registry.

    Manages which federated clients execute adversarial attacks in each round,
    and isolates ground-truth labels for post-aggregation evaluation.

    Attributes
    ----------
    _schedule : Dict[int, Dict[str, BaseAttack]]
        Internal mapping: ``round_id -> {client_id: BaseAttack}``.
    _ground_truth : Dict[int, Dict[str, str]]
        Internal ground-truth mapping: ``round_id -> {client_id: attack_type_str}``.
    """

    def __init__(self) -> None:
        self._schedule: Dict[int, Dict[str, BaseAttack]] = {}
        self._ground_truth: Dict[int, Dict[str, str]] = {}

    def register_attack(
        self,
        round_id: int,
        client_id: str,
        attack: BaseAttack,
    ) -> None:
        """Register an attack instance for a specific client in a federated round.

        Records the attack instance in the round schedule and updates the
        isolated ground-truth registry with ``attack.attack_type``.
        Overwrites any existing registration for the same client and round.

        Parameters
        ----------
        round_id:
            The federated learning round index (must be a non-negative integer).
        client_id:
            The identifier of the federated client.
        attack:
            The concrete :class:`BaseAttack` instance to schedule.
        """
        if not isinstance(round_id, int) or round_id < 0:
            raise ValueError(f"round_id must be a non-negative integer; got {round_id}")
        if not isinstance(client_id, str) or not client_id:
            raise ValueError(f"client_id must be a non-empty string; got {client_id!r}")
        if not isinstance(attack, BaseAttack):
            raise TypeError(f"attack must be an instance of BaseAttack; got {type(attack).__name__}")

        if round_id not in self._schedule:
            self._schedule[round_id] = {}
        if round_id not in self._ground_truth:
            self._ground_truth[round_id] = {}

        self._schedule[round_id][client_id] = attack
        self._ground_truth[round_id][client_id] = attack.attack_type

    def get_attack(self, round_id: int, client_id: str) -> Optional[BaseAttack]:
        """Retrieve the scheduled attack for a client in a specific round.

        If an attack was registered, returns the :class:`BaseAttack` instance.
        If the client is unassigned, returns ``None`` and implicitly records
        ``"honest"`` in the isolated ground-truth registry for evaluation.

        Parameters
        ----------
        round_id:
            The federated learning round index.
        client_id:
            The identifier of the federated client.

        Returns
        -------
        Optional[BaseAttack]
            The scheduled attack instance, or ``None`` if the client is honest.
        """
        if round_id not in self._ground_truth:
            self._ground_truth[round_id] = {}

        round_schedule = self._schedule.get(round_id, {})
        attack = round_schedule.get(client_id)

        if attack is None:
            # If not already recorded as malicious, record as honest for this round
            if client_id not in self._ground_truth[round_id]:
                self._ground_truth[round_id][client_id] = "honest"
            return None

        return attack

    def is_malicious(self, round_id: int, client_id: str) -> bool:
        """Query whether a client is scheduled as malicious in a given round.

        Parameters
        ----------
        round_id:
            The federated learning round index.
        client_id:
            The identifier of the federated client.

        Returns
        -------
        bool
            ``True`` if an attack is registered for that client in that round,
            ``False`` otherwise.
        """
        return client_id in self._schedule.get(round_id, {})

    def get_ground_truth_for_evaluation(self) -> Dict[int, Dict[str, str]]:
        """Return an isolated deep copy of the ground-truth attack registry.

        CRITICAL INTEGRATION CONTRACT NOTICE (Contract v2.0, Person 2):
        ---------------------------------------------------------------
        This method is STRICTLY RESERVED for Person 4 (Backend / Evaluation /
        Dashboard Metrics) to compute post-hoc accuracy, ROC curves, and confusion
        matrices after federated aggregation.

        THIS METHOD MUST NEVER BE CALLED BY PERSON 3 (FedSentinel Detection Engine)
        DURING INFERENCE. Under Section 5 and Section 27 of the contract, the
        detector must operate solely on ``ModelUpdate[]`` and extract features
        without knowledge of actual attack assignments.

        Returns
        -------
        Dict[int, Dict[str, str]]
            A deep copy of the ground-truth dictionary mapping:
            ``round_id -> {client_id: attack_type_str}``.
            External mutations to the returned dictionary will not affect
            internal state.
        """
        return copy.deepcopy(self._ground_truth)

    def clear(self) -> None:
        """Clear all registered attack schedules and ground-truth records."""
        self._schedule.clear()
        self._ground_truth.clear()
