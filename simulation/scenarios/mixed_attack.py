"""
simulation/scenarios/mixed_attack.py
====================================
Mixed adversarial scenario builder (concurrent Byzantine sign flip and Backdoor attacks).

Contract reference: FedSentinel Team Engineering Contract v2.0 — Person 2
(Adversarial ML / Attack Engineer)
"""

from __future__ import annotations

from typing import List

from core.attacks.attack_manager import AttackManager
from core.attacks.backdoor import BackdoorAttack
from core.attacks.model_poisoning import SignFlipAttack


def build_mixed_attack_scenario(
    client_ids: List[str],
    byzantine_clients: List[str],
    backdoor_clients: List[str],
    start_round: int = 3,
    total_rounds: int = 10,
) -> AttackManager:
    """Schedule concurrent Byzantine and Backdoor attacks across separate client pools.

    Parameters
    ----------
    client_ids:
        List of all simulated client IDs.
    byzantine_clients:
        Subset of clients scheduled to perform Byzantine sign-flip attacks.
    backdoor_clients:
        Subset of clients scheduled to perform Backdoor watermark attacks.
    start_round:
        Round number at which both attack modes activate (default: 3).
    total_rounds:
        Total number of federated rounds in the simulation (default: 10).

    Returns
    -------
    AttackManager
        Pre-configured attack manager ready for simulation execution.
    """
    if not isinstance(total_rounds, int) or total_rounds < 1:
        raise ValueError(f"total_rounds must be an integer >= 1; got {total_rounds}")
    if not isinstance(start_round, int) or start_round < 1:
        raise ValueError(f"start_round must be an integer >= 1; got {start_round}")
    if start_round > total_rounds:
        raise ValueError(f"start_round ({start_round}) cannot exceed total_rounds ({total_rounds})")
    if not client_ids:
        raise ValueError("client_ids must be non-empty")
    if not byzantine_clients and not backdoor_clients:
        raise ValueError("Must specify at least one byzantine or backdoor client")

    client_set = set(client_ids)
    byz_set = set(byzantine_clients)
    back_set = set(backdoor_clients)

    missing_byz = [c for c in byz_set if c not in client_set]
    if missing_byz:
        raise ValueError(f"Byzantine clients not in client_ids: {missing_byz}")

    missing_back = [c for c in back_set if c not in client_set]
    if missing_back:
        raise ValueError(f"Backdoor clients not in client_ids: {missing_back}")

    overlap = byz_set.intersection(back_set)
    if overlap:
        raise ValueError(f"Clients cannot be assigned both Byzantine and Backdoor attacks: {list(overlap)}")

    manager = AttackManager()

    for r in range(1, total_rounds + 1):
        for cid in client_ids:
            if r >= start_round:
                if cid in byz_set:
                    attack = SignFlipAttack(client_id=cid)
                    manager.register_attack(round_id=r, client_id=cid, attack=attack)
                    continue
                elif cid in back_set:
                    attack = BackdoorAttack(client_id=cid)
                    manager.register_attack(round_id=r, client_id=cid, attack=attack)
                    continue

            # Record honest
            _ = manager.get_attack(round_id=r, client_id=cid)

    return manager
