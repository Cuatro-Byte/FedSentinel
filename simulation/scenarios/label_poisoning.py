"""
simulation/scenarios/label_poisoning.py
=======================================
Label poisoning scenario builder (LabelFlipAttack).

Contract reference: FedSentinel Team Engineering Contract v2.0 — Person 2
(Adversarial ML / Attack Engineer)
"""

from __future__ import annotations

from typing import List

from core.attacks.attack_manager import AttackManager
from core.attacks.label_poisoning import LabelFlipAttack


def build_label_poisoning_scenario(
    client_ids: List[str],
    target_clients: List[str],
    source_class: int = 1,
    target_class: int = 7,
    poison_rate: float = 1.0,
    start_round: int = 3,
    total_rounds: int = 10,
) -> AttackManager:
    """Schedule a label-flipping data poisoning attack on target clients.

    Parameters
    ----------
    client_ids:
        List of all simulated client IDs.
    target_clients:
        Subset of client IDs designated as label poisoners.
    source_class:
        Original honest class to corrupt. Default: 1.
    target_class:
        Malicious replacement class. Default: 7.
    poison_rate:
        Fraction of source class samples to flip in [0.0, 1.0]. Default: 1.0.
    start_round:
        Round number at which poisoning activates. Default: 3.
    total_rounds:
        Total number of federated rounds in the simulation. Default: 10.

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
    if not target_clients:
        raise ValueError("target_clients must be non-empty")

    client_set = set(client_ids)
    missing = [c for c in target_clients if c not in client_set]
    if missing:
        raise ValueError(f"Target clients not present in client_ids: {missing}")

    manager = AttackManager()

    for r in range(1, total_rounds + 1):
        for cid in client_ids:
            if r >= start_round and cid in target_clients:
                attack = LabelFlipAttack(
                    client_id=cid,
                    source_class=source_class,
                    target_class=target_class,
                    poison_rate=poison_rate,
                )
                manager.register_attack(round_id=r, client_id=cid, attack=attack)
            else:
                _ = manager.get_attack(round_id=r, client_id=cid)

    return manager
