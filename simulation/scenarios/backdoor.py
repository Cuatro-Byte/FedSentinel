"""
simulation/scenarios/backdoor.py
================================
Backdoor trigger watermark scenario builder (Contract Section 28 standard demo flow).

Contract reference: FedSentinel Team Engineering Contract v2.0 — Person 2
(Adversarial ML / Attack Engineer)
"""

from __future__ import annotations

from typing import List, Optional

from core.attacks.attack_manager import AttackManager
from core.attacks.backdoor import BackdoorAttack


def build_backdoor_scenario(
    client_ids: List[str],
    target_clients: Optional[List[str]] = None,
    start_round: int = 5,
    total_rounds: int = 10,
    target_label: int = 0,
    poison_rate: float = 0.3,
    intensity: float = 0.8,
) -> AttackManager:
    """Build the canonical FedSentinel Backdoor attack scenario (Contract Section 28).

    Deploys a backdoor watermark trigger attack on target clients starting at round 5.
    By default, targets client 'C17' (or the first client if 'C17' is not in client_ids).

    Parameters
    ----------
    client_ids:
        List of all simulated client IDs.
    target_clients:
        Optional list of adversarial clients. If None, defaults to ['C17'] if present
        in client_ids, or [client_ids[0]].
    start_round:
        Round at which the backdoor attack activates (default: 5).
    total_rounds:
        Total rounds in the federated simulation (default: 10).
    target_label:
        Target class for the trigger watermark (default: 0).
    poison_rate:
        Fraction of training data stamped with the trigger (default: 0.3).
    intensity:
        Trigger pixel value intensity (default: 0.8).

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

    client_set = set(client_ids)

    if target_clients is None:
        if "C17" in client_set:
            targets = ["C17"]
        else:
            targets = [client_ids[0]]
    else:
        if not target_clients:
            raise ValueError("target_clients cannot be an empty list when provided")
        missing = [c for c in target_clients if c not in client_set]
        if missing:
            raise ValueError(f"Target clients not present in client_ids: {missing}")
        targets = list(target_clients)

    manager = AttackManager()

    for r in range(1, total_rounds + 1):
        for cid in client_ids:
            if r >= start_round and cid in targets:
                attack = BackdoorAttack(
                    client_id=cid,
                    target_label=target_label,
                    poison_rate=poison_rate,
                    trigger_value=intensity,
                )
                manager.register_attack(round_id=r, client_id=cid, attack=attack)
            else:
                _ = manager.get_attack(round_id=r, client_id=cid)

    return manager
