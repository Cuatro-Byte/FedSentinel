"""
simulation/scenarios/model_poisoning.py
=======================================
Model poisoning scenario builder (ScalingAttack / SignFlipAttack).

Contract reference: FedSentinel Team Engineering Contract v2.0 — Person 2
(Adversarial ML / Attack Engineer)
"""

from __future__ import annotations

from typing import List

from core.attacks.attack_manager import AttackManager
from core.attacks.model_poisoning import ScalingAttack, SignFlipAttack


def build_model_poisoning_scenario(
    client_ids: List[str],
    target_clients: List[str],
    start_round: int = 3,
    total_rounds: int = 10,
    scale_factor: float = 10.0,
    use_sign_flip: bool = False,
) -> AttackManager:
    """Schedule a model poisoning attack on selected target clients from start_round.

    Parameters
    ----------
    client_ids:
        List of all simulated client IDs.
    target_clients:
        Subset of client IDs designated as adversarial model poisoners.
    start_round:
        Round number at which poisoning activates (1-indexed). Default: 3.
    total_rounds:
        Total number of federated rounds in the simulation. Default: 10.
    scale_factor:
        Amplification scale factor (or intensity if use_sign_flip is True). Default: 10.0.
    use_sign_flip:
        If True, deploys SignFlipAttack; otherwise deploys ScalingAttack. Default: False.

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
                if use_sign_flip:
                    attack = SignFlipAttack(client_id=cid, intensity=scale_factor)
                else:
                    attack = ScalingAttack(client_id=cid, scale_factor=scale_factor)
                manager.register_attack(round_id=r, client_id=cid, attack=attack)
            else:
                _ = manager.get_attack(round_id=r, client_id=cid)

    return manager
