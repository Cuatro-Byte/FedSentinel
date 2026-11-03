"""
simulation/scenarios/sybil_scenario.py
======================================
Sybil attack scenario builder (coordinated multi-identity update alignment).

Contract reference: FedSentinel Team Engineering Contract v2.0 — Person 2
(Adversarial ML / Attack Engineer)
"""

from __future__ import annotations

from typing import List

from core.attacks.attack_manager import AttackManager
from core.attacks.sybil import SybilAttack


def build_sybil_scenario(
    client_ids: List[str],
    target_clients: List[str],
    start_round: int = 1,
    total_rounds: int = 10,
    shared_seed: int = 42,
    scale: float = 1.0,
    noise_std: float = 1e-4,
    target_bias: float = -0.5,
) -> AttackManager:
    """Schedule a Sybil attack on selected target clients from start_round.

    Parameters
    ----------
    client_ids:
        List of all simulated client IDs.
    target_clients:
        Subset of client IDs designated as colluding Sybil attackers (minimum 2).
    start_round:
        Round number at which the attack activates (1-indexed). Default: 1.
    total_rounds:
        Total number of federated rounds in the simulation. Default: 10.
    shared_seed:
        Seed shared across all Sybil clients to generate the common direction. Default: 42.
    scale:
        Scaling factor for Sybil updates. Default: 1.0.
    noise_std:
        Standard deviation of per-client micro-noise. Default: 1e-4.
    target_bias:
        Directional bias for the shared vector. Default: -0.5.

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
    if not target_clients or len(target_clients) < 2:
        raise ValueError(
            f"Sybil scenario requires at least 2 target clients; got {len(target_clients) if target_clients else 0}"
        )

    client_set = set(client_ids)
    missing = [c for c in target_clients if c not in client_set]
    if missing:
        raise ValueError(f"Target clients not present in client_ids: {missing}")

    manager = AttackManager()

    for r in range(1, total_rounds + 1):
        for cid in client_ids:
            if r >= start_round and cid in target_clients:
                attack = SybilAttack(
                    client_id=cid,
                    intensity=scale,
                    shared_seed=shared_seed,
                    scale=scale,
                    noise_std=noise_std,
                    target_bias=target_bias,
                )
                manager.register_attack(round_id=r, client_id=cid, attack=attack)
            else:
                _ = manager.get_attack(round_id=r, client_id=cid)

    return manager
