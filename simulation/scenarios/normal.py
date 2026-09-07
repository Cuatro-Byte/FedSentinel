"""
simulation/scenarios/normal.py
==============================
Baseline honest federated learning scenario builder (no adversarial attacks).

Contract reference: FedSentinel Team Engineering Contract v2.0 — Person 2
(Adversarial ML / Attack Engineer)
"""

from __future__ import annotations

from typing import List

from core.attacks.attack_manager import AttackManager


def build_normal_scenario(
    total_rounds: int,
    client_ids: List[str],
) -> AttackManager:
    """Build an AttackManager instance with zero attacks (100% honest baseline).

    Parameters
    ----------
    total_rounds:
        Total number of federated rounds in the simulation (must be >= 1).
    client_ids:
        List of all participating simulated client IDs.

    Returns
    -------
    AttackManager
        An AttackManager with no attacks registered, where every client in every
        round is recorded as honest in the evaluation ground truth.
    """
    if not isinstance(total_rounds, int) or total_rounds < 1:
        raise ValueError(f"total_rounds must be an integer >= 1; got {total_rounds}")
    if not client_ids:
        raise ValueError("client_ids must be a non-empty list of client IDs")

    manager = AttackManager()

    # Pre-populate evaluation ground truth as honest across all rounds and clients
    for r in range(1, total_rounds + 1):
        for cid in client_ids:
            _ = manager.get_attack(r, cid)

    return manager
