"""
simulation/scenarios/sleeper_scenario.py
========================================
Sleeper adversarial attack scenario builder for FedSentinel.

Designed specifically to exercise Person 3's Impact Estimator and Person 1's
Selective Recovery loop (Contract Sections 4.2, 20, 28):
1. Rounds 1 to (stealth_round - 1): Target client acts honestly to build a high reputation score.
2. Round stealth_round: Covert low-rate backdoor injected to slip past anomaly detection.
3. Round escalation_round to total_rounds: High-intensity scaling attack deliberately
   triggers global validation loss degradation, activating recovery.

Contract reference: FedSentinel Team Engineering Contract v2.0 — Person 2
(Adversarial ML / Attack Engineer)
"""

from __future__ import annotations

from typing import List

from core.attacks.attack_manager import AttackManager
from core.attacks.backdoor import BackdoorAttack
from core.attacks.model_poisoning import ScalingAttack


def build_sleeper_scenario(
    client_ids: List[str],
    target_client: str = "C17",
    stealth_round: int = 4,
    escalation_round: int = 5,
    total_rounds: int = 10,
    target_label: int = 0,
    stealth_poison_rate: float = 0.2,
    escalation_scale: float = 15.0,
) -> AttackManager:
    """Build the Sleeper scenario to test Impact Estimation and Selective Recovery.

    Parameters
    ----------
    client_ids:
        List of all simulated client IDs.
    target_client:
        The client ID acting as the sleeper agent (default: "C17").
    stealth_round:
        The round in which covert backdoor poisoning occurs (default: 4).
    escalation_round:
        The round in which high-intensity model scaling explodes (default: 5).
    total_rounds:
        Total rounds in the federated simulation (default: 10).
    target_label:
        Target backdoor class label (default: 0).
    stealth_poison_rate:
        Covert low poison rate for stealth round (default: 0.2).
    escalation_scale:
        Aggressive scale multiplier for escalation rounds (default: 15.0).

    Returns
    -------
    AttackManager
        Pre-configured attack manager ready for simulation execution.
    """
    if not isinstance(total_rounds, int) or total_rounds < 1:
        raise ValueError(f"total_rounds must be an integer >= 1; got {total_rounds}")
    if not client_ids:
        raise ValueError("client_ids must be a non-empty list")
    if target_client not in set(client_ids):
        raise ValueError(f"target_client '{target_client}' must be in client_ids")

    if not (1 < stealth_round < escalation_round <= total_rounds):
        raise ValueError(
            f"Expected 1 < stealth_round ({stealth_round}) < escalation_round ({escalation_round}) "
            f"<= total_rounds ({total_rounds})"
        )

    manager = AttackManager()

    for r in range(1, total_rounds + 1):
        for cid in client_ids:
            if cid == target_client:
                if r == stealth_round:
                    attack = BackdoorAttack(
                        client_id=cid,
                        target_label=target_label,
                        poison_rate=stealth_poison_rate,
                    )
                    manager.register_attack(round_id=r, client_id=cid, attack=attack)
                elif r >= escalation_round:
                    attack = ScalingAttack(
                        client_id=cid,
                        scale_factor=escalation_scale,
                    )
                    manager.register_attack(round_id=r, client_id=cid, attack=attack)
                else:
                    # Honest round (rounds 1 to stealth_round - 1)
                    _ = manager.get_attack(round_id=r, client_id=cid)
            else:
                # All other clients remain honest
                _ = manager.get_attack(round_id=r, client_id=cid)

    return manager
