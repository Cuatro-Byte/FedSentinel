"""
simulation/scenarios/__init__.py
================================
Public surface for the FedSentinel simulation scenarios package.

Exposes scenario builder functions corresponding to all primary threat scenarios:
* :func:`build_normal_scenario`           — Honest baseline (no attacks)
* :func:`build_model_poisoning_scenario`  — Scaling and sign-flip model poisoning
* :func:`build_label_poisoning_scenario`  — Targeted label flipping
* :func:`build_backdoor_scenario`         — Backdoor trigger watermarking (Contract Sec 28)
* :func:`build_mixed_attack_scenario`     — Concurrent multi-vector attacks
* :func:`build_sleeper_scenario`          — Sleeper agent for Impact & Recovery evaluation

Contract reference: FedSentinel Team Engineering Contract v2.0 — Person 2
"""

from simulation.scenarios.backdoor import build_backdoor_scenario
from simulation.scenarios.label_poisoning import build_label_poisoning_scenario
from simulation.scenarios.mixed_attack import build_mixed_attack_scenario
from simulation.scenarios.model_poisoning import build_model_poisoning_scenario
from simulation.scenarios.normal import build_normal_scenario
from simulation.scenarios.sleeper_scenario import build_sleeper_scenario
from simulation.scenarios.sybil_scenario import build_sybil_scenario

__all__: list[str] = [
    "build_normal_scenario",
    "build_model_poisoning_scenario",
    "build_label_poisoning_scenario",
    "build_backdoor_scenario",
    "build_mixed_attack_scenario",
    "build_sleeper_scenario",
    "build_sybil_scenario",
]
