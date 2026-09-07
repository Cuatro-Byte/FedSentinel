"""
core/attacks/__init__.py
========================
Public surface of the ``core.attacks`` package.

Phase 1 exports: :class:`BaseAttack`
Phase 2 exports: :class:`SignFlipAttack`, :class:`ScalingAttack`,
                 :class:`AdaptiveStealthAttack`
Phase 3 exports: :class:`GaussianNoiseAttack`, :class:`ZeroUpdateAttack`,
                 :class:`ExtremeValueAttack`
Phase 4 exports: :class:`LabelFlipAttack`, :class:`LabelFlipDatasetWrapper`,
                 :class:`BackdoorAttack`, :class:`BackdoorDatasetWrapper`
Phase 5 exports: :class:`AttackManager`

Contract reference: FedSentinel Team Engineering Contract v2.0 — Person 2
"""

from core.attacks.attack_manager import AttackManager
from core.attacks.backdoor import (
    BackdoorAttack,
    BackdoorDatasetWrapper,
)
from core.attacks.base_attack import BaseAttack
from core.attacks.byzantine import (
    ExtremeValueAttack,
    GaussianNoiseAttack,
    ZeroUpdateAttack,
)
from core.attacks.label_poisoning import (
    LabelFlipAttack,
    LabelFlipDatasetWrapper,
)
from core.attacks.model_poisoning import (
    AdaptiveStealthAttack,
    ScalingAttack,
    SignFlipAttack,
)

__all__: list[str] = [
    "BaseAttack",
    # Phase 2 — model & gradient poisoning
    "SignFlipAttack",
    "ScalingAttack",
    "AdaptiveStealthAttack",
    # Phase 3 — Byzantine fault attacks
    "GaussianNoiseAttack",
    "ZeroUpdateAttack",
    "ExtremeValueAttack",
    # Phase 4 — Data poisoning & backdoor trigger attacks
    "LabelFlipAttack",
    "LabelFlipDatasetWrapper",
    "BackdoorAttack",
    "BackdoorDatasetWrapper",
    # Phase 5 — Centralized scheduling & ground-truth isolation
    "AttackManager",
]
