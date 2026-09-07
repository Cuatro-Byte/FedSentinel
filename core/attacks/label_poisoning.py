"""
core/attacks/label_poisoning.py
================================
Dataset-level label poisoning attack implementations for FedSentinel Phase 4.

Provides:
* :class:`LabelFlipDatasetWrapper` — PyTorch Dataset wrapper injecting label flips.
* :class:`LabelFlipAttack`         — Concrete BaseAttack implementing apply_data_attack.

Contract reference: FedSentinel Team Engineering Contract v2.0 — Person 2
(Adversarial ML / Attack Engineer)

Dependency policy
-----------------
Strictly PyTorch, NumPy, and standard library.
CRITICAL: ``is_malicious`` ground-truth must NEVER appear in any returned object.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from core.attacks.base_attack import BaseAttack


class LabelFlipDatasetWrapper(Dataset):
    """PyTorch Dataset wrapper for targeted label flipping.

    Wraps an existing dataset. For samples where the ground truth label equals
    ``source_class``, the label is replaced with ``target_class`` with probability
    ``poison_rate``.

    Parameters
    ----------
    dataset:
        The underlying PyTorch dataset providing ``(x, y)`` tuples.
    source_class:
        The class label to target for poisoning.
    target_class:
        The replacement malicious class label.
    poison_rate:
        Probability in ``[0.0, 1.0]`` of flipping a matching label. Default: 1.0.
    seed:
        Optional PRNG seed for deterministic index-based reproducibility.
    """

    def __init__(
        self,
        dataset: Dataset,
        source_class: int,
        target_class: int,
        poison_rate: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        if isinstance(dataset, (str, bytes)) or not (hasattr(dataset, "__len__") and hasattr(dataset, "__getitem__")):
            raise TypeError("dataset must implement __len__ and __getitem__ and not be string/bytes")
        if not isinstance(source_class, (int, np.integer)):
            raise TypeError(f"source_class must be int; got {type(source_class).__name__}")
        if not isinstance(target_class, (int, np.integer)):
            raise TypeError(f"target_class must be int; got {type(target_class).__name__}")
        if not isinstance(poison_rate, (int, float)):
            raise TypeError(f"poison_rate must be float; got {type(poison_rate).__name__}")
        if not (0.0 <= float(poison_rate) <= 1.0):
            raise ValueError(f"poison_rate must be in [0.0, 1.0]; got {poison_rate}")
        if seed is not None and not isinstance(seed, (int, np.integer)):
            raise TypeError(f"seed must be int or None; got {type(seed).__name__}")

        self.dataset: Dataset = dataset
        self.source_class: int = int(source_class)
        self.target_class: int = int(target_class)
        self.poison_rate: float = float(poison_rate)
        self.seed: Optional[int] = int(seed) if seed is not None else None

    def __len__(self) -> int:
        return len(self.dataset)

    def _should_poison(self, idx: int) -> bool:
        if self.poison_rate <= 0.0:
            return False
        if self.poison_rate >= 1.0:
            return True

        if self.seed is not None:
            # Deterministic per index for consistent reproducibility
            rng = np.random.default_rng((self.seed + idx * 1000003) & 0xFFFFFFFF)
            return bool(rng.random() < self.poison_rate)
        else:
            return bool(np.random.random() < self.poison_rate)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        x, y = self.dataset[idx]

        # Extract numerical label if y is a PyTorch tensor
        label_val = int(y.item()) if isinstance(y, torch.Tensor) else int(y)

        if label_val == self.source_class and self._should_poison(idx):
            if isinstance(y, torch.Tensor):
                y_poisoned = torch.tensor(self.target_class, dtype=y.dtype, device=y.device)
            else:
                y_poisoned = self.target_class
            return x, y_poisoned

        return x, y


class LabelFlipAttack(BaseAttack):
    """Data-plane targeted label flipping attack.

    Wraps the client's local training dataset with a :class:`LabelFlipDatasetWrapper`
    prior to local training. During aggregation, the model update itself is passed
    unmodified because the adversarial influence has already manifested in the
    trained weights.

    Parameters
    ----------
    client_id:
        Unique identifier of the simulated malicious client.
    source_class:
        The honest source class to poison.
    target_class:
        The target adversarial class to assign.
    poison_rate:
        Fraction of ``source_class`` samples to flip in ``[0.0, 1.0]``. Default: 1.0.
    seed:
        Optional PRNG seed for deterministic dataset poisoning.
    """

    def __init__(
        self,
        client_id: str,
        source_class: int,
        target_class: int,
        poison_rate: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        if not isinstance(source_class, (int, np.integer)):
            raise TypeError(f"source_class must be int; got {type(source_class).__name__}")
        if not isinstance(target_class, (int, np.integer)):
            raise TypeError(f"target_class must be int; got {type(target_class).__name__}")
        if not isinstance(poison_rate, (int, float)):
            raise TypeError(f"poison_rate must be float; got {type(poison_rate).__name__}")
        if not (0.0 <= float(poison_rate) <= 1.0):
            raise ValueError(f"poison_rate must be in [0.0, 1.0]; got {poison_rate}")
        if seed is not None and not isinstance(seed, (int, np.integer)):
            raise TypeError(f"seed must be int or None; got {type(seed).__name__}")

        super().__init__(
            client_id=client_id,
            intensity=float(poison_rate),
            attack_type="label_flip",
        )

        self.source_class: int = int(source_class)
        self.target_class: int = int(target_class)
        self.poison_rate: float = float(poison_rate)
        self.seed: Optional[int] = int(seed) if seed is not None else None

    def apply_data_attack(self, dataset: Dataset) -> LabelFlipDatasetWrapper:
        """Wrap the local dataset with label-flipping logic before training.

        Parameters
        ----------
        dataset:
            PyTorch dataset instance to wrap.

        Returns
        -------
        LabelFlipDatasetWrapper
            The wrapped dataset with targeted labels flipped according to poison_rate.
        """
        return LabelFlipDatasetWrapper(
            dataset=dataset,
            source_class=self.source_class,
            target_class=self.target_class,
            poison_rate=self.poison_rate,
            seed=self.seed,
        )

    def apply_update_attack(
        self,
        delta: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Update-plane fallback: returns delta unmodified.

        Label poisoning works on the data plane before local training; the
        resulting parameter delta naturally embodies the poisoned distribution.
        """
        return delta
