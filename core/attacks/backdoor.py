"""
core/attacks/backdoor.py
========================
Backdoor watermark trigger dataset wrapper and attack for FedSentinel Phase 4.

Provides:
* :class:`BackdoorDatasetWrapper` — PyTorch Dataset wrapper injecting visual trigger patterns.
* :class:`BackdoorAttack`         — Concrete BaseAttack implementing apply_data_attack.

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


class BackdoorDatasetWrapper(Dataset):
    """PyTorch Dataset wrapper for injecting backdoor watermark triggers into images.

    Wraps an existing dataset. With probability ``poison_rate``, an image tensor
    is stamped with a top-left square trigger watermark of size
    ``trigger_size x trigger_size`` filled with ``trigger_value``, and its label
    is relabeled to ``target_label``.

    Supports both 3D ``(C, H, W)`` and 2D ``(H, W)`` image tensors.
    Underlying dataset tensors are never mutated in-place.

    Parameters
    ----------
    dataset:
        The underlying PyTorch dataset providing ``(x, y)`` tuples.
    target_label:
        The adversarial target label to embed on poisoned items. Default: 0.
    poison_rate:
        Fraction of items to poison in ``[0.0, 1.0]``. Default: 0.3.
    trigger_size:
        Side length in pixels of the square trigger pattern. Default: 3.
    trigger_value:
        Pixel intensity assigned to the trigger watermark. Default: 1.0.
    seed:
        Optional PRNG seed for deterministic index-based reproducibility.
    """

    def __init__(
        self,
        dataset: Dataset,
        target_label: int = 0,
        poison_rate: float = 0.3,
        trigger_size: int = 3,
        trigger_value: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        if isinstance(dataset, (str, bytes)) or not (hasattr(dataset, "__len__") and hasattr(dataset, "__getitem__")):
            raise TypeError("dataset must implement __len__ and __getitem__ and not be string/bytes")
        if not isinstance(target_label, (int, np.integer)):
            raise TypeError(f"target_label must be int; got {type(target_label).__name__}")
        if not isinstance(poison_rate, (int, float)):
            raise TypeError(f"poison_rate must be float; got {type(poison_rate).__name__}")
        if not (0.0 <= float(poison_rate) <= 1.0):
            raise ValueError(f"poison_rate must be in [0.0, 1.0]; got {poison_rate}")
        if not isinstance(trigger_size, (int, np.integer)):
            raise TypeError(f"trigger_size must be int; got {type(trigger_size).__name__}")
        if trigger_size < 1:
            raise ValueError(f"trigger_size must be positive; got {trigger_size}")
        if not isinstance(trigger_value, (int, float)):
            raise TypeError(f"trigger_value must be numeric; got {type(trigger_value).__name__}")
        if seed is not None and not isinstance(seed, (int, np.integer)):
            raise TypeError(f"seed must be int or None; got {type(seed).__name__}")

        self.dataset: Dataset = dataset
        self.target_label: int = int(target_label)
        self.poison_rate: float = float(poison_rate)
        self.trigger_size: int = int(trigger_size)
        self.trigger_value: float = float(trigger_value)
        self.seed: Optional[int] = int(seed) if seed is not None else None

    def __len__(self) -> int:
        return len(self.dataset)

    def _should_poison(self, idx: int) -> bool:
        if self.poison_rate <= 0.0:
            return False
        if self.poison_rate >= 1.0:
            return True

        if self.seed is not None:
            rng = np.random.default_rng((self.seed + idx * 1000003) & 0xFFFFFFFF)
            return bool(rng.random() < self.poison_rate)
        else:
            return bool(np.random.random() < self.poison_rate)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        x, y = self.dataset[idx]

        if not self._should_poison(idx):
            return x, y

        # Ensure x is a torch tensor
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)

        # Clone x to avoid mutating underlying dataset memory
        x_poisoned = x.clone()

        if x_poisoned.ndim == 3:
            # (C, H, W)
            x_poisoned[:, :self.trigger_size, :self.trigger_size] = self.trigger_value
        elif x_poisoned.ndim == 2:
            # (H, W)
            x_poisoned[:self.trigger_size, :self.trigger_size] = self.trigger_value
        else:
            raise ValueError(f"Expected 2D or 3D image tensor, got shape {x_poisoned.shape}")

        if isinstance(y, torch.Tensor):
            y_poisoned = torch.tensor(self.target_label, dtype=y.dtype, device=y.device)
        else:
            y_poisoned = self.target_label

        return x_poisoned, y_poisoned


class BackdoorAttack(BaseAttack):
    """Data-plane backdoor watermark trigger injection attack.

    Wraps the client's local training dataset with a :class:`BackdoorDatasetWrapper`
    prior to local training, embedding a trigger watermark into a fraction of images
    paired with ``target_label``.

    Parameters
    ----------
    client_id:
        Unique identifier of the simulated malicious client.
    target_label:
        The target class to correlate with the watermark trigger. Default: 0.
    poison_rate:
        Fraction of local dataset items to watermark in ``[0.0, 1.0]``. Default: 0.3.
    trigger_size:
        Side length in pixels of the square trigger watermark. Default: 3.
    trigger_value:
        Pixel intensity assigned to the trigger pixels. Default: 1.0.
    seed:
        Optional PRNG seed for deterministic dataset poisoning.
    """

    def __init__(
        self,
        client_id: str,
        target_label: int = 0,
        poison_rate: float = 0.3,
        trigger_size: int = 3,
        trigger_value: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        if not isinstance(target_label, (int, np.integer)):
            raise TypeError(f"target_label must be int; got {type(target_label).__name__}")
        if not isinstance(poison_rate, (int, float)):
            raise TypeError(f"poison_rate must be float; got {type(poison_rate).__name__}")
        if not (0.0 <= float(poison_rate) <= 1.0):
            raise ValueError(f"poison_rate must be in [0.0, 1.0]; got {poison_rate}")
        if not isinstance(trigger_size, (int, np.integer)):
            raise TypeError(f"trigger_size must be int; got {type(trigger_size).__name__}")
        if trigger_size < 1:
            raise ValueError(f"trigger_size must be positive; got {trigger_size}")
        if not isinstance(trigger_value, (int, float)):
            raise TypeError(f"trigger_value must be numeric; got {type(trigger_value).__name__}")
        if seed is not None and not isinstance(seed, (int, np.integer)):
            raise TypeError(f"seed must be int or None; got {type(seed).__name__}")

        super().__init__(
            client_id=client_id,
            intensity=float(poison_rate),
            attack_type="backdoor",
        )

        self.target_label: int = int(target_label)
        self.poison_rate: float = float(poison_rate)
        self.trigger_size: int = int(trigger_size)
        self.trigger_value: float = float(trigger_value)
        self.seed: Optional[int] = int(seed) if seed is not None else None

    def apply_data_attack(self, dataset: Dataset) -> BackdoorDatasetWrapper:
        """Wrap the local dataset with backdoor watermark stamping logic.

        Parameters
        ----------
        dataset:
            PyTorch dataset instance to wrap.

        Returns
        -------
        BackdoorDatasetWrapper
            The wrapped dataset with trigger watermarks injected.
        """
        return BackdoorDatasetWrapper(
            dataset=dataset,
            target_label=self.target_label,
            poison_rate=self.poison_rate,
            trigger_size=self.trigger_size,
            trigger_value=self.trigger_value,
            seed=self.seed,
        )

    def apply_update_attack(
        self,
        delta: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Update-plane fallback: returns delta unmodified.

        Backdoor injection operates on the training dataset; the resulting model
        delta naturally embodies the backdoor trigger association.
        """
        return delta
