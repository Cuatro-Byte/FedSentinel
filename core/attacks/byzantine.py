"""
core/attacks/byzantine.py
=========================
Concrete Byzantine fault attack implementations for FedSentinel Phase 3.

All classes inherit from :class:`core.attacks.base_attack.BaseAttack` and
operate on the update-plane hook ``apply_update_attack``.

Attack catalogue
----------------
* :class:`GaussianNoiseAttack` — injects additive Gaussian noise into model updates.
* :class:`ZeroUpdateAttack`    — replaces model updates with all-zero tensors (freezing/sabotage).
* :class:`ExtremeValueAttack`   — floods model updates with extreme values (gradient explosion).

Contract reference: FedSentinel Team Engineering Contract v2.0 — Person 2
(Adversarial ML / Attack Engineer)

Dependency policy
-----------------
Strictly limited to NumPy and the Python standard library.
CRITICAL: ``is_malicious`` ground-truth must NEVER appear in any returned object.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from core.attacks.base_attack import BaseAttack


# ---------------------------------------------------------------------------
# GaussianNoiseAttack
# ---------------------------------------------------------------------------

class GaussianNoiseAttack(BaseAttack):
    """Byzantine Gaussian noise injection attack.

    Adds Gaussian noise drawn from Normal(mean, std) to each parameter array
    in the client's model delta. Simulates noisy sensor channels, hardware degradation,
    or deliberate sabotage aimed at degrading global model accuracy.

    Update rule
    -----------
    .. code-block:: text

        noise = rng.normal(loc=mean, scale=std, size=v.shape).astype(v.dtype)
        mutated[k] = v + noise

    Parameters
    ----------
    client_id:
        Unique identifier of the simulated Byzantine client.
    mean:
        Mean of the Gaussian distribution (default: 0.0).
    std:
        Standard deviation of the Gaussian distribution (default: 1.0).
        Reflected as ``self.intensity``.
    seed:
        Optional integer seed for PRNG reproducibility (Contract Section 49).
    """

    def __init__(
        self,
        client_id: str,
        mean: float = 0.0,
        std: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        if not isinstance(mean, (int, float)):
            raise TypeError(f"mean must be numeric; got {type(mean).__name__}")
        if not isinstance(std, (int, float)):
            raise TypeError(f"std must be numeric; got {type(std).__name__}")
        if std < 0.0:
            raise ValueError(f"std must be non-negative; got {std}")
        if seed is not None and not isinstance(seed, (int, np.integer)):
            raise TypeError(f"seed must be an integer or None; got {type(seed).__name__}")

        super().__init__(
            client_id=client_id,
            intensity=float(std),
            attack_type="byzantine_gaussian",
        )

        self.mean: float = float(mean)
        self.std: float = float(std)
        self.seed: Optional[int] = int(seed) if seed is not None else None

    def apply_update_attack(
        self,
        delta: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Return parameter deltas perturbed by additive Gaussian noise.

        Parameters
        ----------
        delta:
            Dictionary mapping parameter names to update arrays (w_local - w_global).

        Returns
        -------
        Dict[str, np.ndarray]
            A new dictionary where each array has added Gaussian noise.
            Preserves shapes, keys, and dtypes. Original delta is not mutated.
        """
        if not delta:
            return {}

        rng = np.random.default_rng(self.seed)
        mutated: Dict[str, np.ndarray] = {}

        for k, v in delta.items():
            if self.std == 0.0 and self.mean == 0.0:
                mutated[k] = v.copy()
            else:
                noise = rng.normal(loc=self.mean, scale=self.std, size=v.shape).astype(v.dtype)
                mutated[k] = (v + noise).astype(v.dtype)

        return mutated


# ---------------------------------------------------------------------------
# ZeroUpdateAttack
# ---------------------------------------------------------------------------

class ZeroUpdateAttack(BaseAttack):
    """Byzantine zero update attack (freezing / free-riding / denial of service).

    Submits an update of all zeros for all model parameters. This simulates
    a dead or sabotaging client that contributes no gradient information while
    diluting honest updates during federated aggregation.

    Update rule
    -----------
    .. code-block:: text

        mutated[k] = np.zeros_like(v)

    Parameters
    ----------
    client_id:
        Unique identifier of the simulated Byzantine client.
    """

    def __init__(self, client_id: str) -> None:
        super().__init__(
            client_id=client_id,
            intensity=0.0,
            attack_type="byzantine_zero",
        )

    def apply_update_attack(
        self,
        delta: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Return zero-filled parameter deltas.

        Parameters
        ----------
        delta:
            Dictionary mapping parameter names to update arrays.

        Returns
        -------
        Dict[str, np.ndarray]
            A new dictionary with identical keys, shapes, and dtypes,
            where all elements are 0.0. Original delta is not mutated.
        """
        return {k: np.zeros_like(v) for k, v in delta.items()}


# ---------------------------------------------------------------------------
# ExtremeValueAttack
# ---------------------------------------------------------------------------

class ExtremeValueAttack(BaseAttack):
    """Byzantine extreme value attack (gradient explosion / numerical overflow).

    Sets all parameter updates to an abnormally high or low extreme value.
    This simulates malicious numerical sabotage or corrupted hardware attempting
    to cause numerical overflow (NaN/Inf) and destroy the global model parameters.

    Update rule
    -----------
    .. code-block:: text

        mutated[k] = np.full_like(v, fill_value=extreme_val, dtype=v.dtype)

    Parameters
    ----------
    client_id:
        Unique identifier of the simulated Byzantine client.
    extreme_val:
        The extreme value to assign to all parameters (default: 1e6).
        Reflected as ``self.intensity``.
    """

    def __init__(self, client_id: str, extreme_val: float = 1e6) -> None:
        if not isinstance(extreme_val, (int, float)):
            raise TypeError(f"extreme_val must be numeric; got {type(extreme_val).__name__}")

        super().__init__(
            client_id=client_id,
            intensity=float(extreme_val),
            attack_type="byzantine_extreme",
        )

        self.extreme_val: float = float(extreme_val)

    def apply_update_attack(
        self,
        delta: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Return parameter deltas populated entirely with the extreme value.

        Parameters
        ----------
        delta:
            Dictionary mapping parameter names to update arrays.

        Returns
        -------
        Dict[str, np.ndarray]
            A new dictionary with identical keys, shapes, and dtypes,
            where all elements are set to ``self.extreme_val``.
            Original delta is not mutated.
        """
        return {
            k: np.full_like(v, fill_value=self.extreme_val, dtype=v.dtype)
            for k, v in delta.items()
        }
