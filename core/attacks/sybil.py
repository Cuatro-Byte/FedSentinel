"""
core/attacks/sybil.py
=====================
Concrete Sybil attack implementation for FedSentinel.

A Sybil attack coordinates multiple malicious identities to submit highly
correlated parameter updates that share a targeted adversarial direction,
while keeping updates structurally valid and subtly differentiated.

Contract reference: FedSentinel Team Engineering Contract v2.0 — Person 2
(Adversarial ML / Attack Engineer)
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import numpy as np

from core.attacks.base_attack import BaseAttack


class SybilAttack(BaseAttack):
    """Sybil attack coordinating update directions across multiple clients.

    Subclasses :class:`core.attacks.base_attack.BaseAttack` and operates on the
    update-plane hook ``apply_update_attack``. Data-plane hook ``apply_data_attack``
    is a no-op identity pass-through.

    Update rule:
    Each Sybil client generates updates oriented along a deterministic shared
    adversarial direction derived from ``shared_seed``, scaled by ``self.scale``
    relative to the client's delta norm (or an absolute scale if zero), plus
    a tiny independent Gaussian micro-noise (jitter) so updates are not
    byte-for-byte identical.

    Parameters
    ----------
    client_id:
        Identifier of the owning client.
    intensity:
        Attack intensity multiplier. Defaults to 1.0.
    shared_seed:
        Seed shared across all colluding Sybil clients to generate the common
        adversarial perturbation direction. Defaults to 42.
    scale:
        Scale multiplier applied to the shared direction. Defaults to 1.0.
    noise_std:
        Standard deviation of per-client micro-noise. Defaults to 1e-4.
    target_bias:
        Directional bias added to the generated direction prior to normalization.
        Defaults to -0.5.
    is_active:
        Boolean flag to enable/disable attack mutations. Defaults to True.
    """

    def __init__(
        self,
        client_id: str = "sybil_client",
        intensity: float = 1.0,
        shared_seed: int = 42,
        scale: float = 1.0,
        noise_std: float = 1e-4,
        target_bias: float = -0.5,
        is_active: bool = True,
    ) -> None:
        super().__init__(
            client_id=client_id,
            intensity=intensity,
            attack_type="sybil",
        )
        self.shared_seed = int(shared_seed)
        self.scale = float(scale)
        self.noise_std = float(noise_std)
        self.target_bias = float(target_bias)
        self.is_active = bool(is_active)
        self._shared_direction: Optional[Dict[str, np.ndarray]] = None

    def apply_data_attack(self, dataset: Any) -> Any:
        """Data-plane attack hook (no-op for update-level Sybil attack)."""
        return dataset

    def apply_update_attack(
        self, delta: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Mutate delta to align with the coordinated Sybil direction.

        Parameters
        ----------
        delta:
            Dictionary mapping parameter names to update arrays (w_local - w_global).

        Returns
        -------
        Dict[str, np.ndarray]
            Mutated parameter delta matching all input keys, shapes, and dtypes.
        """
        if not self.is_active or not delta:
            return delta

        # Lazily construct deterministic shared direction across all parameters
        if self._shared_direction is None or set(self._shared_direction.keys()) != set(delta.keys()):
            self._shared_direction = {}
            for idx, (k, v) in enumerate(sorted(delta.items())):
                # Derive deterministic seed per parameter key from shared_seed
                key_hash = sum((i + 1) * ord(c) for i, c in enumerate(k))
                tensor_seed = int((self.shared_seed * 10007 + idx * 37 + key_hash) % (2**31 - 1))
                rng = np.random.RandomState(tensor_seed)
                raw = rng.standard_normal(size=v.shape).astype(np.float64)
                biased = raw + self.target_bias
                norm = float(np.linalg.norm(biased))
                if norm > 1e-12:
                    self._shared_direction[k] = biased / norm
                else:
                    self._shared_direction[k] = np.ones(v.shape, dtype=np.float64) / np.sqrt(max(v.size, 1))

        mutated: Dict[str, np.ndarray] = {}
        for k, v in delta.items():
            dtype = v.dtype
            client_norm = float(np.linalg.norm(v))
            effective_norm = max(client_norm, 1.0) * self.scale * self.intensity
            shared_dir = self._shared_direction[k]

            # Add subtle stochastic micro-noise so updates are non-identical byte-for-byte
            jitter = np.random.randn(*v.shape).astype(np.float64) * self.noise_std
            combined = (shared_dir * effective_norm) + jitter

            # Ensure all values are finite and preserve input dtype and shape
            out_arr = np.nan_to_num(combined, nan=0.0, posinf=1.0, neginf=-1.0).astype(dtype)
            mutated[k] = out_arr

        return mutated
