"""
core/attacks/model_poisoning.py
================================
Concrete model-poisoning attack implementations for FedSentinel Phase 2.

All classes inherit from :class:`core.attacks.base_attack.BaseAttack` and
operate exclusively on the update-plane hook ``apply_update_attack``.

Attack catalogue
----------------
* :class:`SignFlipAttack`   — negates and scales the gradient direction.
* :class:`ScalingAttack`    — boosts the update magnitude to overwhelm FedAvg.
* :class:`AdaptiveStealthAttack` — inverts direction while clamping the L2 norm
  to stay below a detection threshold.

Contract reference: FedSentinel Team Engineering Contract v2.0 — Person 2
(Adversarial ML / Attack Engineer)

Dependency policy
-----------------
Only NumPy and the standard library are used here.  No PyTorch, no scikit-learn,
no SciPy.  This keeps the attack module fully portable and independently testable.

CRITICAL: ``is_malicious`` ground-truth must NEVER appear in any returned object.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from core.attacks.base_attack import BaseAttack


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _global_l2_norm(delta: Dict[str, np.ndarray]) -> float:
    """Compute the global L2 norm across all concatenated parameter arrays.

    Parameters
    ----------
    delta:
        Dictionary mapping parameter names to NumPy arrays.

    Returns
    -------
    float
        The scalar L2 norm ``‖vec(delta)‖₂``.  Returns ``0.0`` for an
        empty dictionary or an all-zero delta.
    """
    if not delta:
        return 0.0
    flat = np.concatenate([v.ravel().astype(np.float64) for v in delta.values()])
    return float(np.sqrt(np.dot(flat, flat)))


# ---------------------------------------------------------------------------
# SignFlipAttack
# ---------------------------------------------------------------------------

class SignFlipAttack(BaseAttack):
    """Byzantine sign-flip / gradient inversion attack.

    Negates the direction of every parameter delta and scales by
    ``self.intensity``.  This is equivalent to a server that pushes the
    global model in the *opposite* direction to honest clients, degrading
    convergence.

    Update rule
    -----------
    .. code-block:: text

        mutated[k] = -1.0 * delta[k] * intensity

    Parameters
    ----------
    client_id:
        Unique identifier of the simulated malicious client.
    intensity:
        Scalar multiplier applied after sign inversion.  ``intensity=1.0``
        (default) is a pure sign flip; values ``> 1.0`` additionally
        amplify the poisoned update.

    Notes
    -----
    * Preserves original array shapes, NumPy dtypes (e.g. ``float32``), and
      dictionary key order.
    * Does NOT attach ``is_malicious`` or any ground-truth label to the output.
    """

    def __init__(self, client_id: str, intensity: float = 1.0) -> None:
        super().__init__(
            client_id=client_id,
            intensity=intensity,
            attack_type="sign_flip",
        )

    def apply_update_attack(
        self,
        delta: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Return sign-flipped (and intensity-scaled) parameter deltas.

        Parameters
        ----------
        delta:
            Dictionary mapping parameter names to their update arrays
            (``w_local − w_global``).

        Returns
        -------
        Dict[str, numpy.ndarray]
            A new dictionary with the same keys and shapes as ``delta``,
            where every array has been negated and multiplied by
            ``self.intensity``.  Original ``delta`` arrays are not mutated.
        """
        return {
            k: (-1.0 * v * self.intensity).astype(v.dtype)
            for k, v in delta.items()
        }


# ---------------------------------------------------------------------------
# ScalingAttack
# ---------------------------------------------------------------------------

class ScalingAttack(BaseAttack):
    """Update-boosting / gradient scaling attack.

    Multiplies every parameter delta by ``scale_factor`` to overcome the
    dilution effect of FedAvg.  When ``n`` clients contribute equally but
    one submits a delta scaled by ``n * scale_factor``, the poisoned update
    dominates the aggregation.

    Update rule
    -----------
    .. code-block:: text

        mutated[k] = delta[k] * scale_factor

    Parameters
    ----------
    client_id:
        Unique identifier of the simulated malicious client.
    scale_factor:
        Multiplicative amplification applied to every parameter delta.
        Default is ``10.0``, meaning the poisoned update has 10× the L2
        norm of the honest update, making it dominant in FedAvg.

    Notes
    -----
    * ``scale_factor`` is stored as both ``self.scale_factor`` (semantic
      attribute) and ``self.intensity`` (base-class attribute) so that
      :meth:`to_dict` remains consistent with the base contract.
    * Preserves original array shapes and NumPy dtypes.
    * Does NOT attach ``is_malicious`` to the output.
    """

    def __init__(self, client_id: str, scale_factor: float = 10.0) -> None:
        super().__init__(
            client_id=client_id,
            intensity=scale_factor,
            attack_type="scaling",
        )
        self.scale_factor: float = float(scale_factor)

    def apply_update_attack(
        self,
        delta: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Return amplified parameter deltas.

        Parameters
        ----------
        delta:
            Dictionary mapping parameter names to their update arrays.

        Returns
        -------
        Dict[str, numpy.ndarray]
            A new dictionary with the same keys and shapes as ``delta``,
            where every array has been multiplied by ``self.scale_factor``.
            Original ``delta`` arrays are not mutated.
        """
        return {
            k: (v * self.scale_factor).astype(v.dtype)
            for k, v in delta.items()
        }


# ---------------------------------------------------------------------------
# AdaptiveStealthAttack
# ---------------------------------------------------------------------------

class AdaptiveStealthAttack(BaseAttack):
    """Adaptive stealth attack: inverts direction while clamping the L2 norm.

    This attack attempts to mislead the global model (by inverting gradient
    signs) while simultaneously evading norm-based anomaly detectors by
    ensuring the poisoned update's global L2 norm does not exceed a multiple
    of the original honest update's norm.

    Algorithm
    ---------
    1. Compute the **baseline norm** ``‖delta‖₂`` across all parameters.
    2. Invert direction:
       ``poisoned[k] = -1.0 * delta[k] * intensity``.
    3. Compute the L2 norm of ``poisoned``.
    4. If ``poisoned_norm > baseline_norm * max_norm_ratio``, rescale:
       ``poisoned[k] *= (baseline_norm * max_norm_ratio) / poisoned_norm``.
    5. If ``baseline_norm == 0.0``, skip step 4 (no division by zero).

    Parameters
    ----------
    client_id:
        Unique identifier of the simulated malicious client.
    max_norm_ratio:
        The maximum allowed ratio of the poisoned update's L2 norm to the
        baseline (honest) update's L2 norm.  Values slightly above ``1.0``
        (default ``1.1``) produce an attack that looks statistically similar
        to an honest update while still inverting gradient direction.
    intensity:
        Internal amplification applied before norm clamping.  The final
        norm is always clamped to ``baseline_norm * max_norm_ratio``, so
        ``intensity`` only affects the *direction* (sign) when
        ``max_norm_ratio`` is close to 1.

    Notes
    -----
    * Preserves original array keys, shapes, and NumPy dtypes.
    * The norm clamping uses ``float64`` arithmetic internally and casts
      back to the original dtype of each array.
    * Does NOT attach ``is_malicious`` or any ground-truth label.
    """

    def __init__(
        self,
        client_id: str,
        max_norm_ratio: float = 1.1,
        intensity: float = 1.0,
    ) -> None:
        super().__init__(
            client_id=client_id,
            intensity=intensity,
            attack_type="adaptive_stealth",
        )
        if not isinstance(max_norm_ratio, (int, float)):
            raise TypeError(
                f"max_norm_ratio must be numeric; got {type(max_norm_ratio).__name__}"
            )
        self.max_norm_ratio: float = float(max_norm_ratio)

    def apply_update_attack(
        self,
        delta: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Return norm-clamped, direction-inverted parameter deltas.

        Parameters
        ----------
        delta:
            Dictionary mapping parameter names to their update arrays
            (``w_local − w_global``).

        Returns
        -------
        Dict[str, numpy.ndarray]
            A new dictionary with the same keys and shapes as ``delta``.
            Each array has its direction inverted (and intensity-scaled).
            If the resulting global L2 norm would exceed
            ``baseline_norm * max_norm_ratio``, all arrays are rescaled so
            the final norm exactly equals that ceiling.  Original ``delta``
            arrays are not mutated.
        """
        if not delta:
            return {}

        # Step 1 — baseline norm of the honest delta.
        baseline_norm: float = _global_l2_norm(delta)

        # Step 2 — invert and scale (working in float64 for numerical stability,
        #           original dtype is restored at the end).
        poisoned: Dict[str, np.ndarray] = {
            k: (-1.0 * v.astype(np.float64) * self.intensity)
            for k, v in delta.items()
        }

        # Step 3 — norm of the poisoned update.
        poisoned_norm: float = _global_l2_norm(poisoned)

        # Step 4 — norm clamping (skip if baseline is zero to avoid ÷0).
        if baseline_norm > 0.0:
            norm_ceiling: float = baseline_norm * self.max_norm_ratio
            if poisoned_norm > norm_ceiling:
                rescale: float = norm_ceiling / poisoned_norm
                poisoned = {k: v * rescale for k, v in poisoned.items()}

        # Step 5 — restore original dtypes.
        return {k: poisoned[k].astype(delta[k].dtype) for k in delta}
