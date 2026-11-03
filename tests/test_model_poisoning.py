"""
tests/test_model_poisoning.py
==============================
Unit tests for ``core.attacks.model_poisoning``:
    * :class:`~core.attacks.model_poisoning.SignFlipAttack`
    * :class:`~core.attacks.model_poisoning.ScalingAttack`
    * :class:`~core.attacks.model_poisoning.AdaptiveStealthAttack`

Run with::

    pytest tests/test_model_poisoning.py -v

Test sections
-------------
A  Package import & class hierarchy
B  SignFlipAttack — sign inversion, intensity, shapes/dtypes, edge cases
C  ScalingAttack  — scalar multiplication, norm scaling, shapes/dtypes
D  AdaptiveStealthAttack — direction inversion, norm clamping, edge cases
E  Cross-class contract invariants (no is_malicious, dtype/shape preservation)

Contract reference: FedSentinel Team Engineering Contract v2.0 — Person 2
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np
import pytest

from core.attacks.base_attack import BaseAttack
from core.attacks.model_poisoning import (
    AdaptiveStealthAttack,
    ScalingAttack,
    SignFlipAttack,
    _global_l2_norm,
)
from core.attacks import (
    SignFlipAttack as SignFlipFromInit,
    ScalingAttack as ScalingFromInit,
    AdaptiveStealthAttack as AdaptiveFromInit,
)


# ---------------------------------------------------------------------------
# Shared fixtures & helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(seed=2024)


def _make_delta(
    dtype: np.dtype = np.float32,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Return a realistic multi-layer model delta dictionary."""
    rng = np.random.default_rng(seed=seed)
    return {
        "conv1.weight": rng.standard_normal((32, 3, 3, 3)).astype(dtype),   # 4-D
        "conv1.bias":   rng.standard_normal((32,)).astype(dtype),            # 1-D
        "fc1.weight":   rng.standard_normal((128, 288)).astype(dtype),       # 2-D
        "fc1.bias":     rng.standard_normal((128,)).astype(dtype),           # 1-D
        "fc2.weight":   rng.standard_normal((10, 128)).astype(dtype),        # 2-D
        "fc2.bias":     rng.standard_normal((10,)).astype(dtype),            # 1-D
    }


def _global_l2(delta: Dict[str, np.ndarray]) -> float:
    """Inline reference implementation of global L2 norm (for test assertions)."""
    flat = np.concatenate([v.ravel().astype(np.float64) for v in delta.values()])
    return float(np.linalg.norm(flat))


def _snapshot(delta: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Deep copy of delta for mutation-safety checks."""
    return {k: v.copy() for k, v in delta.items()}


# ---------------------------------------------------------------------------
# Section A — Package imports & class hierarchy
# ---------------------------------------------------------------------------

class TestImportsAndHierarchy:
    """All three classes must be importable from both module and package."""

    def test_signflip_from_module(self) -> None:
        assert SignFlipAttack is not None

    def test_scaling_from_module(self) -> None:
        assert ScalingAttack is not None

    def test_adaptive_from_module(self) -> None:
        assert AdaptiveStealthAttack is not None

    def test_signflip_from_package_init(self) -> None:
        assert SignFlipFromInit is SignFlipAttack

    def test_scaling_from_package_init(self) -> None:
        assert ScalingFromInit is ScalingAttack

    def test_adaptive_from_package_init(self) -> None:
        assert AdaptiveFromInit is AdaptiveStealthAttack

    def test_signflip_is_baseattack(self) -> None:
        assert issubclass(SignFlipAttack, BaseAttack)

    def test_scaling_is_baseattack(self) -> None:
        assert issubclass(ScalingAttack, BaseAttack)

    def test_adaptive_is_baseattack(self) -> None:
        assert issubclass(AdaptiveStealthAttack, BaseAttack)


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

class TestGlobalL2Norm:
    """Tests for the internal _global_l2_norm helper."""

    def test_empty_dict_returns_zero(self) -> None:
        assert _global_l2_norm({}) == 0.0

    def test_matches_reference(self) -> None:
        delta = _make_delta()
        assert _global_l2_norm(delta) == pytest.approx(_global_l2(delta), rel=1e-5)

    def test_all_zeros_returns_zero(self) -> None:
        delta = {"w": np.zeros((10, 10), dtype=np.float32)}
        assert _global_l2_norm(delta) == pytest.approx(0.0)

    def test_single_element(self) -> None:
        delta = {"w": np.array([3.0], dtype=np.float32)}
        assert _global_l2_norm(delta) == pytest.approx(3.0)

    def test_multidimensional_matches_numpy(self) -> None:
        delta = _make_delta(dtype=np.float64, seed=99)
        expected = float(np.linalg.norm(
            np.concatenate([v.ravel() for v in delta.values()])
        ))
        assert _global_l2_norm(delta) == pytest.approx(expected, rel=1e-8)


# ---------------------------------------------------------------------------
# Section B — SignFlipAttack
# ---------------------------------------------------------------------------

class TestSignFlipAttackConstructor:
    """Constructor attribute storage."""

    def test_attack_type(self) -> None:
        a = SignFlipAttack(client_id="c0")
        assert a.attack_type == "sign_flip"

    def test_default_intensity(self) -> None:
        a = SignFlipAttack(client_id="c0")
        assert a.intensity == pytest.approx(1.0)

    def test_custom_intensity(self) -> None:
        a = SignFlipAttack(client_id="c0", intensity=3.5)
        assert a.intensity == pytest.approx(3.5)

    def test_client_id_stored(self) -> None:
        a = SignFlipAttack(client_id="mal_client_1")
        assert a.client_id == "mal_client_1"

    def test_to_dict_attack_type(self) -> None:
        a = SignFlipAttack(client_id="c0", intensity=2.0)
        d = a.to_dict()
        assert d["attack_type"] == "sign_flip"
        assert d["intensity"] == pytest.approx(2.0)


class TestSignFlipAttackBehavior:
    """Core sign-inversion behaviour."""

    def test_sign_inverted_1d(self) -> None:
        a = SignFlipAttack(client_id="c0")
        delta = {"bias": np.array([1.0, -2.0, 3.0], dtype=np.float32)}
        result = a.apply_update_attack(delta)
        np.testing.assert_allclose(result["bias"], [-1.0, 2.0, -3.0], rtol=1e-6)

    def test_sign_inverted_2d(self) -> None:
        a = SignFlipAttack(client_id="c0")
        w = np.array([[1.0, -1.0], [0.5, -0.5]], dtype=np.float32)
        delta = {"w": w}
        result = a.apply_update_attack(delta)
        np.testing.assert_allclose(result["w"], -w, rtol=1e-6)

    def test_sign_inverted_4d(self) -> None:
        a = SignFlipAttack(client_id="c0")
        delta = _make_delta()
        result = a.apply_update_attack(delta)
        np.testing.assert_allclose(
            result["conv1.weight"], -delta["conv1.weight"], rtol=1e-5
        )

    def test_intensity_multiplier(self) -> None:
        """intensity=2.0 should yield -2.0 * v."""
        a = SignFlipAttack(client_id="c0", intensity=2.0)
        delta = _make_delta()
        snapshot = _snapshot(delta)
        result = a.apply_update_attack(delta)
        for k in snapshot:
            np.testing.assert_allclose(result[k], -2.0 * snapshot[k], rtol=1e-5)

    def test_intensity_zero_yields_zeros(self) -> None:
        a = SignFlipAttack(client_id="c0", intensity=0.0)
        delta = _make_delta()
        result = a.apply_update_attack(delta)
        for k in result:
            np.testing.assert_array_equal(result[k], np.zeros_like(delta[k]))

    def test_double_flip_recovers_original(self) -> None:
        """Applying sign flip twice with intensity=1.0 restores the original delta."""
        a = SignFlipAttack(client_id="c0", intensity=1.0)
        delta = _make_delta()
        snapshot = _snapshot(delta)
        double_flipped = a.apply_update_attack(a.apply_update_attack(delta))
        for k in snapshot:
            np.testing.assert_allclose(double_flipped[k], snapshot[k], rtol=1e-5)

    def test_zero_array_no_nan_inf(self) -> None:
        """Zero-valued arrays must not produce NaN or Inf."""
        a = SignFlipAttack(client_id="c0")
        delta = {"w": np.zeros((5, 5), dtype=np.float32)}
        result = a.apply_update_attack(delta)
        assert np.all(np.isfinite(result["w"]))
        np.testing.assert_array_equal(result["w"], np.zeros((5, 5), dtype=np.float32))

    def test_all_keys_preserved(self) -> None:
        delta = _make_delta()
        result = SignFlipAttack(client_id="c0").apply_update_attack(delta)
        assert set(result.keys()) == set(delta.keys())

    def test_returns_new_dict(self) -> None:
        """Must not return the same dict object (avoids silent mutation)."""
        a = SignFlipAttack(client_id="c0")
        delta = _make_delta()
        result = a.apply_update_attack(delta)
        assert result is not delta

    def test_does_not_mutate_original(self) -> None:
        a = SignFlipAttack(client_id="c0")
        delta = _make_delta()
        snapshot = _snapshot(delta)
        _ = a.apply_update_attack(delta)
        for k in snapshot:
            np.testing.assert_array_equal(delta[k], snapshot[k])

    def test_empty_delta_returns_empty(self) -> None:
        a = SignFlipAttack(client_id="c0")
        result = a.apply_update_attack({})
        assert result == {}


class TestSignFlipAttackDtypeShape:
    """Dtype and shape preservation."""

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_dtype_preserved(self, dtype) -> None:
        a = SignFlipAttack(client_id="c0")
        delta = _make_delta(dtype=dtype)
        result = a.apply_update_attack(delta)
        for k in delta:
            assert result[k].dtype == dtype, (
                f"dtype mismatch for '{k}': expected {dtype}, got {result[k].dtype}"
            )

    def test_shape_preserved_all_layers(self) -> None:
        a = SignFlipAttack(client_id="c0")
        delta = _make_delta()
        result = a.apply_update_attack(delta)
        for k in delta:
            assert result[k].shape == delta[k].shape


# ---------------------------------------------------------------------------
# Section C — ScalingAttack
# ---------------------------------------------------------------------------

class TestScalingAttackConstructor:
    """Constructor attribute storage."""

    def test_attack_type(self) -> None:
        a = ScalingAttack(client_id="c0")
        assert a.attack_type == "scaling"

    def test_default_scale_factor(self) -> None:
        a = ScalingAttack(client_id="c0")
        assert a.scale_factor == pytest.approx(10.0)

    def test_intensity_equals_scale_factor(self) -> None:
        """intensity must mirror scale_factor for to_dict() consistency."""
        a = ScalingAttack(client_id="c0", scale_factor=5.0)
        assert a.intensity == pytest.approx(5.0)
        assert a.scale_factor == pytest.approx(5.0)

    def test_custom_scale_factor(self) -> None:
        a = ScalingAttack(client_id="c0", scale_factor=20.0)
        assert a.scale_factor == pytest.approx(20.0)

    def test_to_dict_intensity(self) -> None:
        a = ScalingAttack(client_id="c1", scale_factor=7.5)
        d = a.to_dict()
        assert d["intensity"] == pytest.approx(7.5)
        assert d["attack_type"] == "scaling"


class TestScalingAttackBehavior:
    """Core multiplicative behaviour."""

    def test_exact_multiplication_1d(self) -> None:
        a = ScalingAttack(client_id="c0", scale_factor=3.0)
        delta = {"b": np.array([1.0, 2.0, -3.0], dtype=np.float32)}
        result = a.apply_update_attack(delta)
        np.testing.assert_allclose(result["b"], [3.0, 6.0, -9.0], rtol=1e-6)

    def test_exact_multiplication_2d(self) -> None:
        a = ScalingAttack(client_id="c0", scale_factor=5.0)
        w = np.array([[1.0, 0.5], [-1.0, -0.5]], dtype=np.float32)
        result = a.apply_update_attack({"w": w})
        np.testing.assert_allclose(result["w"], 5.0 * w, rtol=1e-6)

    def test_scale_factor_one_is_identity(self) -> None:
        a = ScalingAttack(client_id="c0", scale_factor=1.0)
        delta = _make_delta()
        snapshot = _snapshot(delta)
        result = a.apply_update_attack(delta)
        for k in snapshot:
            np.testing.assert_allclose(result[k], snapshot[k], rtol=1e-6)

    def test_scale_factor_zero_yields_zeros(self) -> None:
        a = ScalingAttack(client_id="c0", scale_factor=0.0)
        delta = _make_delta()
        result = a.apply_update_attack(delta)
        for k in result:
            np.testing.assert_array_equal(result[k], np.zeros_like(delta[k]))

    def test_global_l2_norm_scales_by_factor(self) -> None:
        """Global L2 norm must scale exactly by scale_factor."""
        scale = 7.0
        a = ScalingAttack(client_id="c0", scale_factor=scale)
        delta = _make_delta()
        baseline_norm = _global_l2(delta)
        result = a.apply_update_attack(delta)
        result_norm = _global_l2(result)
        assert result_norm == pytest.approx(baseline_norm * scale, rel=1e-5)

    def test_all_keys_preserved(self) -> None:
        delta = _make_delta()
        result = ScalingAttack(client_id="c0").apply_update_attack(delta)
        assert set(result.keys()) == set(delta.keys())

    def test_returns_new_dict(self) -> None:
        a = ScalingAttack(client_id="c0")
        delta = _make_delta()
        result = a.apply_update_attack(delta)
        assert result is not delta

    def test_does_not_mutate_original(self) -> None:
        a = ScalingAttack(client_id="c0", scale_factor=10.0)
        delta = _make_delta()
        snapshot = _snapshot(delta)
        _ = a.apply_update_attack(delta)
        for k in snapshot:
            np.testing.assert_array_equal(delta[k], snapshot[k])

    def test_empty_delta_returns_empty(self) -> None:
        result = ScalingAttack(client_id="c0").apply_update_attack({})
        assert result == {}

    def test_negative_scale_factor(self) -> None:
        """Negative scale factor should negate and amplify."""
        a = ScalingAttack(client_id="c0", scale_factor=-2.0)
        delta = {"w": np.array([1.0, -1.0], dtype=np.float32)}
        result = a.apply_update_attack(delta)
        np.testing.assert_allclose(result["w"], [-2.0, 2.0], rtol=1e-6)


class TestScalingAttackDtypeShape:
    """Dtype and shape preservation."""

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_dtype_preserved(self, dtype) -> None:
        a = ScalingAttack(client_id="c0", scale_factor=3.0)
        delta = _make_delta(dtype=dtype)
        result = a.apply_update_attack(delta)
        for k in delta:
            assert result[k].dtype == dtype

    def test_shape_preserved_all_layers(self) -> None:
        a = ScalingAttack(client_id="c0")
        delta = _make_delta()
        result = a.apply_update_attack(delta)
        for k in delta:
            assert result[k].shape == delta[k].shape

    def test_4d_conv_weight_shape(self) -> None:
        a = ScalingAttack(client_id="c0")
        w4d = RNG.standard_normal((64, 3, 5, 5)).astype(np.float32)
        result = a.apply_update_attack({"conv.weight": w4d})
        assert result["conv.weight"].shape == (64, 3, 5, 5)


# ---------------------------------------------------------------------------
# Section D — AdaptiveStealthAttack
# ---------------------------------------------------------------------------

class TestAdaptiveStealthAttackConstructor:
    """Constructor attribute storage and validation."""

    def test_attack_type(self) -> None:
        a = AdaptiveStealthAttack(client_id="c0")
        assert a.attack_type == "adaptive_stealth"

    def test_default_max_norm_ratio(self) -> None:
        a = AdaptiveStealthAttack(client_id="c0")
        assert a.max_norm_ratio == pytest.approx(1.1)

    def test_custom_max_norm_ratio(self) -> None:
        a = AdaptiveStealthAttack(client_id="c0", max_norm_ratio=0.9)
        assert a.max_norm_ratio == pytest.approx(0.9)

    def test_default_intensity(self) -> None:
        a = AdaptiveStealthAttack(client_id="c0")
        assert a.intensity == pytest.approx(1.0)

    def test_custom_intensity(self) -> None:
        a = AdaptiveStealthAttack(client_id="c0", intensity=2.5)
        assert a.intensity == pytest.approx(2.5)

    def test_invalid_max_norm_ratio_raises(self) -> None:
        with pytest.raises(TypeError, match="max_norm_ratio"):
            AdaptiveStealthAttack(client_id="c0", max_norm_ratio="high")  # type: ignore

    def test_to_dict_no_extra_fields(self) -> None:
        a = AdaptiveStealthAttack(client_id="c0")
        d = a.to_dict()
        assert set(d.keys()) == {"client_id", "attack_type", "intensity"}


class TestAdaptiveStealthAttackDirectionInversion:
    """Direction inversion (gradient sign flip)."""

    def test_direction_inverted_1d(self) -> None:
        a = AdaptiveStealthAttack(client_id="c0", max_norm_ratio=100.0)
        delta = {"b": np.array([1.0, -2.0, 3.0], dtype=np.float32)}
        result = a.apply_update_attack(delta)
        # With a very large max_norm_ratio the clamp never activates.
        assert result["b"][0] < 0, "Positive value should become negative after inversion"
        assert result["b"][1] > 0, "Negative value should become positive after inversion"
        assert result["b"][2] < 0, "Positive value should become negative after inversion"

    def test_inverted_direction_dot_product_negative(self) -> None:
        """Dot product of original and mutated updates must be ≤ 0 (opposite direction)."""
        a = AdaptiveStealthAttack(client_id="c0", max_norm_ratio=100.0)
        delta = _make_delta()
        result = a.apply_update_attack(delta)
        for k in delta:
            dot = float(np.dot(delta[k].ravel(), result[k].ravel()))
            assert dot <= 0.0, f"Expected opposing direction for '{k}', got dot={dot}"


class TestAdaptiveStealthAttackNormClamping:
    """Norm clamping: mutated norm ≤ baseline_norm * max_norm_ratio."""

    def test_norm_clamped_when_intensity_large(self) -> None:
        """High intensity should trigger clamping so norm stays at ceiling."""
        ratio = 1.1
        a = AdaptiveStealthAttack(client_id="c0", max_norm_ratio=ratio, intensity=50.0)
        delta = _make_delta()
        baseline_norm = _global_l2(delta)
        result = a.apply_update_attack(delta)
        mutated_norm = _global_l2(result)
        ceiling = baseline_norm * ratio
        assert mutated_norm <= ceiling + 1e-4, (
            f"mutated_norm={mutated_norm:.6f} exceeds ceiling={ceiling:.6f}"
        )

    def test_norm_at_ceiling_when_clamped(self) -> None:
        """When clamping triggers, mutated norm must equal ceiling precisely."""
        ratio = 1.05
        a = AdaptiveStealthAttack(client_id="c0", max_norm_ratio=ratio, intensity=100.0)
        delta = _make_delta()
        baseline_norm = _global_l2(delta)
        result = a.apply_update_attack(delta)
        mutated_norm = _global_l2(result)
        ceiling = baseline_norm * ratio
        assert mutated_norm == pytest.approx(ceiling, rel=1e-4), (
            f"Expected norm≈{ceiling:.6f}, got {mutated_norm:.6f}"
        )

    def test_no_clamp_when_ratio_large(self) -> None:
        """With a huge max_norm_ratio, no clamping should occur."""
        ratio = 1000.0
        intensity = 1.0
        a = AdaptiveStealthAttack(client_id="c0", max_norm_ratio=ratio, intensity=intensity)
        delta = _make_delta()
        baseline_norm = _global_l2(delta)
        result = a.apply_update_attack(delta)
        mutated_norm = _global_l2(result)
        # With intensity=1 and ratio=1000, unclamped norm ≈ baseline_norm.
        assert mutated_norm < baseline_norm * ratio

    def test_clamp_with_ratio_below_one(self) -> None:
        """max_norm_ratio < 1 forces the update to be *smaller* than the original."""
        ratio = 0.5
        a = AdaptiveStealthAttack(client_id="c0", max_norm_ratio=ratio, intensity=10.0)
        delta = _make_delta()
        baseline_norm = _global_l2(delta)
        result = a.apply_update_attack(delta)
        mutated_norm = _global_l2(result)
        assert mutated_norm <= baseline_norm * ratio + 1e-4


class TestAdaptiveStealthAttackEdgeCases:
    """Edge cases: zero norm, empty dict, single-element arrays."""

    def test_zero_baseline_norm_no_div_by_zero(self) -> None:
        """If all delta values are zero, baseline_norm=0 — must not raise ZeroDivisionError."""
        a = AdaptiveStealthAttack(client_id="c0", max_norm_ratio=1.1, intensity=5.0)
        delta = {
            "w": np.zeros((4, 4), dtype=np.float32),
            "b": np.zeros((4,), dtype=np.float32),
        }
        result = a.apply_update_attack(delta)
        for k in result:
            assert np.all(np.isfinite(result[k])), f"Non-finite values in '{k}'"

    def test_zero_baseline_norm_returns_poisoned_directly(self) -> None:
        """When baseline_norm==0, clamping is skipped; poisoned = -v * intensity."""
        intensity = 3.0
        a = AdaptiveStealthAttack(client_id="c0", max_norm_ratio=1.1, intensity=intensity)
        delta = {"w": np.zeros((3,), dtype=np.float32)}
        result = a.apply_update_attack(delta)
        np.testing.assert_array_equal(result["w"], np.zeros((3,), dtype=np.float32))

    def test_empty_delta_returns_empty(self) -> None:
        a = AdaptiveStealthAttack(client_id="c0")
        assert a.apply_update_attack({}) == {}

    def test_single_element_array(self) -> None:
        ratio = 1.2
        a = AdaptiveStealthAttack(client_id="c0", max_norm_ratio=ratio, intensity=10.0)
        delta = {"w": np.array([2.0], dtype=np.float32)}
        result = a.apply_update_attack(delta)
        baseline_norm = abs(float(delta["w"][0]))
        mutated_norm = abs(float(result["w"][0]))
        assert mutated_norm <= baseline_norm * ratio + 1e-5

    def test_single_element_direction_inverted(self) -> None:
        a = AdaptiveStealthAttack(client_id="c0", max_norm_ratio=100.0, intensity=1.0)
        delta = {"w": np.array([5.0], dtype=np.float32)}
        result = a.apply_update_attack(delta)
        assert float(result["w"][0]) < 0.0

    def test_does_not_mutate_original_delta(self) -> None:
        a = AdaptiveStealthAttack(client_id="c0", max_norm_ratio=1.1, intensity=3.0)
        delta = _make_delta()
        snapshot = _snapshot(delta)
        _ = a.apply_update_attack(delta)
        for k in snapshot:
            np.testing.assert_array_equal(delta[k], snapshot[k])

    def test_returns_new_dict(self) -> None:
        a = AdaptiveStealthAttack(client_id="c0")
        delta = _make_delta()
        result = a.apply_update_attack(delta)
        assert result is not delta

    def test_all_keys_preserved(self) -> None:
        delta = _make_delta()
        result = AdaptiveStealthAttack(client_id="c0").apply_update_attack(delta)
        assert set(result.keys()) == set(delta.keys())


class TestAdaptiveStealthAttackDtypeShape:
    """Dtype and shape preservation."""

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_dtype_preserved(self, dtype) -> None:
        a = AdaptiveStealthAttack(client_id="c0")
        delta = _make_delta(dtype=dtype)
        result = a.apply_update_attack(delta)
        for k in delta:
            assert result[k].dtype == dtype, (
                f"dtype mismatch for '{k}': expected {dtype}, got {result[k].dtype}"
            )

    def test_shape_preserved_all_layers(self) -> None:
        a = AdaptiveStealthAttack(client_id="c0")
        delta = _make_delta()
        result = a.apply_update_attack(delta)
        for k in delta:
            assert result[k].shape == delta[k].shape

    def test_4d_shape_preserved(self) -> None:
        a = AdaptiveStealthAttack(client_id="c0")
        w4d = RNG.standard_normal((16, 8, 3, 3)).astype(np.float32)
        result = a.apply_update_attack({"w": w4d})
        assert result["w"].shape == (16, 8, 3, 3)


# ---------------------------------------------------------------------------
# Section E — Cross-class contract invariants
# ---------------------------------------------------------------------------

class TestContractInvariants:
    """Explicit enforcement of Contract v2.0 rules across all three classes."""

    @pytest.mark.parametrize("AttackCls, kwargs", [
        (SignFlipAttack,         {"client_id": "c0"}),
        (ScalingAttack,          {"client_id": "c0"}),
        (AdaptiveStealthAttack,  {"client_id": "c0"}),
    ])
    def test_no_is_malicious_attribute(self, AttackCls, kwargs) -> None:
        a = AttackCls(**kwargs)
        assert not hasattr(a, "is_malicious"), (
            f"{AttackCls.__name__} must never expose 'is_malicious'"
        )

    @pytest.mark.parametrize("AttackCls, kwargs", [
        (SignFlipAttack,         {"client_id": "c0"}),
        (ScalingAttack,          {"client_id": "c0"}),
        (AdaptiveStealthAttack,  {"client_id": "c0"}),
    ])
    def test_to_dict_no_is_malicious(self, AttackCls, kwargs) -> None:
        a = AttackCls(**kwargs)
        assert "is_malicious" not in a.to_dict()

    @pytest.mark.parametrize("AttackCls, kwargs", [
        (SignFlipAttack,         {"client_id": "c0"}),
        (ScalingAttack,          {"client_id": "c0"}),
        (AdaptiveStealthAttack,  {"client_id": "c0"}),
    ])
    def test_result_dict_no_is_malicious(self, AttackCls, kwargs) -> None:
        a = AttackCls(**kwargs)
        delta = _make_delta()
        result = a.apply_update_attack(delta)
        assert "is_malicious" not in result

    @pytest.mark.parametrize("AttackCls, kwargs", [
        (SignFlipAttack,         {"client_id": "c0"}),
        (ScalingAttack,          {"client_id": "c0"}),
        (AdaptiveStealthAttack,  {"client_id": "c0"}),
    ])
    def test_output_arrays_are_finite(self, AttackCls, kwargs) -> None:
        """All returned arrays must contain only finite values."""
        a = AttackCls(**kwargs)
        delta = _make_delta()
        result = a.apply_update_attack(delta)
        for k, arr in result.items():
            assert np.all(np.isfinite(arr)), (
                f"{AttackCls.__name__}: non-finite values in key '{k}'"
            )

    @pytest.mark.parametrize("AttackCls, kwargs", [
        (SignFlipAttack,         {"client_id": "c0"}),
        (ScalingAttack,          {"client_id": "c0"}),
        (AdaptiveStealthAttack,  {"client_id": "c0"}),
    ])
    def test_output_dtype_is_float32(self, AttackCls, kwargs) -> None:
        a = AttackCls(**kwargs)
        delta = _make_delta(dtype=np.float32)
        result = a.apply_update_attack(delta)
        for k, arr in result.items():
            assert arr.dtype == np.float32, (
                f"{AttackCls.__name__}: key '{k}' has dtype {arr.dtype}, expected float32"
            )

    @pytest.mark.parametrize("AttackCls, kwargs", [
        (SignFlipAttack,         {"client_id": "c0"}),
        (ScalingAttack,          {"client_id": "c0"}),
        (AdaptiveStealthAttack,  {"client_id": "c0"}),
    ])
    def test_output_dtype_is_float64(self, AttackCls, kwargs) -> None:
        a = AttackCls(**kwargs)
        delta = _make_delta(dtype=np.float64)
        result = a.apply_update_attack(delta)
        for k, arr in result.items():
            assert arr.dtype == np.float64, (
                f"{AttackCls.__name__}: key '{k}' has dtype {arr.dtype}, expected float64"
            )

    @pytest.mark.parametrize("AttackCls, kwargs", [
        (SignFlipAttack,         {"client_id": "c0"}),
        (ScalingAttack,          {"client_id": "c0"}),
        (AdaptiveStealthAttack,  {"client_id": "c0"}),
    ])
    def test_to_dict_has_exactly_three_keys(self, AttackCls, kwargs) -> None:
        """No extra metadata must leak into to_dict()."""
        a = AttackCls(**kwargs)
        d = a.to_dict()
        assert len(d) == 3
        assert set(d.keys()) == {"client_id", "attack_type", "intensity"}
