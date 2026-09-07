"""
tests/test_byzantine.py
=======================
Unit tests for ``core.attacks.byzantine``:
    * :class:`~core.attacks.byzantine.GaussianNoiseAttack`
    * :class:`~core.attacks.byzantine.ZeroUpdateAttack`
    * :class:`~core.attacks.byzantine.ExtremeValueAttack`

Run with::

    pytest tests/test_byzantine.py -v

Test sections
-------------
A  Package imports & class hierarchy
B  GaussianNoiseAttack — perturbation, shape/dtype, determinism, zero-std
C  ZeroUpdateAttack — zeroing out, shape/dtype preservation
D  ExtremeValueAttack — numerical flooding, shape/dtype preservation
E  Edge cases (empty delta, multi-dimensional shapes: 1D, 2D, 4D)
F  Contract invariants (no is_malicious, non-mutation of inputs, return fresh dicts)

Contract reference: FedSentinel Team Engineering Contract v2.0 — Person 2
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pytest

from core.attacks.base_attack import BaseAttack
from core.attacks.byzantine import (
    ExtremeValueAttack,
    GaussianNoiseAttack,
    ZeroUpdateAttack,
)
from core.attacks import (
    ExtremeValueAttack as ExtremeFromInit,
    GaussianNoiseAttack as GaussianFromInit,
    ZeroUpdateAttack as ZeroFromInit,
)


# ---------------------------------------------------------------------------
# Shared fixtures & helpers
# ---------------------------------------------------------------------------

def _make_delta(dtype: np.dtype = np.float32, seed: int = 42) -> Dict[str, np.ndarray]:
    """Return a multi-layer realistic parameter delta dictionary."""
    rng = np.random.default_rng(seed=seed)
    return {
        "conv1.weight": rng.standard_normal((16, 3, 5, 5)).astype(dtype),  # 4D
        "conv1.bias":   rng.standard_normal((16,)).astype(dtype),           # 1D
        "fc1.weight":   rng.standard_normal((64, 400)).astype(dtype),       # 2D
        "fc1.bias":     rng.standard_normal((64,)).astype(dtype),           # 1D
        "fc2.weight":   rng.standard_normal((10, 64)).astype(dtype),        # 2D
        "fc2.bias":     rng.standard_normal((10,)).astype(dtype),           # 1D
    }


def _snapshot(delta: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Deep-copy snapshot for immutability verification."""
    return {k: v.copy() for k, v in delta.items()}


# ---------------------------------------------------------------------------
# Section A — Package imports & class hierarchy
# ---------------------------------------------------------------------------

class TestImportsAndHierarchy:
    """Check module exports, package exports, and BaseAttack inheritance."""

    def test_import_from_module(self) -> None:
        assert GaussianNoiseAttack is not None
        assert ZeroUpdateAttack is not None
        assert ExtremeValueAttack is not None

    def test_import_from_package_init(self) -> None:
        assert GaussianFromInit is GaussianNoiseAttack
        assert ZeroFromInit is ZeroUpdateAttack
        assert ExtremeFromInit is ExtremeValueAttack

    def test_subclass_of_base_attack(self) -> None:
        assert issubclass(GaussianNoiseAttack, BaseAttack)
        assert issubclass(ZeroUpdateAttack, BaseAttack)
        assert issubclass(ExtremeValueAttack, BaseAttack)


# ---------------------------------------------------------------------------
# Section B — GaussianNoiseAttack
# ---------------------------------------------------------------------------

class TestGaussianNoiseAttackConstructor:
    """Constructor validation and attribute storage."""

    def test_default_attributes(self) -> None:
        attack = GaussianNoiseAttack(client_id="c1")
        assert attack.client_id == "c1"
        assert attack.attack_type == "byzantine_gaussian"
        assert attack.mean == 0.0
        assert attack.std == 1.0
        assert attack.intensity == 1.0
        assert attack.seed is None

    def test_custom_attributes(self) -> None:
        attack = GaussianNoiseAttack(client_id="c2", mean=0.5, std=2.5, seed=123)
        assert attack.client_id == "c2"
        assert attack.mean == 0.5
        assert attack.std == 2.5
        assert attack.intensity == 2.5
        assert attack.seed == 123

    def test_invalid_mean_raises(self) -> None:
        with pytest.raises(TypeError, match="mean"):
            GaussianNoiseAttack(client_id="c1", mean="invalid")  # type: ignore

    def test_invalid_std_type_raises(self) -> None:
        with pytest.raises(TypeError, match="std"):
            GaussianNoiseAttack(client_id="c1", std="invalid")  # type: ignore

    def test_negative_std_raises(self) -> None:
        with pytest.raises(ValueError, match="std must be non-negative"):
            GaussianNoiseAttack(client_id="c1", std=-0.5)

    def test_invalid_seed_type_raises(self) -> None:
        with pytest.raises(TypeError, match="seed"):
            GaussianNoiseAttack(client_id="c1", seed="seed")  # type: ignore

    def test_to_dict_format(self) -> None:
        attack = GaussianNoiseAttack(client_id="c1", std=1.75)
        d = attack.to_dict()
        assert d == {
            "client_id": "c1",
            "attack_type": "byzantine_gaussian",
            "intensity": 1.75,
        }


class TestGaussianNoiseAttackBehavior:
    """Noise addition, shapes, dtypes, and determinism."""

    def test_output_values_are_perturbed(self) -> None:
        attack = GaussianNoiseAttack(client_id="c1", std=1.0, seed=10)
        delta = _make_delta()
        result = attack.apply_update_attack(delta)

        for k in delta:
            # Noise should perturb values away from original
            assert not np.allclose(result[k], delta[k])

    def test_deterministic_output_with_same_seed(self) -> None:
        delta = _make_delta()
        attack1 = GaussianNoiseAttack(client_id="c1", std=1.5, seed=999)
        attack2 = GaussianNoiseAttack(client_id="c2", std=1.5, seed=999)

        res1 = attack1.apply_update_attack(delta)
        res2 = attack2.apply_update_attack(delta)

        for k in delta:
            np.testing.assert_array_equal(res1[k], res2[k])

    def test_different_seeds_produce_different_noise(self) -> None:
        delta = _make_delta()
        attack1 = GaussianNoiseAttack(client_id="c1", seed=1)
        attack2 = GaussianNoiseAttack(client_id="c1", seed=2)

        res1 = attack1.apply_update_attack(delta)
        res2 = attack2.apply_update_attack(delta)

        for k in delta:
            assert not np.array_equal(res1[k], res2[k])

    def test_zero_std_and_zero_mean_preserves_values_exactly(self) -> None:
        attack = GaussianNoiseAttack(client_id="c1", mean=0.0, std=0.0)
        delta = _make_delta()
        result = attack.apply_update_attack(delta)

        for k in delta:
            np.testing.assert_array_equal(result[k], delta[k])

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_dtypes_preserved(self, dtype) -> None:
        attack = GaussianNoiseAttack(client_id="c1", seed=42)
        delta = _make_delta(dtype=dtype)
        result = attack.apply_update_attack(delta)

        for k in delta:
            assert result[k].dtype == dtype

    def test_shapes_preserved(self) -> None:
        attack = GaussianNoiseAttack(client_id="c1", seed=42)
        delta = _make_delta()
        result = attack.apply_update_attack(delta)

        for k in delta:
            assert result[k].shape == delta[k].shape


# ---------------------------------------------------------------------------
# Section C — ZeroUpdateAttack
# ---------------------------------------------------------------------------

class TestZeroUpdateAttack:
    """ZeroUpdateAttack verification."""

    def test_constructor_and_to_dict(self) -> None:
        attack = ZeroUpdateAttack(client_id="c_zero")
        assert attack.client_id == "c_zero"
        assert attack.attack_type == "byzantine_zero"
        assert attack.intensity == 0.0

        d = attack.to_dict()
        assert d == {
            "client_id": "c_zero",
            "attack_type": "byzantine_zero",
            "intensity": 0.0,
        }

    def test_all_elements_are_zero(self) -> None:
        attack = ZeroUpdateAttack(client_id="c_zero")
        delta = _make_delta()
        result = attack.apply_update_attack(delta)

        for k, v in result.items():
            assert np.all(v == 0.0)
            np.testing.assert_array_equal(v, np.zeros_like(delta[k]))

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_dtypes_preserved(self, dtype) -> None:
        attack = ZeroUpdateAttack(client_id="c_zero")
        delta = _make_delta(dtype=dtype)
        result = attack.apply_update_attack(delta)

        for k in delta:
            assert result[k].dtype == dtype

    def test_shapes_and_keys_preserved(self) -> None:
        attack = ZeroUpdateAttack(client_id="c_zero")
        delta = _make_delta()
        result = attack.apply_update_attack(delta)

        assert set(result.keys()) == set(delta.keys())
        for k in delta:
            assert result[k].shape == delta[k].shape


# ---------------------------------------------------------------------------
# Section D — ExtremeValueAttack
# ---------------------------------------------------------------------------

class TestExtremeValueAttack:
    """ExtremeValueAttack verification."""

    def test_constructor_defaults(self) -> None:
        attack = ExtremeValueAttack(client_id="c_ext")
        assert attack.client_id == "c_ext"
        assert attack.attack_type == "byzantine_extreme"
        assert attack.extreme_val == 1e6
        assert attack.intensity == 1e6

    def test_constructor_custom_val(self) -> None:
        attack = ExtremeValueAttack(client_id="c_ext", extreme_val=-99999.0)
        assert attack.extreme_val == -99999.0
        assert attack.intensity == -99999.0

    def test_invalid_val_raises(self) -> None:
        with pytest.raises(TypeError, match="extreme_val"):
            ExtremeValueAttack(client_id="c_ext", extreme_val="extreme")  # type: ignore

    def test_to_dict(self) -> None:
        attack = ExtremeValueAttack(client_id="c_ext", extreme_val=5e5)
        assert attack.to_dict() == {
            "client_id": "c_ext",
            "attack_type": "byzantine_extreme",
            "intensity": 5e5,
        }

    def test_all_elements_match_extreme_val(self) -> None:
        val = 1e6
        attack = ExtremeValueAttack(client_id="c_ext", extreme_val=val)
        delta = _make_delta()
        result = attack.apply_update_attack(delta)

        for k, v in result.items():
            assert np.all(v == val)
            assert v.shape == delta[k].shape

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_dtypes_preserved(self, dtype) -> None:
        attack = ExtremeValueAttack(client_id="c_ext", extreme_val=1e4)
        delta = _make_delta(dtype=dtype)
        result = attack.apply_update_attack(delta)

        for k in delta:
            assert result[k].dtype == dtype


# ---------------------------------------------------------------------------
# Section E — Edge Cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Empty delta and multi-dimensional tensor handling."""

    @pytest.mark.parametrize("attack", [
        GaussianNoiseAttack(client_id="c1", seed=42),
        ZeroUpdateAttack(client_id="c2"),
        ExtremeValueAttack(client_id="c3"),
    ])
    def test_empty_delta_returns_empty(self, attack: BaseAttack) -> None:
        result = attack.apply_update_attack({})
        assert result == {}
        assert isinstance(result, dict)

    @pytest.mark.parametrize("attack", [
        GaussianNoiseAttack(client_id="c1", seed=42),
        ZeroUpdateAttack(client_id="c2"),
        ExtremeValueAttack(client_id="c3"),
    ])
    def test_multidimensional_tensors(self, attack: BaseAttack) -> None:
        delta = {
            "1d_bias": np.ones((10,), dtype=np.float32),
            "2d_dense": np.ones((32, 64), dtype=np.float32),
            "4d_conv": np.ones((8, 4, 3, 3), dtype=np.float32),
        }
        result = attack.apply_update_attack(delta)

        assert result["1d_bias"].shape == (10,)
        assert result["2d_dense"].shape == (32, 64)
        assert result["4d_conv"].shape == (8, 4, 3, 3)


# ---------------------------------------------------------------------------
# Section F — Contract Invariants
# ---------------------------------------------------------------------------

class TestContractInvariants:
    """Strict adherence to Team Engineering Contract v2.0."""

    @pytest.mark.parametrize("attack_cls, kwargs", [
        (GaussianNoiseAttack, {"client_id": "c1", "seed": 42}),
        (ZeroUpdateAttack,    {"client_id": "c2"}),
        (ExtremeValueAttack,   {"client_id": "c3"}),
    ])
    def test_no_is_malicious_attribute(self, attack_cls, kwargs) -> None:
        attack = attack_cls(**kwargs)
        assert not hasattr(attack, "is_malicious")

    @pytest.mark.parametrize("attack_cls, kwargs", [
        (GaussianNoiseAttack, {"client_id": "c1", "seed": 42}),
        (ZeroUpdateAttack,    {"client_id": "c2"}),
        (ExtremeValueAttack,   {"client_id": "c3"}),
    ])
    def test_to_dict_has_no_is_malicious(self, attack_cls, kwargs) -> None:
        attack = attack_cls(**kwargs)
        d = attack.to_dict()
        assert "is_malicious" not in d
        assert set(d.keys()) == {"client_id", "attack_type", "intensity"}

    @pytest.mark.parametrize("attack_cls, kwargs", [
        (GaussianNoiseAttack, {"client_id": "c1", "seed": 42}),
        (ZeroUpdateAttack,    {"client_id": "c2"}),
        (ExtremeValueAttack,   {"client_id": "c3"}),
    ])
    def test_result_dict_has_no_is_malicious(self, attack_cls, kwargs) -> None:
        attack = attack_cls(**kwargs)
        delta = _make_delta()
        result = attack.apply_update_attack(delta)
        assert "is_malicious" not in result

    @pytest.mark.parametrize("attack_cls, kwargs", [
        (GaussianNoiseAttack, {"client_id": "c1", "seed": 42}),
        (ZeroUpdateAttack,    {"client_id": "c2"}),
        (ExtremeValueAttack,   {"client_id": "c3"}),
    ])
    def test_does_not_mutate_input_delta(self, attack_cls, kwargs) -> None:
        attack = attack_cls(**kwargs)
        delta = _make_delta()
        snapshot = _snapshot(delta)

        _ = attack.apply_update_attack(delta)

        for k in snapshot:
            np.testing.assert_array_equal(delta[k], snapshot[k])

    @pytest.mark.parametrize("attack_cls, kwargs", [
        (GaussianNoiseAttack, {"client_id": "c1", "seed": 42}),
        (ZeroUpdateAttack,    {"client_id": "c2"}),
        (ExtremeValueAttack,   {"client_id": "c3"}),
    ])
    def test_returns_fresh_dict(self, attack_cls, kwargs) -> None:
        attack = attack_cls(**kwargs)
        delta = _make_delta()
        result = attack.apply_update_attack(delta)
        assert result is not delta
