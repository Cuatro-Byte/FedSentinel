"""
tests/test_base_attack.py
=========================
Unit tests for ``core.attacks.base_attack.BaseAttack``.

Run with::

    pytest tests/test_base_attack.py -v

All tests are self-contained and do NOT require a GPU or a real
``torch.utils.data.Dataset`` — lightweight fakes / mocks are used instead.

Coverage checklist (aligned with Phase 1 contract deliverables):
    a) Concrete subclass without overrides behaves as an identity pass-through.
    b) ``apply_data_attack`` returns the *identical* dataset reference.
    c) ``apply_update_attack`` returns the *identical* dict reference with
       keys and values completely unchanged.
    d) ``to_dict()`` returns the expected structure and correct Python types.
    e) Overriding ``apply_update_attack`` in a dummy attack mutates NumPy
       arrays as expected.

Contract reference: FedSentinel Team Engineering Contract v2.0 — Person 2
"""

from __future__ import annotations

import copy
from typing import Dict

import numpy as np
import pytest

from core.attacks.base_attack import BaseAttack
from core.attacks import BaseAttack as BaseAttackFromInit  # package re-export


# ---------------------------------------------------------------------------
# Shared fixtures & helpers
# ---------------------------------------------------------------------------


class _ConcreteAttack(BaseAttack):
    """Minimal concrete subclass that does NOT override any attack hooks.

    Used to verify identity (pass-through) behaviour of the base class.
    """


class _ScalingAttack(BaseAttack):
    """Dummy attack that multiplies every parameter delta by ``intensity``.

    Used to verify that an overriding subclass can mutate NumPy arrays.
    """

    def __init__(self, client_id: str, intensity: float = 2.0) -> None:
        super().__init__(client_id=client_id, intensity=intensity, attack_type="scaling")

    def apply_update_attack(
        self, delta: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Scale every parameter delta by ``self.intensity``."""
        return {k: v * self.intensity for k, v in delta.items()}


class _FakeDataset:
    """Lightweight stand-in for ``torch.utils.data.Dataset``."""

    def __init__(self, data: list) -> None:
        self._data = data

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int):
        return self._data[idx]


def _make_delta() -> Dict[str, np.ndarray]:
    """Return a realistic-looking model delta dictionary."""
    rng = np.random.default_rng(seed=42)
    return {
        "layer1.weight": rng.standard_normal((64, 32)).astype(np.float32),
        "layer1.bias": rng.standard_normal((64,)).astype(np.float32),
        "layer2.weight": rng.standard_normal((10, 64)).astype(np.float32),
        "layer2.bias": rng.standard_normal((10,)).astype(np.float32),
    }


# ---------------------------------------------------------------------------
# Section A — Package import & subclassing
# ---------------------------------------------------------------------------


class TestPackageExport:
    """Verify that BaseAttack is properly accessible via the package __init__."""

    def test_import_from_package_init(self) -> None:
        """``from core.attacks import BaseAttack`` must resolve to the same class."""
        assert BaseAttackFromInit is BaseAttack

    def test_is_abc_subclass(self) -> None:
        """``BaseAttack`` must be a subclass of ``ABC`` (is registered as an ABC)."""
        import abc
        assert issubclass(BaseAttack, abc.ABC)

    def test_base_attack_directly_instantiable_with_identity_hooks(self) -> None:
        """``BaseAttack`` uses identity-fallback (non-abstract) hooks by design.

        The contract requires that the *base* implementations work as
        pass-throughs so that benign clients and test harnesses can use
        ``BaseAttack`` (or a trivial subclass) without overriding anything.
        Direct instantiation of ``BaseAttack`` must therefore succeed.
        """
        # BaseAttack has no @abstractmethod, so direct instantiation is valid.
        # We verify the object is fully functional with identity hooks.
        attack = BaseAttack(client_id="client_direct")  # type: ignore[abstract]
        assert attack.client_id == "client_direct"
        assert attack.attack_type == "base"
        dataset = _FakeDataset(data=[1, 2, 3])
        assert attack.apply_data_attack(dataset) is dataset

    def test_concrete_subclass_instantiates(self) -> None:
        """A concrete subclass (no method overrides) must instantiate without error."""
        attack = _ConcreteAttack(client_id="client_0")
        assert attack is not None


# ---------------------------------------------------------------------------
# Section B — Constructor validation
# ---------------------------------------------------------------------------


class TestConstructor:
    """Verify constructor argument storage and validation."""

    def test_stores_client_id(self) -> None:
        attack = _ConcreteAttack(client_id="client_7")
        assert attack.client_id == "client_7"

    def test_stores_intensity_default(self) -> None:
        attack = _ConcreteAttack(client_id="client_0")
        assert attack.intensity == 1.0

    def test_stores_intensity_custom(self) -> None:
        attack = _ConcreteAttack(client_id="client_0", intensity=0.5)
        assert attack.intensity == pytest.approx(0.5)

    def test_stores_attack_type_default(self) -> None:
        attack = _ConcreteAttack(client_id="client_0")
        assert attack.attack_type == "base"

    def test_stores_attack_type_custom(self) -> None:
        attack = _ConcreteAttack(client_id="client_0", attack_type="test_attack")
        assert attack.attack_type == "test_attack"

    def test_intensity_stored_as_float(self) -> None:
        """Even when passed as int, intensity must be stored as float."""
        attack = _ConcreteAttack(client_id="client_0", intensity=2)
        assert isinstance(attack.intensity, float)
        assert attack.intensity == 2.0

    def test_invalid_client_id_empty_string_raises(self) -> None:
        with pytest.raises(ValueError, match="client_id"):
            _ConcreteAttack(client_id="")

    def test_invalid_client_id_non_string_raises(self) -> None:
        with pytest.raises((ValueError, TypeError)):
            _ConcreteAttack(client_id=123)  # type: ignore[arg-type]

    def test_invalid_intensity_raises(self) -> None:
        with pytest.raises(TypeError, match="intensity"):
            _ConcreteAttack(client_id="client_0", intensity="high")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Section C — apply_data_attack (identity pass-through)
# ---------------------------------------------------------------------------


class TestApplyDataAttack:
    """Tests for the base-class apply_data_attack identity behaviour (req. b)."""

    def test_returns_same_object_reference(self) -> None:
        """Base apply_data_attack must return the *identical* dataset object."""
        attack = _ConcreteAttack(client_id="client_0")
        dataset = _FakeDataset(data=list(range(100)))
        result = attack.apply_data_attack(dataset)
        assert result is dataset

    def test_does_not_modify_dataset_length(self) -> None:
        attack = _ConcreteAttack(client_id="client_0")
        dataset = _FakeDataset(data=list(range(50)))
        result = attack.apply_data_attack(dataset)
        assert len(result) == 50

    def test_dataset_items_unchanged(self) -> None:
        attack = _ConcreteAttack(client_id="client_0")
        original_data = [(i, i % 10) for i in range(20)]
        dataset = _FakeDataset(data=original_data)
        result = attack.apply_data_attack(dataset)
        for idx in range(len(original_data)):
            assert result[idx] == original_data[idx]

    def test_none_passthrough_is_not_applicable(self) -> None:
        """Verify that a non-None object is required (interface sanity check)."""
        attack = _ConcreteAttack(client_id="client_0")
        # Passing a plain object to verify the identity contract still holds
        fake = object()
        result = attack.apply_data_attack(fake)  # type: ignore[arg-type]
        assert result is fake


# ---------------------------------------------------------------------------
# Section D — apply_update_attack (identity pass-through)
# ---------------------------------------------------------------------------


class TestApplyUpdateAttackIdentity:
    """Tests for the base-class apply_update_attack identity behaviour (req. c)."""

    def test_returns_same_dict_reference(self) -> None:
        """Base apply_update_attack must return the *identical* dict object."""
        attack = _ConcreteAttack(client_id="client_0")
        delta = _make_delta()
        result = attack.apply_update_attack(delta)
        assert result is delta

    def test_keys_unchanged(self) -> None:
        """Returned dict must have exactly the same keys as the input."""
        attack = _ConcreteAttack(client_id="client_0")
        delta = _make_delta()
        original_keys = set(delta.keys())
        result = attack.apply_update_attack(delta)
        assert set(result.keys()) == original_keys

    def test_values_unchanged_by_reference(self) -> None:
        """Each value in the returned dict must be the same ndarray object."""
        attack = _ConcreteAttack(client_id="client_0")
        delta = _make_delta()
        result = attack.apply_update_attack(delta)
        for key in delta:
            assert result[key] is delta[key], (
                f"Array for key '{key}' is not the same object after identity call"
            )

    def test_values_numerically_identical(self) -> None:
        """Values must not be modified (all elements equal)."""
        attack = _ConcreteAttack(client_id="client_0")
        delta = _make_delta()
        snapshot = {k: v.copy() for k, v in delta.items()}
        result = attack.apply_update_attack(delta)
        for key in snapshot:
            np.testing.assert_array_equal(result[key], snapshot[key])

    def test_empty_delta_passthrough(self) -> None:
        """An empty delta dict must also be returned unchanged."""
        attack = _ConcreteAttack(client_id="client_0")
        delta: Dict[str, np.ndarray] = {}
        result = attack.apply_update_attack(delta)
        assert result is delta
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Section E — to_dict()
# ---------------------------------------------------------------------------


class TestToDict:
    """Tests for metadata serialisation (req. d)."""

    def test_returns_dict(self) -> None:
        attack = _ConcreteAttack(client_id="client_0")
        assert isinstance(attack.to_dict(), dict)

    def test_expected_keys(self) -> None:
        attack = _ConcreteAttack(client_id="client_0")
        result = attack.to_dict()
        assert set(result.keys()) == {"client_id", "attack_type", "intensity"}

    def test_exactly_three_keys(self) -> None:
        """Ensure no extra keys are silently added (e.g., is_malicious)."""
        attack = _ConcreteAttack(client_id="client_0")
        result = attack.to_dict()
        assert len(result) == 3

    def test_client_id_value_and_type(self) -> None:
        attack = _ConcreteAttack(client_id="my_client")
        result = attack.to_dict()
        assert result["client_id"] == "my_client"
        assert isinstance(result["client_id"], str)

    def test_attack_type_value_and_type(self) -> None:
        attack = _ConcreteAttack(client_id="client_0", attack_type="test_type")
        result = attack.to_dict()
        assert result["attack_type"] == "test_type"
        assert isinstance(result["attack_type"], str)

    def test_intensity_value_and_type(self) -> None:
        attack = _ConcreteAttack(client_id="client_0", intensity=0.75)
        result = attack.to_dict()
        assert result["intensity"] == pytest.approx(0.75)
        assert isinstance(result["intensity"], float)

    def test_to_dict_does_not_contain_is_malicious(self) -> None:
        """Critical contract rule: is_malicious must NEVER appear in to_dict()."""
        attack = _ConcreteAttack(client_id="client_0")
        result = attack.to_dict()
        assert "is_malicious" not in result

    def test_to_dict_is_independent_copy(self) -> None:
        """Mutating the returned dict must not affect the attack object."""
        attack = _ConcreteAttack(client_id="client_0")
        result = attack.to_dict()
        result["client_id"] = "tampered"
        # The original attribute must be unchanged
        assert attack.client_id == "client_0"

    def test_defaults_in_to_dict(self) -> None:
        attack = _ConcreteAttack(client_id="alpha")
        result = attack.to_dict()
        assert result == {
            "client_id": "alpha",
            "attack_type": "base",
            "intensity": 1.0,
        }


# ---------------------------------------------------------------------------
# Section F — Overriding attack hook (dummy scaling attack) (req. e)
# ---------------------------------------------------------------------------


class TestOverriddenApplyUpdateAttack:
    """Tests for a concrete override that mutates NumPy arrays (req. e)."""

    def test_scaling_attack_returns_new_dict(self) -> None:
        """A scaling override must return a *new* dict (not the same reference)."""
        attack = _ScalingAttack(client_id="client_1", intensity=2.0)
        delta = _make_delta()
        result = attack.apply_update_attack(delta)
        assert result is not delta

    def test_scaling_attack_preserves_keys(self) -> None:
        attack = _ScalingAttack(client_id="client_1", intensity=2.0)
        delta = _make_delta()
        result = attack.apply_update_attack(delta)
        assert set(result.keys()) == set(delta.keys())

    def test_scaling_attack_doubles_values(self) -> None:
        attack = _ScalingAttack(client_id="client_1", intensity=2.0)
        delta = _make_delta()
        original_snapshot = {k: v.copy() for k, v in delta.items()}
        result = attack.apply_update_attack(delta)
        for key in original_snapshot:
            np.testing.assert_allclose(
                result[key],
                original_snapshot[key] * 2.0,
                rtol=1e-6,
                err_msg=f"Values for key '{key}' were not doubled correctly",
            )

    def test_scaling_attack_zero_intensity(self) -> None:
        """intensity=0 should produce all-zero arrays."""
        attack = _ScalingAttack(client_id="client_1", intensity=0.0)
        delta = _make_delta()
        result = attack.apply_update_attack(delta)
        for key in result:
            np.testing.assert_array_equal(
                result[key],
                np.zeros_like(delta[key]),
            )

    def test_scaling_attack_negative_intensity(self) -> None:
        """Negative intensity should negate and scale the arrays."""
        attack = _ScalingAttack(client_id="client_1", intensity=-1.0)
        delta = _make_delta()
        snapshot = {k: v.copy() for k, v in delta.items()}
        result = attack.apply_update_attack(delta)
        for key in snapshot:
            np.testing.assert_allclose(
                result[key],
                -snapshot[key],
                rtol=1e-6,
            )

    def test_scaling_attack_values_are_ndarrays(self) -> None:
        """All values in the returned dict must be numpy.ndarray instances."""
        attack = _ScalingAttack(client_id="client_1", intensity=3.0)
        delta = _make_delta()
        result = attack.apply_update_attack(delta)
        for key, val in result.items():
            assert isinstance(val, np.ndarray), (
                f"Value for key '{key}' is {type(val).__name__}, expected ndarray"
            )

    def test_scaling_attack_does_not_modify_original_delta(self) -> None:
        """The original delta dict must be unchanged after the scaling attack."""
        attack = _ScalingAttack(client_id="client_1", intensity=5.0)
        delta = _make_delta()
        original_snapshot = {k: v.copy() for k, v in delta.items()}
        _ = attack.apply_update_attack(delta)
        for key in original_snapshot:
            np.testing.assert_array_equal(delta[key], original_snapshot[key])

    def test_scaling_attack_to_dict_reflects_overridden_attack_type(self) -> None:
        attack = _ScalingAttack(client_id="client_1", intensity=2.0)
        result = attack.to_dict()
        assert result["attack_type"] == "scaling"
        assert result["intensity"] == pytest.approx(2.0)
        assert result["client_id"] == "client_1"

    def test_scaling_attack_shape_preserved(self) -> None:
        """Array shapes must be preserved after scaling."""
        attack = _ScalingAttack(client_id="client_1", intensity=2.0)
        delta = _make_delta()
        result = attack.apply_update_attack(delta)
        for key in delta:
            assert result[key].shape == delta[key].shape


# ---------------------------------------------------------------------------
# Section G — Contract invariant: no is_malicious leakage
# ---------------------------------------------------------------------------


class TestContractInvariants:
    """Explicit checks for Person 2 contract rules."""

    def test_base_attack_has_no_is_malicious_attribute(self) -> None:
        attack = _ConcreteAttack(client_id="client_0")
        assert not hasattr(attack, "is_malicious"), (
            "BaseAttack must never expose an 'is_malicious' attribute"
        )

    def test_to_dict_has_no_is_malicious_key(self) -> None:
        attack = _ConcreteAttack(client_id="client_0")
        assert "is_malicious" not in attack.to_dict()

    def test_scaling_attack_has_no_is_malicious_attribute(self) -> None:
        attack = _ScalingAttack(client_id="client_1")
        assert not hasattr(attack, "is_malicious")

    def test_apply_update_attack_result_has_no_is_malicious_key(self) -> None:
        attack = _ScalingAttack(client_id="client_1")
        delta = _make_delta()
        result = attack.apply_update_attack(delta)
        assert "is_malicious" not in result
