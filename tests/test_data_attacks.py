"""
tests/test_data_attacks.py
==========================
Unit tests for data-plane poisoning attacks in ``core.attacks``:
    * :class:`~core.attacks.label_poisoning.LabelFlipDatasetWrapper`
    * :class:`~core.attacks.label_poisoning.LabelFlipAttack`
    * :class:`~core.attacks.backdoor.BackdoorDatasetWrapper`
    * :class:`~core.attacks.backdoor.BackdoorAttack`

Run with::

    pytest tests/test_data_attacks.py -v

Test sections
-------------
A  Package imports & class inheritance
B  LabelFlipDatasetWrapper & LabelFlipAttack (targeted flipping, non-targets untouched, rates 0.0 and 1.0)
C  BackdoorDatasetWrapper & BackdoorAttack (trigger geometry, cloning immutability, 2D/3D shapes, rates)
D  Update-plane fallbacks (apply_update_attack returns delta unmodified)
E  Contract invariants (no is_malicious, input validation, to_dict structure)

Contract reference: FedSentinel Team Engineering Contract v2.0 — Person 2
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from core.attacks.backdoor import (
    BackdoorAttack,
    BackdoorDatasetWrapper,
)
from core.attacks.base_attack import BaseAttack
from core.attacks.label_poisoning import (
    LabelFlipAttack,
    LabelFlipDatasetWrapper,
)
from core.attacks import (
    BackdoorAttack as BackdoorAttackFromInit,
    BackdoorDatasetWrapper as BackdoorWrapperFromInit,
    LabelFlipAttack as LabelFlipAttackFromInit,
    LabelFlipDatasetWrapper as LabelFlipWrapperFromInit,
)


# ---------------------------------------------------------------------------
# Synthetic Dataset Fixtures
# ---------------------------------------------------------------------------

class SyntheticClassificationDataset(Dataset):
    """Simple in-memory classification dataset."""

    def __init__(self, samples: List[Tuple[torch.Tensor, int]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.samples[idx]


@pytest.fixture
def dummy_classification_dataset() -> SyntheticClassificationDataset:
    """Dataset of 20 samples with classes 0, 1, 2, 3."""
    samples = []
    for i in range(20):
        x = torch.full((4,), float(i), dtype=torch.float32)
        y = i % 4
        samples.append((x, y))
    return SyntheticClassificationDataset(samples)


@pytest.fixture
def dummy_image_dataset_3d() -> SyntheticClassificationDataset:
    """Dataset of 10 3-channel images (3, 16, 16) with all zeros."""
    samples = []
    for i in range(10):
        x = torch.zeros((3, 16, 16), dtype=torch.float32)
        y = i % 2
        samples.append((x, y))
    return SyntheticClassificationDataset(samples)


@pytest.fixture
def dummy_image_dataset_2d() -> SyntheticClassificationDataset:
    """Dataset of 10 2-channel/grayscale images (16, 16) with all zeros."""
    samples = []
    for i in range(10):
        x = torch.zeros((16, 16), dtype=torch.float32)
        y = 1
        samples.append((x, y))
    return SyntheticClassificationDataset(samples)


# ---------------------------------------------------------------------------
# Section A — Package Imports & Inheritance
# ---------------------------------------------------------------------------

class TestImportsAndInheritance:
    """Verify package re-exports and class relationships."""

    def test_package_exports(self) -> None:
        assert LabelFlipAttackFromInit is LabelFlipAttack
        assert LabelFlipWrapperFromInit is LabelFlipDatasetWrapper
        assert BackdoorAttackFromInit is BackdoorAttack
        assert BackdoorWrapperFromInit is BackdoorDatasetWrapper

    def test_inheritance(self) -> None:
        assert issubclass(LabelFlipAttack, BaseAttack)
        assert issubclass(BackdoorAttack, BaseAttack)
        assert issubclass(LabelFlipDatasetWrapper, Dataset)
        assert issubclass(BackdoorDatasetWrapper, Dataset)


# ---------------------------------------------------------------------------
# Section B — LabelFlipDatasetWrapper & LabelFlipAttack
# ---------------------------------------------------------------------------

class TestLabelFlipAttack:
    """Tests for label poisoning mechanics."""

    def test_constructor_and_to_dict(self) -> None:
        attack = LabelFlipAttack(
            client_id="c1",
            source_class=1,
            target_class=7,
            poison_rate=0.8,
            seed=42,
        )
        assert attack.client_id == "c1"
        assert attack.source_class == 1
        assert attack.target_class == 7
        assert attack.poison_rate == 0.8
        assert attack.intensity == 0.8
        assert attack.attack_type == "label_flip"
        assert attack.to_dict() == {
            "client_id": "c1",
            "attack_type": "label_flip",
            "intensity": 0.8,
        }

    def test_full_poison_rate_flips_all_source_samples(
        self, dummy_classification_dataset: SyntheticClassificationDataset
    ) -> None:
        attack = LabelFlipAttack(client_id="c1", source_class=1, target_class=9, poison_rate=1.0)
        wrapped = attack.apply_data_attack(dummy_classification_dataset)
        assert len(wrapped) == len(dummy_classification_dataset)

        for i in range(len(wrapped)):
            _, orig_y = dummy_classification_dataset[i]
            _, poisoned_y = wrapped[i]

            if orig_y == 1:
                assert poisoned_y == 9
            else:
                assert poisoned_y == orig_y

    def test_zero_poison_rate_flips_nothing(
        self, dummy_classification_dataset: SyntheticClassificationDataset
    ) -> None:
        attack = LabelFlipAttack(client_id="c1", source_class=1, target_class=9, poison_rate=0.0)
        wrapped = attack.apply_data_attack(dummy_classification_dataset)

        for i in range(len(wrapped)):
            _, orig_y = dummy_classification_dataset[i]
            _, poisoned_y = wrapped[i]
            assert poisoned_y == orig_y

    def test_non_source_classes_never_flipped(
        self, dummy_classification_dataset: SyntheticClassificationDataset
    ) -> None:
        attack = LabelFlipAttack(client_id="c1", source_class=0, target_class=5, poison_rate=1.0)
        wrapped = attack.apply_data_attack(dummy_classification_dataset)

        for i in range(len(wrapped)):
            _, orig_y = dummy_classification_dataset[i]
            _, poisoned_y = wrapped[i]
            if orig_y != 0:
                assert poisoned_y == orig_y

    def test_deterministic_with_seed(
        self, dummy_classification_dataset: SyntheticClassificationDataset
    ) -> None:
        wrapped1 = LabelFlipDatasetWrapper(
            dummy_classification_dataset,
            source_class=2,
            target_class=8,
            poison_rate=0.5,
            seed=1234,
        )
        wrapped2 = LabelFlipDatasetWrapper(
            dummy_classification_dataset,
            source_class=2,
            target_class=8,
            poison_rate=0.5,
            seed=1234,
        )

        labels1 = [wrapped1[i][1] for i in range(len(wrapped1))]
        labels2 = [wrapped2[i][1] for i in range(len(wrapped2))]
        assert labels1 == labels2

    def test_tensor_label_preservation(self) -> None:
        samples = [(torch.zeros(2), torch.tensor(3, dtype=torch.int64))]
        dataset = SyntheticClassificationDataset(samples)
        wrapped = LabelFlipDatasetWrapper(dataset, source_class=3, target_class=0, poison_rate=1.0)
        _, y = wrapped[0]
        assert isinstance(y, torch.Tensor)
        assert y.item() == 0
        assert y.dtype == torch.int64


# ---------------------------------------------------------------------------
# Section C — BackdoorDatasetWrapper & BackdoorAttack
# ---------------------------------------------------------------------------

class TestBackdoorAttack:
    """Tests for backdoor trigger injection mechanics."""

    def test_constructor_and_to_dict(self) -> None:
        attack = BackdoorAttack(
            client_id="c2",
            target_label=0,
            poison_rate=0.4,
            trigger_size=4,
            trigger_value=2.0,
            seed=77,
        )
        assert attack.client_id == "c2"
        assert attack.target_label == 0
        assert attack.poison_rate == 0.4
        assert attack.trigger_size == 4
        assert attack.trigger_value == 2.0
        assert attack.intensity == 0.4
        assert attack.attack_type == "backdoor"
        assert attack.to_dict() == {
            "client_id": "c2",
            "attack_type": "backdoor",
            "intensity": 0.4,
        }

    def test_trigger_geometry_3d(
        self, dummy_image_dataset_3d: SyntheticClassificationDataset
    ) -> None:
        trigger_val = 1.0
        size = 3
        attack = BackdoorAttack(
            client_id="c2",
            target_label=9,
            poison_rate=1.0,
            trigger_size=size,
            trigger_value=trigger_val,
        )
        wrapped = attack.apply_data_attack(dummy_image_dataset_3d)

        for i in range(len(wrapped)):
            x_poisoned, y_poisoned = wrapped[i]
            assert y_poisoned == 9

            # Verify top-left trigger is stamped across all channels
            trigger_region = x_poisoned[:, :size, :size]
            assert torch.all(trigger_region == trigger_val)

            # Verify non-trigger area remains clean (0.0)
            assert torch.all(x_poisoned[:, size:, size:] == 0.0)

    def test_trigger_geometry_2d(
        self, dummy_image_dataset_2d: SyntheticClassificationDataset
    ) -> None:
        trigger_val = 5.0
        size = 4
        attack = BackdoorAttack(
            client_id="c2",
            target_label=0,
            poison_rate=1.0,
            trigger_size=size,
            trigger_value=trigger_val,
        )
        wrapped = attack.apply_data_attack(dummy_image_dataset_2d)

        for i in range(len(wrapped)):
            x_poisoned, y_poisoned = wrapped[i]
            assert y_poisoned == 0
            assert torch.all(x_poisoned[:size, :size] == trigger_val)
            assert torch.all(x_poisoned[size:, size:] == 0.0)

    def test_cloning_prevents_in_place_mutation_of_underlying_data(
        self, dummy_image_dataset_3d: SyntheticClassificationDataset
    ) -> None:
        attack = BackdoorAttack(
            client_id="c2",
            target_label=9,
            poison_rate=1.0,
            trigger_size=3,
            trigger_value=1.0,
        )
        wrapped = attack.apply_data_attack(dummy_image_dataset_3d)

        # Trigger item access
        _ = wrapped[0]

        # Underlying original sample must remain all zeros
        orig_x, _ = dummy_image_dataset_3d[0]
        assert torch.all(orig_x == 0.0)

    def test_poison_rate_zero_poisons_nothing(
        self, dummy_image_dataset_3d: SyntheticClassificationDataset
    ) -> None:
        attack = BackdoorAttack(
            client_id="c2",
            target_label=9,
            poison_rate=0.0,
            trigger_size=3,
        )
        wrapped = attack.apply_data_attack(dummy_image_dataset_3d)

        for i in range(len(wrapped)):
            x, y = wrapped[i]
            orig_x, orig_y = dummy_image_dataset_3d[i]
            assert y == orig_y
            assert torch.all(x == orig_x)


# ---------------------------------------------------------------------------
# Section D — Update-Plane Fallbacks
# ---------------------------------------------------------------------------

class TestUpdatePlaneFallbacks:
    """Ensure data attacks return parameter deltas unmodified."""

    def test_label_flip_apply_update_attack_is_identity(self) -> None:
        attack = LabelFlipAttack(client_id="c1", source_class=0, target_class=1)
        delta: Dict[str, np.ndarray] = {
            "w": np.ones((4, 4), dtype=np.float32),
            "b": np.zeros((4,), dtype=np.float32),
        }
        result = attack.apply_update_attack(delta)
        assert result is delta
        np.testing.assert_array_equal(result["w"], delta["w"])

    def test_backdoor_apply_update_attack_is_identity(self) -> None:
        attack = BackdoorAttack(client_id="c2")
        delta: Dict[str, np.ndarray] = {
            "layer.weight": np.random.randn(8, 8).astype(np.float32),
        }
        result = attack.apply_update_attack(delta)
        assert result is delta


# ---------------------------------------------------------------------------
# Section E — Contract Invariants & Validation
# ---------------------------------------------------------------------------

class TestContractInvariantsAndValidation:
    """Contract rules, lack of is_malicious, and boundary checks."""

    @pytest.mark.parametrize("attack_cls, kwargs", [
        (LabelFlipAttack, {"client_id": "c1", "source_class": 0, "target_class": 1}),
        (BackdoorAttack,  {"client_id": "c2"}),
    ])
    def test_no_is_malicious_attribute_or_dict_key(self, attack_cls, kwargs) -> None:
        attack = attack_cls(**kwargs)
        assert not hasattr(attack, "is_malicious")
        d = attack.to_dict()
        assert "is_malicious" not in d
        assert set(d.keys()) == {"client_id", "attack_type", "intensity"}

    def test_wrapper_no_is_malicious_attribute(
        self, dummy_classification_dataset: SyntheticClassificationDataset
    ) -> None:
        w1 = LabelFlipDatasetWrapper(dummy_classification_dataset, 0, 1)
        w2 = BackdoorDatasetWrapper(dummy_classification_dataset, 0)
        assert not hasattr(w1, "is_malicious")
        assert not hasattr(w2, "is_malicious")

    def test_label_flip_invalid_poison_rate_raises(self) -> None:
        with pytest.raises(ValueError, match=r"poison_rate must be in \[0.0, 1.0\]"):
            LabelFlipAttack(client_id="c1", source_class=0, target_class=1, poison_rate=1.5)

        with pytest.raises(ValueError, match=r"poison_rate must be in \[0.0, 1.0\]"):
            LabelFlipDatasetWrapper(SyntheticClassificationDataset([]), 0, 1, poison_rate=-0.1)

    def test_backdoor_invalid_trigger_size_raises(self) -> None:
        with pytest.raises(ValueError, match="trigger_size must be positive"):
            BackdoorAttack(client_id="c2", trigger_size=0)

        with pytest.raises(ValueError, match="trigger_size must be positive"):
            BackdoorDatasetWrapper(SyntheticClassificationDataset([]), trigger_size=-2)

    def test_wrapper_invalid_dataset_type_raises(self) -> None:
        with pytest.raises(TypeError, match="dataset must implement __len__ and __getitem__"):
            LabelFlipDatasetWrapper("not_a_dataset", 0, 1)  # type: ignore

        with pytest.raises(TypeError, match="dataset must implement __len__ and __getitem__"):
            BackdoorDatasetWrapper(None)  # type: ignore
