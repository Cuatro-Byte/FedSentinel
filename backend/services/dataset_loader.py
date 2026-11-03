"""
Dataset loader and partitioning infrastructure for FedSentinel simulations.

Provides deterministic, labelled training and evaluation datasets suitable for
Person 1's SimpleCNN (1x28x28 grayscale, 10 classes).
Handles client dataset partitioning via P1's DatasetPartitioner and mapping
to canonical client IDs ('client-0', 'client-1', ...).
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from core.federated.partitioner import DatasetPartitioner


def generate_pattern_dataset(
    sample_count: int = 500,
    num_classes: int = 10,
    seed: int = 42,
) -> TensorDataset:
    """Generate a reproducible, structured 28x28 classification dataset.

    Each class is stamped with a distinct spatial pattern/activation block so that
    the model learns genuine decision boundaries, resulting in real, measurable
    accuracy and loss during evaluation.
    """
    generator = torch.Generator().manual_seed(seed)
    images = torch.randn(sample_count, 1, 28, 28, generator=generator) * 0.1
    labels = torch.randint(0, num_classes, (sample_count,), generator=generator)

    # Embed distinctive class patterns in 28x28 space
    for idx in range(sample_count):
        cls_idx = int(labels[idx].item())
        row_start = 4 + (cls_idx % 5) * 4
        col_start = 4 + (cls_idx // 5) * 8
        images[idx, 0, row_start : row_start + 4, col_start : col_start + 4] += 3.0

    return TensorDataset(images, labels)


def get_default_datasets(
    train_samples: int = 500,
    val_samples: int = 100,
    eval_samples: int = 100,
    num_classes: int = 10,
    seed: int = 42,
) -> tuple[Dataset, Dataset, Dataset]:
    """Return distinct training, validation, and evaluation datasets."""
    train_ds = generate_pattern_dataset(sample_count=train_samples, num_classes=num_classes, seed=seed)
    val_ds = generate_pattern_dataset(sample_count=val_samples, num_classes=num_classes, seed=seed + 2000)
    eval_ds = generate_pattern_dataset(sample_count=eval_samples, num_classes=num_classes, seed=seed + 1000)
    return train_ds, val_ds, eval_ds


def partition_for_clients(
    dataset: Dataset,
    client_count: int,
    seed: int = 42,
) -> dict[str, Dataset]:
    """Partition a dataset among client_count clients using P1's DatasetPartitioner.

    Maps generated partitioner keys ('client_001', 'client_002', ...) to
    canonical FedSentinel client IDs ('client-0', 'client-1', ...).
    """
    if client_count < 1:
        raise ValueError(f"client_count must be at least 1, got {client_count}")

    partitioner = DatasetPartitioner(num_clients=client_count, seed=seed)
    raw_partitions = partitioner.partition(dataset)

    # Sort partitioner keys ('client_001', 'client_002', ...) to ensure deterministic order
    sorted_keys = sorted(raw_partitions.keys())
    canonical_partitions: dict[str, Dataset] = {}

    for idx, raw_key in enumerate(sorted_keys):
        canonical_id = f"client-{idx}"
        canonical_partitions[canonical_id] = raw_partitions[raw_key]

    return canonical_partitions
