from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from datasets.change_dataset import ChangeDetectionDataset
from utils.augmentations import build_train_transforms, build_val_transforms


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def build_change_dataset(
    data_root: str | Path,
    split: str,
    modalities: Sequence[str] | None = None,
    modality_aliases: dict[str, Sequence[str]] | None = None,
    transform=None,
    normalize: bool = False,
    channel_means: Sequence[float] | None = None,
    channel_stds: Sequence[float] | None = None,
    recursive: bool = True,
    skip_failed_samples: bool = True,
) -> ChangeDetectionDataset:
    split_dir = Path(data_root) / split
    return ChangeDetectionDataset(
        split_dir=split_dir,
        modalities=modalities,
        modality_aliases=modality_aliases,
        transform=transform,
        normalize=normalize,
        channel_means=channel_means,
        channel_stds=channel_stds,
        recursive=recursive,
        skip_failed_samples=skip_failed_samples,
    )


def build_dataloader(
    dataset: Dataset,
    batch_size: int = 2,
    shuffle: bool = False,
    num_workers: int = 0,
    seed: int = 42,
    sample_limit: int | None = None,
    pin_memory: bool | None = None,
) -> DataLoader:
    if sample_limit is not None and sample_limit < len(dataset):
        subset_indices = _sample_subset_indices(len(dataset), sample_limit, seed)
        dataset = Subset(dataset, subset_indices)

    generator = torch.Generator()
    generator.manual_seed(seed)

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=bool(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=False,
    )


def build_train_dataloader(
    data_root: str | Path,
    batch_size: int = 2,
    num_workers: int = 0,
    seed: int = 42,
    sample_limit: int | None = None,
    crop_size: int | None = None,
    modalities: Sequence[str] | None = None,
    modality_aliases: dict[str, Sequence[str]] | None = None,
    normalize: bool = False,
    channel_means: Sequence[float] | None = None,
    channel_stds: Sequence[float] | None = None,
) -> DataLoader:
    dataset = build_change_dataset(
        data_root=data_root,
        split="train",
        modalities=modalities,
        modality_aliases=modality_aliases,
        transform=build_train_transforms(crop_size=crop_size),
        normalize=normalize,
        channel_means=channel_means,
        channel_stds=channel_stds,
    )
    return build_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        seed=seed,
        sample_limit=sample_limit,
    )


def build_val_dataloader(
    data_root: str | Path,
    batch_size: int = 2,
    num_workers: int = 0,
    seed: int = 42,
    sample_limit: int | None = None,
    crop_size: int | None = None,
    split: str = "val",
    modalities: Sequence[str] | None = None,
    modality_aliases: dict[str, Sequence[str]] | None = None,
    normalize: bool = False,
    channel_means: Sequence[float] | None = None,
    channel_stds: Sequence[float] | None = None,
) -> DataLoader:
    dataset = build_change_dataset(
        data_root=data_root,
        split=split,
        modalities=modalities,
        modality_aliases=modality_aliases,
        transform=build_val_transforms(crop_size=crop_size),
        normalize=normalize,
        channel_means=channel_means,
        channel_stds=channel_stds,
    )
    return build_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        seed=seed,
        sample_limit=sample_limit,
    )


def _sample_subset_indices(dataset_size: int, sample_limit: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    indices = rng.sample(range(dataset_size), k=sample_limit)
    indices.sort()
    return indices
