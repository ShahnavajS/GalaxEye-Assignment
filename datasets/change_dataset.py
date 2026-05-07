from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.dataset_utils import is_image_file, load_image_array
from utils.label_utils import remap_mask, validate_mask_values

LOGGER = logging.getLogger(__name__)

CANONICAL_MODALITY_ORDER = ("eo_pre", "eo_post", "sar_pre", "sar_post")

DEFAULT_MODALITY_ALIASES = {
    "eo_pre": ("eo_pre", "eo-pre", "pre_eo", "pre-eo", "eo_pre_event", "pre-event"),
    "eo_post": ("eo_post", "eo-post", "post_eo", "post-eo", "eo_post_event", "post-event-eo"),
    "sar_pre": ("sar_pre", "sar-pre", "pre_sar", "pre-sar", "sar_pre_event", "pre-event-sar"),
    "sar_post": ("sar_post", "sar-post", "post_sar", "post-sar", "sar_post_event", "post-event"),
}

DEFAULT_MASK_ALIASES = ("target", "mask", "label", "labels", "annotation", "gt")


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    modality_paths: dict[str, Path]
    mask_path: Path


def _normalize_token(token: str) -> str:
    return token.lower().replace("_", "-").replace(" ", "")


def _build_alias_lookup(alias_map: dict[str, Sequence[str]]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for canonical_name, aliases in alias_map.items():
        for alias in aliases:
            lookup[_normalize_token(alias)] = canonical_name
    return lookup


class ChangeDetectionDataset(Dataset):
    def __init__(
        self,
        split_dir: str | Path,
        modalities: Sequence[str] | None = None,
        transform: Any | None = None,
        normalize: bool = False,
        channel_means: Sequence[float] | None = None,
        channel_stds: Sequence[float] | None = None,
        recursive: bool = True,
        skip_failed_samples: bool = True,
        max_read_retries: int = 3,
        modality_aliases: dict[str, Sequence[str]] | None = None,
        mask_aliases: Sequence[str] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.split_dir = Path(split_dir)
        self.transform = transform
        self.normalize = normalize
        self.channel_means = tuple(channel_means) if channel_means is not None else None
        self.channel_stds = tuple(channel_stds) if channel_stds is not None else None
        self.recursive = recursive
        self.skip_failed_samples = skip_failed_samples
        self.max_read_retries = max(1, max_read_retries)
        self.logger = logger or LOGGER

        alias_map = modality_aliases or DEFAULT_MODALITY_ALIASES
        self.mask_aliases = {_normalize_token(alias) for alias in (mask_aliases or DEFAULT_MASK_ALIASES)}
        self.alias_lookup = _build_alias_lookup(alias_map)

        discovered_records, discovered_modalities = self._discover_samples()
        self.modalities = self._resolve_modalities(modalities, discovered_modalities)
        self.samples, self.skipped_samples = self._filter_complete_samples(discovered_records)

        if not self.samples:
            raise RuntimeError(f"No valid samples found under {self.split_dir}")

        self.modality_channels = self._infer_modality_channels()
        self.channel_slices = self._build_channel_slices()
        self.num_channels = sum(self.modality_channels.values())
        self._validate_normalization()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        max_attempts = min(self.max_read_retries, len(self.samples))

        for attempt in range(max_attempts):
            sample_index = (index + attempt) % len(self.samples)
            record = self.samples[sample_index]
            try:
                image, mask = self._load_sample_arrays(record)
                image, mask = self._apply_transform(image, mask)
                image = self._prepare_image(image)
                mask = self._prepare_mask(mask)
                return {
                    "image": image,
                    "mask": mask,
                    "sample_id": record.sample_id,
                }
            except Exception as exc:
                if not self.skip_failed_samples or attempt == max_attempts - 1:
                    raise RuntimeError(f"Failed to load sample '{record.sample_id}': {exc}") from exc
                self.logger.warning("Skipping sample '%s': %s", record.sample_id, exc)

        raise RuntimeError("Unable to fetch a valid sample.")

    def describe_channels(self) -> list[dict[str, int | str]]:
        description: list[dict[str, int | str]] = []
        for modality_name in self.modalities:
            channel_slice = self.channel_slices[modality_name]
            description.append(
                {
                    "name": modality_name,
                    "start": int(channel_slice.start),
                    "end": int(channel_slice.stop),
                    "channels": int(channel_slice.stop - channel_slice.start),
                }
            )
        return description

    def split_image_by_modality(self, image: torch.Tensor | np.ndarray) -> dict[str, torch.Tensor | np.ndarray]:
        if isinstance(image, np.ndarray):
            channels_first = image if image.shape[0] == self.num_channels else np.transpose(image, (2, 0, 1))
        else:
            channels_first = image

        split: dict[str, torch.Tensor | np.ndarray] = {}
        for modality_name, channel_slice in self.channel_slices.items():
            split[modality_name] = channels_first[channel_slice, ...]
        return split

    def _discover_samples(self) -> tuple[list[SampleRecord], list[str]]:
        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split directory does not exist: {self.split_dir}")

        grouped_modalities: dict[str, dict[str, Path]] = {}
        grouped_masks: dict[str, Path] = {}
        pattern = "**/*" if self.recursive else "*"

        for file_path in sorted(self.split_dir.glob(pattern)):
            if not is_image_file(file_path):
                continue

            sample_id = file_path.stem
            role = self._infer_role(file_path)
            if role is None:
                continue

            if role == "mask":
                if sample_id in grouped_masks:
                    self.logger.warning("Duplicate mask for sample '%s'. Keeping the first file.", sample_id)
                    continue
                grouped_masks[sample_id] = file_path
                continue

            sample_modalities = grouped_modalities.setdefault(sample_id, {})
            if role in sample_modalities:
                self.logger.warning(
                    "Duplicate modality '%s' for sample '%s'. Keeping the first file.",
                    role,
                    sample_id,
                )
                continue
            sample_modalities[role] = file_path

        discovered_modalities = sorted(
            {
                modality_name
                for sample_modalities in grouped_modalities.values()
                for modality_name in sample_modalities
            },
            key=lambda name: CANONICAL_MODALITY_ORDER.index(name) if name in CANONICAL_MODALITY_ORDER else len(CANONICAL_MODALITY_ORDER),
        )

        records: list[SampleRecord] = []
        for sample_id, sample_modalities in sorted(grouped_modalities.items()):
            mask_path = grouped_masks.get(sample_id)
            if mask_path is None:
                continue
            records.append(
                SampleRecord(
                    sample_id=sample_id,
                    modality_paths=sample_modalities,
                    mask_path=mask_path,
                )
            )

        return records, discovered_modalities

    def _resolve_modalities(
        self,
        requested_modalities: Sequence[str] | None,
        discovered_modalities: list[str],
    ) -> list[str]:
        if requested_modalities is None:
            modalities = [name for name in CANONICAL_MODALITY_ORDER if name in discovered_modalities]
        else:
            modalities = [name for name in requested_modalities if name in CANONICAL_MODALITY_ORDER]

        if not modalities:
            raise RuntimeError(
                "No usable modalities were found. Pass explicit modality names or update the alias mapping."
            )

        return modalities

    def _filter_complete_samples(
        self,
        discovered_records: list[SampleRecord],
    ) -> tuple[list[SampleRecord], list[dict[str, Any]]]:
        complete_samples: list[SampleRecord] = []
        skipped_samples: list[dict[str, Any]] = []

        for record in discovered_records:
            missing_modalities = [name for name in self.modalities if name not in record.modality_paths]
            if missing_modalities:
                skipped_samples.append(
                    {
                        "sample_id": record.sample_id,
                        "missing_modalities": missing_modalities,
                    }
                )
                continue

            complete_samples.append(
                SampleRecord(
                    sample_id=record.sample_id,
                    modality_paths={name: record.modality_paths[name] for name in self.modalities},
                    mask_path=record.mask_path,
                )
            )

        return complete_samples, skipped_samples

    def _infer_modality_channels(self) -> dict[str, int]:
        channel_counts: dict[str, int] = {}

        for modality_name in self.modalities:
            for record in self.samples:
                array = load_image_array(record.modality_paths[modality_name], logger=self.logger)
                if array is None:
                    continue
                channel_counts[modality_name] = self._channel_count(array)
                break

        if set(channel_counts) != set(self.modalities):
            missing = [name for name in self.modalities if name not in channel_counts]
            raise RuntimeError(f"Could not infer channel counts for modalities: {missing}")

        return channel_counts

    def _build_channel_slices(self) -> dict[str, slice]:
        channel_slices: dict[str, slice] = {}
        start = 0
        for modality_name in self.modalities:
            width = self.modality_channels[modality_name]
            channel_slices[modality_name] = slice(start, start + width)
            start += width
        return channel_slices

    def _validate_normalization(self) -> None:
        if not self.normalize:
            return

        if self.channel_means is None or self.channel_stds is None:
            return

        if len(self.channel_means) != self.num_channels or len(self.channel_stds) != self.num_channels:
            raise ValueError(
                "Normalization statistics must match the concatenated channel count. "
                f"Expected {self.num_channels} values."
            )

    def _infer_role(self, file_path: Path) -> str | None:
        relative_parts = file_path.relative_to(self.split_dir).parts[:-1]
        for part in relative_parts:
            normalized = _normalize_token(part)
            if normalized in self.mask_aliases:
                return "mask"
            if normalized in self.alias_lookup:
                return self.alias_lookup[normalized]
        return None

    def _load_sample_arrays(self, record: SampleRecord) -> tuple[np.ndarray, np.ndarray]:
        modality_arrays: list[np.ndarray] = []
        spatial_shape: tuple[int, int] | None = None

        for modality_name in self.modalities:
            array = load_image_array(record.modality_paths[modality_name], logger=self.logger)
            if array is None:
                raise OSError(f"Failed to read {record.modality_paths[modality_name]}")

            if array.ndim == 2:
                array = array[..., None]
            elif array.ndim != 3:
                raise ValueError(f"Unsupported shape for modality '{modality_name}': {array.shape}")

            current_shape = tuple(int(size) for size in array.shape[:2])
            if spatial_shape is None:
                spatial_shape = current_shape
            elif current_shape != spatial_shape:
                raise ValueError(
                    f"Spatial mismatch for sample '{record.sample_id}': {current_shape} vs {spatial_shape}"
                )

            modality_arrays.append(array)

        mask = load_image_array(record.mask_path, logger=self.logger)
        if mask is None:
            raise OSError(f"Failed to read {record.mask_path}")
        if mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask[..., 0]
        if mask.ndim != 2:
            raise ValueError(f"Mask must be 2D after loading, got shape {mask.shape}")

        is_valid_mask, invalid_values = validate_mask_values(mask)
        if not is_valid_mask:
            raise ValueError(f"Unexpected mask values: {invalid_values}")

        remapped_mask = remap_mask(mask)
        image = np.concatenate(modality_arrays, axis=2)
        return image, remapped_mask

    def _apply_transform(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]:
        if self.transform is None:
            return image, mask

        transformed = self.transform(image=image, mask=mask)
        return transformed["image"], transformed["mask"]

    def _prepare_image(self, image: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(image, torch.Tensor):
            image_tensor = image.float()
        else:
            image_array = self._scale_to_unit_range(image)
            if self.normalize and self.channel_means is not None and self.channel_stds is not None:
                mean = np.asarray(self.channel_means, dtype=np.float32).reshape(1, 1, -1)
                std = np.asarray(self.channel_stds, dtype=np.float32).reshape(1, 1, -1)
                image_array = (image_array - mean) / np.clip(std, a_min=1e-6, a_max=None)
            image_tensor = torch.from_numpy(np.ascontiguousarray(np.transpose(image_array, (2, 0, 1)))).float()

        return image_tensor

    def _prepare_mask(self, mask: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(mask, torch.Tensor):
            mask_tensor = mask.float()
        else:
            mask_tensor = torch.from_numpy(np.ascontiguousarray(mask)).float()

        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0)
        elif mask_tensor.ndim == 3 and mask_tensor.shape[-1] == 1:
            mask_tensor = mask_tensor.permute(2, 0, 1)

        return mask_tensor

    @staticmethod
    def _scale_to_unit_range(image: np.ndarray) -> np.ndarray:
        array = np.asarray(image)
        if array.ndim == 2:
            array = array[..., None]

        if np.issubdtype(array.dtype, np.integer):
            denominator = float(np.iinfo(array.dtype).max)
            return array.astype(np.float32) / max(denominator, 1.0)

        return array.astype(np.float32)

    @staticmethod
    def _channel_count(array: np.ndarray) -> int:
        if array.ndim == 2:
            return 1
        if array.ndim == 3:
            return int(array.shape[2])
        raise ValueError(f"Unsupported array shape: {array.shape}")
