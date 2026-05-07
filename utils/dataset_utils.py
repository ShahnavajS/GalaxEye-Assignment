from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
MASK_MODALITY_HINTS = ("mask", "target", "label", "labels", "annotation", "gt")


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES


def is_mask_modality(modality_name: str) -> bool:
    lower_name = modality_name.lower()
    return any(hint in lower_name for hint in MASK_MODALITY_HINTS)


def infer_modality_name(split_dir: Path, file_path: Path) -> str:
    relative_path = file_path.relative_to(split_dir)
    if len(relative_path.parts) == 1:
        return "root"
    return relative_path.parts[0]


def sample_id_from_path(file_path: Path) -> str:
    return file_path.stem


def index_split_directory(
    split_dir: Path,
    recursive: bool = True,
    logger: logging.Logger | None = None,
) -> tuple[dict[str, dict[str, Path]], list[str]]:
    logger = logger or logging.getLogger(__name__)
    pattern = "**/*" if recursive else "*"
    sample_index: dict[str, dict[str, Path]] = {}
    warnings: list[str] = []

    for file_path in sorted(split_dir.glob(pattern)):
        if not is_image_file(file_path):
            continue

        sample_id = sample_id_from_path(file_path)
        modality_name = infer_modality_name(split_dir, file_path)
        sample_files = sample_index.setdefault(sample_id, {})

        if modality_name in sample_files:
            message = (
                f"Duplicate modality '{modality_name}' for sample '{sample_id}' in "
                f"split '{split_dir.name}'. Keeping the first file."
            )
            warnings.append(message)
            logger.warning(message)
            continue

        sample_files[modality_name] = file_path

    return sample_index, warnings


def get_expected_modalities(sample_index: dict[str, dict[str, Path]]) -> list[str]:
    modality_names = set()
    for sample_files in sample_index.values():
        modality_names.update(sample_files)
    return sorted(modality_names)


def find_missing_modalities(
    sample_index: dict[str, dict[str, Path]],
    expected_modalities: list[str] | None = None,
) -> list[dict[str, object]]:
    expected = expected_modalities or get_expected_modalities(sample_index)
    missing: list[dict[str, object]] = []

    for sample_id, sample_files in sorted(sample_index.items()):
        absent = [name for name in expected if name not in sample_files]
        if absent:
            missing.append(
                {
                    "sample_id": sample_id,
                    "missing_modalities": absent,
                }
            )

    return missing


def load_image_array(
    image_path: Path,
    logger: logging.Logger | None = None,
) -> np.ndarray | None:
    logger = logger or logging.getLogger(__name__)
    try:
        with Image.open(image_path) as image:
            return np.array(image)
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        logger.warning("Failed to read %s: %s", image_path, exc)
        return None


def describe_array(array: np.ndarray) -> dict[str, object]:
    array = np.asarray(array)
    if array.ndim == 2:
        channels = 1
    elif array.ndim == 3:
        channels = int(array.shape[2])
    else:
        channels = 0

    return {
        "shape": tuple(int(size) for size in array.shape),
        "spatial_shape": tuple(int(size) for size in array.shape[:2]),
        "channels": channels,
        "dtype": str(array.dtype),
        "min": float(array.min()),
        "max": float(array.max()),
    }
