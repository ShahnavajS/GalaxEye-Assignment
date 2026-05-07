from __future__ import annotations

from typing import Iterable

import numpy as np

RAW_LABEL_NAMES = {
    0: "background",
    1: "intact",
    2: "damaged",
    3: "destroyed",
}

BINARY_LABEL_NAMES = {
    0: "no_change",
    1: "change",
}

LABEL_REMAP = {
    0: 0,
    1: 0,
    2: 1,
    3: 1,
}

_RAW_ALLOWED_VALUES = set(LABEL_REMAP)
_BINARY_ALLOWED_VALUES = {0, 1}
_REMAP_LUT = np.zeros(256, dtype=np.uint8)
for source_value, target_value in LABEL_REMAP.items():
    _REMAP_LUT[source_value] = target_value


def get_unique_values(mask: np.ndarray) -> list[int]:
    values = np.unique(np.asarray(mask))
    return [int(value) for value in values.tolist()]


def validate_mask_values(
    mask: np.ndarray,
    allowed_values: Iterable[int] | None = None,
) -> tuple[bool, list[int]]:
    allowed = set(_RAW_ALLOWED_VALUES if allowed_values is None else allowed_values)
    present = set(get_unique_values(mask))
    invalid = sorted(present - allowed)
    return len(invalid) == 0, invalid


def remap_mask(mask: np.ndarray) -> np.ndarray:
    mask_array = np.asarray(mask)
    is_valid, invalid_values = validate_mask_values(mask_array)
    if not is_valid:
        raise ValueError(f"Mask contains unexpected values: {invalid_values}")
    return _REMAP_LUT[mask_array].astype(np.uint8, copy=False)


def is_binary_mask(mask: np.ndarray) -> bool:
    return set(get_unique_values(mask)).issubset(_BINARY_ALLOWED_VALUES)
