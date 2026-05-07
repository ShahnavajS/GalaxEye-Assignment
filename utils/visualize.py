from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from utils.label_utils import remap_mask

RAW_MASK_COLORS = {
    0: (0.0, 0.0, 0.0),
    1: (0.25, 0.7, 0.35),
    2: (0.95, 0.65, 0.15),
    3: (0.85, 0.2, 0.2),
}

BINARY_MASK_COLORS = {
    0: (0.0, 0.0, 0.0),
    1: (0.95, 0.2, 0.2),
}

ERROR_COLORS = {
    "tp": (0.2, 0.8, 0.2),
    "fp": (1.0, 0.8, 0.0),
    "fn": (0.1, 0.6, 1.0),
}


def _percentile_normalize(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32, copy=False)
    low, high = np.percentile(image, [2, 98])
    if high <= low:
        return np.zeros_like(image, dtype=np.float32)
    image = np.clip(image, low, high)
    return (image - low) / (high - low)


def to_display_image(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)

    if array.ndim == 2:
        gray = _percentile_normalize(array)
        return np.repeat(gray[..., None], 3, axis=2)

    if array.ndim == 3 and array.shape[2] == 1:
        gray = _percentile_normalize(array[..., 0])
        return np.repeat(gray[..., None], 3, axis=2)

    if array.ndim == 3 and array.shape[2] >= 3:
        rgb = array[..., :3].astype(np.float32, copy=False)
        channels = [_percentile_normalize(rgb[..., index]) for index in range(3)]
        return np.stack(channels, axis=2)

    raise ValueError(f"Unsupported image shape for visualization: {array.shape}")


def mask_to_color(mask: np.ndarray, remapped: bool = False) -> np.ndarray:
    array = np.asarray(mask)
    palette = BINARY_MASK_COLORS if remapped else RAW_MASK_COLORS
    color_mask = np.zeros((*array.shape, 3), dtype=np.float32)

    for value, color in palette.items():
        color_mask[array == value] = color

    return color_mask


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.35,
    remapped: bool = False,
) -> np.ndarray:
    base = to_display_image(image)
    overlay = mask_to_color(mask, remapped=remapped)

    if remapped:
        foreground = np.asarray(mask) == 1
    else:
        foreground = np.asarray(mask) != 0

    blended = base.copy()
    blended[foreground] = (1.0 - alpha) * base[foreground] + alpha * overlay[foreground]
    return np.clip(blended, 0.0, 1.0)


def probability_to_heatmap(probability_map: np.ndarray, cmap_name: str = "inferno") -> np.ndarray:
    probability_map = np.asarray(probability_map, dtype=np.float32)
    probability_map = np.clip(probability_map, 0.0, 1.0)
    cmap = plt.get_cmap(cmap_name)
    return cmap(probability_map)[..., :3].astype(np.float32)


def overlay_heatmap(
    image: np.ndarray,
    probability_map: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    base = to_display_image(image)
    heatmap = probability_to_heatmap(probability_map)
    return np.clip((1.0 - alpha) * base + alpha * heatmap, 0.0, 1.0)


def error_map_to_color(target_mask: np.ndarray, prediction_mask: np.ndarray) -> np.ndarray:
    target_mask = np.asarray(target_mask) >= 0.5
    prediction_mask = np.asarray(prediction_mask) >= 0.5
    color_map = np.zeros((*target_mask.shape, 3), dtype=np.float32)

    true_positive = np.logical_and(target_mask, prediction_mask)
    false_positive = np.logical_and(~target_mask, prediction_mask)
    false_negative = np.logical_and(target_mask, ~prediction_mask)

    color_map[true_positive] = ERROR_COLORS["tp"]
    color_map[false_positive] = ERROR_COLORS["fp"]
    color_map[false_negative] = ERROR_COLORS["fn"]
    return color_map


def overlay_error_map(
    image: np.ndarray,
    target_mask: np.ndarray,
    prediction_mask: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    base = to_display_image(image)
    error_map = error_map_to_color(target_mask, prediction_mask)
    foreground = np.any(error_map > 0, axis=2)
    blended = base.copy()
    blended[foreground] = (1.0 - alpha) * base[foreground] + alpha * error_map[foreground]
    return np.clip(blended, 0.0, 1.0)


def plot_sample(
    eo_pre: np.ndarray | None = None,
    eo_post: np.ndarray | None = None,
    sar_pre: np.ndarray | None = None,
    sar_post: np.ndarray | None = None,
    raw_mask: np.ndarray | None = None,
    remapped_mask: np.ndarray | None = None,
    prediction_mask: np.ndarray | None = None,
    probability_map: np.ndarray | None = None,
    sample_id: str | None = None,
    overlay_alpha: float = 0.35,
):
    panels: list[tuple[str, np.ndarray]] = []

    if eo_pre is not None:
        panels.append(("EO pre-event", to_display_image(eo_pre)))
    if eo_post is not None:
        panels.append(("EO post-event", to_display_image(eo_post)))
    if sar_pre is not None:
        panels.append(("SAR pre-event", to_display_image(sar_pre)))
    if sar_post is not None:
        panels.append(("SAR post-event", to_display_image(sar_post)))
    if raw_mask is not None:
        panels.append(("Raw mask", mask_to_color(raw_mask, remapped=False)))
    if remapped_mask is not None:
        panels.append(("Remapped mask", mask_to_color(remapped_mask, remapped=True)))
    if prediction_mask is not None:
        panels.append(("Prediction mask", mask_to_color(prediction_mask, remapped=True)))
    if probability_map is not None:
        panels.append(("Probability heatmap", probability_to_heatmap(probability_map)))

    overlay_base = eo_pre if eo_pre is not None else eo_post
    if overlay_base is None:
        overlay_base = sar_pre if sar_pre is not None else sar_post

    if overlay_base is not None and raw_mask is not None:
        panels.append(
            (
                "Raw overlay",
                overlay_mask(overlay_base, raw_mask, alpha=overlay_alpha, remapped=False),
            )
        )
    if overlay_base is not None and remapped_mask is not None:
        panels.append(
            (
                "Target overlay",
                overlay_mask(
                    overlay_base,
                    remapped_mask,
                    alpha=overlay_alpha,
                    remapped=True,
                ),
            )
        )
    if overlay_base is not None and prediction_mask is not None:
        panels.append(
            (
                "Prediction overlay",
                overlay_mask(
                    overlay_base,
                    prediction_mask,
                    alpha=overlay_alpha,
                    remapped=True,
                ),
            )
        )
    if overlay_base is not None and probability_map is not None:
        panels.append(
            (
                "Heatmap overlay",
                overlay_heatmap(
                    overlay_base,
                    probability_map,
                    alpha=overlay_alpha,
                ),
            )
        )
    if overlay_base is not None and remapped_mask is not None and prediction_mask is not None:
        panels.append(
            (
                "Error overlay",
                overlay_error_map(
                    overlay_base,
                    remapped_mask,
                    prediction_mask,
                    alpha=overlay_alpha,
                ),
            )
        )

    if not panels:
        raise ValueError("No inputs were provided for visualization.")

    figure, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
    if len(panels) == 1:
        axes = [axes]

    for axis, (title, image) in zip(axes, panels):
        axis.imshow(image)
        axis.set_title(title)
        axis.axis("off")

    if sample_id:
        figure.suptitle(sample_id, fontsize=14)

    figure.tight_layout()
    return figure


def save_figure(figure, output_path: str | Path, dpi: int = 160) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def ensure_binary_mask(mask: np.ndarray) -> np.ndarray:
    mask_array = np.asarray(mask)
    unique_values = set(np.unique(mask_array).tolist())
    if unique_values.issubset({0, 1}):
        return mask_array.astype(np.uint8, copy=False)
    return remap_mask(mask_array)


def save_contact_sheet(
    image_paths: list[str | Path],
    output_path: str | Path,
    columns: int = 3,
    padding: int = 12,
) -> None:
    paths = [Path(path) for path in image_paths if Path(path).exists()]
    if not paths:
        return

    with ExitStack() as stack:
        images = [stack.enter_context(Image.open(path)).convert("RGB") for path in paths]
        max_width = max(image.width for image in images)
        max_height = max(image.height for image in images)
        columns = max(columns, 1)
        rows = int(np.ceil(len(images) / columns))

        canvas = Image.new(
            "RGB",
            (
                columns * max_width + padding * (columns + 1),
                rows * max_height + padding * (rows + 1),
            ),
            color=(255, 255, 255),
        )

        for index, image in enumerate(images):
            row = index // columns
            column = index % columns
            x = padding + column * (max_width + padding)
            y = padding + row * (max_height + padding)
            canvas.paste(image, (x, y))

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output_path)
