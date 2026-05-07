from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from project_bootstrap import ensure_local_packages  # noqa: E402

ensure_local_packages()

import numpy as np  # noqa: E402

from utils.dataset_utils import (  # noqa: E402
    index_split_directory,
    is_mask_modality,
    load_image_array,
)
from utils.dataset_source import resolve_dataset_root  # noqa: E402
from utils.label_utils import (  # noqa: E402
    get_unique_values,
    is_binary_mask,
    remap_mask,
    validate_mask_values,
)
from utils.visualize import plot_sample, save_figure  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run quick dataset sanity checks and save sample plots.")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--data-root", type=Path, default=None, help="Optional explicit dataset root.")
    parser.add_argument("--split", type=str, default="train", help="Split to sample from.")
    parser.add_argument("--num-samples", type=int, default=4, help="Number of random samples to inspect.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sample selection.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/sanity_checks"),
        help="Directory to save visualization panels.",
    )
    parser.add_argument("--overlay-alpha", type=float, default=0.35, help="Overlay opacity.")
    return parser.parse_args()


def setup_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    return logging.getLogger("sanity_check")


def map_modalities_for_plot(modality_arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray | None]:
    plot_inputs = {
        "eo_pre": None,
        "eo_post": None,
        "sar_pre": None,
        "sar_post": None,
    }

    for modality_name, array in modality_arrays.items():
        lower_name = modality_name.lower()
        if "eo" in lower_name and "pre" in lower_name:
            plot_inputs["eo_pre"] = array
        elif "eo" in lower_name and "post" in lower_name:
            plot_inputs["eo_post"] = array
        elif "sar" in lower_name and "pre" in lower_name:
            plot_inputs["sar_pre"] = array
        elif "sar" in lower_name and "post" in lower_name:
            plot_inputs["sar_post"] = array
        elif lower_name == "pre-event" and array.ndim == 3:
            plot_inputs["eo_pre"] = array
        elif lower_name == "post-event" and array.ndim == 2:
            plot_inputs["sar_post"] = array
        elif lower_name == "post-event" and array.ndim == 3:
            plot_inputs["eo_post"] = array

    return plot_inputs


def validate_spatial_alignment(modality_arrays: dict[str, np.ndarray]) -> tuple[bool, dict[str, list[int]]]:
    spatial_shapes = {name: list(array.shape[:2]) for name, array in modality_arrays.items()}
    is_aligned = len({tuple(shape) for shape in spatial_shapes.values()}) <= 1
    return is_aligned, spatial_shapes


def main() -> None:
    args = parse_args()
    logger = setup_logger()

    if args.data_root is not None:
        data_root = args.data_root.resolve()
    else:
        from utils.runtime import load_config  # noqa: E402

        config = load_config(args.config)
        data_root = resolve_dataset_root(config, download_if_missing=True, logger=logger)

    split_dir = data_root / args.split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory does not exist: {split_dir}")

    sample_index, _ = index_split_directory(split_dir, recursive=True, logger=logger)
    if not sample_index:
        raise RuntimeError(f"No image files found under {split_dir}")

    sample_ids = sorted(sample_index)
    rng = random.Random(args.seed)
    chosen_ids = rng.sample(sample_ids, k=min(args.num_samples, len(sample_ids)))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for sample_id in chosen_ids:
        sample_files = sample_index[sample_id]
        modality_arrays: dict[str, np.ndarray] = {}
        raw_mask = None

        for modality_name, file_path in sorted(sample_files.items()):
            array = load_image_array(file_path, logger=logger)
            if array is None:
                raise RuntimeError(f"Could not load {file_path}")

            modality_arrays[modality_name] = array
            if is_mask_modality(modality_name):
                raw_mask = array

        if raw_mask is None:
            raise RuntimeError(f"No mask modality found for sample {sample_id}")

        is_valid_mask, invalid_values = validate_mask_values(raw_mask)
        if not is_valid_mask:
            raise ValueError(f"Sample {sample_id} has invalid mask values: {invalid_values}")

        remapped_mask = remap_mask(raw_mask)
        if not is_binary_mask(remapped_mask):
            raise ValueError(f"Sample {sample_id} did not become binary after remapping.")

        is_aligned, spatial_shapes = validate_spatial_alignment(modality_arrays)
        if not is_aligned:
            logger.warning("Spatial mismatch for %s: %s", sample_id, spatial_shapes)

        plot_inputs = map_modalities_for_plot(
            {name: array for name, array in modality_arrays.items() if not is_mask_modality(name)}
        )
        figure = plot_sample(
            eo_pre=plot_inputs["eo_pre"],
            eo_post=plot_inputs["eo_post"],
            sar_pre=plot_inputs["sar_pre"],
            sar_post=plot_inputs["sar_post"],
            raw_mask=raw_mask,
            remapped_mask=remapped_mask,
            sample_id=sample_id,
            overlay_alpha=args.overlay_alpha,
        )

        output_path = args.output_dir / f"{sample_id}.png"
        save_figure(figure, output_path)

        logger.info(
            "Saved %s | shapes=%s | raw_mask_values=%s | remapped_values=%s",
            output_path,
            spatial_shapes,
            get_unique_values(raw_mask),
            get_unique_values(remapped_mask),
        )


if __name__ == "__main__":
    main()
