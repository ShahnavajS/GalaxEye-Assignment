from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from project_bootstrap import ensure_local_packages  # noqa: E402

ensure_local_packages()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import Subset  # noqa: E402

from datasets.build import build_change_dataset, build_dataloader  # noqa: E402
from utils.augmentations import build_train_transforms, build_val_transforms  # noqa: E402
from utils.dataset_source import resolve_dataset_root  # noqa: E402
from utils.visualize import plot_sample, save_figure  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the change-detection dataloader on a tiny subset.")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--data-root", type=Path, default=None, help="Optional explicit dataset root.")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--sample-limit", type=int, default=4, help="How many samples to use for validation.")
    parser.add_argument("--batch-size", type=int, default=2, help="Mini-batch size.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count.")
    parser.add_argument("--num-batches", type=int, default=2, help="How many batches to inspect.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--crop-size", type=int, default=512, help="Optional crop size for transforms.")
    parser.add_argument("--normalize", action="store_true", help="Apply channel-wise normalization if stats are given.")
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=None,
        choices=["eo_pre", "eo_post", "sar_pre", "sar_post"],
        help="Optional canonical modality list. Defaults to auto-detected modalities.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/dataloader_checks"),
        help="Where to save visualized batch samples.",
    )
    parser.add_argument("--overlay-alpha", type=float, default=0.35, help="Overlay opacity for saved figures.")
    return parser.parse_args()


def setup_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    return logging.getLogger("check_dataloader")


def to_plot_array(modality_tensor: torch.Tensor | None) -> np.ndarray | None:
    if modality_tensor is None:
        return None

    array = modality_tensor.detach().cpu().numpy()
    if array.ndim == 3 and array.shape[0] == 1:
        return array[0]
    if array.ndim == 3:
        return np.transpose(array, (1, 2, 0))
    return array


def unwrap_dataset(dataset) -> object:
    if isinstance(dataset, Subset):
        return dataset.dataset
    return dataset


def validate_mask_binary(mask_tensor: torch.Tensor) -> None:
    mask_values = set(torch.unique(mask_tensor).cpu().tolist())
    if not mask_values.issubset({0.0, 1.0}):
        raise ValueError(f"Mask contains non-binary values: {sorted(mask_values)}")


def build_visualization(dataset, image_tensor: torch.Tensor, mask_tensor: torch.Tensor, sample_id: str, overlay_alpha: float):
    split_modalities = dataset.split_image_by_modality(image_tensor)
    binary_mask = mask_tensor.detach().cpu().numpy()
    if binary_mask.ndim == 3 and binary_mask.shape[0] == 1:
        binary_mask = binary_mask[0]

    return plot_sample(
        eo_pre=to_plot_array(split_modalities.get("eo_pre")),
        eo_post=to_plot_array(split_modalities.get("eo_post")),
        sar_pre=to_plot_array(split_modalities.get("sar_pre")),
        sar_post=to_plot_array(split_modalities.get("sar_post")),
        raw_mask=None,
        remapped_mask=binary_mask,
        sample_id=sample_id,
        overlay_alpha=overlay_alpha,
    )


def main() -> None:
    args = parse_args()
    logger = setup_logger()

    if not 2 <= args.sample_limit <= 10:
        raise ValueError("--sample-limit must stay between 2 and 10 for this validation phase.")

    if args.data_root is not None:
        data_root = args.data_root.resolve()
    else:
        from utils.runtime import load_config  # noqa: E402

        config = load_config(args.config)
        data_root = resolve_dataset_root(config, download_if_missing=True, logger=logger)

    transforms = build_train_transforms(crop_size=args.crop_size) if args.split == "train" else build_val_transforms(crop_size=args.crop_size)
    dataset = build_change_dataset(
        data_root=data_root,
        split=args.split,
        modalities=args.modalities,
        transform=transforms,
        normalize=args.normalize,
    )
    loader = build_dataloader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=args.split == "train",
        num_workers=args.num_workers,
        seed=args.seed,
        sample_limit=args.sample_limit,
    )

    base_dataset = unwrap_dataset(loader.dataset)

    logger.info("Indexed samples in split '%s': %d", args.split, len(dataset))
    logger.info("Validation subset size: %d", min(args.sample_limit, len(dataset)))
    logger.info("Image tensor shape per sample: [%d, H, W]", base_dataset.num_channels)
    logger.info("Mask tensor shape per sample: [1, H, W]")
    logger.info("Channel layout: %s", base_dataset.describe_channels())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    saved_count = 0

    for batch_index, batch in enumerate(loader):
        if batch_index >= args.num_batches:
            break

        images = batch["image"]
        masks = batch["mask"]
        sample_ids = batch["sample_id"]

        if images.ndim != 4:
            raise ValueError(f"Expected image batch shape [B, C, H, W], got {tuple(images.shape)}")
        if masks.ndim != 4:
            raise ValueError(f"Expected mask batch shape [B, 1, H, W], got {tuple(masks.shape)}")

        validate_mask_binary(masks)

        channel_min = images.amin(dim=(0, 2, 3)).cpu().tolist()
        channel_max = images.amax(dim=(0, 2, 3)).cpu().tolist()
        mask_values = torch.unique(masks).cpu().tolist()

        logger.info(
            "Batch %d | image_shape=%s | mask_shape=%s | image_dtype=%s | mask_dtype=%s",
            batch_index,
            tuple(images.shape),
            tuple(masks.shape),
            images.dtype,
            masks.dtype,
        )
        logger.info(
            "Batch %d | image_min=%.4f | image_max=%.4f | channel_min=%s | channel_max=%s | mask_values=%s",
            batch_index,
            float(images.min().item()),
            float(images.max().item()),
            [round(value, 4) for value in channel_min],
            [round(value, 4) for value in channel_max],
            mask_values,
        )

        for sample_offset in range(images.shape[0]):
            if saved_count >= args.sample_limit:
                break

            figure = build_visualization(
                dataset=base_dataset,
                image_tensor=images[sample_offset],
                mask_tensor=masks[sample_offset],
                sample_id=sample_ids[sample_offset],
                overlay_alpha=args.overlay_alpha,
            )
            output_path = args.output_dir / f"batch_{batch_index:02d}_{sample_offset:02d}_{sample_ids[sample_offset]}.png"
            save_figure(figure, output_path)
            logger.info("Saved %s", output_path)
            saved_count += 1


if __name__ == "__main__":
    main()
