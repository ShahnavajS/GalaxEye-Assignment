from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from project_bootstrap import ensure_local_packages

ensure_local_packages()

import cv2
import numpy as np
import torch
from torch.utils.data import Subset

from datasets.build import build_change_dataset, build_dataloader
from models import build_segmentation_model
from utils.augmentations import build_val_transforms
from utils.checkpoint import read_checkpoint
from utils.dataset_source import resolve_dataset_root
from utils.device import ensure_cuda, log_device_info, log_gpu_memory, resolve_amp_enabled, resolve_device, set_seed
from utils.error_analysis import (
    attach_artifact_path,
    build_generalization_report,
    rank_cases,
    summarize_by_scene,
    summarize_case,
)
from utils.experiment import build_run_directories
from utils.logger import create_logger, save_json
from utils.reporting import save_rows_csv
from utils.runtime import (
    load_config,
    resolve_checkpoint_config,
    resolve_checkpoint_threshold,
    resolve_experiment_name,
)
from utils.visualize import plot_sample, save_contact_sheet, save_figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference and save qualitative outputs.")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


def to_plot_array(modality_tensor: torch.Tensor | np.ndarray | None) -> np.ndarray | None:
    if modality_tensor is None:
        return None
    if isinstance(modality_tensor, torch.Tensor):
        array = modality_tensor.detach().cpu().numpy()
    else:
        array = modality_tensor

    if array.ndim == 3 and array.shape[0] == 1:
        return array[0]
    if array.ndim == 3 and array.shape[0] in {1, 3}:
        return np.transpose(array, (1, 2, 0))
    return array


def save_mask(mask_path: Path, mask: np.ndarray) -> None:
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(mask_path), (mask.astype(np.uint8) * 255))


def build_ranked_groups(cases: list[dict[str, float | int | str]], top_k: int) -> dict[str, list[dict[str, float | int | str]]]:
    return {
        "false_positives": rank_cases(cases, key="false_positive_pixels", top_k=top_k),
        "false_negatives": rank_cases(cases, key="false_negative_pixels", top_k=top_k),
        "uncertain": rank_cases(cases, key="uncertain_ratio", top_k=top_k),
        "worst_iou": rank_cases(cases, key="iou", top_k=top_k, reverse=False),
        "best_iou": rank_cases(cases, key="iou", top_k=top_k, reverse=True),
    }


def copy_ranked_examples(group_name: str, items: list[dict[str, float | int | str]], output_dir: Path) -> list[str]:
    group_dir = output_dir / group_name
    group_dir.mkdir(parents=True, exist_ok=True)
    copied_paths: list[str] = []
    for item in items:
        source = Path(str(item["grid_path"]))
        if not source.exists():
            continue
        destination = group_dir / source.name
        shutil.copy2(source, destination)
        copied_paths.append(str(destination))
    return copied_paths


def select_split_subset_indices(config: dict[str, Any], checkpoint: dict[str, Any], split: str) -> list[int] | None:
    if split == str(config["data"]["train_split"]):
        return checkpoint.get("train_subset_indices")
    if split == str(config["data"]["val_split"]):
        return checkpoint.get("val_subset_indices")
    return None


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config["project"]["seed"]), deterministic=bool(config["project"].get("deterministic", True)))
    device = resolve_device(args.device)
    ensure_cuda(device, require_cuda=bool(args.require_cuda or config["project"].get("require_cuda", False)))

    checkpoint = read_checkpoint(args.checkpoint, map_location=device)
    checkpoint_config = resolve_checkpoint_config(config, checkpoint)
    experiment_name = resolve_experiment_name(config, checkpoint_path=args.checkpoint)
    output_root = Path(args.output_dir or config["paths"]["output_root"])
    directories = build_run_directories(output_root=output_root, experiment_name=experiment_name, path_config=config["paths"])
    run_root = directories["inference_dir"]
    grid_dir = run_root / "grids"
    mask_dir = run_root / "masks"
    montage_dir = run_root / "montages"
    failure_dir = run_root / "failure_examples"
    logger, _ = create_logger(run_root / "logs", name="inference")
    log_device_info(logger, device)
    dataset_root = resolve_dataset_root(config, download_if_missing=True, logger=logger)

    split = args.split or str(config["data"]["val_split"])
    image_size = int(args.image_size or config["data"]["image_size"])
    sample_limit = args.sample_limit if args.sample_limit is not None else config["inference"]["max_visualizations"]
    threshold = resolve_checkpoint_threshold(args.threshold, config, checkpoint)
    use_amp = resolve_amp_enabled(bool(config.get("inference", {}).get("amp", config["training"].get("amp", False))), device)

    dataset = build_change_dataset(
        data_root=dataset_root,
        split=split,
        modalities=checkpoint_config["modalities"],
        transform=build_val_transforms(crop_size=image_size),
    )
    model = build_segmentation_model(
        architecture=checkpoint_config["architecture"],
        encoder_name=checkpoint_config["encoder_name"],
        encoder_weights=checkpoint_config["encoder_weights"],
        in_channels=dataset.num_channels,
        classes=checkpoint_config["output_classes"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    subset_indices = select_split_subset_indices(config, checkpoint, split)
    inference_dataset = dataset if not subset_indices else Subset(dataset, subset_indices)
    loader = build_dataloader(
        dataset=inference_dataset,
        batch_size=int(config["data"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["data"].get("num_workers", 0)),
        seed=int(config["project"]["seed"]),
        sample_limit=None if subset_indices else sample_limit,
    )

    uncertainty_band = float(config["inference"]["uncertainty_band"])
    summaries: list[dict[str, float | int | str]] = []
    autocast_enabled = bool(use_amp and device.type == "cuda")

    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            sample_ids = batch["sample_id"]

            with torch.autocast(device_type=device.type, enabled=autocast_enabled):
                logits = model(images)

            probabilities = torch.sigmoid(logits)
            predictions = (probabilities >= threshold).float()

            for sample_offset in range(images.shape[0]):
                sample_id = sample_ids[sample_offset]
                probability_map = probabilities[sample_offset, 0].detach().cpu().numpy()
                prediction_map = predictions[sample_offset, 0].detach().cpu().numpy()
                target_map = masks[sample_offset, 0].detach().cpu().numpy()

                save_mask(mask_dir / f"{sample_id}.png", prediction_map)

                split_modalities = dataset.split_image_by_modality(images[sample_offset].detach().cpu())
                figure = plot_sample(
                    eo_pre=to_plot_array(split_modalities.get("eo_pre")),
                    eo_post=to_plot_array(split_modalities.get("eo_post")),
                    sar_pre=to_plot_array(split_modalities.get("sar_pre")),
                    sar_post=to_plot_array(split_modalities.get("sar_post")),
                    remapped_mask=target_map,
                    prediction_mask=prediction_map,
                    probability_map=probability_map,
                    sample_id=sample_id,
                )
                grid_path = grid_dir / f"{sample_id}.png"
                save_figure(figure, grid_path)

                summary = summarize_case(
                    sample_id=sample_id,
                    probabilities=probability_map,
                    targets=target_map,
                    threshold=threshold,
                    uncertainty_band=uncertainty_band,
                )
                summary = attach_artifact_path(summary, "grid_path", grid_path)
                summary = attach_artifact_path(summary, "mask_path", mask_dir / f"{sample_id}.png")
                summaries.append(summary)

    top_k = int(config["inference"]["top_k_failures"])
    ranked_groups = build_ranked_groups(summaries, top_k=top_k)
    scene_summaries = summarize_by_scene(summaries)
    generalization_report = build_generalization_report(summaries, scene_summaries, top_k=top_k)

    montage_dir.mkdir(parents=True, exist_ok=True)
    failure_dir.mkdir(parents=True, exist_ok=True)
    for group_name, items in ranked_groups.items():
        copied_paths = copy_ranked_examples(group_name, items, failure_dir)
        save_contact_sheet(
            image_paths=copied_paths,
            output_path=montage_dir / f"{group_name}.png",
            columns=int(config["inference"]["grid_columns"]),
        )

    save_rows_csv(run_root / "case_summary.csv", summaries)
    save_rows_csv(run_root / "scene_summary.csv", scene_summaries)
    save_rows_csv(run_root / "top_false_positives.csv", ranked_groups["false_positives"])
    save_rows_csv(run_root / "top_false_negatives.csv", ranked_groups["false_negatives"])
    save_rows_csv(run_root / "best_iou_cases.csv", ranked_groups["best_iou"])
    save_rows_csv(run_root / "worst_iou_cases.csv", ranked_groups["worst_iou"])
    save_json(
        run_root / "failure_report.json",
        {
            **ranked_groups,
            "all_cases": summaries,
            "scene_summary": scene_summaries,
            "generalization_report": generalization_report,
            "threshold": threshold,
            "split": split,
        },
    )

    logger.info("Using checkpoint subset: %s", bool(subset_indices))
    logger.info("Inference threshold: %.2f", threshold)
    logger.info("Saved inference grids to %s", grid_dir)
    logger.info("Saved prediction masks to %s", mask_dir)
    logger.info("Saved failure montages to %s", montage_dir)
    logger.info("Saved failure report to %s", run_root / "failure_report.json")
    log_gpu_memory(logger, device, prefix=f"Inference {split} GPU")


if __name__ == "__main__":
    main()
