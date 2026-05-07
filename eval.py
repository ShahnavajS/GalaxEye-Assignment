from __future__ import annotations

import argparse
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
from utils.metrics import (
    binary_confusion_counts,
    build_thresholds,
    initialize_threshold_search,
    merge_confusion_counts,
    metrics_from_confusion_counts,
    summarize_threshold_search,
    update_threshold_search,
)
from utils.reporting import (
    save_confusion_matrix_csv,
    save_confusion_matrix_figure,
    save_metrics_csv,
    save_rows_csv,
)
from utils.runtime import (
    load_config,
    resolve_checkpoint_config,
    resolve_checkpoint_threshold,
    resolve_experiment_name,
    resolve_splits,
)
from utils.visualize import plot_sample, save_figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a GalaxEye checkpoint.")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=None)
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


def build_threshold_search_config(config: dict[str, Any]) -> dict[str, Any] | None:
    search_config = config.get("threshold_search", {})
    if not search_config.get("enabled", False):
        return None

    thresholds = build_thresholds(
        start=float(search_config["start"]),
        end=float(search_config["end"]),
        step=float(search_config["step"]),
    )
    return {
        "metric": str(search_config["metric"]),
        "thresholds": thresholds,
    }


def select_split_subset_indices(config: dict[str, Any], checkpoint: dict[str, Any], split: str) -> list[int] | None:
    if split == str(config["data"]["train_split"]):
        return checkpoint.get("train_subset_indices")
    if split == str(config["data"]["val_split"]):
        return checkpoint.get("val_subset_indices")
    return None


def resolve_modality_aliases(config: dict[str, Any]) -> dict[str, list[str]] | None:
    aliases = config["data"].get("modality_aliases")
    if not aliases:
        return None
    return {str(name): [str(alias) for alias in values] for name, values in aliases.items()}


def evaluate_split(
    *,
    config: dict[str, Any],
    checkpoint: dict[str, Any],
    checkpoint_config: dict[str, Any],
    dataset_root: Path,
    split: str,
    image_size: int,
    sample_limit: int | None,
    threshold: float,
    device: torch.device,
    use_amp: bool,
    directories: dict[str, Path],
    logger,
) -> dict[str, Any]:
    split_dir = directories["eval_dir"] / split
    visual_dir = split_dir / "visualizations"
    mask_dir = split_dir / "prediction_masks"
    split_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_change_dataset(
        data_root=dataset_root,
        split=split,
        modalities=checkpoint_config["modalities"],
        modality_aliases=resolve_modality_aliases(config),
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
    eval_dataset = dataset if not subset_indices else Subset(dataset, subset_indices)
    loader = build_dataloader(
        dataset=eval_dataset,
        batch_size=int(config["data"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["data"].get("num_workers", 0)),
        seed=int(config["project"]["seed"]),
        sample_limit=None if subset_indices else sample_limit,
    )

    confusion = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    threshold_search_config = build_threshold_search_config(config)
    search_state = None if threshold_search_config is None else initialize_threshold_search(threshold_search_config["thresholds"])

    summaries: list[dict[str, Any]] = []
    max_visualizations = int(config.get("evaluation", {}).get("max_visualizations", 12))
    save_prediction_masks = bool(config.get("evaluation", {}).get("save_prediction_masks", True))
    saved_visualizations = 0
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
            batch_counts = binary_confusion_counts(logits, masks, threshold=threshold, from_logits=True)
            merge_confusion_counts(confusion, batch_counts)

            if search_state is not None:
                update_threshold_search(search_state, logits, masks, from_logits=True)

            for sample_offset in range(images.shape[0]):
                sample_id = sample_ids[sample_offset]
                probability_map = probabilities[sample_offset, 0].detach().cpu().numpy()
                prediction_map = predictions[sample_offset, 0].detach().cpu().numpy()
                target_map = masks[sample_offset, 0].detach().cpu().numpy()

                if save_prediction_masks:
                    save_mask(mask_dir / f"{sample_id}.png", prediction_map)

                summary = summarize_case(
                    sample_id=sample_id,
                    probabilities=probability_map,
                    targets=target_map,
                    threshold=threshold,
                    uncertainty_band=float(config.get("inference", {}).get("uncertainty_band", 0.1)),
                )

                if saved_visualizations < max_visualizations:
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
                    grid_path = visual_dir / f"{sample_id}.png"
                    save_figure(figure, grid_path)
                    summary = attach_artifact_path(summary, "grid_path", grid_path)
                    saved_visualizations += 1

                summaries.append(summary)

    metrics = metrics_from_confusion_counts(confusion)
    metrics.update(
        {
            "split": split,
            "threshold": float(threshold),
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "checkpoint_best_metric": float(checkpoint.get("best_metric", 0.0)),
            "used_checkpoint_subset": bool(subset_indices),
            "num_samples": len(eval_dataset),
            "device": str(device),
        }
    )

    if search_state is not None:
        search_summary = summarize_threshold_search(search_state, metric_name=str(threshold_search_config["metric"]))
        metrics["search_threshold"] = float(search_summary["best_threshold"])
        metrics["search_iou"] = float(search_summary["best_metrics"]["iou"])
        metrics["search_precision"] = float(search_summary["best_metrics"]["precision"])
        metrics["search_recall"] = float(search_summary["best_metrics"]["recall"])
        metrics["search_f1"] = float(search_summary["best_metrics"]["f1"])
        save_json(split_dir / "threshold_search.json", search_summary["curve"])

    scene_summaries = summarize_by_scene(summaries)
    analysis = {
        "false_positives": rank_cases(summaries, key="false_positive_pixels", top_k=int(config.get("inference", {}).get("top_k_failures", 5))),
        "false_negatives": rank_cases(summaries, key="false_negative_pixels", top_k=int(config.get("inference", {}).get("top_k_failures", 5))),
        "worst_iou": rank_cases(summaries, key="iou", top_k=int(config.get("inference", {}).get("top_k_failures", 5)), reverse=False),
        "scene_summary": scene_summaries,
        "generalization_report": build_generalization_report(
            summaries,
            scene_summaries,
            top_k=int(config.get("inference", {}).get("top_k_failures", 5)),
        ),
    }

    save_json(split_dir / "metrics.json", metrics)
    save_metrics_csv(split_dir / "metrics.csv", metrics)
    save_confusion_matrix_csv(confusion, split_dir / "confusion_matrix.csv")
    save_confusion_matrix_figure(confusion, split_dir / "confusion_matrix.png", normalize=False)
    save_confusion_matrix_figure(confusion, split_dir / "confusion_matrix_normalized.png", normalize=True)
    save_rows_csv(split_dir / "case_summary.csv", summaries)
    save_rows_csv(split_dir / "scene_summary.csv", scene_summaries)
    save_rows_csv(split_dir / "top_false_positives.csv", analysis["false_positives"])
    save_rows_csv(split_dir / "top_false_negatives.csv", analysis["false_negatives"])
    save_rows_csv(split_dir / "worst_iou_cases.csv", analysis["worst_iou"])
    save_json(split_dir / "analysis.json", analysis)

    logger.info(
        "Split=%s | threshold=%.2f | samples=%d | IoU=%.4f | Precision=%.4f | Recall=%.4f | F1=%.4f | visuals=%d",
        split,
        threshold,
        len(eval_dataset),
        metrics["iou"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
        saved_visualizations,
    )
    if "search_threshold" in metrics:
        logger.info(
            "Split=%s | threshold search | best_threshold=%.2f | search_iou=%.4f | search_f1=%.4f",
            split,
            metrics["search_threshold"],
            metrics["search_iou"],
            metrics["search_f1"],
        )

    log_gpu_memory(logger, device, prefix=f"Eval {split} GPU")
    return metrics


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
    logger, _ = create_logger(directories["eval_dir"] / str(config["paths"]["log_subdir"]), name="eval")
    log_device_info(logger, device)
    dataset_root = resolve_dataset_root(config, download_if_missing=True, logger=logger)

    splits = resolve_splits(config, args.splits)
    image_size = int(args.image_size or config["data"]["image_size"])
    sample_limit = args.sample_limit
    threshold = resolve_checkpoint_threshold(args.threshold, config, checkpoint)
    use_amp = resolve_amp_enabled(bool(config.get("evaluation", {}).get("amp", config["training"].get("amp", False))), device)

    split_metrics: list[dict[str, Any]] = []
    for split in splits:
        split_metrics.append(
            evaluate_split(
                config=config,
                checkpoint=checkpoint,
                checkpoint_config=checkpoint_config,
                dataset_root=dataset_root,
                split=split,
                image_size=image_size,
                sample_limit=sample_limit,
                threshold=threshold,
                device=device,
                use_amp=use_amp,
                directories=directories,
                logger=logger,
            )
        )

    save_rows_csv(directories["eval_dir"] / "metrics_summary.csv", split_metrics)
    save_json(directories["eval_dir"] / "metrics_summary.json", split_metrics)
    logger.info("Saved split summaries to %s", directories["eval_dir"])


if __name__ == "__main__":
    main()
