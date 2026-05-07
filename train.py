from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from project_bootstrap import ensure_local_packages

ensure_local_packages()

import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import Subset
from tqdm import tqdm

from datasets.build import build_change_dataset, build_dataloader
from models import build_segmentation_model
from utils.augmentations import build_train_transforms, build_val_transforms
from utils.checkpoint import build_checkpoint_state, load_checkpoint, save_checkpoint
from utils.dataset_utils import load_image_array
from utils.dataset_source import resolve_dataset_root
from utils.device import ensure_cuda, log_device_info, log_gpu_memory, resolve_amp_enabled, resolve_device, set_seed
from utils.experiment import append_experiment_index, build_run_directories, save_experiment_snapshot
from utils.label_utils import remap_mask
from utils.logger import append_csv_row, create_logger, save_json
from utils.losses import build_loss
from utils.metrics import (
    binary_confusion_counts,
    build_thresholds,
    initialize_threshold_search,
    merge_confusion_counts,
    metrics_from_confusion_counts,
    summarize_threshold_search,
    update_threshold_search,
)
from utils.runtime import load_config, resolve_experiment_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the GalaxEye change detection baseline.")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--train-sample-limit", type=int, default=None)
    parser.add_argument("--val-sample-limit", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--model-architecture", type=str, default=None)
    parser.add_argument("--encoder-name", type=str, default=None)
    parser.add_argument("--loss-name", type=str, default=None)
    parser.add_argument("--modalities", nargs="+", default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


def resolve_modalities(config: dict[str, Any], args: argparse.Namespace) -> list[str] | None:
    if args.modalities:
        return list(args.modalities)
    modalities = config["data"].get("modalities")
    if modalities:
        return list(modalities)
    return None


def resolve_modality_aliases(config: dict[str, Any]) -> dict[str, list[str]] | None:
    aliases = config["data"].get("modality_aliases")
    if not aliases:
        return None
    return {str(name): [str(alias) for alias in values] for name, values in aliases.items()}


def select_subset_indices(dataset, sample_limit: int | None, seed: int) -> list[int] | None:
    if sample_limit is None or sample_limit >= len(dataset):
        return None
    rng = random.Random(seed)
    indices = rng.sample(range(len(dataset)), k=sample_limit)
    indices.sort()
    return indices


def maybe_subset_dataset(dataset, subset_indices: list[int] | None):
    if subset_indices is None:
        return dataset
    return Subset(dataset, subset_indices)


def compute_pos_weight(
    dataset,
    subset_indices: list[int] | None = None,
    max_pos_weight: float | None = None,
) -> tuple[float, float]:
    if subset_indices is None:
        records = dataset.samples
    else:
        records = [dataset.samples[index] for index in subset_indices]

    positive_pixels = 0
    total_pixels = 0
    for record in records:
        mask = load_image_array(record.mask_path, logger=dataset.logger)
        if mask is None:
            continue
        if mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask[..., 0]
        remapped = remap_mask(mask)
        positive_pixels += int(remapped.sum())
        total_pixels += int(remapped.size)

    negative_pixels = max(total_pixels - positive_pixels, 0)
    raw_pos_weight = 1.0 if positive_pixels == 0 else max(negative_pixels / positive_pixels, 1.0)
    effective_pos_weight = raw_pos_weight if max_pos_weight is None else min(raw_pos_weight, float(max_pos_weight))
    return float(raw_pos_weight), float(effective_pos_weight)


def build_scheduler(optimizer: torch.optim.Optimizer, scheduler_config: dict[str, Any], epochs: int):
    name = str(scheduler_config["name"]).lower()
    if name == "none":
        return None
    if name == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode=str(scheduler_config["mode"]),
            factor=float(scheduler_config["factor"]),
            patience=int(scheduler_config["patience"]),
            min_lr=float(scheduler_config["min_lr"]),
        )
    if name == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=int(scheduler_config.get("t_max", epochs)),
            eta_min=float(scheduler_config["min_lr"]),
        )
    raise ValueError(f"Unsupported scheduler: {name}")


def create_scaler(enabled: bool, device: torch.device):
    if device.type == "cuda":
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return None


def get_current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


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


def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    threshold: float,
    use_amp: bool,
    scaler,
    grad_clip_norm: float | None,
    logger,
    epoch: int,
    split_name: str,
    log_every_steps: int,
    threshold_search_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_batches = 0
    confusion = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    search_state = None if is_training or threshold_search_config is None else initialize_threshold_search(threshold_search_config["thresholds"])

    progress = tqdm(loader, desc=f"{split_name.capitalize()} {epoch}", leave=False)
    autocast_enabled = bool(use_amp and device.type == "cuda")

    for step, batch in enumerate(progress, start=1):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            with torch.autocast(device_type=device.type, enabled=autocast_enabled):
                logits = model(images)
                loss = criterion(logits, masks)

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss encountered during {split_name} at epoch {epoch}, step {step}.")

            if is_training:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    if grad_clip_norm and grad_clip_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip_norm and grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    optimizer.step()

        total_loss += float(loss.item())
        total_batches += 1

        detached_logits = logits.detach()
        detached_masks = masks.detach()
        batch_counts = binary_confusion_counts(detached_logits, detached_masks, threshold=threshold, from_logits=True)
        merge_confusion_counts(confusion, batch_counts)

        if search_state is not None:
            update_threshold_search(search_state, detached_logits, detached_masks, from_logits=True)

        if is_training:
            progress.set_postfix(loss=f"{loss.item():.4f}", lr=f"{get_current_lr(optimizer):.2e}")
        else:
            progress.set_postfix(loss=f"{loss.item():.4f}")

        if step % max(log_every_steps, 1) == 0:
            logger.info(
                "%s epoch %d | step %d/%d | loss=%.4f",
                split_name,
                epoch,
                step,
                len(loader),
                float(loss.item()),
            )

    metrics: dict[str, Any] = metrics_from_confusion_counts(confusion)
    metrics["loss"] = total_loss / max(total_batches, 1)

    if search_state is not None:
        search_summary = summarize_threshold_search(
            search_state=search_state,
            metric_name=str(threshold_search_config["metric"]),
        )
        metrics.update(
            {
                "search_threshold": float(search_summary["best_threshold"]),
                "search_iou": float(search_summary["best_metrics"]["iou"]),
                "search_precision": float(search_summary["best_metrics"]["precision"]),
                "search_recall": float(search_summary["best_metrics"]["recall"]),
                "search_f1": float(search_summary["best_metrics"]["f1"]),
                "threshold_curve": search_summary["curve"],
            }
        )

    return metrics


def build_summary_payload(
    experiment_name: str,
    resolved_config: dict[str, Any],
    best_metric: float,
    best_metrics: dict[str, float],
    best_threshold: float,
    best_epoch: int,
    stopped_early: bool,
    checkpoint_path: Path,
) -> dict[str, Any]:
    return {
        "experiment_name": experiment_name,
        "architecture": resolved_config["model"]["architecture"],
        "encoder_name": resolved_config["model"]["encoder_name"],
        "loss_name": resolved_config["loss"]["name"],
        "modalities": resolved_config["data"].get("modalities"),
        "train_sample_limit": resolved_config["runtime"]["train_sample_limit"],
        "val_sample_limit": resolved_config["runtime"]["val_sample_limit"],
        "best_metric_name": resolved_config["training"]["monitor"],
        "best_metric": float(best_metric),
        "best_epoch": int(best_epoch),
        "best_threshold": float(best_threshold),
        "best_iou": float(best_metrics.get("iou", 0.0)),
        "best_precision": float(best_metrics.get("precision", 0.0)),
        "best_recall": float(best_metrics.get("recall", 0.0)),
        "best_f1": float(best_metrics.get("f1", 0.0)),
        "stopped_early": bool(stopped_early),
        "checkpoint_path": str(checkpoint_path),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    seed = int(config["project"]["seed"])
    set_seed(seed, deterministic=bool(config["project"].get("deterministic", True)))

    device = resolve_device(args.device)
    require_cuda = bool(args.require_cuda or config["project"].get("require_cuda", False))
    ensure_cuda(device, require_cuda=require_cuda)
    output_root = Path(args.output_dir or config["paths"]["output_root"])
    experiment_name = resolve_experiment_name(config, explicit_name=args.experiment_name)
    directories = build_run_directories(output_root=output_root, experiment_name=experiment_name, path_config=config["paths"])

    logger, log_path = create_logger(directories["log_dir"], name=f"train_{experiment_name}")
    log_device_info(logger, device)

    image_size = int(args.image_size or config["data"]["image_size"])
    batch_size = int(args.batch_size or config["data"]["batch_size"])
    epochs = int(args.epochs or config["training"]["epochs"])
    learning_rate = float(args.lr or config["optimizer"]["lr"])
    threshold = float(args.threshold or config["training"]["threshold"])
    train_sample_limit = args.train_sample_limit if args.train_sample_limit is not None else config["data"].get("train_sample_limit")
    val_sample_limit = args.val_sample_limit if args.val_sample_limit is not None else config["data"].get("val_sample_limit")
    grad_clip_norm = float(config["training"]["grad_clip_norm"])
    requested_train_amp = bool(args.amp or config["training"].get("amp", False))
    use_train_amp = resolve_amp_enabled(requested_train_amp, device)
    use_eval_amp = resolve_amp_enabled(bool(config.get("evaluation", {}).get("amp", requested_train_amp)), device)
    architecture = str(args.model_architecture or config["model"]["architecture"])
    encoder_name = str(args.encoder_name or config["model"]["encoder_name"])
    loss_name = str(args.loss_name or config["loss"]["name"])
    modalities = resolve_modalities(config, args)
    modality_aliases = resolve_modality_aliases(config)

    logger.info(
        "Training configuration | architecture=%s | encoder=%s | batch_size=%d | image_size=%d | amp_train=%s | amp_eval=%s",
        architecture,
        encoder_name,
        batch_size,
        image_size,
        use_train_amp,
        use_eval_amp,
    )

    dataset_root = resolve_dataset_root(config, download_if_missing=True, logger=logger)

    augmentation_config = config.get("augmentation", {})
    train_transform = build_train_transforms(
        crop_size=image_size,
        use_transpose=bool(augmentation_config.get("use_transpose", True)),
        affine_probability=float(augmentation_config.get("affine_probability", 0.25)),
        affine_scale=float(augmentation_config.get("affine_scale", 0.05)),
        affine_translate=float(augmentation_config.get("affine_translate", 0.05)),
        horizontal_flip_probability=float(augmentation_config.get("horizontal_flip_probability", 0.5)),
        vertical_flip_probability=float(augmentation_config.get("vertical_flip_probability", 0.5)),
        rotate90_probability=float(augmentation_config.get("rotate90_probability", 0.5)),
    )
    val_transform = build_val_transforms(crop_size=image_size)

    train_dataset = build_change_dataset(
        data_root=dataset_root,
        split=config["data"]["train_split"],
        modalities=modalities,
        modality_aliases=modality_aliases,
        transform=train_transform,
    )
    val_dataset = build_change_dataset(
        data_root=dataset_root,
        split=config["data"]["val_split"],
        modalities=modalities,
        modality_aliases=modality_aliases,
        transform=val_transform,
    )

    train_subset_indices = select_subset_indices(train_dataset, train_sample_limit, seed=seed)
    val_subset_indices = select_subset_indices(val_dataset, val_sample_limit, seed=seed + 1)
    train_dataset_for_loader = maybe_subset_dataset(train_dataset, train_subset_indices)
    val_dataset_for_loader = maybe_subset_dataset(val_dataset, val_subset_indices)

    num_workers = int(config["data"].get("num_workers", 0))
    train_loader = build_dataloader(
        dataset=train_dataset_for_loader,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        seed=seed,
        sample_limit=None,
    )
    val_loader = build_dataloader(
        dataset=val_dataset_for_loader,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        seed=seed,
        sample_limit=None,
    )

    raw_pos_weight = 1.0
    pos_weight = None
    if config["loss"].get("use_pos_weight", False):
        raw_pos_weight, pos_weight = compute_pos_weight(
            train_dataset,
            train_subset_indices,
            max_pos_weight=config["loss"].get("pos_weight_max"),
        )

    logger.info(
        "Train samples: %d | Val samples: %d | Input channels: %d | Modalities: %s",
        len(train_dataset_for_loader),
        len(val_dataset_for_loader),
        train_dataset.num_channels,
        train_dataset.modalities,
    )
    logger.info("Channel layout: %s", train_dataset.describe_channels())
    logger.info("raw_pos_weight: %.4f | effective_pos_weight: %.4f", raw_pos_weight, float(pos_weight or 1.0))

    model = build_segmentation_model(
        architecture=architecture,
        encoder_name=encoder_name,
        encoder_weights=config["model"]["encoder_weights"],
        in_channels=train_dataset.num_channels,
        classes=int(config["model"]["output_classes"]),
    ).to(device)

    criterion = build_loss(
        name=loss_name,
        bce_weight=float(config["loss"]["bce_weight"]),
        dice_weight=float(config["loss"]["dice_weight"]),
        focal_weight=float(config["loss"]["focal_weight"]),
        focal_gamma=float(config["loss"]["focal_gamma"]),
        focal_alpha=float(config["loss"]["focal_alpha"]),
        pos_weight=pos_weight,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )
    scheduler = build_scheduler(optimizer, config["scheduler"], epochs=epochs)
    scaler = create_scaler(enabled=use_train_amp, device=device)
    threshold_search_config = build_threshold_search_config(config)

    start_epoch = 1
    best_metric = float("-inf")
    best_epoch = 0
    best_threshold = threshold
    best_metrics = {"iou": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    early_stopping_config = config["training"].get("early_stopping", {})
    early_stopping_enabled = bool(early_stopping_config.get("enabled", False))
    early_stopping_patience = int(early_stopping_config.get("patience", 0))
    early_stopping_min_delta = float(early_stopping_config.get("min_delta", 0.0))
    early_stopping_start_epoch = int(early_stopping_config.get("start_epoch", 1))
    epochs_without_improvement = 0
    stopped_early = False

    resume_path = args.resume or config["training"]["resume_from"]
    if resume_path:
        checkpoint = load_checkpoint(
            checkpoint_path=resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=device,
        )
        start_epoch = int(checkpoint["epoch"]) + 1
        best_metric = float(checkpoint.get("best_metric", best_metric))
        best_epoch = int(checkpoint.get("best_epoch", best_epoch))
        best_threshold = float(checkpoint.get("best_threshold", threshold))
        best_metrics = checkpoint.get("best_metrics", best_metrics)
        logger.info("Resumed from %s at epoch %d", resume_path, start_epoch)

    history_csv = directories["metrics_dir"] / str(config["logging"]["csv_name"])
    best_ckpt_path = directories["checkpoint_dir"] / "best.pt"
    latest_ckpt_path = directories["checkpoint_dir"] / "latest.pt"
    monitor_key = str(config["training"]["monitor"])

    resolved_config = {
        **config,
        "model": {
            **config["model"],
            "architecture": architecture,
            "encoder_name": encoder_name,
        },
        "loss": {
            **config["loss"],
            "name": loss_name,
        },
        "data": {
            **config["data"],
            "modalities": train_dataset.modalities,
        },
        "logging": {
            **config["logging"],
            "experiment_name": experiment_name,
        },
            "runtime": {
                "image_size": image_size,
                "batch_size": batch_size,
                "train_sample_limit": train_sample_limit,
                "val_sample_limit": val_sample_limit,
                "dataset_root": str(dataset_root),
                "device": str(device),
                "require_cuda": require_cuda,
                "amp_train": use_train_amp,
                "amp_eval": use_eval_amp,
                "learning_rate": learning_rate,
                "threshold": threshold,
            },
    }
    save_experiment_snapshot(directories["run_root"], resolved_config)

    for epoch in range(start_epoch, epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            threshold=threshold,
            use_amp=use_train_amp,
            scaler=scaler,
            grad_clip_norm=grad_clip_norm,
            logger=logger,
            epoch=epoch,
            split_name="train",
            log_every_steps=int(config["logging"]["log_every_steps"]),
            threshold_search_config=None,
        )
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
            threshold=threshold,
            use_amp=use_eval_amp,
            scaler=None,
            grad_clip_norm=None,
            logger=logger,
            epoch=epoch,
            split_name="val",
            log_every_steps=int(config["logging"]["log_every_steps"]),
            threshold_search_config=threshold_search_config,
        )

        threshold_curve = val_metrics.pop("threshold_curve", None)
        if threshold_curve is not None:
            save_json(directories["metrics_dir"] / f"threshold_search_epoch_{epoch:03d}.json", threshold_curve)

        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_metrics[monitor_key])
            else:
                scheduler.step()

        fixed_threshold_message = (
            f"val_iou={val_metrics['iou']:.4f} | val_precision={val_metrics['precision']:.4f} | "
            f"val_recall={val_metrics['recall']:.4f} | val_f1={val_metrics['f1']:.4f}"
        )
        if "search_threshold" in val_metrics:
            fixed_threshold_message += (
                f" | search_threshold={val_metrics['search_threshold']:.2f} | "
                f"search_iou={val_metrics['search_iou']:.4f} | "
                f"search_f1={val_metrics['search_f1']:.4f}"
            )

        logger.info(
            "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | %s | lr=%.2e",
            epoch,
            epochs,
            train_metrics["loss"],
            val_metrics["loss"],
            fixed_threshold_message,
            get_current_lr(optimizer),
        )
        log_gpu_memory(logger, device, prefix=f"Epoch {epoch} GPU")

        append_csv_row(
            history_csv,
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_iou": train_metrics["iou"],
                "train_precision": train_metrics["precision"],
                "train_recall": train_metrics["recall"],
                "train_f1": train_metrics["f1"],
                "val_loss": val_metrics["loss"],
                "val_iou": val_metrics["iou"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_f1": val_metrics["f1"],
                "search_threshold": val_metrics.get("search_threshold"),
                "search_iou": val_metrics.get("search_iou"),
                "search_precision": val_metrics.get("search_precision"),
                "search_recall": val_metrics.get("search_recall"),
                "search_f1": val_metrics.get("search_f1"),
                "lr": get_current_lr(optimizer),
            },
        )

        current_metric = float(val_metrics[monitor_key])
        has_improved = current_metric > (best_metric + early_stopping_min_delta)
        if has_improved:
            best_metric = current_metric
            best_epoch = epoch
            best_metrics = {
                "iou": float(val_metrics["search_iou"] if monitor_key.startswith("search_") else val_metrics["iou"]),
                "precision": float(val_metrics["search_precision"] if monitor_key.startswith("search_") else val_metrics["precision"]),
                "recall": float(val_metrics["search_recall"] if monitor_key.startswith("search_") else val_metrics["recall"]),
                "f1": float(val_metrics["search_f1"] if monitor_key.startswith("search_") else val_metrics["f1"]),
            }
            best_threshold = float(val_metrics.get("search_threshold", threshold))
            if threshold_curve is not None:
                save_json(directories["metrics_dir"] / "best_threshold_search.json", threshold_curve)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        checkpoint_state = build_checkpoint_state(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_metric=best_metric,
            best_metrics=best_metrics,
            config=resolved_config,
            extra_state={
                "train_subset_indices": train_subset_indices,
                "val_subset_indices": val_subset_indices,
                "input_channels": train_dataset.num_channels,
                "active_modalities": train_dataset.modalities,
                "best_epoch": best_epoch,
                "best_threshold": best_threshold,
            },
        )
        save_checkpoint(checkpoint_state, latest_ckpt_path)

        if has_improved:
            save_checkpoint(checkpoint_state, best_ckpt_path)

        if early_stopping_enabled and epoch >= early_stopping_start_epoch and epochs_without_improvement >= early_stopping_patience:
            stopped_early = True
            logger.info(
                "Early stopping at epoch %d after %d epochs without improvement in %s.",
                epoch,
                epochs_without_improvement,
                monitor_key,
            )
            break

    summary = build_summary_payload(
        experiment_name=experiment_name,
        resolved_config=resolved_config,
        best_metric=best_metric,
        best_metrics=best_metrics,
        best_threshold=best_threshold,
        best_epoch=best_epoch,
        stopped_early=stopped_early,
        checkpoint_path=best_ckpt_path,
    )
    save_json(directories["run_root"] / "summary.json", summary)
    append_experiment_index(
        output_root=output_root,
        row=summary,
        filename=str(config["logging"].get("experiment_index", "experiment_index.csv")),
    )

    logger.info("Saved latest checkpoint to %s", latest_ckpt_path)
    logger.info("Saved best checkpoint to %s", best_ckpt_path)
    logger.info("Best epoch: %d | Best threshold: %.2f | Best metric (%s): %.4f", best_epoch, best_threshold, monitor_key, best_metric)
    logger.info("Training log saved to %s", log_path)


if __name__ == "__main__":
    main()
