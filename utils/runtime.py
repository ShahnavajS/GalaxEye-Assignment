from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_experiment_name(
    config: dict[str, Any],
    checkpoint_path: str | Path | None = None,
    explicit_name: str | None = None,
) -> str:
    if explicit_name:
        return str(explicit_name)

    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path).resolve()
        checkpoint_dir_name = str(config["paths"]["checkpoint_subdir"])
        if checkpoint_path.parent.name == checkpoint_dir_name:
            return checkpoint_path.parent.parent.name

    return str(config["logging"]["experiment_name"])


def resolve_checkpoint_config(base_config: dict[str, Any], checkpoint: dict[str, Any]) -> dict[str, Any]:
    saved_config = checkpoint.get("config", {})
    model_config = saved_config.get("model", {})
    data_config = saved_config.get("data", {})
    return {
        "architecture": str(model_config.get("architecture", base_config["model"]["architecture"])),
        "encoder_name": str(model_config.get("encoder_name", base_config["model"]["encoder_name"])),
        "encoder_weights": model_config.get("encoder_weights", base_config["model"]["encoder_weights"]),
        "output_classes": int(model_config.get("output_classes", base_config["model"]["output_classes"])),
        "modalities": checkpoint.get("active_modalities") or data_config.get("modalities") or base_config["data"].get("modalities"),
    }


def resolve_checkpoint_threshold(
    requested_threshold: float | None,
    config: dict[str, Any],
    checkpoint: dict[str, Any],
) -> float:
    if requested_threshold is not None:
        return float(requested_threshold)
    if config.get("threshold_search", {}).get("apply_best_threshold_on_eval", False):
        return float(checkpoint.get("best_threshold", config["training"]["threshold"]))
    return float(config["training"]["threshold"])


def resolve_splits(config: dict[str, Any], requested_splits: list[str] | None = None) -> list[str]:
    if requested_splits:
        return list(requested_splits)

    splits = [str(config["data"]["val_split"])]
    test_split = config["data"].get("test_split")
    if test_split:
        splits.append(str(test_split))
    return splits
