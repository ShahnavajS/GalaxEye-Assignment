from __future__ import annotations

from pathlib import Path
from typing import Any

from utils.logger import append_csv_row, save_json


def build_run_directories(
    output_root: str | Path,
    experiment_name: str,
    path_config: dict[str, Any],
) -> dict[str, Path]:
    output_root = Path(output_root)
    run_root = output_root / experiment_name
    directories = {
        "run_root": run_root,
        "checkpoint_dir": run_root / str(path_config["checkpoint_subdir"]),
        "log_dir": run_root / str(path_config["log_subdir"]),
        "metrics_dir": run_root / str(path_config["metrics_subdir"]),
        "inference_dir": run_root / str(path_config["inference_subdir"]),
        "eval_dir": run_root / "eval",
        "report_dir": run_root / "report",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return directories


def save_experiment_snapshot(run_root: str | Path, config: dict[str, Any]) -> Path:
    snapshot_path = Path(run_root) / "resolved_config.json"
    save_json(snapshot_path, config)
    return snapshot_path


def append_experiment_index(
    output_root: str | Path,
    row: dict[str, Any],
    filename: str = "experiment_index.csv",
) -> Path:
    index_path = Path(output_root) / filename
    append_csv_row(index_path, row)
    return index_path
