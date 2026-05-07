from __future__ import annotations

import os
from collections import deque
from pathlib import Path
from typing import Any


def _expand_path(path_value: str | os.PathLike[str] | None) -> Path | None:
    if path_value in (None, ""):
        return None
    expanded = os.path.expandvars(os.path.expanduser(str(path_value)))
    return Path(expanded)


def get_split_names(data_config: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for key in ("train_split", "val_split", "test_split"):
        value = data_config.get(key)
        if value and value not in names:
            names.append(str(value))
    return names


def is_dataset_root(path: str | Path, split_names: list[str]) -> bool:
    path = Path(path)
    if not path.exists() or not path.is_dir():
        return False
    return all((path / split_name).is_dir() for split_name in split_names)


def find_dataset_root(
    base_path: str | Path | None,
    split_names: list[str],
    max_depth: int = 2,
) -> Path | None:
    if base_path is None:
        return None

    root = _expand_path(base_path)
    if root is None or not root.exists():
        return None

    if is_dataset_root(root, split_names):
        return root.resolve()

    queue: deque[tuple[Path, int]] = deque([(root, 0)])
    visited: set[Path] = set()

    while queue:
        current_path, depth = queue.popleft()
        if current_path in visited or depth >= max_depth:
            continue
        visited.add(current_path)

        try:
            children = sorted(child for child in current_path.iterdir() if child.is_dir())
        except OSError:
            continue

        for child in children:
            if is_dataset_root(child, split_names):
                return child.resolve()
            queue.append((child, depth + 1))

    return None


def download_hf_dataset(
    repo_id: str,
    *,
    cache_dir: str | Path | None = None,
    local_dir: str | Path | None = None,
    revision: str | None = None,
    force_download: bool = False,
    logger=None,
) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required for Hugging Face dataset downloads. "
            "Install it with `pip install huggingface-hub`."
        ) from exc

    cache_dir_path = _expand_path(cache_dir)
    local_dir_path = _expand_path(local_dir)

    download_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        cache_dir=None if cache_dir_path is None else str(cache_dir_path),
        local_dir=None if local_dir_path is None else str(local_dir_path),
        revision=revision,
        force_download=force_download,
    )
    resolved_path = Path(download_path).resolve()

    if logger is not None:
        logger.info("Dataset available at %s", resolved_path)

    return resolved_path


def resolve_dataset_root(
    config: dict[str, Any],
    *,
    download_if_missing: bool = True,
    force_download: bool = False,
    logger=None,
) -> Path:
    data_config = config.get("data", {})
    path_config = config.get("paths", {})
    split_names = get_split_names(data_config)
    source = str(data_config.get("source", "auto")).lower()

    candidate_paths = [
        path_config.get("dataset_root"),
        path_config.get("data_root"),
        data_config.get("local_dataset_root"),
        data_config.get("hf_local_dir"),
    ]

    for candidate in candidate_paths:
        resolved_root = find_dataset_root(candidate, split_names)
        if resolved_root is not None:
            if logger is not None:
                logger.info("Using dataset root: %s", resolved_root)
            return resolved_root

    if source == "local":
        raise FileNotFoundError(
            "Could not locate a local dataset root with the configured split folders. "
            "Set `paths.dataset_root` or `data.local_dataset_root`."
        )

    if source not in {"auto", "hf"}:
        raise ValueError(f"Unsupported data.source value: {source}")

    if not download_if_missing or not bool(data_config.get("download_on_missing", True)):
        raise FileNotFoundError(
            "Dataset root was not found locally and automatic download is disabled."
        )

    repo_id = data_config.get("hf_repo_id")
    if not repo_id:
        raise ValueError("`data.hf_repo_id` must be set when using Hugging Face dataset mode.")

    downloaded_path = download_hf_dataset(
        repo_id=str(repo_id),
        cache_dir=data_config.get("hf_cache_dir"),
        local_dir=data_config.get("hf_local_dir"),
        revision=data_config.get("hf_revision"),
        force_download=force_download,
        logger=logger,
    )

    resolved_root = find_dataset_root(downloaded_path, split_names)
    if resolved_root is None:
        raise FileNotFoundError(
            f"Downloaded dataset from {repo_id}, but could not find split folders {split_names} under {downloaded_path}."
        )

    if logger is not None:
        logger.info("Resolved downloaded dataset root: %s", resolved_root)

    return resolved_root
