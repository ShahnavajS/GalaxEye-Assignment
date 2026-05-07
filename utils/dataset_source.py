from __future__ import annotations

import os
import zipfile
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


def _has_split_archives(path: Path, split_names: list[str]) -> bool:
    return all((path / f"{split_name}.zip").is_file() for split_name in split_names)


def _default_extract_root(
    *,
    repo_id: str,
    cache_dir: str | Path | None = None,
    local_dir: str | Path | None = None,
) -> Path:
    local_dir_path = _expand_path(local_dir)
    if local_dir_path is not None:
        return local_dir_path

    cache_root = _expand_path(cache_dir) or Path.home() / ".cache" / "huggingface"
    safe_repo_name = repo_id.replace("/", "__")
    return cache_root / "datasets_extracted" / safe_repo_name


def _extract_archive(archive_path: Path, split_name: str, extract_root: Path) -> None:
    extract_root.mkdir(parents=True, exist_ok=True)
    split_root = extract_root / split_name

    with zipfile.ZipFile(archive_path) as archive:
        member_names = [name for name in archive.namelist() if name and not name.endswith("/")]
        has_top_level_split_dir = any(Path(name).parts and Path(name).parts[0] == split_name for name in member_names)

        if has_top_level_split_dir:
            archive.extractall(extract_root)
        else:
            split_root.mkdir(parents=True, exist_ok=True)
            archive.extractall(split_root)


def extract_split_archives(
    archive_root: str | Path,
    *,
    split_names: list[str],
    repo_id: str,
    cache_dir: str | Path | None = None,
    local_dir: str | Path | None = None,
    force_extract: bool = False,
    logger=None,
) -> Path | None:
    archive_root = Path(archive_root)
    if not _has_split_archives(archive_root, split_names):
        return None

    extract_root = _default_extract_root(repo_id=repo_id, cache_dir=cache_dir, local_dir=local_dir).resolve()

    for split_name in split_names:
        split_dir = extract_root / split_name
        archive_path = archive_root / f"{split_name}.zip"

        if split_dir.is_dir() and not force_extract:
            if logger is not None:
                logger.info("Reusing extracted split: %s", split_dir)
            continue

        if logger is not None:
            logger.info("Extracting %s to %s", archive_path.name, extract_root)
        _extract_archive(archive_path, split_name, extract_root)

    resolved_root = find_dataset_root(extract_root, split_names, max_depth=2)
    if resolved_root is None:
        raise FileNotFoundError(
            f"Extracted archives from {archive_root}, but could not find split folders {split_names} under {extract_root}."
        )

    return resolved_root


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
    force_extract: bool = False,
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
        resolved_root = extract_split_archives(
            downloaded_path,
            split_names=split_names,
            repo_id=str(repo_id),
            cache_dir=data_config.get("hf_cache_dir"),
            local_dir=data_config.get("hf_local_dir"),
            force_extract=force_extract,
            logger=logger,
        )
    if resolved_root is None:
        raise FileNotFoundError(
            f"Downloaded dataset from {repo_id}, but could not find split folders {split_names} under {downloaded_path}."
        )

    if logger is not None:
        logger.info("Resolved downloaded dataset root: %s", resolved_root)

    return resolved_root
