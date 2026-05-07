from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def resolve_device(requested_device: str = "auto") -> torch.device:
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested_device)


def resolve_amp_enabled(enabled: bool, device: torch.device) -> bool:
    return bool(enabled and device.type == "cuda")


def ensure_cuda(device: torch.device, require_cuda: bool = False) -> None:
    if require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA is required for this run, but no CUDA device is available.")


def get_device_payload(device: torch.device) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
    }

    if device.type != "cuda":
        payload["mode"] = "cpu_fallback"
        return payload

    index = 0 if device.index is None else int(device.index)
    properties = torch.cuda.get_device_properties(index)
    payload.update(
        {
            "mode": "cuda",
            "device_index": index,
            "device_name": torch.cuda.get_device_name(index),
            "total_memory_gb": round(properties.total_memory / (1024**3), 2),
            "capability": f"{properties.major}.{properties.minor}",
        }
    )
    return payload


def log_device_info(logger, device: torch.device) -> None:
    payload = get_device_payload(device)
    if device.type == "cuda":
        logger.info(
            "Using CUDA device %s | name=%s | memory=%.2f GB | capability=%s",
            payload["device_index"],
            payload["device_name"],
            payload["total_memory_gb"],
            payload["capability"],
        )
    else:
        logger.warning("CUDA unavailable. Falling back to CPU execution.")


def gpu_memory_stats(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {}

    index = 0 if device.index is None else int(device.index)
    return {
        "allocated_gb": round(torch.cuda.memory_allocated(index) / (1024**3), 3),
        "reserved_gb": round(torch.cuda.memory_reserved(index) / (1024**3), 3),
        "max_allocated_gb": round(torch.cuda.max_memory_allocated(index) / (1024**3), 3),
        "max_reserved_gb": round(torch.cuda.max_memory_reserved(index) / (1024**3), 3),
    }


def log_gpu_memory(logger, device: torch.device, prefix: str) -> None:
    stats = gpu_memory_stats(device)
    if not stats:
        return
    logger.info(
        "%s | allocated=%.3f GB | reserved=%.3f GB | max_allocated=%.3f GB | max_reserved=%.3f GB",
        prefix,
        stats["allocated_gb"],
        stats["reserved_gb"],
        stats["max_allocated_gb"],
        stats["max_reserved_gb"],
    )
