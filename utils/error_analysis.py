from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


def to_numpy(array: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def extract_scene_id(sample_id: str) -> str:
    match = re.search(r"(scene_\d+)", sample_id)
    if match:
        return match.group(1)
    return "unknown_scene"


def build_error_maps(
    probabilities: torch.Tensor | np.ndarray,
    targets: torch.Tensor | np.ndarray,
    threshold: float = 0.5,
    uncertainty_band: float = 0.1,
) -> dict[str, np.ndarray]:
    probabilities = to_numpy(probabilities).astype(np.float32, copy=False)
    targets = (to_numpy(targets) >= 0.5).astype(np.uint8, copy=False)

    predicted = (probabilities >= threshold).astype(np.uint8)
    false_positive = np.logical_and(predicted == 1, targets == 0).astype(np.uint8)
    false_negative = np.logical_and(predicted == 0, targets == 1).astype(np.uint8)
    true_positive = np.logical_and(predicted == 1, targets == 1).astype(np.uint8)
    uncertain = (np.abs(probabilities - threshold) <= uncertainty_band).astype(np.uint8)

    return {
        "predicted": predicted,
        "target": targets,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "uncertain": uncertain,
    }


def summarize_case(
    sample_id: str,
    probabilities: torch.Tensor | np.ndarray,
    targets: torch.Tensor | np.ndarray,
    threshold: float = 0.5,
    uncertainty_band: float = 0.1,
) -> dict[str, float | int | str]:
    maps = build_error_maps(
        probabilities=probabilities,
        targets=targets,
        threshold=threshold,
        uncertainty_band=uncertainty_band,
    )

    tp = int(maps["true_positive"].sum())
    fp = int(maps["false_positive"].sum())
    fn = int(maps["false_negative"].sum())
    target_pixels = int(maps["target"].sum())
    predicted_pixels = int(maps["predicted"].sum())
    total_pixels = int(maps["target"].size)
    union = tp + fp + fn
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    iou = tp / max(union, 1)
    f1 = (2.0 * precision * recall) / max(precision + recall, 1e-8)

    probability_map = to_numpy(probabilities).astype(np.float32, copy=False)
    return {
        "sample_id": sample_id,
        "scene_id": extract_scene_id(sample_id),
        "true_positive_pixels": tp,
        "false_positive_pixels": fp,
        "false_negative_pixels": fn,
        "error_pixels": fp + fn,
        "target_pixels": target_pixels,
        "predicted_pixels": predicted_pixels,
        "target_ratio": float(target_pixels / max(total_pixels, 1)),
        "predicted_ratio": float(predicted_pixels / max(total_pixels, 1)),
        "uncertain_pixels": int(maps["uncertain"].sum()),
        "uncertain_ratio": float(maps["uncertain"].sum() / max(total_pixels, 1)),
        "mean_probability": float(probability_map.mean()),
        "std_probability": float(probability_map.std()),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def attach_artifact_path(
    case_summary: dict[str, float | int | str],
    key: str,
    artifact_path: str | Path,
) -> dict[str, float | int | str]:
    payload = dict(case_summary)
    payload[key] = str(artifact_path)
    return payload


def rank_cases(
    cases: list[dict[str, float | int | str]],
    key: str,
    top_k: int = 5,
    reverse: bool = True,
) -> list[dict[str, float | int | str]]:
    return sorted(cases, key=lambda item: float(item[key]), reverse=reverse)[:top_k]


def summarize_by_scene(cases: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    grouped: dict[str, list[dict[str, float | int | str]]] = defaultdict(list)
    for case in cases:
        grouped[str(case["scene_id"])].append(case)

    summaries: list[dict[str, float | int | str]] = []
    for scene_id, items in sorted(grouped.items()):
        tp = sum(int(item["true_positive_pixels"]) for item in items)
        fp = sum(int(item["false_positive_pixels"]) for item in items)
        fn = sum(int(item["false_negative_pixels"]) for item in items)
        total_target = sum(int(item["target_pixels"]) for item in items)
        total_predicted = sum(int(item["predicted_pixels"]) for item in items)
        total_error = sum(int(item["error_pixels"]) for item in items)

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        iou = tp / max(tp + fp + fn, 1)
        f1 = (2.0 * precision * recall) / max(precision + recall, 1e-8)

        summaries.append(
            {
                "scene_id": scene_id,
                "num_samples": len(items),
                "true_positive_pixels": tp,
                "false_positive_pixels": fp,
                "false_negative_pixels": fn,
                "target_pixels": total_target,
                "predicted_pixels": total_predicted,
                "error_pixels": total_error,
                "mean_uncertain_ratio": float(np.mean([float(item["uncertain_ratio"]) for item in items])),
                "mean_target_ratio": float(np.mean([float(item["target_ratio"]) for item in items])),
                "mean_predicted_ratio": float(np.mean([float(item["predicted_ratio"]) for item in items])),
                "mean_probability": float(np.mean([float(item["mean_probability"]) for item in items])),
                "iou": float(iou),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
        )

    return summaries


def build_generalization_report(
    cases: list[dict[str, float | int | str]],
    scene_summaries: list[dict[str, float | int | str]],
    top_k: int = 5,
) -> dict[str, object]:
    return {
        "scene_summary": scene_summaries,
        "worst_scenes_by_f1": rank_cases(scene_summaries, key="f1", top_k=top_k, reverse=False),
        "worst_scenes_by_iou": rank_cases(scene_summaries, key="iou", top_k=top_k, reverse=False),
        "most_uncertain_scenes": rank_cases(scene_summaries, key="mean_uncertain_ratio", top_k=top_k),
        "worst_cases_by_iou": rank_cases(cases, key="iou", top_k=top_k, reverse=False),
        "highest_false_positive_cases": rank_cases(cases, key="false_positive_pixels", top_k=top_k),
        "highest_false_negative_cases": rank_cases(cases, key="false_negative_pixels", top_k=top_k),
        "most_uncertain_cases": rank_cases(cases, key="uncertain_ratio", top_k=top_k),
    }
