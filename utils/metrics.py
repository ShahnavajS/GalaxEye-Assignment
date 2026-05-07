from __future__ import annotations

import torch


def sigmoid_probabilities(predictions: torch.Tensor, from_logits: bool = True) -> torch.Tensor:
    if from_logits:
        return torch.sigmoid(predictions)
    return predictions


def _prepare_binary_predictions(
    predictions: torch.Tensor,
    threshold: float = 0.5,
    from_logits: bool = True,
) -> torch.Tensor:
    predictions = sigmoid_probabilities(predictions, from_logits=from_logits)
    return (predictions >= threshold).to(torch.int64)


def _prepare_binary_targets(targets: torch.Tensor) -> torch.Tensor:
    return (targets >= 0.5).to(torch.int64)


@torch.no_grad()
def binary_confusion_counts(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    from_logits: bool = True,
) -> dict[str, int]:
    predicted_mask = _prepare_binary_predictions(predictions, threshold=threshold, from_logits=from_logits)
    target_mask = _prepare_binary_targets(targets)

    if predicted_mask.ndim == 4 and predicted_mask.shape[1] == 1:
        predicted_mask = predicted_mask[:, 0]
    if target_mask.ndim == 4 and target_mask.shape[1] == 1:
        target_mask = target_mask[:, 0]

    tp = torch.logical_and(predicted_mask == 1, target_mask == 1).sum().item()
    fp = torch.logical_and(predicted_mask == 1, target_mask == 0).sum().item()
    fn = torch.logical_and(predicted_mask == 0, target_mask == 1).sum().item()
    tn = torch.logical_and(predicted_mask == 0, target_mask == 0).sum().item()

    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def metrics_from_confusion_counts(counts: dict[str, int], eps: float = 1e-7) -> dict[str, float]:
    tp = float(counts["tp"])
    fp = float(counts["fp"])
    fn = float(counts["fn"])

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    f1 = (2.0 * precision * recall) / (precision + recall + eps)

    return {
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


@torch.no_grad()
def compute_binary_segmentation_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    from_logits: bool = True,
) -> dict[str, float]:
    counts = binary_confusion_counts(predictions, targets, threshold=threshold, from_logits=from_logits)
    metrics = metrics_from_confusion_counts(counts)
    return {**counts, **metrics}


def merge_confusion_counts(total: dict[str, int], batch_counts: dict[str, int]) -> dict[str, int]:
    for key in ("tp", "fp", "fn", "tn"):
        total[key] = total.get(key, 0) + int(batch_counts[key])
    return total


def build_thresholds(start: float = 0.3, end: float = 0.8, step: float = 0.05) -> list[float]:
    if step <= 0:
        raise ValueError("Threshold step must be positive.")
    if end < start:
        raise ValueError("Threshold search end must be greater than or equal to start.")

    thresholds: list[float] = []
    value = float(start)
    while value <= float(end) + 1e-8:
        thresholds.append(round(value, 6))
        value += float(step)
    return thresholds


def initialize_threshold_search(thresholds: list[float]) -> list[dict[str, object]]:
    return [
        {
            "threshold": float(threshold),
            "counts": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
        }
        for threshold in thresholds
    ]


@torch.no_grad()
def update_threshold_search(
    search_state: list[dict[str, object]],
    predictions: torch.Tensor,
    targets: torch.Tensor,
    from_logits: bool = True,
) -> None:
    probabilities = sigmoid_probabilities(predictions, from_logits=from_logits)
    for item in search_state:
        threshold = float(item["threshold"])
        counts = binary_confusion_counts(probabilities, targets, threshold=threshold, from_logits=False)
        merge_confusion_counts(item["counts"], counts)


def summarize_threshold_search(
    search_state: list[dict[str, object]],
    metric_name: str = "f1",
) -> dict[str, object]:
    if metric_name not in {"iou", "precision", "recall", "f1"}:
        raise ValueError(f"Unsupported threshold search metric: {metric_name}")

    curve: list[dict[str, float]] = []
    best_item: dict[str, float] | None = None

    for item in search_state:
        threshold = float(item["threshold"])
        counts = item["counts"]
        metrics = metrics_from_confusion_counts(counts)
        row = {
            "threshold": threshold,
            "iou": metrics["iou"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
        }
        curve.append(row)

        if best_item is None or row[metric_name] > best_item[metric_name]:
            best_item = row

    if best_item is None:
        raise RuntimeError("Threshold search did not receive any data.")

    return {
        "metric_name": metric_name,
        "best_threshold": float(best_item["threshold"]),
        "best_metrics": {
            "iou": float(best_item["iou"]),
            "precision": float(best_item["precision"]),
            "recall": float(best_item["recall"]),
            "f1": float(best_item["f1"]),
        },
        "curve": curve,
    }
