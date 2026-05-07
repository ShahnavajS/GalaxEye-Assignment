from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_LABELS = ("No Change", "Change")


def confusion_matrix_from_counts(counts: dict[str, int]) -> np.ndarray:
    return np.asarray(
        [
            [int(counts["tn"]), int(counts["fp"])],
            [int(counts["fn"]), int(counts["tp"])],
        ],
        dtype=np.int64,
    )


def save_metrics_csv(metrics_path: str | Path, row: dict[str, Any]) -> None:
    metrics_path = Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def save_rows_csv(rows_path: str | Path, rows: list[dict[str, Any]]) -> None:
    rows_path = Path(rows_path)
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        rows_path.write_text("", encoding="utf-8")
        return

    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with rows_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_confusion_matrix_csv(
    counts: dict[str, int],
    output_path: str | Path,
    labels: tuple[str, str] = DEFAULT_LABELS,
) -> None:
    matrix = confusion_matrix_from_counts(counts)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual/predicted", *labels])
        writer.writerow([labels[0], int(matrix[0, 0]), int(matrix[0, 1])])
        writer.writerow([labels[1], int(matrix[1, 0]), int(matrix[1, 1])])


def save_confusion_matrix_figure(
    counts: dict[str, int],
    output_path: str | Path,
    labels: tuple[str, str] = DEFAULT_LABELS,
    normalize: bool = False,
) -> None:
    matrix = confusion_matrix_from_counts(counts).astype(np.float32)
    display = matrix.copy()
    if normalize:
        row_sums = np.clip(display.sum(axis=1, keepdims=True), a_min=1.0, a_max=None)
        display = display / row_sums

    figure, axis = plt.subplots(figsize=(5, 4.5))
    image = axis.imshow(display, cmap="Blues")
    plt.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    axis.set_xticks(range(len(labels)))
    axis.set_yticks(range(len(labels)))
    axis.set_xticklabels(labels)
    axis.set_yticklabels(labels)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_title("Pixel Confusion Matrix")

    for row in range(display.shape[0]):
        for column in range(display.shape[1]):
            raw_value = int(matrix[row, column])
            shown_value = f"{display[row, column]:.3f}" if normalize else f"{raw_value}"
            axis.text(column, row, shown_value, ha="center", va="center", color="black")

    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def load_csv_rows(csv_path: str | Path) -> list[dict[str, str]]:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def save_markdown_table(rows: list[dict[str, Any]], output_path: str | Path, columns: list[str]) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        output_path.write_text("No experiment rows found.\n", encoding="utf-8")
        return

    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = [
        "| " + " | ".join(str(row.get(column, "")) for column in columns) + " |"
        for row in rows
    ]
    output_path.write_text("\n".join([header, divider, *body]) + "\n", encoding="utf-8")
