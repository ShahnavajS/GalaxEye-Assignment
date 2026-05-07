from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.logger import save_json
from utils.reporting import load_csv_rows, save_markdown_table, save_rows_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export report-ready tables and selected artifacts.")
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--index-path", type=Path, default=Path("outputs/experiment_index.csv"))
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def to_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("-inf")


def copy_if_exists(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_directory_contents(source_dir: Path, destination_dir: Path) -> None:
    if not source_dir.exists():
        return
    destination_dir.mkdir(parents=True, exist_ok=True)
    for path in source_dir.iterdir():
        if path.is_file():
            shutil.copy2(path, destination_dir / path.name)


def main() -> None:
    args = parse_args()
    rows = load_csv_rows(args.index_path)
    output_dir = args.output_root / "report_assets"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not rows:
        save_json(output_dir / "report_summary.json", {"message": "No experiment rows found."})
        return

    ranked_rows = sorted(rows, key=lambda row: to_float(row, "best_f1"), reverse=True)
    top_rows = ranked_rows[: max(args.top_k, 1)]

    save_rows_csv(output_dir / "experiment_table.csv", ranked_rows)
    save_markdown_table(
        ranked_rows,
        output_dir / "experiment_table.md",
        columns=[
            "experiment_name",
            "architecture",
            "encoder_name",
            "loss_name",
            "modalities",
            "best_threshold",
            "best_iou",
            "best_precision",
            "best_recall",
            "best_f1",
        ],
    )
    save_rows_csv(output_dir / "top_experiments.csv", top_rows)

    best_row = top_rows[0]
    best_experiment_name = str(best_row["experiment_name"])
    best_experiment_dir = args.output_root / best_experiment_name
    best_output_dir = output_dir / "best_experiment"

    copy_if_exists(best_experiment_dir / "summary.json", best_output_dir / "summary.json")
    copy_if_exists(best_experiment_dir / "eval" / "metrics_summary.json", best_output_dir / "metrics_summary.json")
    copy_if_exists(best_experiment_dir / "eval" / "metrics_summary.csv", best_output_dir / "metrics_summary.csv")
    copy_if_exists(best_experiment_dir / "inference" / "failure_report.json", best_output_dir / "failure_report.json")
    copy_directory_contents(best_experiment_dir / "inference" / "montages", best_output_dir / "montages")

    save_json(
        output_dir / "report_summary.json",
        {
            "best_experiment": best_row,
            "num_experiments": len(ranked_rows),
            "top_experiments": top_rows,
        },
    )


if __name__ == "__main__":
    main()
