from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from project_bootstrap import ensure_local_packages  # noqa: E402

ensure_local_packages()

import numpy as np  # noqa: E402

from utils.dataset_utils import (  # noqa: E402
    describe_array,
    find_missing_modalities,
    get_expected_modalities,
    index_split_directory,
    is_mask_modality,
    load_image_array,
)
from utils.dataset_source import resolve_dataset_root  # noqa: E402
from utils.label_utils import (  # noqa: E402
    BINARY_LABEL_NAMES,
    RAW_LABEL_NAMES,
    get_unique_values,
    remap_mask,
    validate_mask_values,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect GalaxEye dataset structure and masks.")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--data-root", type=Path, default=None, help="Optional explicit dataset root.")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Split folders to inspect.",
    )
    parser.add_argument(
        "--save-json",
        type=Path,
        default=None,
        help="Optional path to save the inspection report as JSON.",
    )
    parser.add_argument(
        "--max-examples-with-issues",
        type=int,
        default=20,
        help="Maximum number of corrupted files or missing-sample examples to keep in the report.",
    )
    return parser.parse_args()


def setup_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    return logging.getLogger("explore_dataset")


def counter_to_dict(counter: Counter) -> dict[str, int]:
    return {str(key): int(counter[key]) for key in sorted(counter)}


def summarize_class_counts(
    counts: Counter,
    label_names: dict[int, str],
) -> dict[str, dict[str, float | int]]:
    total = sum(counts.values())
    summary: dict[str, dict[str, float | int]] = {}
    for value in sorted(counts):
        summary[str(value)] = {
            "name": label_names.get(int(value), "unknown"),
            "count": int(counts[value]),
            "fraction": round(float(counts[value] / total), 6) if total else 0.0,
        }
    return summary


def inspect_split(
    split_dir: Path,
    logger: logging.Logger,
    max_examples_with_issues: int,
) -> dict[str, object]:
    sample_index, index_warnings = index_split_directory(split_dir, recursive=True, logger=logger)
    expected_modalities = get_expected_modalities(sample_index)
    missing_modalities = find_missing_modalities(sample_index, expected_modalities)

    modality_shapes: dict[str, Counter] = defaultdict(Counter)
    modality_channels: dict[str, Counter] = defaultdict(Counter)
    modality_dtypes: dict[str, Counter] = defaultdict(Counter)
    mask_unique_values: dict[str, set[int]] = defaultdict(set)

    raw_class_counts: Counter = Counter()
    remapped_class_counts: Counter = Counter()
    corrupted_files: list[str] = []
    invalid_masks: list[dict[str, object]] = []
    spatial_mismatches: list[dict[str, object]] = []

    for sample_id, sample_files in sorted(sample_index.items()):
        sample_spatial_shapes: dict[str, tuple[int, int]] = {}

        for modality_name, file_path in sorted(sample_files.items()):
            array = load_image_array(file_path, logger=logger)
            if array is None:
                corrupted_files.append(str(file_path))
                continue

            metadata = describe_array(array)
            modality_shapes[modality_name][str(metadata["shape"])] += 1
            modality_channels[modality_name][str(metadata["channels"])] += 1
            modality_dtypes[modality_name][str(metadata["dtype"])] += 1
            sample_spatial_shapes[modality_name] = tuple(metadata["spatial_shape"])

            if not is_mask_modality(modality_name):
                continue

            unique_values = get_unique_values(array)
            mask_unique_values[modality_name].update(unique_values)

            is_valid, invalid_values = validate_mask_values(array)
            if not is_valid:
                invalid_masks.append(
                    {
                        "sample_id": sample_id,
                        "path": str(file_path),
                        "invalid_values": invalid_values,
                    }
                )
                continue

            raw_values, raw_counts = np_unique_counts(array)
            raw_class_counts.update(dict(zip(raw_values, raw_counts)))

            remapped = remap_mask(array)
            remapped_values, remapped_counts = np_unique_counts(remapped)
            remapped_class_counts.update(dict(zip(remapped_values, remapped_counts)))

        if len(set(sample_spatial_shapes.values())) > 1:
            spatial_mismatches.append(
                {
                    "sample_id": sample_id,
                    "spatial_shapes": {name: list(shape) for name, shape in sample_spatial_shapes.items()},
                }
            )

    return {
        "split": split_dir.name,
        "sample_count": len(sample_index),
        "modalities": expected_modalities,
        "missing_modalities_count": len(missing_modalities),
        "missing_modalities_examples": missing_modalities[:max_examples_with_issues],
        "index_warnings": index_warnings[:max_examples_with_issues],
        "corrupted_file_count": len(corrupted_files),
        "corrupted_file_examples": corrupted_files[:max_examples_with_issues],
        "invalid_mask_count": len(invalid_masks),
        "invalid_mask_examples": invalid_masks[:max_examples_with_issues],
        "spatial_mismatch_count": len(spatial_mismatches),
        "spatial_mismatch_examples": spatial_mismatches[:max_examples_with_issues],
        "modality_shapes": {name: counter_to_dict(counter) for name, counter in modality_shapes.items()},
        "modality_channels": {name: counter_to_dict(counter) for name, counter in modality_channels.items()},
        "modality_dtypes": {name: counter_to_dict(counter) for name, counter in modality_dtypes.items()},
        "mask_unique_values": {
            name: sorted(int(value) for value in values)
            for name, values in mask_unique_values.items()
        },
        "raw_class_distribution": summarize_class_counts(raw_class_counts, RAW_LABEL_NAMES),
        "remapped_class_distribution": summarize_class_counts(remapped_class_counts, BINARY_LABEL_NAMES),
    }


def np_unique_counts(array) -> tuple[list[int], list[int]]:
    values, counts = np.unique(array, return_counts=True)
    return [int(value) for value in values.tolist()], [int(count) for count in counts.tolist()]


def print_report(report: dict[str, object]) -> None:
    print("\nDataset inspection summary")
    print("=" * 80)

    for split_name, summary in report["splits"].items():
        print(f"\n[{split_name}]")
        print(f"Samples: {summary['sample_count']}")
        print(f"Modalities: {', '.join(summary['modalities'])}")
        print(f"Missing modality groups: {summary['missing_modalities_count']}")
        print(f"Corrupted files: {summary['corrupted_file_count']}")
        print(f"Invalid masks: {summary['invalid_mask_count']}")
        print(f"Spatial mismatches: {summary['spatial_mismatch_count']}")

        print("Shapes by modality:")
        for modality_name, shapes in summary["modality_shapes"].items():
            print(f"  - {modality_name}: {shapes}")

        print("Channels by modality:")
        for modality_name, channels in summary["modality_channels"].items():
            print(f"  - {modality_name}: {channels}")

        if summary["mask_unique_values"]:
            print("Mask values by modality:")
            for modality_name, values in summary["mask_unique_values"].items():
                print(f"  - {modality_name}: {values}")

        if summary["raw_class_distribution"]:
            print("Raw class distribution:")
            for label_value, stats in summary["raw_class_distribution"].items():
                print(
                    f"  - {label_value} ({stats['name']}): "
                    f"count={stats['count']}, fraction={stats['fraction']}"
                )

        if summary["remapped_class_distribution"]:
            print("Remapped class distribution:")
            for label_value, stats in summary["remapped_class_distribution"].items():
                print(
                    f"  - {label_value} ({stats['name']}): "
                    f"count={stats['count']}, fraction={stats['fraction']}"
                )


def main() -> None:
    args = parse_args()
    logger = setup_logger()

    if args.data_root is not None:
        data_root = args.data_root.resolve()
    else:
        from utils.runtime import load_config  # noqa: E402

        config = load_config(args.config)
        data_root = resolve_dataset_root(config, download_if_missing=True, logger=logger)

    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    report: dict[str, object] = {
        "data_root": str(data_root),
        "splits": {},
    }

    for split_name in args.splits:
        split_dir = data_root / split_name
        if not split_dir.exists():
            logger.warning("Split directory not found: %s", split_dir)
            continue

        logger.info("Inspecting %s", split_dir)
        report["splits"][split_name] = inspect_split(
            split_dir=split_dir,
            logger=logger,
            max_examples_with_issues=args.max_examples_with_issues,
        )

    print_report(report)

    if args.save_json is not None:
        output_path = args.save_json.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info("Saved report to %s", output_path)


if __name__ == "__main__":
    main()
