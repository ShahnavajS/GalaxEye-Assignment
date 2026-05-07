from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from project_bootstrap import ensure_local_packages  # noqa: E402

ensure_local_packages()

from utils.dataset_source import resolve_dataset_root  # noqa: E402
from utils.logger import create_logger  # noqa: E402
from utils.runtime import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download or reuse the official Hugging Face dataset.")
    parser.add_argument("--config", type=Path, default=Path("configs/final_config.yaml"))
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    log_dir = Path(config["paths"]["output_root"]) / "dataset_setup_logs"
    logger, _ = create_logger(log_dir, name="prepare_hf_dataset")

    dataset_root = resolve_dataset_root(
        config,
        download_if_missing=True,
        force_download=args.force_download,
        logger=logger,
    )

    payload = {
        "dataset_root": str(dataset_root),
        "source": str(config["data"].get("source", "auto")),
        "hf_repo_id": config["data"].get("hf_repo_id"),
    }

    print(json.dumps(payload, indent=2))

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
