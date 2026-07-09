from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


STAGES = [
    "01_download.py",
    "02_normalize.py",
    "03_clean.py",
    "04_filter.py",
    "05_dedup.py",
    "06_split.py",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="dataset_builder/config.yaml")
    parser.add_argument(
        "--start-at",
        choices=STAGES,
        default=STAGES[0],
        help="Resume the pipeline from this stage.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent / "scripts"
    start_index = STAGES.index(args.start_at)
    for stage in STAGES[start_index:]:
        script = script_dir / stage
        print(f"\n=== {stage} ===")
        subprocess.run([sys.executable, str(script), "--config", args.config], check=True)


if __name__ == "__main__":
    main()
