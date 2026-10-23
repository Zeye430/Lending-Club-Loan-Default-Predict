"""Run the complete pre-pipeline data-discovery workflow in order."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
STAGES = [
    "01_download_data.py",
    "02_validate_raw_data.py",
    "03_build_column_catalog.py",
    "04_analyze_target_cohorts.py",
    "05_run_eda.py",
    "06_compare_original_approach.py",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Start from raw-data validation when the source snapshot already exists.",
    )
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--max-sample-rows", type=int, default=200_000)
    args = parser.parse_args()

    stages = STAGES[1:] if args.skip_download else STAGES
    for stage in stages:
        command = [sys.executable, str(SCRIPT_DIR / stage)]
        if stage in {"02_validate_raw_data.py", "04_analyze_target_cohorts.py", "05_run_eda.py"}:
            command.extend(["--chunk-size", str(args.chunk_size)])
        if stage == "05_run_eda.py":
            command.extend(["--max-sample-rows", str(args.max_sample_rows)])

        print(f"\nRunning {stage}", flush=True)
        subprocess.run(command, check=True)

    print("\nData-discovery workflow complete.")


if __name__ == "__main__":
    main()
