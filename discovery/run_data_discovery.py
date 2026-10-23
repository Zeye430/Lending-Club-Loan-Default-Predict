"""Run the raw-data audit in order and publish its report.

The EDA notebook (discovery/eda.ipynb) runs separately because it reads the built
cohort: python -m lc_risk.data.cohort, then execute the notebook.
"""

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
    "05_write_audit_report.py",
]
CHUNKED_STAGES = {"02_validate_raw_data.py", "04_analyze_target_cohorts.py"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Start from raw-data validation when the source snapshot already exists.",
    )
    parser.add_argument("--chunk-size", type=int, default=100_000)
    args = parser.parse_args()

    stages = STAGES[1:] if args.skip_download else STAGES
    for stage in stages:
        command = [sys.executable, str(SCRIPT_DIR / stage)]
        if stage in CHUNKED_STAGES:
            command.extend(["--chunk-size", str(args.chunk_size)])

        print(f"\nRunning {stage}", flush=True)
        subprocess.run(command, check=True)

    print("\nData-discovery audit complete.")


if __name__ == "__main__":
    main()
