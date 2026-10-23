"""Download and verify the immutable Lending Club source snapshot."""

from __future__ import annotations

import argparse
import shutil
from datetime import UTC, datetime
from pathlib import Path

from _common import (
    AUDIT_DIR,
    DICTIONARY_FILE,
    RAW_DIR,
    RAW_FILE,
    detect_file_format,
    ensure_output_dirs,
    sha256_file,
    write_json,
)

KAGGLE_DATASET = "ethon0426/lending-club-20072020q1"


def find_largest(root: Path, suffixes: tuple[str, ...]) -> Path:
    candidates = [
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in suffixes
    ]
    if not candidates:
        raise FileNotFoundError(f"No file with suffix {suffixes} found under {root}")
    return max(candidates, key=lambda path: path.stat().st_size)


def copy_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".download")
    shutil.copy2(source, temp_path)
    temp_path.replace(destination)


def download(force: bool = False) -> None:
    ensure_output_dirs()
    have_files = RAW_FILE.exists() and DICTIONARY_FILE.exists()

    if have_files and not force:
        print("Raw dataset and dictionary already exist; download skipped.")
        download_path = None
    else:
        try:
            import kagglehub
        except ModuleNotFoundError as exc:
            raise SystemExit(
                "kagglehub is required for downloading. Run: python -m pip install kagglehub"
            ) from exc

        print(f"Downloading Kaggle dataset: {KAGGLE_DATASET}")
        download_path = Path(kagglehub.dataset_download(KAGGLE_DATASET))
        source_data = find_largest(download_path, (".gzip", ".gz", ".csv"))
        source_dictionary = find_largest(download_path, (".xlsx",))
        copy_atomic(source_data, RAW_FILE)
        copy_atomic(source_dictionary, DICTIONARY_FILE)
        print(f"Copied raw data to {RAW_FILE}")
        print(f"Copied dictionary to {DICTIONARY_FILE}")

    manifest = {
        "dataset": KAGGLE_DATASET,
        "downloaded_at_utc": datetime.now(UTC).isoformat(),
        "kaggle_cache_path": str(download_path) if download_path else None,
        "raw_file": str(RAW_FILE.relative_to(RAW_DIR.parent.parent)),
        "raw_file_size_bytes": RAW_FILE.stat().st_size,
        "raw_file_sha256": sha256_file(RAW_FILE),
        "raw_file_detected_format": detect_file_format(RAW_FILE),
        "dictionary_file": str(DICTIONARY_FILE.relative_to(RAW_DIR.parent.parent)),
        "dictionary_file_size_bytes": DICTIONARY_FILE.stat().st_size,
        "dictionary_file_sha256": sha256_file(DICTIONARY_FILE),
    }
    write_json(manifest, AUDIT_DIR / "source_manifest.json")

    print("Source verification complete:")
    print(f"  data format : {manifest['raw_file_detected_format']}")
    print(f"  data size   : {manifest['raw_file_size_bytes']:,} bytes")
    print(f"  sha256      : {manifest['raw_file_sha256']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download again and replace the existing raw snapshot.",
    )
    args = parser.parse_args()
    download(force=args.force)


if __name__ == "__main__":
    main()
