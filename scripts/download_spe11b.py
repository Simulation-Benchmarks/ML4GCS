"""Download SPE11B benchmark participant data from a config file."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path


DEFAULT_CONFIG = Path("configs/spe11b_download.json")


def download_file(url: str, destination: Path) -> None:
    subprocess.run(
        ["curl", "-L", "-o", str(destination), url],
        check=True,
    )


def unzip_archive(archive: Path, output_dir: Path) -> None:
    subprocess.run(
        ["unzip", "-q", "-o", str(archive), "-d", str(output_dir)],
        check=True,
    )


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="JSON config file with the file IDs to download.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    output_dir = Path(config.get("output_dir", "spe11b"))
    file_ids = [int(file_id) for file_id in config["file_ids"]]

    if output_dir.exists():
        print(f"{output_dir} already exists; skipping download.")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        for file_id in file_ids:
            archive = tmpdir_path / f"{file_id}.zip"
            url = f"https://darus.uni-stuttgart.de/api/access/datafile/{file_id}"
            print(f"Downloading {file_id}...")
            download_file(url, archive)
            unzip_archive(archive, output_dir)

    print(f"Finished downloading into {output_dir}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
