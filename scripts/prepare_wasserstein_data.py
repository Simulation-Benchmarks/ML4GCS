"""Prepare SPE11B spatial-map pairs and Wasserstein-distance labels."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ml4gcs.data import (  # noqa: E402
    WassersteinDistanceStore,
    build_pair_dataset,
    load_map_values,
    read_map_records,
    save_wasserstein_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-list", type=Path, default=REPO_ROOT / "scripts/map_files.txt")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        action="append",
        dest="data_roots",
        help="Root containing participant directories; may be repeated.",
    )
    parser.add_argument(
        "--distance-root",
        type=Path,
        action="append",
        dest="distance_roots",
        help="Root containing dense Wasserstein tables; may be repeated.",
    )
    parser.add_argument("--field-name", default="tmCO2 [kg]")
    parser.add_argument("--height", type=int, default=120)
    parser.add_argument("--width", type=int, default=840)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--include-self", action="store_true")
    parser.add_argument("--ordered-pairs", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_roots = args.data_roots or [
        REPO_ROOT / "spe11b/spe11b",
        REPO_ROOT / "spe11b",
        Path("/home/jovyan/shared_folder/data/spe11b/spe11b"),
        Path("/home/jovyan/shared_folder/data/spe11b"),
    ]
    distance_roots = args.distance_roots or [
        REPO_ROOT / "spe11b/dense",
        Path("/home/jovyan/shared_folder/evaluation/spe11b/dense"),
    ]
    records = read_map_records(
        args.map_list,
        data_roots,
        strict=not args.allow_missing,
    )
    values = load_map_values(
        records,
        field_name=args.field_name,
        expected_size=args.height * args.width,
    )
    dataset = build_pair_dataset(
        records,
        values,
        WassersteinDistanceStore(distance_roots),
        include_self=args.include_self,
        unique_pairs=not args.ordered_pairs,
        grid_shape=(args.height, args.width),
    )
    save_wasserstein_dataset(
        dataset,
        args.output_dir / "wasserstein_pairs.npz",
        args.output_dir / "wasserstein_pairs.json",
    )
    print(f"Prepared {len(dataset.pairs)} pairs with input shape {dataset.X.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
