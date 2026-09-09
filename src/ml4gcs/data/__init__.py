"""Data loading helpers for SPE11B."""

from .discovery import (
    discover_fluidflower_spatial_map_paths,
    discover_spatial_map_paths,
    find_data_root,
    find_fluidflower_data_root,
    find_spe11b_data_root,
)
from .spatial_map import (
    FLUIDFLOWER_SPATIAL_MAP_TIME_RE,
    SPE11B_SPATIAL_MAP_TIME_RE,
    SpatialMapSnapshot,
    load_spatial_map_csv,
    parse_fluidflower_spatial_map_time,
    parse_spatial_map_time,
)
from .wasserstein import (
    MapRecord,
    PairRecord,
    WassersteinDataset,
    WassersteinDistanceStore,
    build_pair_dataset,
    load_wasserstein_arrays,
    load_map_values,
    parse_spe11b_time,
    read_map_records,
    save_wasserstein_dataset,
)
from .index import SpatialMapRef, build_spatial_map_index, group_spatial_map_index

__all__ = [
    "FLUIDFLOWER_SPATIAL_MAP_TIME_RE",
    "find_data_root",
    "find_fluidflower_data_root",
    "SPE11B_SPATIAL_MAP_TIME_RE",
    "SpatialMapSnapshot",
    "SpatialMapRef",
    "build_spatial_map_index",
    "discover_fluidflower_spatial_map_paths",
    "discover_spatial_map_paths",
    "find_spe11b_data_root",
    "group_spatial_map_index",
    "load_spatial_map_csv",
    "parse_fluidflower_spatial_map_time",
    "parse_spatial_map_time",
    "MapRecord",
    "PairRecord",
    "WassersteinDataset",
    "WassersteinDistanceStore",
    "build_pair_dataset",
    "load_map_values",
    "load_wasserstein_arrays",
    "parse_spe11b_time",
    "read_map_records",
    "save_wasserstein_dataset",
]
