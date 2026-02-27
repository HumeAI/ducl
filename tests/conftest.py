"""Synthetic data fixtures for dashboard tests.

Generates Feather files matching the pwalk2 schema with controllable structure.
Tests use small pruning thresholds so synthetic data stays tiny.

Fixture tree (~50 files across 3 levels):
    /root/
      /root/big_a/              (many files, survives pruning)
        /root/big_a/sub1/       (survives pruning)
        /root/big_a/sub2/       (survives pruning)
        /root/big_a/sub3/       (pruned into __rest__)
      /root/big_b/              (survives pruning)
      /root/small_c/            (pruned into __rest__)
      /root/file_at_root.dat    (file directly in root)

Files have varied extensions (tar.gz, wav, mp3, txt, dat) and sizes (10KB to 5GB).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.feather as pf
import pytest

from ducl.schema import HIST_EDGES, AGG_GROUP_COLS, EXAMPLES_TOP_K
from ducl.ext import normalize_ext_expr

# Small thresholds for testing
TEST_MIN_FOLDER_SIZE = 1024  # 1 KB
TEST_MAX_CHILDREN = 3
TEST_MAX_TREE_LEVELS = 7
TEST_TOP_N_EXTENSIONS = 4
TEST_TOP_N_FOLDERS = 4


def _make_file_row(path: str, size: int, ext: str = "") -> dict:
    """Create a file row matching the pwalk2 schema subset needed by dashboard.py."""
    return {
        "inode_id": abs(hash(path)) % (2**63),
        "parent_inode_id": abs(hash(os.path.dirname(path))) % (2**63),
        "depth": path.count("/") - 1,
        "path": path,
        "ext": ext,
        "uid": 1000,
        "gid": 1000,
        "size": size,
        "fs_id": 1,
        "blocks": (size + 4095) // 4096,
        "nlink": 1,
        "mode": "100644",
        "atime": 1700000000,
        "mtime": 1700000000,
        "ctime": 1700000000,
        "child_count": -1,  # file
        "dir_total_size": 0,
    }


def _make_dir_row(path: str, child_count: int = 0) -> dict:
    """Create a directory row."""
    return {
        "inode_id": abs(hash(path + "/dir")) % (2**63),
        "parent_inode_id": abs(hash(os.path.dirname(path))) % (2**63),
        "depth": path.count("/") - 1,
        "path": path,
        "ext": "",
        "uid": 1000,
        "gid": 1000,
        "size": 4096,
        "fs_id": 1,
        "blocks": 8,
        "nlink": 2 + child_count,
        "mode": "040755",
        "atime": 1700000000,
        "mtime": 1700000000,
        "ctime": 1700000000,
        "child_count": child_count,
        "dir_total_size": 0,
    }


# The actual file entries for the synthetic tree
SYNTHETIC_FILES: list[dict] = []
SYNTHETIC_DIRS: list[dict] = []

# ---- Directories ----
SYNTHETIC_DIRS.extend([
    _make_dir_row("/root", 4),        # big_a, big_b, small_c, file_at_root.dat
    _make_dir_row("/root/big_a", 3),  # sub1, sub2, sub3
    _make_dir_row("/root/big_a/sub1", 0),
    _make_dir_row("/root/big_a/sub2", 0),
    _make_dir_row("/root/big_a/sub3", 0),
    _make_dir_row("/root/big_b", 0),
    _make_dir_row("/root/small_c", 0),
])

# ---- Files in /root/ (directly) ----
SYNTHETIC_FILES.append(_make_file_row("/root/file_at_root.dat", 50_000, "dat"))

# ---- Files in /root/big_a/sub1/ (audio folder -- big files) ----
SYNTHETIC_FILES.extend([
    _make_file_row("/root/big_a/sub1/track1.wav", 5_000_000_000, "wav"),   # 5 GB -- bin 11
    _make_file_row("/root/big_a/sub1/track2.wav", 2_000_000_000, "wav"),   # 2 GB -- bin 9
    _make_file_row("/root/big_a/sub1/track3.mp3", 500_000_000, "mp3"),     # 500 MB -- bin 8
    _make_file_row("/root/big_a/sub1/track4.mp3", 50_000_000, "mp3"),      # 50 MB -- bin 6
    _make_file_row("/root/big_a/sub1/notes.txt", 200_000, "txt"),          # 200 KB -- bin 1
    _make_file_row("/root/big_a/sub1/readme.txt", 500, "txt"),             # 500 B -- bin 0
    _make_file_row("/root/big_a/sub1/archive.tar.gz", 100_000_000, "gz"),  # 100 MB -- bin 6 (ext normalized to tar.gz)
    _make_file_row("/root/big_a/sub1/data.dat", 5_000_000, "dat"),         # 5 MB -- bin 4
])

# ---- Files in /root/big_a/sub2/ (more moderate files) ----
SYNTHETIC_FILES.extend([
    _make_file_row("/root/big_a/sub2/video.mp4", 1_500_000_000, "mp4"),  # 1.5 GB -- bin 9
    _make_file_row("/root/big_a/sub2/audio.wav", 800_000_000, "wav"),     # 800 MB -- bin 8
    _make_file_row("/root/big_a/sub2/clip.mp3", 30_000_000, "mp3"),       # 30 MB -- bin 5
    _make_file_row("/root/big_a/sub2/log.txt", 2_000_000, "txt"),         # 2 MB -- bin 3
    _make_file_row("/root/big_a/sub2/config", 100, ""),                   # 100 B -- bin 0, ext = "(no ext)"
])

# ---- Files in /root/big_a/sub3/ (small folder -- pruned into __rest__) ----
SYNTHETIC_FILES.extend([
    _make_file_row("/root/big_a/sub3/tiny1.txt", 100, "txt"),     # bin 0
    _make_file_row("/root/big_a/sub3/tiny2.txt", 200, "txt"),     # bin 0
    _make_file_row("/root/big_a/sub3/tiny3.dat", 300, "dat"),     # bin 0
])

# ---- Files in /root/big_a/ directly (immediate files for non-leaf histogram) ----
SYNTHETIC_FILES.extend([
    _make_file_row("/root/big_a/index.dat", 10_000_000, "dat"),    # 10 MB -- bin 4
    _make_file_row("/root/big_a/backup.tar.gz", 400_000_000, "gz"),  # 400 MB -- bin 8 (ext tar.gz)
])

# ---- Files in /root/big_b/ ----
SYNTHETIC_FILES.extend([
    _make_file_row("/root/big_b/dataset.tar.gz", 3_000_000_000, "gz"),  # 3 GB -- bin 9 (ext tar.gz)
    _make_file_row("/root/big_b/model.dat", 1_000_000_000, "dat"),       # 1 GB -- bin 8
    _make_file_row("/root/big_b/results.txt", 5_000_000, "txt"),         # 5 MB -- bin 4
    _make_file_row("/root/big_b/summary.mp3", 15_000_000, "mp3"),        # 15 MB -- bin 5
    _make_file_row("/root/big_b/README", 800, ""),                       # bin 0, ext = "(no ext)"
])

# ---- Files in /root/small_c/ (small total -- pruned into root's __rest__) ----
SYNTHETIC_FILES.extend([
    _make_file_row("/root/small_c/a.txt", 50, "txt"),
    _make_file_row("/root/small_c/b.txt", 60, "txt"),
    _make_file_row("/root/small_c/c.dat", 70, "dat"),
])

ALL_ROWS = SYNTHETIC_DIRS + SYNTHETIC_FILES

# Pre-compute expected total file size (files only)
EXPECTED_TOTAL_SIZE = sum(f["size"] for f in SYNTHETIC_FILES)


def _write_feather(rows: list[dict], path: str) -> None:
    """Write rows to Feather using pyarrow schema matching csv2feather.py."""
    schema = pa.schema([
        ("inode_id", pa.uint64()),
        ("parent_inode_id", pa.uint64()),
        ("depth", pa.int32()),
        ("path", pa.utf8()),
        ("ext", pa.utf8()),
        ("uid", pa.uint32()),
        ("gid", pa.uint32()),
        ("size", pa.int64()),
        ("fs_id", pa.uint64()),
        ("blocks", pa.int64()),
        ("nlink", pa.uint32()),
        ("mode", pa.utf8()),
        ("atime", pa.int64()),
        ("mtime", pa.int64()),
        ("ctime", pa.int64()),
        ("child_count", pa.int64()),
        ("dir_total_size", pa.int64()),
    ])
    arrays = {}
    for field in schema:
        col_name = field.name
        values = [r[col_name] for r in rows]
        arrays[col_name] = pa.array(values, type=field.type)
    table = pa.table(arrays, schema=schema)
    pf.write_feather(table, path, compression="zstd")


@pytest.fixture
def synthetic_feather(tmp_path: Path) -> Path:
    """Write the full synthetic tree to a Feather file, return its path."""
    out = tmp_path / "scan.feather"
    _write_feather(ALL_ROWS, str(out))
    return out


@pytest.fixture
def build_result(tmp_path: Path, synthetic_feather: Path) -> Path:
    """Run build() with test thresholds, return the output directory."""
    from ducl.dashboard import build

    out_dir = tmp_path / "dashboard_data"
    build(
        synthetic_feather,
        out_dir,
        max_tree_levels=TEST_MAX_TREE_LEVELS,
        max_children=TEST_MAX_CHILDREN,
        min_folder_size=TEST_MIN_FOLDER_SIZE,
        top_n_extensions=TEST_TOP_N_EXTENSIONS,
        top_n_folders=TEST_TOP_N_FOLDERS,
    )
    return out_dir


# ---- Update fixtures ----

# For update tests: a "rescan" of /root/big_b with modified files
RESCAN_FILES: list[dict] = []
RESCAN_DIRS: list[dict] = []

RESCAN_DIRS.append(_make_dir_row("/root/big_b", 0))

RESCAN_FILES.extend([
    # Same files but different sizes (simulating changes)
    _make_file_row("/root/big_b/dataset.tar.gz", 4_000_000_000, "gz"),  # was 3 GB, now 4 GB
    _make_file_row("/root/big_b/model.dat", 500_000_000, "dat"),         # was 1 GB, now 500 MB
    _make_file_row("/root/big_b/results.txt", 5_000_000, "txt"),         # unchanged
    _make_file_row("/root/big_b/summary.mp3", 15_000_000, "mp3"),        # unchanged
    _make_file_row("/root/big_b/README", 800, ""),                       # unchanged
    # New file
    _make_file_row("/root/big_b/extra.wav", 200_000_000, "wav"),         # 200 MB -- new
])

RESCAN_ALL = RESCAN_DIRS + RESCAN_FILES
RESCAN_TOTAL_SIZE = sum(f["size"] for f in RESCAN_FILES)
ORIGINAL_BIG_B_SIZE = sum(
    f["size"] for f in SYNTHETIC_FILES if f["path"].startswith("/root/big_b/")
)


def _generate_agg_sidecars(feather_path: Path) -> None:
    """Generate .agg.parquet and .examples.parquet alongside a feather file.

    Mimics what csv2feather does during streaming aggregation, but reads
    from the already-written feather file. Used by test fixtures.
    """
    df = pl.read_ipc(str(feather_path))
    df = df.filter(pl.col("child_count") == -1)

    # Compute derived columns (same as process_batch)
    df = df.select("path", "ext", "size").with_columns(
        pl.col("path").str.replace(r"/[^/]+$", "").alias("dir_path"),
        pl.col("path").str.extract(r"([^/]+)/[^/]+$").alias("leaf_folder"),
        pl.col("path").str.extract(r"([^/]+)$").alias("filename"),
    ).with_columns(
        normalize_ext_expr(),
        (pl.col("dir_path").str.count_matches("/") + 2).cast(pl.Int32).alias("n_components"),
    )

    sizes_np = df["size"].to_numpy().astype(float)
    bins = np.digitize(sizes_np, HIST_EDGES).astype(np.int8)
    df = df.with_columns(pl.lit(pl.Series("size_bin", bins)).alias("size_bin"))

    # Agg
    agg_df = (
        df.group_by(AGG_GROUP_COLS)
        .agg(
            pl.len().cast(pl.Int64).alias("file_count"),
            pl.col("size").sum().alias("total_size"),
        )
    )

    # Examples: top-3 largest per (dir_path, size_bin)
    examples_df = (
        df.sort("size", descending=True)
        .group_by(["dir_path", "size_bin"])
        .head(EXAMPLES_TOP_K)
        .select("dir_path", "size_bin", "filename", "ext", "size")
    )

    base = str(feather_path).removesuffix(".feather")
    agg_df.write_parquet(base + ".agg.parquet")
    examples_df.write_parquet(base + ".examples.parquet")


@pytest.fixture
def build_result_agg(tmp_path: Path, synthetic_feather: Path) -> Path:
    """Generate agg sidecars, then run build() which auto-detects the fast path."""
    from ducl.dashboard import build

    _generate_agg_sidecars(synthetic_feather)
    out_dir = tmp_path / "dashboard_data_agg"
    build(
        synthetic_feather,
        out_dir,
        max_tree_levels=TEST_MAX_TREE_LEVELS,
        max_children=TEST_MAX_CHILDREN,
        min_folder_size=TEST_MIN_FOLDER_SIZE,
        top_n_extensions=TEST_TOP_N_EXTENSIONS,
        top_n_folders=TEST_TOP_N_FOLDERS,
    )
    return out_dir


@pytest.fixture
def rescan_feather(tmp_path: Path) -> Path:
    """Write the rescan data for /root/big_b to a Feather file."""
    out = tmp_path / "rescan_big_b.feather"
    _write_feather(RESCAN_ALL, str(out))
    return out


@pytest.fixture
def updated_result(build_result: Path, rescan_feather: Path) -> Path:
    """Run update() on the build result, return the dashboard directory."""
    from ducl.dashboard import update

    update(build_result, rescan_feather)
    return build_result
