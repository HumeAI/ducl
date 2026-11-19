"""Shared constants: column names, Arrow schema, histogram edges, aggregation keys."""

from __future__ import annotations

import numpy as np
import pyarrow as pa

# ---------------------------------------------------------------------------
# Column names of the canonical 17-column table (SCHEMA order)
# ---------------------------------------------------------------------------

COLUMN_NAMES = [
    "inode_id",
    "parent_inode_id",
    "depth",
    "path",
    "ext",
    "uid",
    "gid",
    "size",
    "fs_id",
    "blocks",
    "nlink",
    "mode",
    "atime",
    "mtime",
    "ctime",
    "child_count",
    "dir_total_size",
]

# ---------------------------------------------------------------------------
# Arrow schema for raw Feather files
# ---------------------------------------------------------------------------

SCHEMA = pa.schema(
    [
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
    ]
)

# ---------------------------------------------------------------------------
# Histogram bin edges (shared by csv2feather, dashboard, preprocess, etc.)
# ---------------------------------------------------------------------------

HIST_EDGES = np.array([
    100 * 1024,        # 100 KB
    300 * 1024,        # 300 KB
    1   * 1024**2,     # 1 MB
    3   * 1024**2,     # 3 MB
    10  * 1024**2,     # 10 MB
    30  * 1024**2,     # 30 MB
    100 * 1024**2,     # 100 MB
    300 * 1024**2,     # 300 MB
    1   * 1024**3,     # 1 GB
    3   * 1024**3,     # 3 GB
    10  * 1024**3,     # 10 GB
], dtype=float)

# ---------------------------------------------------------------------------
# Aggregation constants
# ---------------------------------------------------------------------------

AGG_GROUP_COLS = ["dir_path", "ext", "leaf_folder", "size_bin", "n_components"]
EXAMPLES_TOP_K = 3
