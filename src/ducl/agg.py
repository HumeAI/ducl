"""Batch processing for aggregation sidecars."""

from __future__ import annotations

import numpy as np
import polars as pl
import pyarrow as pa

from .schema import AGG_GROUP_COLS, EXAMPLES_TOP_K, HIST_EDGES
from .ext import normalize_ext_expr

# How a hard-linked file's bytes are charged to the directories that link it.
# The scan has one row per directory ENTRY, each carrying the inode's full size, so
# summing `size` counts an inode once per link (6.2 TB / 28M entries too many on the
# 2026-11-06 /mnt/weka capture).
#
#   split  each link is charged size // nlink. Stateless (a row needs only its own
#          nlink), so it is deterministic and two captures of an unchanged tree agree
#          exactly. Totals are exact when every link is inside the scan; a link outside
#          it takes its share with it.
#   first  the first link met in stream order is charged the whole file, later ones 0.
#          Whole files, like du -- but WHICH directory pays depends on walk order,
#          which differs between scans (phantom moves in a diff), and it needs a set
#          of every hard-linked inode seen (~1 GB at 15M inodes).
#   full   every link is charged the full size (du -l; ducl's behaviour before 0.3.1).
#
# Only `total_size` is affected. file_count still counts entries, and size_bin and the
# examples keep the file's real size.
HARDLINK_MODES = ("split", "first", "full")
DEFAULT_HARDLINKS = "split"


def check_hardlinks(mode: str) -> str:
    if mode not in HARDLINK_MODES:
        raise ValueError(f"hardlinks must be one of {HARDLINK_MODES}, got {mode!r}")
    return mode


def charged_size_expr(mode: str = DEFAULT_HARDLINKS) -> pl.Expr:
    """`size` as charged to this entry, for a frame holding a whole capture.

    split/full are row-wise. `first` uses the frame's own row order, so it is only
    meaningful when the frame is the complete capture (streaming callers use FirstSeen).
    nlink 0 (S3 objects, synthetic rows) is treated as 1.
    """
    check_hardlinks(mode)
    size = pl.col("size")
    if mode == "split":
        return (size // pl.col("nlink").cast(pl.Int64).clip(lower_bound=1)).alias("size")
    if mode == "first":
        repeat = (pl.col("nlink") > 1) & ~pl.struct("fs_id", "inode_id").is_first_distinct()
        return pl.when(repeat).then(0).otherwise(size).alias("size")
    return size


class FirstSeen:
    """Which (fs_id, inode_id) pairs a stream has already produced -- `first` mode's state.

    Per-filesystem sets of inode numbers. Only entries with nlink > 1 are ever offered, so
    the cost is one set operation per hard link, not per file.
    """

    def __init__(self) -> None:
        self._seen: dict[int, set[int]] = {}

    def repeats(self, fs_id: np.ndarray, inode_id: np.ndarray) -> np.ndarray:
        """Bool mask: True where this entry's inode was already seen (here or earlier)."""
        out = np.zeros(len(inode_id), dtype=bool)
        for fs in np.unique(fs_id).tolist():
            idx = np.flatnonzero(fs_id == fs)
            seen = self._seen.setdefault(fs, set())
            for i, ino in zip(idx.tolist(), inode_id[idx].tolist()):
                if ino in seen:
                    out[i] = True
                else:
                    seen.add(ino)
        return out


def process_batch(
    batch: pa.RecordBatch,
    hardlinks: str = DEFAULT_HARDLINKS,
    first_seen: FirstSeen | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Process a single pyarrow RecordBatch into agg and example DataFrames.

    *hardlinks* picks how hard-linked files are charged (see HARDLINK_MODES); `first`
    needs the stream's shared *first_seen*.

    Returns (agg_df, examples_df).
    """
    check_hardlinks(hardlinks)
    df = pl.from_arrow(batch)

    # Filter to files only
    df = df.filter(pl.col("child_count") == -1)
    if df.height == 0:
        agg_empty = pl.DataFrame(schema={
            "dir_path": pl.Utf8, "ext": pl.Utf8, "leaf_folder": pl.Utf8,
            "size_bin": pl.Int8, "n_components": pl.Int32,
            "file_count": pl.Int64, "total_size": pl.Int64,
        })
        ex_empty = pl.DataFrame(schema={
            "dir_path": pl.Utf8, "size_bin": pl.Int8, "filename": pl.Utf8,
            "ext": pl.Utf8, "size": pl.Int64,
        })
        return agg_empty, ex_empty

    # `charged`: the bytes this entry contributes to total_size. `size` stays the file's
    # real size for binning and examples.
    if hardlinks == "first":
        if first_seen is None:
            raise ValueError("hardlinks='first' needs a FirstSeen shared across the stream")
        charged = df["size"].to_numpy().copy()
        cand = np.flatnonzero(df["nlink"].to_numpy() > 1)
        if cand.size:
            rep = first_seen.repeats(df["fs_id"].to_numpy()[cand], df["inode_id"].to_numpy()[cand])
            charged[cand[rep]] = 0
        df = df.with_columns(pl.Series("charged", charged, dtype=pl.Int64))
    else:
        df = df.with_columns(charged_size_expr(hardlinks).alias("charged"))

    # Compute derived columns
    df = df.select("path", "ext", "size", "charged").with_columns(
        pl.col("path").str.replace(r"/[^/]+$", "").alias("dir_path"),
        pl.col("path").str.extract(r"([^/]+)/[^/]+$").alias("leaf_folder"),
        pl.col("path").str.extract(r"([^/]+)$").alias("filename"),
    ).with_columns(
        normalize_ext_expr(),
        (pl.col("dir_path").str.count_matches("/") + 2).cast(pl.Int32).alias("n_components"),
    )

    # Assign size bins
    sizes_np = df["size"].to_numpy().astype(float)
    bins = np.digitize(sizes_np, HIST_EDGES).astype(np.int8)
    df = df.with_columns(pl.lit(pl.Series("size_bin", bins)).alias("size_bin"))

    # Agg: group by (dir_path, ext, leaf_folder, size_bin, n_components)
    agg_df = (
        df.group_by(AGG_GROUP_COLS)
        .agg(
            pl.len().cast(pl.Int64).alias("file_count"),
            pl.col("charged").sum().alias("total_size"),
        )
    )

    # Examples: top-3 largest files per (dir_path, size_bin)
    examples_df = (
        df.sort("size", descending=True)
        .group_by(["dir_path", "size_bin"])
        .head(EXAMPLES_TOP_K)
        .select("dir_path", "size_bin", "filename", "ext", "size")
    )

    return agg_df, examples_df


def compact_agg(batches: list[pl.DataFrame]) -> pl.DataFrame:
    """Re-aggregate accumulated agg batches to bound memory."""
    combined = pl.concat(batches)
    return (
        combined.group_by(AGG_GROUP_COLS)
        .agg(
            pl.col("file_count").sum().alias("file_count"),
            pl.col("total_size").sum().alias("total_size"),
        )
    )


def compact_examples(batches: list[pl.DataFrame]) -> pl.DataFrame:
    """Re-pick top-3 from accumulated example batches."""
    combined = pl.concat(batches)
    return (
        combined.sort("size", descending=True)
        .group_by(["dir_path", "size_bin"])
        .head(EXAMPLES_TOP_K)
    )


class SidecarBuilder:
    """Accumulate the .agg / .examples sidecars from a stream of scan batches.

    Owns the hard-link policy and its state, and compacts every few batches to bound
    memory. Used by the scan writer and by `ducl agg` (re-aggregating a stored feather).
    """

    COMPACT_EVERY = 5

    def __init__(self, hardlinks: str = DEFAULT_HARDLINKS) -> None:
        self.hardlinks = check_hardlinks(hardlinks)
        self._first_seen = FirstSeen() if hardlinks == "first" else None
        self._agg: list[pl.DataFrame] = []
        self._examples: list[pl.DataFrame] = []

    def add(self, batch: pa.RecordBatch) -> None:
        agg_df, ex_df = process_batch(batch, self.hardlinks, self._first_seen)
        self._agg.append(agg_df)
        self._examples.append(ex_df)
        if len(self._agg) >= self.COMPACT_EVERY:
            self._agg = [compact_agg(self._agg)]
            self._examples = [compact_examples(self._examples)]

    def finish(self) -> tuple[pl.DataFrame, pl.DataFrame] | None:
        """(agg, examples), or None when no batch was added."""
        if not self._agg:
            return None
        return compact_agg(self._agg), compact_examples(self._examples)
