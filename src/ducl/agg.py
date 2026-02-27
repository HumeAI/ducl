"""Batch processing for aggregation sidecars."""

from __future__ import annotations

import numpy as np
import polars as pl
import pyarrow as pa

from .schema import AGG_GROUP_COLS, EXAMPLES_TOP_K, HIST_EDGES
from .ext import normalize_ext_expr


def process_batch(batch: pa.RecordBatch) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Process a single pyarrow RecordBatch into agg and example DataFrames.

    Returns (agg_df, examples_df).
    """
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

    # Compute derived columns
    df = df.select("path", "ext", "size").with_columns(
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
            pl.col("size").sum().alias("total_size"),
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
