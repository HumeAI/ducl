"""Extension normalization (Polars expression)."""

from __future__ import annotations

import polars as pl


def normalize_ext_expr() -> pl.Expr:
    """Polars expression that normalizes the ext column.

    Handles compound extensions (tar.gz/bz2/zst), missing -> '(no ext)', lowercase.
    """
    return (
        pl.when(pl.col("path").str.ends_with(".tar.gz")).then(pl.lit("tar.gz"))
        .when(pl.col("path").str.ends_with(".tar.bz2")).then(pl.lit("tar.bz2"))
        .when(pl.col("path").str.ends_with(".tar.zst")).then(pl.lit("tar.zst"))
        .when((pl.col("ext") == "") | pl.col("ext").is_null()).then(pl.lit("(no ext)"))
        .otherwise(pl.col("ext").str.to_lowercase())
        .alias("ext")
    )
