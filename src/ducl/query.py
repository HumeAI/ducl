"""General-purpose query engine for raw .feather scan files.

Equivalent to find ... | du pipelines but running against already-scanned data.
"""

from __future__ import annotations

import fnmatch
import os
import re
import sys

import polars as pl

from .ext import normalize_ext_expr
from .fmt import parse_size, fmt_size


def build_filters(
    *,
    dirs: bool = False,
    under: str | None = None,
    path_contains: str | None = None,
    name: str | None = None,
    name_glob: str | None = None,
    min_size: str | None = None,
    max_size: str | None = None,
) -> list[pl.Expr]:
    """Return a list of Polars filter expressions from query parameters."""
    filters: list[pl.Expr] = []

    # --dirs: directories only; default: files only
    if dirs:
        filters.append(pl.col("child_count") >= 0)
    else:
        filters.append(pl.col("child_count") == -1)

    # --under: path prefix
    if under:
        prefix = under
        # Ensure trailing slash so --under /mnt/weka/tmp doesn't match /mnt/weka/tmpXYZ
        if not prefix.endswith("/"):
            prefix += "/"
        filters.append(
            pl.col("path").str.starts_with(prefix) | (pl.col("path") == prefix.rstrip("/"))
        )

    # --path-contains: substring match on full path
    if path_contains:
        filters.append(pl.col("path").str.contains(path_contains, literal=True))

    # --name: exact match on any path component
    if name:
        escaped = re.escape(name)
        pattern = f"(^|/){escaped}($|/)"
        filters.append(pl.col("path").str.contains(pattern))

    # --name-glob: fnmatch glob on any path component
    if name_glob:
        fn_re = fnmatch.translate(name_glob)
        fn_re = fn_re.rstrip("\\Z$").lstrip("^")
        if fn_re.endswith("\\Z"):
            fn_re = fn_re[:-2]
        pattern = f"(^|/)({fn_re})($|/)"
        filters.append(pl.col("path").str.contains(pattern))

    # --min-size / --max-size (use dir_total_size for dirs)
    size_col = "dir_total_size" if dirs else "size"
    if min_size:
        filters.append(pl.col(size_col) >= parse_size(min_size))
    if max_size:
        filters.append(pl.col(size_col) <= parse_size(max_size))

    return filters


def run_query(
    feather: str,
    *,
    dirs: bool = False,
    under: str | None = None,
    path_contains: str | None = None,
    name: str | None = None,
    name_glob: str | None = None,
    ext: list[str] | None = None,
    min_size: str | None = None,
    max_size: str | None = None,
    top: int | None = None,
    delete: bool = False,
) -> None:
    """Query a .feather scan file with du-style output."""
    if delete and dirs:
        raise ValueError("--delete cannot be used with --dirs")

    # Lazy scan
    lf = pl.scan_ipc(feather)

    # Build and apply filters
    filters = build_filters(
        dirs=dirs,
        under=under,
        path_contains=path_contains,
        name=name,
        name_glob=name_glob,
        min_size=min_size,
        max_size=max_size,
    )
    for f in filters:
        lf = lf.filter(f)

    # --ext: needs normalized ext column
    if ext:
        ext_vals = [e.lower().lstrip(".") for e in ext]
        lf = lf.with_columns(normalize_ext_expr()).filter(pl.col("ext").is_in(ext_vals))

    # Use dir_total_size for directories, size for files
    size_col = "dir_total_size" if dirs else "size"

    # Collect
    if top or delete:
        df = lf.select("path", pl.col(size_col).alias("size")).collect()
    else:
        df = lf.select(pl.col(size_col).alias("size")).collect()

    # --top N: limit before delete/summary
    if top:
        df = df.sort("size", descending=True).head(top)

    total_size = df["size"].sum()
    count = len(df)

    if total_size is None:
        total_size = 0

    # --delete: remove matching files
    if delete:
        deleted = 0
        deleted_bytes = 0
        for row in df.iter_rows(named=True):
            path = row["path"]
            size = row["size"]
            try:
                os.remove(path)
                deleted += 1
                deleted_bytes += size
                print(f"Deleted {path} ({fmt_size(size)})")
            except FileNotFoundError:
                print(f"Skipped (not found): {path}", file=sys.stderr)
            except OSError as e:
                print(f"Skipped ({e.strerror}): {path}", file=sys.stderr)
        print(f"\nDeleted {deleted:,} files, {fmt_size(deleted_bytes)} total")
        return

    # Summary
    print(f"{fmt_size(total_size)}  {count:,} {'dirs' if dirs else 'files'}")

    # --top N
    if top and count > 0:
        print()
        for row in df.iter_rows(named=True):
            print(f"  {fmt_size(row['size']):>12}  {row['path']}")
