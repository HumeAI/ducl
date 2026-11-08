#!/usr/bin/env python3
"""Build a compact directory-index sidecar for a ducl feather.

The feather has one row per filesystem entry (files: child_count == -1,
directories: child_count >= 0) in the scanner's (parallel, unordered) order.
This produces a summary with one row per directory:

    dir_path, dir_offset, enddir_offset

where the offsets are *row indices* into the feather **reordered into DFS order**
(componentwise path sort), in which every directory's subtree is a contiguous
[dir_offset, enddir_offset) range -- i.e. dir_offset points at the directory's
own row (its "DIR") and enddir_offset is one past the last row of its subtree
(its "ENDDIR"). The summary is sorted/unique by directory path.

With this sidecar you can binary-search a directory and slice its whole subtree
out of the (DFS-ordered) feather without scanning -- the basis for per-directory
reconcile, lazy subtree loads, etc.

Usage:
    python dir_index.py SCAN.feather [-o SCAN.dirindex.parquet] [--write-sorted SORTED.feather]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import polars as pl

# DFS / componentwise order: replace '/' with a byte below every real path
# character so a directory sorts immediately before its descendants and before
# any sibling (separator becomes the smallest possible character).
_SEP = "\x01"


def build_index(feather: str, out: str, sorted_out: str | None = None) -> dict:
    t0 = time.time()
    df = (
        pl.scan_ipc(feather)
        .select(
            "path",
            "child_count",
            _k=pl.col("path").str.replace_all("/", _SEP, literal=True),
            _depth=pl.col("path").str.count_matches("/"),
        )
        .sort("_k")                       # DFS order: subtrees become contiguous
        .collect()
    )
    n = df.height
    depth = df["_depth"].to_numpy()
    is_dir = (df["child_count"].to_numpy() >= 0)

    # For each directory, find the end of its contiguous subtree:
    # enddir_offset[i] = first j > i with depth[j] <= depth[i]  (else n).
    # Single monotonic-stack pass over the DFS-ordered rows.
    depth_l = depth.tolist()              # python ints -> fast scalar access in loop
    end = np.empty(n, dtype=np.int64)
    stack: list[int] = []
    push = stack.append
    pop = stack.pop
    for i in range(n):
        di = depth_l[i]
        while stack and depth_l[stack[-1]] >= di:
            end[pop()] = i
        push(i)
    for j in stack:
        end[j] = n

    dir_pos = np.nonzero(is_dir)[0]
    summary = pl.DataFrame(
        {
            "dir_path": df["path"].gather(pl.Series(dir_pos)),  # already DFS-sorted
            "dir_offset": dir_pos.astype(np.int64),
            "enddir_offset": end[dir_pos],
        }
    )
    summary.write_parquet(out, compression="zstd")

    if sorted_out:
        df.drop("_k", "_depth").write_ipc(sorted_out, compression="zstd")

    feather_sz = os.path.getsize(feather)
    summary_sz = os.path.getsize(out)
    return {
        "rows": n,
        "dirs": int(is_dir.sum()),
        "files": int((~is_dir).sum()),
        "feather_bytes": feather_sz,
        "summary_bytes": summary_sz,
        "ratio": summary_sz / feather_sz,
        "secs": time.time() - t0,
        "out": out,
    }


def build_runs(feather: str, out_prefix: str) -> dict:
    """Run-list index over the ORIGINAL (unsorted) feather -- no feather sort.

    RLE-encodes the parent column in original row order: each maximal contiguous
    run of equal-parent rows becomes (dir_id, run_offset, run_len). Reads of a
    directory's rows are then exact (no dilution), and large directories that got
    smeared across the parallel output are simply represented by more runs.

    Writes two files:
      <prefix>.dirs.parquet : (dir_id, dir_path)  -- sorted/unique by path
      <prefix>.runs.parquet : (dir_id, run_offset, run_len)  -- sorted by (dir_id, run_offset)
    """
    t0 = time.time()
    df = (
        pl.scan_ipc(feather)
        .select(parent=pl.col("path").str.replace(r"/[^/]+$", ""))
        .with_row_index("off")
        .collect()
    )
    n = df.height

    # Run-length encode the parent column in original order.
    df = df.with_columns(
        _start=(pl.col("parent") != pl.col("parent").shift(1)).fill_null(True)
    ).with_columns(run_id=pl.col("_start").cast(pl.Int64).cum_sum())
    runs = df.group_by("run_id").agg(
        parent=pl.col("parent").first(),
        run_offset=pl.col("off").min(),
        run_len=pl.len(),
    )

    # Only the (small) directory table is sorted -- never the feather.
    dirmap = (
        runs.select("parent").unique().sort("parent").with_row_index("dir_id")
    )  # (dir_id, parent)
    runs = (
        runs.join(dirmap, on="parent")
        .select(pl.col("dir_id").cast(pl.UInt32),
                pl.col("run_offset").cast(pl.UInt64),
                pl.col("run_len").cast(pl.UInt32))
        .sort("dir_id", "run_offset")
    )

    # Cross-check: per-dir run_len totals must equal a direct group-by count,
    # and all runs must cover every row exactly once.
    assert runs["run_len"].sum() == n, "runs do not cover all rows"
    by_runs = runs.group_by("dir_id").agg(c=pl.col("run_len").sum())
    direct = (
        df.group_by("parent").agg(c=pl.len())
        .join(dirmap, on="parent").select("dir_id", c2="c")
    )
    chk = by_runs.join(direct, on="dir_id")
    assert (chk["c"] == chk["c2"]).all(), "run_len totals != direct child counts"

    dirs = dirmap.select(pl.col("dir_id").cast(pl.UInt32), dir_path="parent")

    dirs_path = out_prefix + ".dirs.parquet"
    runs_path = out_prefix + ".runs.parquet"
    dirs.write_parquet(dirs_path, compression="zstd")
    runs.write_parquet(runs_path, compression="zstd")

    feather_sz = os.path.getsize(feather)
    idx_sz = os.path.getsize(dirs_path) + os.path.getsize(runs_path)
    return {
        "rows": n,
        "dirs": dirs.height,
        "runs": runs.height,
        "runs_per_dir": runs.height / max(dirs.height, 1),
        "feather_bytes": feather_sz,
        "dirs_bytes": os.path.getsize(dirs_path),
        "runs_bytes": os.path.getsize(runs_path),
        "index_bytes": idx_sz,
        "ratio": idx_sz / feather_sz,
        "secs": time.time() - t0,
        "dirs_path": dirs_path,
        "runs_path": runs_path,
    }


def _mb(b):
    return b / 1e6


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("feather")
    ap.add_argument("-o", "--output")
    ap.add_argument("--write-sorted", metavar="FILE",
                    help="(range mode) also write the DFS-ordered feather the offsets index into")
    ap.add_argument("--runs", action="store_true",
                    help="build a run-list index over the ORIGINAL feather (no sort)")
    args = ap.parse_args()

    if args.runs:
        prefix = args.output or args.feather.removesuffix(".feather")
        s = build_runs(args.feather, prefix)
        print(f"feather : {args.feather}")
        print(f"  rows  : {s['rows']:,}  (dirs {s['dirs']:,})")
        print(f"  size  : {_mb(s['feather_bytes']):,.1f} MB")
        print(f"run-list index (no feather sort):")
        print(f"  dirs  : {s['dirs']:,} rows -> {_mb(s['dirs_bytes']):.1f} MB  ({s['dirs_path']})")
        print(f"  runs  : {s['runs']:,} rows -> {_mb(s['runs_bytes']):.1f} MB  ({s['runs_path']})")
        print(f"  runs/dir = {s['runs_per_dir']:.2f}")
        print(f"  total : {_mb(s['index_bytes']):.1f} MB  = {100 * s['ratio']:.2f}% of feather "
              f"(~{1 / s['ratio']:.0f}x smaller)  [built in {s['secs']:.1f}s, no feather sort]")
        return

    out = args.output or args.feather.removesuffix(".feather") + ".dirindex.parquet"
    s = build_index(args.feather, out, args.write_sorted)

    print(f"feather : {args.feather}")
    print(f"  rows  : {s['rows']:,}  (files {s['files']:,}, dirs {s['dirs']:,})")
    print(f"  size  : {_mb(s['feather_bytes']):,.1f} MB")
    print(f"summary : {out}")
    print(f"  rows  : {s['dirs']:,}  (one per directory)")
    print(f"  size  : {_mb(s['summary_bytes']):,.1f} MB")
    print(f"  ratio : {100 * s['ratio']:.2f}% of the feather  "
          f"(~{1 / s['ratio']:.0f}x smaller)  [built in {s['secs']:.1f}s]")


if __name__ == "__main__":
    main()
