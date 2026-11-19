#!/usr/bin/env python3
"""Benchmark two ways to compute the rsync first-pass diff of two captures:

  hash  : Polars streaming full-outer JOIN on path (hash join, spillable).
  merge : merge_sorted of two path-sorted sides + a vectorized adjacent-pair
          pass (the sort-merge diff). Assumes inputs already sorted by path.

Both classify every path as ONLY_A / ONLY_B / CHANGED / SAME and must agree.

Usage:
  # controlled, correctness-checked synthetic pair derived from one feather:
  python diff_bench.py --derive SCAN.feather
  # two real captures (e.g. consecutive WEKA scans):
  python diff_bench.py A.feather B.feather
"""

from __future__ import annotations

import argparse
import sys
import time

import polars as pl

KEY = "path"
COLS = (KEY, "size", "mtime")


def _load(feather: str) -> pl.DataFrame:
    return pl.scan_ipc(feather).select(*COLS).collect()


def derive_pair(feather: str):
    """A = the capture; B = a mutated copy with known ONLY_A/ONLY_B/CHANGED/SAME."""
    base = _load(feather).with_columns(_b=(pl.col(KEY).hash() % 20))
    A = base.select(*COLS)
    kept = base.filter(pl.col("_b") != 0)                       # bucket 0 -> ONLY_A (dropped from B)
    changed = kept.with_columns(                                # bucket 1 -> CHANGED
        size=pl.when(pl.col("_b") == 1).then(pl.col("size") + 1).otherwise(pl.col("size")),
        mtime=pl.when(pl.col("_b") == 1).then(pl.col("mtime") + 1).otherwise(pl.col("mtime")),
    ).select(*COLS)
    ghosts = (base.filter(pl.col("_b") == 2)                    # bucket 2 -> extra ONLY_B
              .select((pl.col(KEY) + "/__ghost__").alias(KEY), "size", "mtime"))
    B = pl.concat([changed, ghosts])
    expect = {
        "ONLY_A": base.filter(pl.col("_b") == 0).height,
        "CHANGED": base.filter(pl.col("_b") == 1).height,
        "ONLY_B": ghosts.height,
    }
    return A, B, expect


# ---------------------------------------------------------------------------
# Method 1: streaming hash full-outer join
# ---------------------------------------------------------------------------

def diff_hash(A: pl.DataFrame, B: pl.DataFrame, engine: str) -> pl.DataFrame:
    j = (
        A.lazy().with_columns(_a=pl.lit(True))
        .join(
            B.lazy().with_columns(_b=pl.lit(True)),
            on=KEY, how="full", coalesce=True, suffix="_B",
        )
        .select(
            KEY,
            label=pl.when(pl.col("_a").is_null()).then(pl.lit("ONLY_B"))
            .when(pl.col("_b").is_null()).then(pl.lit("ONLY_A"))
            .when((pl.col("size") == pl.col("size_B")) & (pl.col("mtime") == pl.col("mtime_B")))
            .then(pl.lit("SAME"))
            .otherwise(pl.lit("CHANGED")),
        )
    )
    return j.collect(engine=engine)


# ---------------------------------------------------------------------------
# Method 2: merge_sorted + vectorized adjacent-pair classification
# ---------------------------------------------------------------------------

def diff_merge(A_sorted: pl.DataFrame, B_sorted: pl.DataFrame, engine: str) -> pl.DataFrame:
    At = A_sorted.lazy().select(KEY, side=pl.lit("A"), size="size", mtime="mtime")
    Bt = B_sorted.lazy().select(KEY, side=pl.lit("B"), size="size", mtime="mtime")
    # merge_sorted interleaves into one path-sorted stream; matching paths become
    # adjacent (ties keep A before B). Classify with shifts -- no hash, no group_by.
    m = At.merge_sorted(Bt, key=KEY).with_columns(
        # fill_null(False) so the first/last rows (null shift) aren't dropped by filter
        _same_prev=(pl.col(KEY) == pl.col(KEY).shift(1)).fill_null(False),
        _same_next=(pl.col(KEY) == pl.col(KEY).shift(-1)).fill_null(False),
        _nsize=pl.col("size").shift(-1),
        _nmtime=pl.col("mtime").shift(-1),
    )
    res = (
        m.filter(~pl.col("_same_prev"))            # drop the B row of a matched pair
        .select(
            KEY,
            label=pl.when(pl.col("_same_next"))    # this is the A row of a pair; next is B
            .then(pl.when((pl.col("size") == pl.col("_nsize"))
                          & (pl.col("mtime") == pl.col("_nmtime")))
                  .then(pl.lit("SAME")).otherwise(pl.lit("CHANGED")))
            .when(pl.col("side") == "A").then(pl.lit("ONLY_A"))
            .otherwise(pl.lit("ONLY_B")),
        )
    )
    return res.collect(engine=engine)


def _counts(df: pl.DataFrame) -> dict:
    return dict(df.group_by("label").len().sort("label").iter_rows())


def diff_hash_lazy(a_path: str, b_path: str, engine: str) -> pl.DataFrame:
    """Fully lazy hash diff from scan_ipc -- never materializes either side
    (the streaming engine spills the build side). For WEKA-scale inputs."""
    A = pl.scan_ipc(a_path).select(*COLS).with_columns(_a=pl.lit(True))
    B = pl.scan_ipc(b_path).select(*COLS).with_columns(_b=pl.lit(True))
    j = A.join(B, on=KEY, how="full", coalesce=True, suffix="_B").select(
        label=pl.when(pl.col("_a").is_null()).then(pl.lit("ONLY_B"))
        .when(pl.col("_b").is_null()).then(pl.lit("ONLY_A"))
        .when((pl.col("size") == pl.col("size_B")) & (pl.col("mtime") == pl.col("mtime_B")))
        .then(pl.lit("SAME"))
        .otherwise(pl.lit("CHANGED")),
    )
    return j.group_by("label").len().collect(engine=engine)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a", nargs="?")
    ap.add_argument("b", nargs="?")
    ap.add_argument("--derive", metavar="FEATHER",
                    help="derive a controlled A/B pair from one feather")
    args = ap.parse_args()

    if args.a and args.b and not args.derive:
        # Scale path: stay lazy (scan_ipc -> streaming spilling join). Materializing
        # two 344M-row captures OOMs; this does not.
        print(f"lazy streaming hash diff:\n  A={args.a}\n  B={args.b}", file=sys.stderr)
        t = time.time()
        c = dict(diff_hash_lazy(args.a, args.b, "streaming").iter_rows())
        print(f"hash diff (lazy, streaming): {time.time()-t:6.1f}s   {c}")
        return

    if args.derive:
        print(f"deriving controlled pair from {args.derive} ...", file=sys.stderr)
        A, B, expect = derive_pair(args.derive)
    else:
        ap.error("give --derive FEATHER, or two feathers A B")

    print(f"A rows: {A.height:,}   B rows: {B.height:,}")

    def timed(fn, *a):
        out = {}
        for eng in ("streaming", "in-memory"):
            try:
                t = time.time(); df = fn(*a, eng); dt = time.time() - t
                out[eng] = (dt, _counts(df))
            except Exception as exc:  # noqa: BLE001  (catch Rust PanicException too)
                out[eng] = (None, f"FAILED: {type(exc).__name__}")
        return out

    # --- hash join (on unsorted inputs) ---
    hash_res = timed(diff_hash, A, B)

    # --- sort both sides (merge prerequisite; one-time if stored sorted) ---
    t = time.time()
    As = A.sort(KEY).rechunk(); Bs = B.sort(KEY).rechunk()
    t_sort = time.time() - t

    # --- merge diff (on sorted, rechunked inputs) ---
    merge_res = timed(diff_merge, As, Bs)

    def fmt(res):
        for eng, (dt, c) in res.items():
            ds = f"{dt:6.2f}s" if dt is not None else "   n/a"
            print(f"    {eng:10}: {ds}   {c}")

    print("\nhash full-outer join (unsorted inputs):")
    fmt(hash_res)
    print(f"\nsort both sides       : {t_sort:6.2f}s   (amortized if you store DFS/path-sorted)")
    print("merge_sorted diff (pre-sorted inputs):")
    fmt(merge_res)

    # agreement + correctness on whichever engine succeeded
    ch = next((c for _, c in hash_res.values() if isinstance(c, dict)), None)
    cm = next((c for _, c in merge_res.values() if isinstance(c, dict)), None)
    print(f"\nagree (hash == merge): {ch == cm}")
    if expect and ch:
        print(f"match expected mutations {expect}: "
              f"{all(ch.get(k, 0) == v for k, v in expect.items())}")


if __name__ == "__main__":
    main()
