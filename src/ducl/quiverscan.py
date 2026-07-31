"""Scan a filesystem with quiver's walker instead of bundled pwalk2 + CSV.

pwalk2 emits CSV that pyarrow re-parses into Arrow; quiver's bvm emits Arrow STAT batches
straight off its walker pool, so this deletes the whole CSV hop. The identity fields are
byte-identical between the two scanners -- same `ino`, `parent_ino`, `dev`, `blocks`, `nlink`,
`size`, and the same `mode` (pwalk2 prints it as "%07o", quiver ships the int) -- so the work
here is projection, not re-derivation:

    pwalk2 column      quiver source
    inode_id           ino
    parent_inode_id    parent_ino          (0 for the scan root)
    depth              depth - 1           (pwalk2 puts root's children at 0, root at -1)
    path               root + "/" + path   (quiver's paths are root-relative)
    ext                derived from the basename
    fs_id              dev
    mode               "%07o" of mode
    atime/mtime/ctime  *_ns // 1e9         (pwalk2 is second-precision)
    child_count        group_by(parent_ino).len()      -1 for files
    dir_total_size     group_by(parent_ino).sum(size)   0 for files

Two things quiver does not hand over and this has to add: the SCAN ROOT'S OWN ROW (bvm emits
the tree under the root, not the root itself) and the two directory aggregates, which pwalk2
accumulates in-scanner because it knows when a directory is finished. A streaming pass cannot
know that -- a directory's children may land in any batch -- so the table is materialized
before the aggregates are joined on. That is the same shape ducl already uses for dashboards.
"""

from __future__ import annotations

import os
import sys

import polars as pl
import pyarrow as pa

from .schema import SCHEMA


def _octal7(col: str = "mode") -> pl.Expr:
    """mode -> pwalk2's "%07o". Polars has no octal format, so build the digits."""
    digits = [((pl.col(col).cast(pl.Int64) // (8 ** i)) % 8).cast(pl.Utf8) for i in range(5, -1, -1)]
    return pl.concat_str([pl.lit("0")] + digits).alias("mode")


def _collect(root: str, nworkers: int, bvm_exe: str | None):
    """Drive bvm's SCAN_FS and return every STAT batch as one polars frame."""
    sys.path.insert(0, "/mnt/weka/jpc/src/quiver")
    from quiver.exec import blocks

    exe = bvm_exe or blocks.default_bvm()
    b = blocks._Bvm(exe, nworkers)
    b.scan_fs(0, root)
    parts, rows, last = [], 0, [0]
    while True:
        t, pld = b.read()
        if t is None or t == 1:
            break
        if t != 0:
            continue
        _sid, _blk, df = blocks._stat_df(pld)
        if df.height:
            parts.append(df)
            rows += df.height
            # THROTTLED. bvm emits a batch per directory, so printing per batch cost 1.4 s of
            # a 5.4 s walk -- the progress line was 26% of the scan.
            if rows - last[0] >= 50_000:
                last[0] = rows
                print(f"\r  {rows:,} entries", end="", flush=True, file=sys.stderr)
    print(file=sys.stderr)
    if not parts:
        return pl.DataFrame(schema={"path": pl.Utf8})
    return pl.concat(parts)


def to_ducl_frame(df: pl.DataFrame, root: str, st=None) -> pl.DataFrame:
    """quiver STAT batch(es) -> ducl's pwalk2-compatible table.

    `st` is the scan root's stat taken BEFORE the walk. Walking a directory updates its
    atime, so stat-ing the root afterwards reports the scan's own read rather than the
    state it had going in -- the one column that differed from pwalk2 until it was hoisted.
    """
    root = root.rstrip("/")
    st = st or os.lstat(root)

    # the scan root itself: bvm walks what is UNDER the root, pwalk2 emits the root too,
    # with parent 0 and depth -1
    rootrow = pl.DataFrame({
        "ino": [st.st_ino], "parent_ino": [0], "depth": [0],
        "path": [""], "size": [0], "blocks": [0], "nlink": [st.st_nlink],
        "mode": [st.st_mode], "uid": [st.st_uid], "gid": [st.st_gid], "dev": [st.st_dev],
        "atime_ns": [st.st_atime_ns], "mtime_ns": [st.st_mtime_ns],
        "ctime_ns": [st.st_ctime_ns], "is_dir": [1],
    }).cast({"ino": pl.UInt64, "parent_ino": pl.UInt64, "depth": pl.Int32, "size": pl.Int64,
             "blocks": pl.Int64, "nlink": pl.Int32, "mode": pl.Int32, "uid": pl.Int32,
             "gid": pl.Int32, "dev": pl.UInt64, "is_dir": pl.UInt8})

    keep = ["ino", "parent_ino", "depth", "path", "size", "blocks", "nlink", "mode",
            "uid", "gid", "dev", "atime_ns", "mtime_ns", "ctime_ns", "is_dir"]
    df = pl.concat([rootrow.select(keep), df.select(keep)]) if df.height else rootrow.select(keep)

    # directory aggregates over DIRECT children, which is what pwalk2 accumulates in-scanner
    agg = (df.filter(pl.col("parent_ino") != 0)
             .group_by("parent_ino")
             .agg(pl.len().alias("_cc"), pl.col("size").sum().alias("_dts")))

    full = pl.when(pl.col("path") == "").then(pl.lit(root)) \
             .otherwise(pl.concat_str([pl.lit(root + "/"), pl.col("path")]))
    out = (df.with_columns(full.alias("_full"))
             .join(agg, left_on="ino", right_on="parent_ino", how="left")
             .with_columns(
                 inode_id=pl.col("ino"),
                 parent_inode_id=pl.col("parent_ino"),
                 # pwalk2: root -1, its children 0. quiver: root would be 0, children 1.
                 depth=(pl.col("depth") - 1).cast(pl.Int32),
                 path=pl.col("_full"),
                 # extension = after the last dot, and only when something precedes the dot
                 # (so ".gitignore" has none, matching a dotfile having no extension)
                 ext=pl.when(pl.col("is_dir") == 1).then(pl.lit(""))
                      .otherwise(pl.col("_full").str.extract(r"[^/.]\.([^./]+)$")
                                 .fill_null("")),
                 fs_id=pl.col("dev"),
                 atime=(pl.col("atime_ns") // 1_000_000_000),
                 mtime=(pl.col("mtime_ns") // 1_000_000_000),
                 ctime=(pl.col("ctime_ns") // 1_000_000_000),
                 # files are -1/0; that is how ducl's agg tells files from directories
                 child_count=pl.when(pl.col("is_dir") == 1)
                               .then(pl.col("_cc").fill_null(0)).otherwise(-1).cast(pl.Int64),
                 dir_total_size=pl.when(pl.col("is_dir") == 1)
                                  .then(pl.col("_dts").fill_null(0)).otherwise(0).cast(pl.Int64),
             )
             .with_columns(_octal7())
             .select("inode_id", "parent_inode_id", "depth", "path", "ext", "uid", "gid",
                     "size", "fs_id", "blocks", "nlink", "mode", "atime", "mtime", "ctime",
                     "child_count", "dir_total_size")
             .cast({"inode_id": pl.UInt64, "parent_inode_id": pl.UInt64, "depth": pl.Int32,
                    "uid": pl.UInt32, "gid": pl.UInt32, "size": pl.Int64, "fs_id": pl.UInt64,
                    "blocks": pl.Int64, "nlink": pl.UInt32, "atime": pl.Int64,
                    "mtime": pl.Int64, "ctime": pl.Int64}))
    return out


def scan_filesystem_quiver(path: str, output: str, *, do_agg: bool = True,
                           nworkers: int = 64, bvm_exe: str | None = None) -> None:
    """Scan *path* with quiver's walker and write ducl's Feather (+ agg sidecars)."""
    from .scan import feather_from_batches

    root = os.path.abspath(path)
    st = os.lstat(root)                       # before the walk touches root's atime
    df = _collect(root, nworkers, bvm_exe)
    tbl = to_ducl_frame(df, root, st).to_arrow().cast(SCHEMA)
    feather_from_batches(tbl.to_batches(max_chunksize=1 << 20), output, do_agg=do_agg)
