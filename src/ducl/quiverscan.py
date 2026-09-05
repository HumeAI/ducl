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
    child_count        in_off on dir rows, -1 on files     (SCAN_DIRS; see below)
    dir_total_size     digest on dir rows, 0 on files

STREAMING. The two directory aggregates used to be a `group_by(parent_ino)` over the finished
table, which meant holding the whole thing in memory: fine for a home directory, ~200 GB and
an OOM for the 162M rows of /mnt/weka. They are now computed in the walker, where pwalk2 has
always computed them -- bvm's SCAN_DIRS mode emits each directory's own row from its own walk,
with child_count/dir_total_size already filled in. Batches are cut on directory boundaries, so
one is normally a whole number of complete [directory row][its children] chunks: transformable
on its own, and the unit an rsync-style differ wants to compare. Nothing accumulates; memory
is one batch.

A directory bigger than BVM_SCAN_CHUNK_MAX (262144 rows) IS split -- /mnt/weka's largest holds
3.3M entries and buffering one whole costs ~0.5 GB per walker. It stays reassemblable because
the dir row carries child_count and precedes its children, so a differ reads N rows after a
dir row; to do that across a split it must first demultiplex on the batch's block field, which
identifies the emitting walker (-idx-2). This projection does not care: it is row-wise.

The scan root's row comes from bvm too (SCAN_FS emitted only what was UNDER the root), with
its stat taken before the walk touched its atime. Every row carries `err` -- 0, or -errno for
a directory that could not be read, which keeps its stat and its row and reports 0 children.
"""

from __future__ import annotations

import collections
import errno
import os
import sys

import polars as pl
import pyarrow as pa

from .schema import SCHEMA

# the STAT columns this projection reads; bvm ships more (link, blocks_p, chunks...)
_SRC = ["ino", "parent_ino", "depth", "path", "size", "blocks", "nlink", "mode",
        "uid", "gid", "dev", "atime_ns", "mtime_ns", "ctime_ns", "is_dir",
        "in_off", "digest", "err"]


def _trim_strings(batch: pa.RecordBatch) -> pa.RecordBatch:
    """Shrink string data buffers to the extent the offsets actually describe.

    polars' string builders over-allocate (a `path` buffer came out 121,180,928 bytes for
    121,180,885 bytes of paths). pyarrow's IPC writer compresses the WHOLE buffer, records the
    padded length, and reads its own output back happily -- but polars' reader checks the
    decompressed size against the offsets and rejects the file:

        ComputeError: Malformed IPC file: got unexpected decompressed buffer size
                      121180928, expected 121180885

    ducl's dashboard reads these with polars, so an un-trimmed write is a file ducl cannot
    open -- silently, since the scan itself reports success. Buffer.slice is zero-copy, so
    this costs nothing.
    """
    cols = []
    for arr in batch.columns:
        t = arr.type
        if pa.types.is_string(t) or pa.types.is_large_string(t):
            if arr.offset:
                # a slice shares its parent's buffers, so the trim below would read the wrong
                # offsets entirely. concat_arrays materializes it at offset 0 with exact
                # buffers. Skipping these instead is what let the full-mount write out an
                # untrimmed batch and stay unreadable after the first fix.
                arr = pa.concat_arrays([arr])
            valid, offs, data = arr.buffers()
            w = 8 if pa.types.is_large_string(t) else 4
            n = len(arr)
            end = int.from_bytes(offs[w * n:w * (n + 1)].to_pybytes(), "little")
            if data is not None and data.size > end:
                arr = pa.Array.from_buffers(t, n, [valid, offs, data.slice(0, end)])
        cols.append(arr)
    return pa.RecordBatch.from_arrays(cols, schema=batch.schema)


def _octal7(col: str = "mode") -> pl.Expr:
    """mode -> pwalk2's "%07o". Polars has no octal format, so build the digits."""
    digits = [((pl.col(col).cast(pl.Int64) // (8 ** i)) % 8).cast(pl.Utf8) for i in range(5, -1, -1)]
    return pl.concat_str([pl.lit("0")] + digits).alias("mode")


def to_ducl_frame(df: pl.DataFrame, root: str) -> pl.DataFrame:
    """One SCAN_DIRS STAT batch -> ducl's pwalk2-compatible table.

    Pure per-batch projection: no cross-batch state, because the aggregates arrived with the
    rows. `root` is only needed to turn bvm's root-relative paths absolute.
    """
    root = root.rstrip("/")
    full = pl.when(pl.col("path") == "").then(pl.lit(root)) \
             .otherwise(pl.concat_str([pl.lit(root + "/"), pl.col("path")]))
    isdir = pl.col("is_dir") == 1
    return (df.select(_SRC)
              .with_columns(full.alias("_full"))
              .with_columns(
                  inode_id=pl.col("ino"),
                  parent_inode_id=pl.col("parent_ino"),
                  # pwalk2: root -1, its children 0. quiver: root 0, children 1.
                  depth=(pl.col("depth") - 1).cast(pl.Int32),
                  path=pl.col("_full"),
                  # extension = after the LAST dot of the basename, empty when that dot is the
                  # basename's first character (so ".gitignore" has none). Hence `[^/]` and
                  # not `[^/.]`: the preceding character may itself be a dot -- pwalk2 only
                  # asks whether the dot is at index 0, so `[[...slug]]` is ext "slug]]".
                  # DIRECTORIES GET ONE TOO: pwalk2 reads the basename without looking at the
                  # type, so `foo-1.2.dist-info` is ext "dist-info" on both sides.
                  ext=pl.col("_full").str.extract(r"[^/]\.([^./]+)$").fill_null(""),
                  fs_id=pl.col("dev"),
                  atime=(pl.col("atime_ns") // 1_000_000_000),
                  mtime=(pl.col("mtime_ns") // 1_000_000_000),
                  ctime=(pl.col("ctime_ns") // 1_000_000_000),
                  # files are -1/0; that is how ducl's agg tells files from directories
                  child_count=pl.when(isdir).then(pl.col("in_off")).otherwise(-1).cast(pl.Int64),
                  dir_total_size=pl.when(isdir).then(pl.col("digest")).otherwise(0).cast(pl.Int64),
              )
              .with_columns(_octal7())
              .select("inode_id", "parent_inode_id", "depth", "path", "ext", "uid", "gid",
                      "size", "fs_id", "blocks", "nlink", "mode", "atime", "mtime", "ctime",
                      "child_count", "dir_total_size")
              .cast({"inode_id": pl.UInt64, "parent_inode_id": pl.UInt64, "depth": pl.Int32,
                     "uid": pl.UInt32, "gid": pl.UInt32, "size": pl.Int64, "fs_id": pl.UInt64,
                     "blocks": pl.Int64, "nlink": pl.UInt32, "atime": pl.Int64,
                     "mtime": pl.Int64, "ctime": pl.Int64}))


# COALESCE BEFORE PROJECTING. bvm's walkers flush at 20000 rows EACH, so a scan arrives as
# thousands of small batches, and the projection below is ~20 polars expressions whose
# per-call overhead does not care how many rows it is handed. Transforming per arriving batch
# cost 2.5x the whole pwalk2 pipeline. Concatenating first amortizes it, and is free of the
# thing streaming exists to avoid: the buffer is bounded by this constant, not by the tree.
# Adjacent batches concatenate cleanly -- each is already a whole number of directories, so
# the complete-chunk invariant survives.
_COALESCE_ROWS = 1 << 20


def stream_batches(root: str, nworkers: int = 128, bvm_exe: str | None = None,
                   progress: bool = True):
    """Yield ducl-schema Arrow RecordBatches as bvm's walker produces them.

    This is the whole scanner as far as any consumer is concerned: a differ that wants
    directory chunks can read the same stream and split on `child_count >= 0` (a dir row) --
    each such row is followed by exactly its `child_count` children.
    """
    sys.path.insert(0, "/mnt/weka/jpc/src/quiver")
    from quiver.exec import blocks

    b = blocks._Bvm(bvm_exe or blocks.default_bvm(), nworkers)
    b.scan_dirs(0, root)
    rows, last, held, nheld = 0, 0, [], 0
    nerr = collections.Counter()

    def flush():
        # RECHUNK, or coalescing only half works. pyarrow's to_batches splits on the
        # underlying chunk boundaries and treats max_chunksize as a cap, not a target, so a
        # concat of N bvm batches still yields N Arrow batches however many rows they hold --
        # and ducl's aggregator pays its per-batch cost N times (2.5s of a 3.6s scan here).
        df = pl.concat(held, rechunk=True) if len(held) > 1 else held[0].rechunk()
        held.clear()
        # no max_chunksize: the frame is rechunked, so this is one batch at offset 0 -- and a
        # batch is already sized by _COALESCE_ROWS. Slicing here would only recreate the
        # offset != 0 case _trim_strings has to work around.
        tbl = to_ducl_frame(df, root).to_arrow().cast(SCHEMA)
        return [_trim_strings(b) for b in tbl.to_batches()]

    while True:
        t, pld = b.read()
        if t is None or t == 1:
            break
        if t != 0:
            continue
        _sid, _blk, df = blocks._stat_df(pld)
        if not df.height:
            continue
        rows += df.height
        if df['err'].ne(0).any():        # cheap: one comparison per batch
            nerr.update(-e for e in df.filter(pl.col('err') != 0)['err'])
        held.append(df.select(_SRC))     # project early: the wire carries columns we drop
        nheld += df.height
        if nheld >= _COALESCE_ROWS:
            nheld = 0
            yield from flush()
            # THROTTLED. bvm emits ~a batch per directory, so printing per batch cost 1.4 s
            # of a 5.4 s walk -- the progress line was 26% of the scan.
            if progress and rows - last >= 1_000_000:
                last = rows
                print(f"\r  {rows:,} entries", end="", flush=True, file=sys.stderr)
    if held:
        yield from flush()
    # REPORT WHAT WAS LOST. Every row now carries `err` (0, or -errno), so unreadable
    # directories arrive IN the stream next to the files they sit among -- counted here from
    # `nerr`, not from bvm's separate ERROR(3) channel, which _Bvm.read swallows. A scan that
    # dropped whole subtrees used to look exactly like one that did not: the full mount came
    # back 42% short and still exited 0.
    if nerr:
        summary = ", ".join(f"{errno.errorcode.get(k, k)} {v:,}"
                            for k, v in sorted(nerr.items(), key=lambda kv: -kv[1]))
        print(f"\n  {sum(nerr.values()):,} unreadable directories ({summary})", file=sys.stderr)
    if progress:
        print(f"\r  {rows:,} entries", file=sys.stderr)


def scan_filesystem_quiver(path: str, output: str, *, do_agg: bool = True,
                           nworkers: int = 128, bvm_exe: str | None = None) -> None:
    """Scan *path* with quiver's walker and write ducl's Feather (+ agg sidecars).

    128 is the cheap end of the plateau: wall time is flat from 128 to 1024 walkers
    (0.89/0.80/0.87/0.89 s on 233k entries) while readdir blocking doubles with every doubling
    of threads -- the WEKA metadata service is saturated at ~292k entries/s on a small tree,
    so extra threads only add waiters.
    """
    from .scan import feather_from_batches

    root = os.path.abspath(path)
    feather_from_batches(stream_batches(root, nworkers, bvm_exe), output, do_agg=do_agg)
