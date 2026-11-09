"""Write a stream of Arrow RecordBatches to zstd-compressed Feather (Arrow IPC).

``feather_from_batches`` is the filesystem scanner's writer (quiverscan.py feeds it bvm's
batches, locally and inside Modal jobs). It streams batches to disk,
builds the .agg.parquet / .examples.parquet sidecars as they flow by, and writes every
output atomically (temp name + rename).
"""

from __future__ import annotations

import os
import sys

import polars as pl
import pyarrow as pa
import pyarrow.ipc as ipc

from .schema import SCHEMA
from .wsds_sparse import SparseSizeFixer
from .agg import DEFAULT_HARDLINKS, SidecarBuilder


def _write_parquet_atomic(df: pl.DataFrame, path: str) -> None:
    """write_parquet to ``path + ".tmp"`` then rename, so a crash leaves no partial sidecar."""
    tmp = path + ".tmp"
    try:
        df.write_parquet(tmp)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    os.replace(tmp, path)


def feather_from_batches(batches, output: str, *, do_agg: bool = True,
                         hardlinks: str = DEFAULT_HARDLINKS) -> None:
    """Write an iterable of Arrow RecordBatches to zstd Feather (+ agg sidecars).

    *hardlinks* is the sidecars' hard-link policy (agg.HARDLINK_MODES); the Feather
    itself always stores every entry with its real size and nlink.

    Kept separate from the walker so any batch source writes identical output through
    identical code.
    """
    write_opts = ipc.IpcWriteOptions(compression="zstd")
    total_rows = 0
    batch_count = 0

    sidecars = SidecarBuilder(hardlinks) if do_agg else None
    reader = batches
    sparse_fixer = SparseSizeFixer()

    # Write to a temp name in the same directory and rename into place only once the
    # footer is on disk. A scan that dies mid-write (OOM, preemption) must not leave a
    # date-named .feather behind: downstream tools pick captures by name, and a
    # footerless file fails much later with polars' "InvalidFooter".
    tmp_output = output + ".tmp"
    try:
        with pa.OSFile(tmp_output, "wb") as sink:
            writer = ipc.new_file(sink, SCHEMA, options=write_opts)
            try:
                for batch in reader:
                    batch = sparse_fixer(batch)
                    writer.write_batch(batch)
                    total_rows += batch.num_rows
                    batch_count += 1
                    print(
                        f"\r  {total_rows:,} rows ({batch_count} batches)",
                        end="",
                        flush=True,
                        file=sys.stderr,
                    )
                    if sidecars is not None:
                        sidecars.add(batch)
            finally:
                writer.close()
    except BaseException:
        try:
            os.unlink(tmp_output)
        except OSError:
            pass
        raise
    os.replace(tmp_output, output)

    mb = os.path.getsize(output) / 1e6
    print(
        f"\nWrote {output}: {total_rows:,} rows, {mb:.1f} MB",
        file=sys.stderr,
    )
    sparse_fixer.report()

    if sidecars is not None:
        _write_sidecars(sidecars, output)


def _write_sidecars(sidecars: SidecarBuilder, output: str) -> None:
    """Write <base>.agg.parquet and <base>.examples.parquet next to *output*."""
    result = sidecars.finish()
    if result is None:
        return
    final_agg, final_examples = result
    base = output.removesuffix(".feather")

    agg_path = base + ".agg.parquet"
    _write_parquet_atomic(final_agg, agg_path)
    print(f"Wrote {agg_path}: {final_agg.height:,} agg rows (hardlinks={sidecars.hardlinks})",
          file=sys.stderr)

    examples_path = base + ".examples.parquet"
    _write_parquet_atomic(final_examples, examples_path)
    print(f"Wrote {examples_path}: {final_examples.height:,} example rows", file=sys.stderr)


def reaggregate(feather: str, *, hardlinks: str = DEFAULT_HARDLINKS) -> None:
    """Rebuild the sidecars of an existing capture from its Feather, without rescanning.

    The Feather keeps every entry's real size and nlink, so a capture aggregated under one
    hard-link policy (or by a ducl older than the policy) can be re-cut under another --
    which is what makes it comparable to a newer capture in `ducl diff`.
    """
    sidecars = SidecarBuilder(hardlinks)
    reader = ipc.open_file(feather)
    rows = 0
    for i in range(reader.num_record_batches):
        batch = reader.get_batch(i)
        sidecars.add(batch)
        rows += batch.num_rows
        print(f"\r  {rows:,} rows ({i + 1}/{reader.num_record_batches} batches)",
              end="", flush=True, file=sys.stderr)
    print(file=sys.stderr)
    _write_sidecars(sidecars, feather)
