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
from .agg import process_batch, compact_agg, compact_examples


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


def feather_from_batches(batches, output: str, *, do_agg: bool = True) -> None:
    """Write an iterable of Arrow RecordBatches to zstd Feather (+ agg sidecars).

    Kept separate from the walker so any batch source writes identical output through
    identical code.
    """
    write_opts = ipc.IpcWriteOptions(compression="zstd")
    total_rows = 0
    batch_count = 0

    agg_batches: list[pl.DataFrame] = []
    example_batches: list[pl.DataFrame] = []
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
                    if do_agg:
                        agg_df, ex_df = process_batch(batch)
                        agg_batches.append(agg_df)
                        example_batches.append(ex_df)
                        # Compact every 5 batches to bound memory
                        if len(agg_batches) >= 5:
                            agg_batches = [compact_agg(agg_batches)]
                            example_batches = [compact_examples(example_batches)]
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

    # Write sidecar files
    if do_agg and agg_batches:
        base = output.removesuffix(".feather")

        # Final agg merge
        final_agg = compact_agg(agg_batches)
        agg_path = base + ".agg.parquet"
        _write_parquet_atomic(final_agg, agg_path)
        print(
            f"Wrote {agg_path}: {final_agg.height:,} agg rows",
            file=sys.stderr,
        )

        # Final examples merge
        final_examples = compact_examples(example_batches)
        examples_path = base + ".examples.parquet"
        _write_parquet_atomic(final_examples, examples_path)
        print(
            f"Wrote {examples_path}: {final_examples.height:,} example rows",
            file=sys.stderr,
        )
