"""Convert pwalk2 CSV to zstd-compressed Feather (Arrow IPC).

Provides ``feather_from_csv_stream`` (generic stdin/pipe reader) and
``scan_filesystem`` (runs bundled pwalk2 and pipes into feather writer).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.csv as pcsv
import pyarrow.ipc as ipc

from .schema import COLUMN_NAMES, SCHEMA
from .agg import process_batch, compact_agg, compact_examples


def _pwalk2_binary() -> Path:
    """Locate the bundled pwalk2 binary."""
    return Path(__file__).parent / "pwalk2" / "pwalk2"


def feather_from_csv_stream(
    stream,
    output: str,
    *,
    do_agg: bool = True,
    block_size: int = 256 << 20,
) -> None:
    """Read pwalk2 CSV from *stream* and write a zstd-compressed Feather file.

    Optionally produces sidecar .agg.parquet and .examples.parquet files.
    """
    try:
        reader = pcsv.open_csv(
            stream,
            read_options=pcsv.ReadOptions(
                column_names=COLUMN_NAMES,
                block_size=block_size,
            ),
            parse_options=pcsv.ParseOptions(quote_char='"'),
            convert_options=pcsv.ConvertOptions(column_types=SCHEMA),
        )
    except pa.lib.ArrowInvalid:
        # Empty CSV — write a valid empty Feather file
        write_opts = ipc.IpcWriteOptions(compression="zstd")
        with pa.OSFile(output, "wb") as sink:
            writer = ipc.new_file(sink, SCHEMA, options=write_opts)
            writer.close()
        print(f"\nWrote {output}: 0 rows, 0.0 MB", file=sys.stderr)
        return

    write_opts = ipc.IpcWriteOptions(compression="zstd")
    total_rows = 0
    batch_count = 0

    agg_batches: list[pl.DataFrame] = []
    example_batches: list[pl.DataFrame] = []

    with pa.OSFile(output, "wb") as sink:
        writer = ipc.new_file(sink, SCHEMA, options=write_opts)
        try:
            for batch in reader:
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

    mb = os.path.getsize(output) / 1e6
    print(
        f"\nWrote {output}: {total_rows:,} rows, {mb:.1f} MB",
        file=sys.stderr,
    )

    # Write sidecar files
    if do_agg and agg_batches:
        base = output.removesuffix(".feather")

        # Final agg merge
        final_agg = compact_agg(agg_batches)
        agg_path = base + ".agg.parquet"
        final_agg.write_parquet(agg_path)
        print(
            f"Wrote {agg_path}: {final_agg.height:,} agg rows",
            file=sys.stderr,
        )

        # Final examples merge
        final_examples = compact_examples(example_batches)
        examples_path = base + ".examples.parquet"
        final_examples.write_parquet(examples_path)
        print(
            f"Wrote {examples_path}: {final_examples.height:,} example rows",
            file=sys.stderr,
        )


def scan_filesystem(
    path: str,
    output: str,
    *,
    do_agg: bool = True,
    block_size: int = 256 << 20,
    pwalk2_args: list[str] | None = None,
) -> None:
    """Run the bundled pwalk2 on *path* and pipe output to feather writer."""
    binary = _pwalk2_binary()
    if not binary.exists():
        raise FileNotFoundError(
            f"pwalk2 binary not found at {binary}. "
            "pwalk2 requires Linux and is compiled at install time."
        )

    cmd = [str(binary)]
    if pwalk2_args:
        cmd.extend(pwalk2_args)
    cmd.append(path)

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    try:
        feather_from_csv_stream(
            proc.stdout,
            output,
            do_agg=do_agg,
            block_size=block_size,
        )
    finally:
        proc.wait()
        if proc.returncode != 0:
            print(
                f"pwalk2 exited with code {proc.returncode}",
                file=sys.stderr,
            )
