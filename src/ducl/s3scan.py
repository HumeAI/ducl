"""Scan an S3 bucket into a zstd-compressed Feather (Arrow IPC) file.

Discovers prefixes for parallel listing, then uses worker processes to
paginate list_objects_v2 across prefixes.  Produces the same schema as
the scan module so the dashboard and query tools work unchanged.

Optionally produces sidecar files:
  - <name>.agg.parquet     -- pre-aggregated rows for fast dashboard builds
  - <name>.examples.parquet -- top-3 largest files per (dir_path, size_bin)
"""

from __future__ import annotations

import multiprocessing as mp
import os
import signal
import sys
import time
from collections import defaultdict

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.ipc as ipc

from .schema import SCHEMA, COLUMN_NAMES, HIST_EDGES
from .agg import process_batch, compact_agg, compact_examples


# ---------------------------------------------------------------------------
# S3 client setup
# ---------------------------------------------------------------------------

def make_s3_client(
    endpoint_url: str | None = None,
    profile: str | None = None,
    region: str | None = None,
):
    """Create a boto3 S3 client with adaptive retry."""
    import boto3
    from botocore.config import Config

    config = Config(
        retries={"mode": "adaptive", "max_attempts": 10},
    )
    session = boto3.Session(profile_name=profile, region_name=region)
    kwargs: dict = {"config": config}
    if endpoint_url:
        kwargs["endpoint_url"] = endpoint_url
    return session.client("s3", **kwargs)


# ---------------------------------------------------------------------------
# Prefix discovery
# ---------------------------------------------------------------------------

def discover_prefixes(
    client, bucket: str, root_prefix: str, depth: int
) -> list[str]:
    """BFS discover prefixes to *depth* levels using Delimiter='/'.

    Returns a list of prefix strings for parallel work distribution.
    """
    if depth <= 0:
        return [root_prefix]

    current = [root_prefix]
    for level in range(depth):
        next_level: list[str] = []
        for i, pfx in enumerate(current):
            paginator = client.get_paginator("list_objects_v2")
            pages = paginator.paginate(
                Bucket=bucket, Prefix=pfx, Delimiter="/"
            )
            found_children = False
            for page in pages:
                for cp in page.get("CommonPrefixes", []):
                    next_level.append(cp["Prefix"])
                    found_children = True
            if not found_children:
                # Leaf prefix -- keep it
                next_level.append(pfx)
            print(
                f"\r  level {level + 1}/{depth}: {i + 1}/{len(current)} prefixes ({len(next_level)} found)",
                end="", flush=True, file=sys.stderr,
            )
        print(file=sys.stderr)  # newline after level
        if not next_level:
            break
        current = next_level

    return current if current else [root_prefix]


# ---------------------------------------------------------------------------
# Directory tracker (accumulates synthetic dir rows)
# ---------------------------------------------------------------------------

class DirectoryTracker:
    """Track directories seen across batches.

    Accumulates {dir_path: [child_count, total_size]}.
    child_count = number of direct children (files + sub-dirs).
    total_size  = sum of direct file sizes.
    """

    def __init__(self):
        self._dirs: dict[str, list[int]] = {}  # path -> [child_count, total_size]

    def merge(self, dir_updates: dict[str, list[int]]) -> None:
        """Merge a worker's local dir dict into the global tracker."""
        for path, (count, size) in dir_updates.items():
            if path in self._dirs:
                self._dirs[path][0] += count
                self._dirs[path][1] += size
            else:
                self._dirs[path] = [count, size]

    def to_record_batch(self, path_prefix: str) -> pa.RecordBatch:
        """Convert tracked directories to a RecordBatch matching SCHEMA."""
        if not self._dirs:
            return pa.RecordBatch.from_pydict(
                {name: [] for name in COLUMN_NAMES}, schema=SCHEMA
            )

        paths = []
        exts = []
        sizes = []
        child_counts = []
        dir_total_sizes = []
        depths = []

        for dir_path, (cc, ts) in sorted(self._dirs.items()):
            full_path = path_prefix.rstrip("/") + "/" + dir_path if dir_path else path_prefix
            paths.append(full_path)
            exts.append("")
            sizes.append(0)
            child_counts.append(cc)
            dir_total_sizes.append(ts)
            depths.append(full_path.count("/"))

        n = len(paths)
        return pa.RecordBatch.from_pydict(
            {
                "inode_id": [0] * n,
                "parent_inode_id": [0] * n,
                "depth": depths,
                "path": paths,
                "ext": exts,
                "uid": [0] * n,
                "gid": [0] * n,
                "size": sizes,
                "fs_id": [0] * n,
                "blocks": [0] * n,
                "nlink": [0] * n,
                "mode": [""] * n,
                "atime": [0] * n,
                "mtime": [0] * n,
                "ctime": [0] * n,
                "child_count": child_counts,
                "dir_total_size": dir_total_sizes,
            },
            schema=SCHEMA,
        )


# ---------------------------------------------------------------------------
# S3 objects -> RecordBatch
# ---------------------------------------------------------------------------

def objects_to_batch(
    objects: list[dict], path_prefix: str
) -> tuple[pa.RecordBatch, dict[str, list[int]]]:
    """Convert S3 list_objects_v2 response objects to a RecordBatch.

    Returns (batch, dir_updates) where dir_updates maps dir_path ->
    [child_count_delta, total_size_delta].
    """
    paths = []
    exts = []
    sizes = []
    mtimes = []
    depths = []
    dir_updates: dict[str, list[int]] = defaultdict(lambda: [0, 0])

    for obj in objects:
        key = obj["Key"]

        # Skip zero-byte folder markers (keys ending in '/')
        if key.endswith("/") and obj.get("Size", 0) == 0:
            continue

        full_path = path_prefix.rstrip("/") + "/" + key
        size = obj.get("Size", 0)
        mtime = obj.get("mtime", 0)

        # Extract extension
        basename = key.rsplit("/", 1)[-1]
        if "." in basename:
            ext = basename.rsplit(".", 1)[-1]
        else:
            ext = ""

        paths.append(full_path)
        exts.append(ext)
        sizes.append(size)
        mtimes.append(mtime)
        depths.append(full_path.count("/"))

        # Track parent directory
        dir_path = key.rsplit("/", 1)[0] if "/" in key else ""
        dir_updates[dir_path][0] += 1
        dir_updates[dir_path][1] += size

        # Ensure ancestor dirs exist in tracker
        parts = dir_path.split("/")
        for i in range(len(parts) - 1):
            ancestor = "/".join(parts[: i + 1])
            if ancestor not in dir_updates:
                dir_updates[ancestor] = [0, 0]

    if not paths:
        empty = pa.RecordBatch.from_pydict(
            {name: [] for name in COLUMN_NAMES}, schema=SCHEMA
        )
        return empty, dict(dir_updates)

    n = len(paths)
    batch = pa.RecordBatch.from_pydict(
        {
            "inode_id": [0] * n,
            "parent_inode_id": [0] * n,
            "depth": depths,
            "path": paths,
            "ext": exts,
            "uid": [0] * n,
            "gid": [0] * n,
            "size": sizes,
            "fs_id": [0] * n,
            "blocks": [0] * n,
            "nlink": [0] * n,
            "mode": [""] * n,
            "atime": [0] * n,
            "mtime": mtimes,
            "ctime": [0] * n,
            "child_count": [-1] * n,  # -1 = file
            "dir_total_size": [-1] * n,
        },
        schema=SCHEMA,
    )
    return batch, dict(dir_updates)


# ---------------------------------------------------------------------------
# Worker process
# ---------------------------------------------------------------------------

BATCH_SIZE = 10_000  # objects per batch pushed to queue


def _worker_main(
    endpoint_url, profile, region,
    bucket, path_prefix,
    work_q, result_q,
):
    """Worker process entry point: pull prefixes from work_q, list objects,
    push batches to result_q.  Runs until work_q is empty."""

    # Ignore SIGINT in workers — let main process handle Ctrl-C
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    client = make_s3_client(
        endpoint_url=endpoint_url, profile=profile, region=region,
    )

    total_s3_time = 0.0
    total_queue_time = 0.0
    total_page_count = 0

    while True:
        prefix = work_q.get()  # blocks until work available or sentinel
        if prefix is None:
            break

        paginator = client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

        buffer: list[dict] = []
        s3_time = 0.0
        queue_time = 0.0
        page_count = 0

        page_iter = iter(pages)
        while True:
            t_page = time.monotonic()
            try:
                page = next(page_iter)
            except StopIteration:
                break
            s3_time += time.monotonic() - t_page
            page_count += 1
            for obj in page.get("Contents", []):
                last_modified = obj.get("LastModified")
                buffer.append({
                    "Key": obj["Key"],
                    "Size": obj.get("Size", 0),
                    "mtime": int(last_modified.timestamp()) if last_modified else 0,
                })
                if len(buffer) >= BATCH_SIZE:
                    t_q = time.monotonic()
                    result_q.put(("batch", buffer))
                    queue_time += time.monotonic() - t_q
                    buffer = []

        if buffer:
            t_q = time.monotonic()
            result_q.put(("batch", buffer))
            queue_time += time.monotonic() - t_q

        result_q.put(("prefix_done", prefix))
        total_s3_time += s3_time
        total_queue_time += queue_time
        total_page_count += page_count

    # Send timing stats as final message
    result_q.put(("timing", {
        "s3_time": total_s3_time,
        "queue_time": total_queue_time,
        "page_count": total_page_count,
    }))


# ---------------------------------------------------------------------------
# Main scan orchestration
# ---------------------------------------------------------------------------

def scan_bucket(
    bucket: str,
    output: str,
    prefix: str = "",
    path_prefix: str | None = None,
    workers: int = 32,
    discover_depth: int = 2,
    do_agg: bool = True,
    do_dirs: bool = True,
    endpoint_url: str | None = None,
    profile: str | None = None,
    region: str | None = None,
    max_objects: int = 0,
) -> None:
    """Scan an S3 bucket and write a Feather file.

    If max_objects > 0, stop after approximately that many objects (for benchmarking).
    """
    t0 = time.monotonic()

    if path_prefix is None:
        path_prefix = f"/{bucket}"

    # Main-process client for prefix discovery only
    client = make_s3_client(
        endpoint_url=endpoint_url, profile=profile, region=region,
    )

    # Phase 1: discover prefixes
    print(f"Discovering prefixes (depth={discover_depth})...", file=sys.stderr, flush=True)
    prefixes = discover_prefixes(client, bucket, prefix, discover_depth)
    print(f"  {len(prefixes)} prefixes found", file=sys.stderr, flush=True)

    # Phase 2+3: parallel listing + main process consumption
    work_q = mp.Queue()
    result_q = mp.Queue()
    for pfx in prefixes:
        work_q.put(pfx)
    num_workers = min(workers, len(prefixes))
    for _ in range(num_workers):
        work_q.put(None)  # sentinel: tells worker to exit

    dir_tracker = DirectoryTracker() if do_dirs else None

    write_opts = ipc.IpcWriteOptions(compression="zstd")
    total_rows = 0
    batch_count = 0
    agg_batches: list[pl.DataFrame] = []
    example_batches: list[pl.DataFrame] = []

    # Start worker processes
    procs: list[mp.Process] = []
    for _ in range(num_workers):
        p = mp.Process(
            target=_worker_main,
            args=(endpoint_url, profile, region, bucket, path_prefix, work_q, result_q),
            daemon=True,
        )
        p.start()
        procs.append(p)

    print(f"  {len(procs)} worker processes started", file=sys.stderr, flush=True)

    # Main thread: consume batches and write
    total_prefixes = len(prefixes)
    prefixes_done = 0
    workers_finished = 0
    worker_timings: list[dict] = []

    def _print_progress():
        elapsed = time.monotonic() - t0
        rate = total_rows / elapsed if elapsed > 0 else 0
        print(
            f"\r  {total_rows:,} objects | "
            f"{prefixes_done}/{total_prefixes} prefixes | "
            f"{rate:,.0f} obj/s",
            end="", flush=True, file=sys.stderr,
        )

    # Main-thread timing accumulators
    convert_time = 0.0
    write_time = 0.0
    dir_merge_time = 0.0
    agg_time = 0.0
    idle_time = 0.0

    try:
        with pa.OSFile(output, "wb") as sink:
            writer = ipc.new_file(sink, SCHEMA, options=write_opts)
            try:
                while workers_finished < len(procs):
                    try:
                        t_idle = time.monotonic()
                        msg_type, payload = result_q.get(timeout=1.0)
                        idle_time += time.monotonic() - t_idle
                    except Exception:
                        idle_time += time.monotonic() - t_idle
                        _print_progress()
                        continue

                    if msg_type == "timing":
                        worker_timings.append(payload)
                        workers_finished += 1
                        continue

                    if msg_type == "prefix_done":
                        prefixes_done += 1
                        _print_progress()
                        continue

                    # msg_type == "batch"
                    objects = payload

                    t1 = time.monotonic()
                    batch, dir_updates = objects_to_batch(objects, path_prefix)
                    convert_time += time.monotonic() - t1
                    if batch.num_rows == 0:
                        continue

                    t1 = time.monotonic()
                    writer.write_batch(batch)
                    write_time += time.monotonic() - t1
                    total_rows += batch.num_rows
                    batch_count += 1

                    if dir_tracker is not None:
                        t1 = time.monotonic()
                        dir_tracker.merge(dir_updates)
                        dir_merge_time += time.monotonic() - t1

                    _print_progress()

                    if do_agg:
                        t1 = time.monotonic()
                        agg_df, ex_df = process_batch(batch)
                        agg_batches.append(agg_df)
                        example_batches.append(ex_df)
                        if len(agg_batches) >= 5:
                            agg_batches = [compact_agg(agg_batches)]
                            example_batches = [compact_examples(example_batches)]
                        agg_time += time.monotonic() - t1

                    if max_objects > 0 and total_rows >= max_objects:
                        print(
                            f"\n  Reached --max-objects limit ({max_objects:,}), stopping.",
                            file=sys.stderr, flush=True,
                        )
                        break

                # Phase 4: write directory rows
                if dir_tracker is not None:
                    dir_batch = dir_tracker.to_record_batch(path_prefix)
                    if dir_batch.num_rows > 0:
                        writer.write_batch(dir_batch)
                        total_rows += dir_batch.num_rows
                        print(
                            f"\n  {dir_batch.num_rows:,} synthetic directory rows",
                            file=sys.stderr, flush=True,
                        )
            finally:
                writer.close()
    except KeyboardInterrupt:
        print("\nInterrupted -- partial file written.", file=sys.stderr)

    # Clean up worker processes
    for p in procs:
        p.terminate()
    for p in procs:
        p.join(timeout=2)
    for p in procs:
        if p.is_alive():
            p.kill()

    result_q.cancel_join_thread()
    result_q.close()
    work_q.cancel_join_thread()
    work_q.close()

    mb = os.path.getsize(output) / 1e6
    elapsed = time.monotonic() - t0
    rate = total_rows / elapsed if elapsed > 0 else 0
    print(
        f"\nWrote {output}: {total_rows:,} rows, {mb:.1f} MB  "
        f"[{elapsed:.1f}s, {rate:,.0f} obj/s]",
        file=sys.stderr,
    )

    worker_errors = [p for p in procs if p.exitcode and p.exitcode != 0]
    if worker_errors:
        print(
            f"  {len(worker_errors)} worker(s) exited with errors",
            file=sys.stderr,
        )

    # Timing breakdown
    if worker_timings:
        total_s3 = sum(t["s3_time"] for t in worker_timings)
        total_queue_wait = sum(t["queue_time"] for t in worker_timings)
        total_pages = sum(t["page_count"] for t in worker_timings)
        print(
            f"\nTiming breakdown (wall={elapsed:.1f}s):\n"
            f"  Workers (summed across {len(worker_timings)} processes):\n"
            f"    S3 API:       {total_s3:6.1f}s  ({total_pages} pages, "
            f"{total_pages / elapsed:.0f} pages/s effective)\n"
            f"    queue.put:    {total_queue_wait:6.1f}s  (backpressure)\n"
            f"  Main process:\n"
            f"    queue.get:    {idle_time:6.1f}s  (idle / waiting for data)\n"
            f"    obj->batch:   {convert_time:6.1f}s\n"
            f"    write_batch:  {write_time:6.1f}s\n"
            f"    dir_merge:    {dir_merge_time:6.1f}s\n"
            f"    agg/examples: {agg_time:6.1f}s",
            file=sys.stderr,
        )

    # Write sidecar files
    if do_agg and agg_batches:
        base = output.removesuffix(".feather")

        final_agg = compact_agg(agg_batches)
        agg_path = base + ".agg.parquet"
        final_agg.write_parquet(agg_path)
        print(f"Wrote {agg_path}: {final_agg.height:,} agg rows", file=sys.stderr)

        final_examples = compact_examples(example_batches)
        examples_path = base + ".examples.parquet"
        final_examples.write_parquet(examples_path)
        print(
            f"Wrote {examples_path}: {final_examples.height:,} example rows",
            file=sys.stderr,
        )
