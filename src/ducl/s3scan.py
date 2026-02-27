"""Scan an S3 bucket into a zstd-compressed Feather (Arrow IPC) file.

Discovers prefixes for parallel listing, then uses a thread pool to
paginate list_objects_v2 across prefixes.  Produces the same schema as
the scan module so the dashboard and query tools work unchanged.

Optionally produces sidecar files:
  - <name>.agg.parquet     -- pre-aggregated rows for fast dashboard builds
  - <name>.examples.parquet -- top-3 largest files per (dir_path, size_bin)
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

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
    for _ in range(depth):
        next_level: list[str] = []
        for pfx in current:
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
        self._lock = threading.Lock()
        self._dirs: dict[str, list[int]] = {}  # path -> [child_count, total_size]

    def merge(self, dir_updates: dict[str, list[int]]) -> None:
        """Merge a worker's local dir dict into the global tracker."""
        with self._lock:
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
        last_modified = obj.get("LastModified")

        # Extract extension
        basename = key.rsplit("/", 1)[-1]
        if "." in basename:
            ext = basename.rsplit(".", 1)[-1]
        else:
            ext = ""

        # Mtime as epoch seconds
        mtime = int(last_modified.timestamp()) if last_modified else 0

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
# Worker: list a single prefix
# ---------------------------------------------------------------------------

BATCH_SIZE = 10_000  # objects per batch pushed to queue


def list_prefix_worker(
    client, bucket: str, prefix: str, path_prefix: str, result_queue: queue.Queue
) -> int:
    """Paginate list_objects_v2 for a single prefix, push batches to queue.

    Returns total object count for this prefix.
    """
    paginator = client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

    buffer: list[dict] = []
    total = 0

    for page in pages:
        contents = page.get("Contents", [])
        for obj in contents:
            buffer.append(obj)
            if len(buffer) >= BATCH_SIZE:
                result_queue.put(("batch", buffer))
                total += len(buffer)
                buffer = []

    if buffer:
        result_queue.put(("batch", buffer))
        total += len(buffer)

    return total


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
) -> None:
    """Scan an S3 bucket and write a Feather file."""
    t0 = time.monotonic()

    if path_prefix is None:
        path_prefix = f"/{bucket}"

    client = make_s3_client(
        endpoint_url=endpoint_url, profile=profile, region=region
    )

    # Phase 1: discover prefixes
    print(f"Discovering prefixes (depth={discover_depth})...", file=sys.stderr, flush=True)
    prefixes = discover_prefixes(client, bucket, prefix, discover_depth)
    print(f"  {len(prefixes)} prefixes found", file=sys.stderr, flush=True)

    # Phase 2+3: parallel listing + main thread consumption
    result_q: queue.Queue = queue.Queue(maxsize=workers * 2)
    dir_tracker = DirectoryTracker() if do_dirs else None

    write_opts = ipc.IpcWriteOptions(compression="zstd")
    total_rows = 0
    batch_count = 0
    agg_batches: list[pl.DataFrame] = []
    example_batches: list[pl.DataFrame] = []
    worker_errors: list[str] = []

    # Consumer runs in main thread -- use a sentinel to know when all workers done
    workers_done = threading.Event()

    def _submit_workers():
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = []
            for pfx in prefixes:
                f = pool.submit(
                    list_prefix_worker, client, bucket, pfx, path_prefix, result_q
                )
                futures.append((pfx, f))

            for pfx, f in futures:
                try:
                    f.result()
                except Exception as exc:
                    worker_errors.append(f"{pfx}: {exc}")
                    print(
                        f"  WARNING: worker failed for prefix {pfx!r}: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
            workers_done.set()
            result_q.put(("done", None))

    producer_thread = threading.Thread(target=_submit_workers, daemon=True)
    producer_thread.start()

    # Main thread: consume batches and write
    try:
        with pa.OSFile(output, "wb") as sink:
            writer = ipc.new_file(sink, SCHEMA, options=write_opts)
            try:
                while True:
                    try:
                        msg_type, payload = result_q.get(timeout=1.0)
                    except queue.Empty:
                        if workers_done.is_set():
                            break
                        continue

                    if msg_type == "done":
                        break

                    objects = payload
                    batch, dir_updates = objects_to_batch(objects, path_prefix)
                    if batch.num_rows == 0:
                        continue

                    writer.write_batch(batch)
                    total_rows += batch.num_rows
                    batch_count += 1

                    if dir_tracker is not None:
                        dir_tracker.merge(dir_updates)

                    print(
                        f"\r  {total_rows:,} objects ({batch_count} batches)",
                        end="", flush=True, file=sys.stderr,
                    )

                    if do_agg:
                        agg_df, ex_df = process_batch(batch)
                        agg_batches.append(agg_df)
                        example_batches.append(ex_df)
                        if len(agg_batches) >= 5:
                            agg_batches = [compact_agg(agg_batches)]
                            example_batches = [compact_examples(example_batches)]

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

    producer_thread.join(timeout=5)

    mb = os.path.getsize(output) / 1e6
    elapsed = time.monotonic() - t0
    print(
        f"\nWrote {output}: {total_rows:,} rows, {mb:.1f} MB  [{elapsed:.1f}s]",
        file=sys.stderr,
    )

    if worker_errors:
        print(
            f"  {len(worker_errors)} prefix(es) had errors",
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
