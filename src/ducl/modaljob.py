"""Scan Modal volumes by running pwalk2 *inside* Modal jobs.

Rather than listing volumes remotely over the gRPC API, this fans out one Modal
job per volume.  Each job mounts its volume and runs the bundled ``pwalk2`` (via
``ducl scan``) against the POSIX mount -- listing happens next to the data, in
parallel across volumes -- then returns the compact ``.feather`` (+ aggregation
sidecars) straight to the parent as its return value.  No staging volume or
shared storage is involved.

Each volume is mounted at ``/scan/<name>`` (namespaced so a volume called e.g.
``lib`` can't shadow the container's own ``/lib``).  pwalk2 therefore emits paths
like ``/scan/<name>/...``; when the per-volume feathers are combined locally the
``/scan`` prefix is stripped so each path becomes ``/<name>/...`` -- i.e. the
volume name is the top-level folder in the dashboard.

The in-container image is just Ubuntu 24.04 (glibc 2.39, new enough for the
prebuilt pwalk2 binary) plus ``pip install ducl``.  No compilation needed.
"""

from __future__ import annotations

import io
import os
import sys

import polars as pl
import pyarrow as pa
import pyarrow.ipc as ipc

from .schema import SCHEMA
from .agg import compact_agg, compact_examples

# Where each volume is mounted inside its job (namespaced, not at root).
_MOUNT_BASE = "/scan"
_STRIP = len(_MOUNT_BASE)  # chars of the "/scan" prefix to strip from paths

# Default per-job resources.  pwalk2 is metadata-bound and benefits from cores;
# the feather conversion needs some RAM for big volumes.
DEFAULT_CPU = 16.0
DEFAULT_MEMORY_MB = 32768
DEFAULT_TIMEOUT_S = 4 * 3600
IMAGE_TAG = "ubuntu:24.04"


def _safe_fn_name(volume: str) -> str:
    """Modal function name: alphanumerics/underscores only."""
    return "scan_" + "".join(c if c.isalnum() else "_" for c in volume)


# ---------------------------------------------------------------------------
# App construction (one dynamically-created function per volume)
# ---------------------------------------------------------------------------

def _make_scan_impl(mount_base: str):
    """Build the in-container entry point as a *nested* function.

    Defining it here (rather than at module level) means cloudpickle serializes
    it by value, so the Modal container -- which only has the published ``ducl``
    wheel, not this ``ducl.modaljob`` module -- can run it without importing us.
    It must reference only stdlib and captured locals (``mount_base``), never
    this module's globals.
    """

    def scan_impl(volume: str) -> dict:
        import os
        import subprocess

        mount_root = f"{mount_base}/{volume}"
        out_path = f"/tmp/{volume.replace('/', '_')}.feather"
        proc = subprocess.run(
            ["ducl", "scan", mount_root, "-o", out_path, "--no-dashboard"],
            capture_output=True, text=True,
        )

        def _read(p):
            return open(p, "rb").read() if os.path.exists(p) else None

        base = out_path[: -len(".feather")]
        feather = _read(out_path)

        summary = ""
        for line in proc.stderr.splitlines():
            if "pwalk2 done" in line or "Wrote " in line:
                summary = line.strip()

        return {
            "volume": volume,
            "rc": proc.returncode,
            "ok": proc.returncode == 0 and feather is not None,
            "feather": feather,
            "agg": _read(base + ".agg.parquet"),
            "examples": _read(base + ".examples.parquet"),
            "summary": summary,
            "stderr_tail": "\n".join(
                l for l in proc.stderr.splitlines() if "io_uring_setup failed" not in l
            )[-1500:],
        }

    return scan_impl


def _build_app(
    volumes: list[str],
    *,
    environment: str | None,
    cpu: float,
    memory_mb: int,
    timeout_s: int,
):
    import modal

    py = f"{sys.version_info.major}.{sys.version_info.minor}"  # match local (serialized fns)
    image = modal.Image.from_registry(IMAGE_TAG, add_python=py).pip_install("ducl")
    app = modal.App("ducl-volume-scan")
    scan_impl = _make_scan_impl(_MOUNT_BASE)

    fns = {}
    for name in volumes:
        vol = modal.Volume.from_name(name, environment_name=environment)
        fns[name] = app.function(
            image=image,
            volumes={f"{_MOUNT_BASE}/{name}": vol},
            timeout=timeout_s,
            cpu=cpu,
            memory=memory_mb,
            serialized=True,
            name=_safe_fn_name(name),
        )(scan_impl)
    return app, fns


# ---------------------------------------------------------------------------
# Combine per-volume results (strip /scan prefix) into one capture
# ---------------------------------------------------------------------------

def _strip_prefix_expr(col: str) -> pl.Expr:
    """Drop the leading '/scan' from a path/dir_path column."""
    return pl.col(col).str.slice(_STRIP).alias(col)


def _fix_agg_prefix(df: pl.DataFrame) -> pl.DataFrame:
    """Strip '/scan' from dir_path and recompute the derived n_components.

    ``n_components`` was computed in-container from the ``/scan/<name>/...``
    path, so it is inflated by one; recompute it from the stripped dir_path
    (matching ``agg.process_batch``: file component count = dir slashes + 2).
    The dashboard's tree builder filters on n_components, so a stale value
    spills everything into the "(other)" bucket.
    """
    df = df.with_columns(_strip_prefix_expr("dir_path"))
    if "n_components" in df.columns:
        df = df.with_columns(
            (pl.col("dir_path").str.count_matches("/") + 2).cast(pl.Int32).alias("n_components")
        )
    return df


def _write_feather_bytes(writer, feather_bytes: bytes) -> int:
    """Strip the /scan prefix from a per-volume feather and append its batches."""
    reader = ipc.open_file(pa.BufferReader(feather_bytes))
    rows = 0
    for i in range(reader.num_record_batches):
        df = pl.from_arrow(reader.get_batch(i)).with_columns(_strip_prefix_expr("path"))
        tbl = df.to_arrow().cast(SCHEMA)
        for batch in tbl.to_batches():
            writer.write_batch(batch)
            rows += batch.num_rows
    return rows


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _run_jobs(
    volumes: list[str],
    output: str,
    *,
    environment: str | None,
    do_agg: bool,
    cpu: float,
    memory_mb: int,
    timeout_s: int,
) -> None:
    if not volumes:
        raise ValueError("no volumes to scan")

    app, fns = _build_app(
        volumes, environment=environment, cpu=cpu, memory_mb=memory_mb, timeout_s=timeout_s,
    )

    print(f"Spawning {len(volumes)} Modal scan job(s)...", file=sys.stderr, flush=True)
    total_rows = 0
    n_ok = 0
    agg_parts: list[pl.DataFrame] = []
    ex_parts: list[pl.DataFrame] = []
    write_opts = ipc.IpcWriteOptions(compression="zstd")

    with pa.OSFile(output, "wb") as sink:
        writer = ipc.new_file(sink, SCHEMA, options=write_opts)
        try:
            with app.run():
                # Spawn everything, then collect as each job finishes and combine
                # incrementally so we never hold all per-volume bytes at once.
                calls = {name: fns[name].spawn(name) for name in volumes}
                for done, (name, call) in enumerate(calls.items(), 1):
                    try:
                        res = call.get()
                    except Exception as exc:  # noqa: BLE001
                        print(f"  [{done}/{len(volumes)}] {name}: FAILED ({exc})",
                              file=sys.stderr, flush=True)
                        continue

                    if not res.get("ok"):
                        print(f"  [{done}/{len(volumes)}] {name}: FAILED (rc={res.get('rc')})",
                              file=sys.stderr, flush=True)
                        if res.get("stderr_tail"):
                            print(f"      {res['stderr_tail']}", file=sys.stderr, flush=True)
                        continue

                    rows = _write_feather_bytes(writer, res["feather"])
                    total_rows += rows
                    n_ok += 1
                    if do_agg and res.get("agg"):
                        agg_parts.append(_fix_agg_prefix(pl.read_parquet(io.BytesIO(res["agg"]))))
                    if do_agg and res.get("examples"):
                        ex_parts.append(
                            pl.read_parquet(io.BytesIO(res["examples"]))
                            .with_columns(_strip_prefix_expr("dir_path"))
                        )
                    print(f"  [{done}/{len(volumes)}] {name}: ok ({rows:,} rows)  "
                          f"{res.get('summary','')}", file=sys.stderr, flush=True)
        finally:
            writer.close()

    base = output.removesuffix(".feather")
    if do_agg and agg_parts:
        compact_agg(agg_parts).write_parquet(base + ".agg.parquet")
    if do_agg and ex_parts:
        compact_examples(ex_parts).write_parquet(base + ".examples.parquet")

    mb = os.path.getsize(output) / 1e6
    print(f"\nWrote {output}: {total_rows:,} rows, {mb:.1f} MB "
          f"(from {n_ok}/{len(volumes)} volumes)", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def list_volume_names(environment: str | None = None) -> list[str]:
    """Return the names of all volumes visible in *environment*."""
    import asyncio
    from modal import Volume

    async def _names():
        vols = await Volume.objects.list.aio(environment_name=environment or "")
        infos = await asyncio.gather(*(v.info.aio() for v in vols))
        return [info.name for info in infos]

    return asyncio.run(_names())


def scan_volume_job(
    volume: str,
    output: str,
    *,
    environment: str | None = None,
    do_agg: bool = True,
    cpu: float = DEFAULT_CPU,
    memory_mb: int = DEFAULT_MEMORY_MB,
    timeout_s: int = DEFAULT_TIMEOUT_S,
) -> None:
    """Scan a single Modal volume via a pwalk2 Modal job."""
    _run_jobs(
        [volume], output, environment=environment, do_agg=do_agg,
        cpu=cpu, memory_mb=memory_mb, timeout_s=timeout_s,
    )


def scan_all_volumes_jobs(
    output: str,
    *,
    environment: str | None = None,
    do_agg: bool = True,
    cpu: float = DEFAULT_CPU,
    memory_mb: int = DEFAULT_MEMORY_MB,
    timeout_s: int = DEFAULT_TIMEOUT_S,
) -> None:
    """Scan every visible Modal volume, one pwalk2 job each, into one capture."""
    print("Listing Modal volumes...", file=sys.stderr, flush=True)
    names = sorted(list_volume_names(environment))
    print(f"  {len(names)} volumes found", file=sys.stderr, flush=True)
    _run_jobs(
        names, output, environment=environment, do_agg=do_agg,
        cpu=cpu, memory_mb=memory_mb, timeout_s=timeout_s,
    )
