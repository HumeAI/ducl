#!/usr/bin/env python3
"""Toy model of the scanner / planner / executor architecture for building
rm / cp / rsync on top of a fast walker. Single process, but it does NOT assume
a sequential executor: the executor applies actions out of order within each
epoch, so correctness comes only from the epoch/barrier ordering -- exactly the
property a real parallel (io_uring/thread-pool) executor would need.

  SCANNER   streams record *batches* (Polars frames) as it walks, like pwalk2
            emitting CSV -> PyArrow RecordBatches. A subtree-complete marker
            (kind == "ENDDIR") is emitted in-band after a directory's whole
            subtree has streamed.

  PLANNER   per-batch **Polars** transforms (the realistic stage) that turn
            records into an action table with an `epoch` column. Epoch = the
            barrier generalization: all of epoch e must complete before any of
            epoch e+1; within an epoch actions are mutually independent.
              - plan_rm : files -> UNLINK @ epoch 0; dirs -> RMDIR @ BIG-depth
                          (deepest first, bottom-up).
              - plan_cp : dirs -> MKDIR @ depth; files -> COPY @ depth (top-down).
              - compare : rsync's tree-comparison stage as a Polars **full outer
                          join** of the two trees -> CREATE/UPDATE/SKIP/DELETE.

  EXECUTOR  consumes the action table, applies epoch by epoch (the drain points
            == barriers), and **shuffles within each epoch** to emulate parallel
            out-of-order application. `respect_epochs=False` collapses to one
            epoch to show what breaks without barriers.

Run it:  python toy_fsops.py
"""

from __future__ import annotations

import os
import random
import shutil
import sys
import tempfile

import polars as pl

# ===========================================================================
# Scanner -- streams record batches (Polars frames) with subtree markers.
# ===========================================================================

FILE, DIR, ENDDIR = "FILE", "DIR", "ENDDIR"

REC_SCHEMA = {
    "kind": pl.Utf8, "path": pl.Utf8, "depth": pl.Int32,
    "size": pl.Int64, "mtime": pl.Int64, "child_count": pl.Int64,
}


def _scan_rows(root: str):
    """Yield record dicts depth-first: DIR before its contents, ENDDIR after."""
    def walk(path, depth):
        try:
            entries = list(os.scandir(path))
        except OSError as exc:
            print(f"scan error: {path}: {exc}", file=sys.stderr)
            entries = []
        dirs, files = [], []
        for e in entries:
            (dirs if e.is_dir(follow_symlinks=False) else files).append(e)
        st = os.stat(path, follow_symlinks=False)
        yield dict(kind=DIR, path=path, depth=depth, size=0,
                   mtime=int(st.st_mtime), child_count=len(entries))
        for f in files:
            try:
                s = f.stat(follow_symlinks=False)
            except OSError:
                continue
            yield dict(kind=FILE, path=f.path, depth=depth + 1, size=s.st_size,
                       mtime=int(s.st_mtime), child_count=0)
        for d in dirs:
            yield from walk(d.path, depth + 1)
        yield dict(kind=ENDDIR, path=path, depth=depth, size=0, mtime=0, child_count=0)

    yield from walk(os.path.abspath(root), 0)


def scan_batches(root: str, batch_size: int = 256):
    """Stream the scan as Polars frames (≈ PyArrow RecordBatches)."""
    buf = []
    for rec in _scan_rows(root):
        buf.append(rec)
        if len(buf) >= batch_size:
            yield pl.DataFrame(buf, schema=REC_SCHEMA)
            buf = []
    if buf:
        yield pl.DataFrame(buf, schema=REC_SCHEMA)


# ===========================================================================
# Planners -- per-batch Polars transforms -> action table (op, src, dst, epoch).
# ===========================================================================

ACTION_SCHEMA = {"op": pl.Utf8, "src": pl.Utf8, "dst": pl.Utf8, "epoch": pl.Int64}
BIG = 1_000_000  # depth ceiling so RMDIR epochs (BIG-depth) stay deepest-first


def _concat(parts):
    parts = [p for p in parts if p.height]
    return pl.concat(parts) if parts else pl.DataFrame(schema=ACTION_SCHEMA)


def plan_rm(batches) -> pl.DataFrame:
    out = []
    for df in batches:
        unlinks = df.filter(pl.col("kind") == FILE).select(
            op=pl.lit("UNLINK"), src=pl.col("path"), dst=pl.lit(""),
            epoch=pl.lit(0, dtype=pl.Int64),
        )
        rmdirs = df.filter(pl.col("kind") == ENDDIR).select(
            op=pl.lit("RMDIR"), src=pl.col("path"), dst=pl.lit(""),
            epoch=(pl.lit(BIG) - pl.col("depth")).cast(pl.Int64),
        )
        out.append(_concat([unlinks, rmdirs]))
    return _concat(out)


def plan_cp(batches, src_root: str, dst_root: str) -> pl.DataFrame:
    src_root, dst_root = os.path.abspath(src_root), os.path.abspath(dst_root)
    # dst = dst_root + (absolute src path with the src_root prefix stripped)
    dst_expr = pl.lit(dst_root) + pl.col("path").str.slice(len(src_root))
    out = []
    for df in batches:
        mkdirs = df.filter(pl.col("kind") == DIR).select(
            op=pl.lit("MKDIR"), src=pl.col("path"), dst=dst_expr,
            epoch=pl.col("depth").cast(pl.Int64),
        )
        copies = df.filter(pl.col("kind") == FILE).select(
            op=pl.lit("COPY"), src=pl.col("path"), dst=dst_expr,
            epoch=pl.col("depth").cast(pl.Int64),
        )
        out.append(_concat([mkdirs, copies]))
    return _concat(out)


def _index(root: str) -> pl.DataFrame:
    """Materialize one tree as a relpath-keyed table (a real diff keeps the
    side it compares against; the other side streams)."""
    root = os.path.abspath(root)
    parts = []
    for df in scan_batches(root):
        parts.append(
            df.filter(pl.col("kind") != ENDDIR).select(
                rel=pl.col("path").str.slice(len(root)),
                kind=pl.col("kind"), size=pl.col("size"), mtime=pl.col("mtime"),
            )
        )
    idx = _concat_idx(parts)
    return idx


def _concat_idx(parts):
    parts = [p for p in parts if p.height]
    schema = {"rel": pl.Utf8, "kind": pl.Utf8, "size": pl.Int64, "mtime": pl.Int64}
    return pl.concat(parts) if parts else pl.DataFrame(schema=schema)


def compare(src_root: str, dst_root: str) -> pl.DataFrame:
    """rsync tree-comparison stage as a Polars full outer join -> action plan."""
    s = _index(src_root).rename({"kind": "skind", "size": "ssize", "mtime": "smtime"})
    d = _index(dst_root).rename({"kind": "dkind", "size": "dsize", "mtime": "dmtime"})
    j = s.join(d, on="rel", how="full", coalesce=True)
    return j.select(
        rel="rel",
        op=pl.when(pl.col("skind").is_null()).then(pl.lit("DELETE"))     # only in dst
        .when(pl.col("dkind").is_null()).then(pl.lit("CREATE"))          # only in src
        .when((pl.col("skind") == FILE)
              & ((pl.col("ssize") != pl.col("dsize"))
                 | (pl.col("smtime") != pl.col("dmtime")))).then(pl.lit("UPDATE"))
        .otherwise(pl.lit("SKIP")),
    )


# ===========================================================================
# Executor -- epoch-ordered drains, out-of-order WITHIN each epoch.
# ===========================================================================

def _apply(a: dict, do: bool) -> str:
    if not do:
        return "dry-run"
    try:
        op = a["op"]
        if op == "UNLINK":
            os.unlink(a["src"])
        elif op == "RMDIR":
            os.rmdir(a["src"])
        elif op == "MKDIR":
            os.makedirs(a["dst"], exist_ok=True)
        elif op == "COPY":
            shutil.copy2(a["src"], a["dst"])
        else:
            return "noop"
        return "ok"
    except OSError as exc:
        return f"errno{exc.errno}"


def execute(plan: pl.DataFrame, *, apply=False, respect_epochs=True,
            out_of_order=True, seed=0, log=False):
    """Apply the action table. Epochs are barrier-delimited drain points; within
    an epoch actions run in arbitrary (shuffled) order to model parallelism."""
    results = []
    rng = random.Random(seed)
    epochs = sorted(plan["epoch"].unique().to_list()) if respect_epochs else [None]
    for e in epochs:
        rows = (plan if e is None else plan.filter(pl.col("epoch") == e)).to_dicts()
        if out_of_order:
            rng.shuffle(rows)                 # <-- no reliance on stream order
        for a in rows:
            status = _apply(a, apply)
            results.append((a, status))
            if log:
                tail = f" -> {a['dst']}" if a["dst"] else ""
                print(f"    e{e} {a['op']:7}{a['src'] or a['dst']}{tail} [{status}]")
        if log and respect_epochs:
            print(f"    --- barrier after epoch {e} ---")
    return results


def _failures(results):
    return [(a, s) for a, s in results if s.startswith("errno")]


# ===========================================================================
# Demo / self-test
# ===========================================================================

def _write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").write(data)


def _build_sample(root):
    _write(f"{root}/top.txt", "top")
    _write(f"{root}/a/f1.txt", "hello")
    _write(f"{root}/a/sub1/g1.bin", "x" * 100)
    _write(f"{root}/a/sub2/g2.bin", "y" * 50)
    _write(f"{root}/b/h.txt", "world")


def _snapshot(root):
    return set(_index(root)["rel"].to_list())


def main():
    work = tempfile.mkdtemp(prefix="toy_fsops_")
    try:
        src = os.path.join(work, "src")
        _build_sample(src)

        print(f"# scan stream of {src} (first batch as a Polars frame)")
        first = next(scan_batches(src, batch_size=100))
        with pl.Config(tbl_rows=20, fmt_str_lengths=60):
            print(first.with_columns(rel=pl.col("path").str.slice(len(src)))
                  .select("kind", "depth", "rel", "size", "child_count"))

        # --- cp -------------------------------------------------------------
        dst = os.path.join(work, "copy")
        print(f"\n# cp  {src} -> {dst}   (executor out-of-order within epochs)")
        execute(plan_cp(scan_batches(src), src, dst), apply=True)
        assert _snapshot(src) == _snapshot(dst), "cp did not reproduce the tree"
        print("  OK: copy matches source")

        # --- rsync compare stage --------------------------------------------
        print("\n# rsync compare  src vs copy")
        plan = compare(src, dst)
        print("  ", dict(plan.group_by("op").len().iter_rows()))
        assert set(plan["op"].unique()) == {"SKIP"}

        _write(f"{dst}/a/f1.txt", "hello, longer now")   # -> UPDATE
        os.remove(f"{dst}/b/h.txt")                       # -> CREATE (missing in dst)
        _write(f"{dst}/extra.txt", "only in dst")         # -> DELETE
        print("\n# rsync compare  src vs mutated copy")
        plan = compare(src, dst).filter(pl.col("op") != "SKIP").sort("op", "rel")
        for rel, op in plan.iter_rows():
            print(f"    {op:7}{rel}")
        ops = set(plan["op"].unique())
        assert {"CREATE", "UPDATE", "DELETE"} <= ops, ops

        # --- rm: barriers vs none, under an out-of-order executor -----------
        print("\n# rm: out-of-order executor, with vs without epoch barriers")
        with_dir = os.path.join(work, "rm_with"); shutil.copytree(src, with_dir)
        without_dir = os.path.join(work, "rm_without"); shutil.copytree(src, without_dir)

        r_with = execute(plan_rm(scan_batches(with_dir)),
                         apply=True, out_of_order=True, respect_epochs=True)
        r_without = execute(plan_rm(scan_batches(without_dir)),
                            apply=True, out_of_order=True, respect_epochs=False)

        print(f"  with barriers   : exists={os.path.exists(with_dir)} "
              f"failures={len(_failures(r_with))}")
        print(f"  without barriers: exists={os.path.exists(without_dir)} "
              f"failures={len(_failures(r_without))}")
        assert not os.path.exists(with_dir) and not _failures(r_with)
        assert _failures(r_without), "expected ENOTEMPTY-style failures w/o barriers"

        # cp also needs ordering: collapse epochs -> copies race ahead of mkdirs
        cp_bad = os.path.join(work, "cp_bad")
        r_cpbad = execute(plan_cp(scan_batches(src), src, cp_bad),
                          apply=True, out_of_order=True, respect_epochs=False)
        print(f"  cp w/o barriers : failures={len(_failures(r_cpbad))} "
              f"(copies before mkdir)")
        assert _failures(r_cpbad)

        print("\nAll demos passed.")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
