"""Tests for feather_from_batches (the Feather + sidecar writer)."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pyarrow as pa
import pytest

from ducl.scan import feather_from_batches
from ducl.schema import SCHEMA, COLUMN_NAMES


def _row(inode, parent_inode, depth, path, ext, uid, gid, size, fs_id, blocks, nlink,
         mode, atime, mtime, ctime, child_count, dir_total_size) -> dict:
    """One row of the canonical 17-column table, keyed by COLUMN_NAMES."""
    return dict(zip(COLUMN_NAMES, (inode, parent_inode, depth, path, ext, uid, gid, size,
                                   fs_id, blocks, nlink, mode, atime, mtime, ctime,
                                   child_count, dir_total_size)))


def _batch(rows: list[dict]) -> pa.RecordBatch:
    return pa.RecordBatch.from_pylist(rows, schema=SCHEMA)


T = 1700000000


def test_feather_from_batches_basic(tmp_path: Path):
    """A small batch round-trips through the Feather writer."""
    rows = [
        _row(1, 0, 0, "/data", "", 1000, 1000, 4096, 1, 8, 3, "040755", T, T, T, 2, 0),
        _row(2, 1, 1, "/data/file1.txt", "txt", 1000, 1000, 1024, 1, 2, 1, "100644", T, T, T, -1, 0),
        _row(3, 1, 1, "/data/file2.dat", "dat", 1000, 1000, 2048, 1, 4, 1, "100644", T, T, T, -1, 0),
    ]
    output = str(tmp_path / "test.feather")
    feather_from_batches([_batch(rows)], output, do_agg=False)

    df = pl.read_ipc(output)
    assert len(df) == 3
    assert set(df.columns) == set(COLUMN_NAMES)
    files = df.filter(pl.col("child_count") == -1)
    assert len(files) == 2
    assert files["size"].sum() == 1024 + 2048


def test_feather_from_batches_with_agg(tmp_path: Path):
    """Sidecar .agg.parquet and .examples.parquet are produced and conserve totals."""
    rows = [
        _row(1, 0, 0, "/data", "", 1000, 1000, 4096, 1, 8, 3, "040755", T, T, T, 2, 0),
        _row(2, 1, 1, "/data/a.txt", "txt", 1000, 1000, 5000, 1, 10, 1, "100644", T, T, T, -1, 0),
        _row(3, 1, 1, "/data/b.txt", "txt", 1000, 1000, 3000, 1, 6, 1, "100644", T, T, T, -1, 0),
        _row(4, 1, 1, "/data/c.dat", "dat", 1000, 1000, 10000, 1, 20, 1, "100644", T, T, T, -1, 0),
    ]
    output = str(tmp_path / "test.feather")
    # two batches, so the per-batch aggregation + final compaction both run
    feather_from_batches([_batch(rows[:2]), _batch(rows[2:])], output, do_agg=True)

    agg_path = tmp_path / "test.agg.parquet"
    examples_path = tmp_path / "test.examples.parquet"
    assert agg_path.exists() and examples_path.exists()
    assert not list(tmp_path.glob("*.tmp"))

    agg = pl.read_parquet(agg_path)
    assert agg["total_size"].sum() == 5000 + 3000 + 10000
    assert agg["file_count"].sum() == 3

    examples = pl.read_parquet(examples_path)
    assert examples.height > 0
    assert set(examples.columns) == {"dir_path", "size_bin", "filename", "ext", "size"}


def test_feather_from_batches_empty(tmp_path: Path):
    """No batches at all still yields a valid, empty Feather file."""
    output = str(tmp_path / "empty.feather")
    feather_from_batches([], output, do_agg=False)
    assert len(pl.read_ipc(output)) == 0


def test_feather_from_batches_atomic(tmp_path: Path):
    """A writer that dies mid-stream must leave no .feather (or .tmp) behind."""

    def batches(fail_after: int):
        for i in range(3):
            if i == fail_after:
                raise MemoryError("boom")
            yield pa.RecordBatch.from_pydict({f.name: [] for f in SCHEMA}, schema=SCHEMA)

    ok = tmp_path / "ok.feather"
    feather_from_batches(batches(99), str(ok), do_agg=False)
    assert ok.read_bytes()[-6:] == b"ARROW1"

    bad = tmp_path / "bad.feather"
    with pytest.raises(MemoryError):
        feather_from_batches(batches(1), str(bad), do_agg=False)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ok.feather"]


def test_open_bvm_feature_detection():
    """Older quivers lack qmax/launch: skip qmax with a warning, refuse launch clearly."""
    from types import SimpleNamespace
    from ducl.quiverscan import _open_bvm

    calls = []

    class NewBvm:
        def __init__(self, exe, nworkers=16, budget_mb=512, launch=None, sock=None, qmax=0):
            calls.append((exe, nworkers, launch, qmax))

    class OldBvm:
        def __init__(self, exe, nworkers=16, budget_mb=512):
            calls.append((exe, nworkers))

    new = SimpleNamespace(_Bvm=NewBvm, default_bvm=lambda: "/bvm")
    _open_bvm(new, None, 8, ["sudo", "--"])
    assert calls[-1] == ("/bvm", 8, ["sudo", "--"], 256)

    old = SimpleNamespace(_Bvm=OldBvm, default_bvm=lambda: "/bvm")
    _open_bvm(old, "/my/bvm", 4, None)
    assert calls[-1] == ("/my/bvm", 4)
    with pytest.raises(RuntimeError, match="launch"):
        _open_bvm(old, None, 4, ["sudo", "--"])
