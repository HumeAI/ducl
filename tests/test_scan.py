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


# --- hard links -------------------------------------------------------------------------
# One 9000-byte inode linked from /d/a, /d/b and /d/c, plus an ordinary 500-byte file.

def _hardlink_batches() -> list[pa.RecordBatch]:
    def link(inode, path):
        return _row(inode, 1, 1, path, "bin", 1000, 1000, 9000, 1, 18, 3, "100644", T, T, T, -1, 0)
    first = [
        _row(1, 0, 0, "/d", "", 1000, 1000, 4096, 1, 8, 5, "040755", T, T, T, 3, 0),
        link(7, "/d/a/big.bin"),
        _row(8, 1, 1, "/d/a/solo.txt", "txt", 1000, 1000, 500, 1, 1, 1, "100644", T, T, T, -1, 0),
    ]
    # the other two links arrive in LATER batches: `first` must remember across batches
    return [_batch(first), _batch([link(7, "/d/b/big.bin")]), _batch([link(7, "/d/c/big.bin")])]


def _dir_sizes(tmp_path: Path, name: str = "hl") -> dict[str, int]:
    agg = pl.read_parquet(tmp_path / f"{name}.agg.parquet")
    return dict(agg.group_by("dir_path").agg(pl.col("total_size").sum()).iter_rows())


@pytest.mark.parametrize("mode,expected", [
    ("split", {"/d/a": 3000 + 500, "/d/b": 3000, "/d/c": 3000}),
    ("first", {"/d/a": 9000 + 500, "/d/b": 0, "/d/c": 0}),
    ("full",  {"/d/a": 9000 + 500, "/d/b": 9000, "/d/c": 9000}),
])
def test_hardlinks_charging(tmp_path: Path, mode: str, expected: dict):
    """Each policy charges the shared inode's bytes as documented; split/first conserve them."""
    out = str(tmp_path / "hl.feather")
    feather_from_batches(_hardlink_batches(), out, hardlinks=mode)
    assert _dir_sizes(tmp_path) == expected

    agg = pl.read_parquet(tmp_path / "hl.agg.parquet")
    assert agg["file_count"].sum() == 4                      # entries, whatever the policy
    # the Feather and the examples always keep the real size
    assert pl.read_ipc(out).filter(pl.col("inode_id") == 7)["size"].to_list() == [9000] * 3
    ex = pl.read_parquet(tmp_path / "hl.examples.parquet")
    assert ex.filter(pl.col("filename") == "big.bin")["size"].to_list() == [9000] * 3


@pytest.mark.parametrize("mode", ["split", "first", "full"])
def test_hardlinks_slow_path_matches_sidecar(tmp_path: Path, mode: str):
    """dashboard.load_files (no sidecar) charges exactly what the sidecar path does."""
    from ducl.dashboard import load_files

    out = str(tmp_path / "hl.feather")
    feather_from_batches(_hardlink_batches(), out, hardlinks=mode)
    raw = load_files(out, hardlinks=mode).with_columns(
        pl.col("path").str.replace(r"/[^/]+$", "").alias("dir_path"))
    assert dict(raw.group_by("dir_path").agg(pl.col("size").sum()).iter_rows()) == _dir_sizes(tmp_path)


def test_reaggregate_recuts_sidecars(tmp_path: Path):
    """`ducl agg` rebuilds sidecars from the Feather under another policy, without rescanning."""
    from ducl.scan import reaggregate

    out = str(tmp_path / "hl.feather")
    feather_from_batches(_hardlink_batches(), out, hardlinks="full")
    assert sum(_dir_sizes(tmp_path).values()) == 3 * 9000 + 500
    reaggregate(out, hardlinks="split")
    assert sum(_dir_sizes(tmp_path).values()) == 9000 + 500
    assert not list(tmp_path.glob("*.tmp"))


def test_hardlinks_rejects_unknown_mode(tmp_path: Path):
    with pytest.raises(ValueError, match="hardlinks"):
        feather_from_batches(_hardlink_batches(), str(tmp_path / "x.feather"), hardlinks="dedupe")
    assert not list(tmp_path.iterdir())          # rejected before anything is written
