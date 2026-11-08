"""Tests for feather_from_csv_stream."""

from __future__ import annotations

import io
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.ipc as ipc
import pytest

from ducl.scan import feather_from_csv_stream, pwalk2_command
from ducl.schema import SCHEMA, COLUMN_NAMES


def _make_csv_line(
    inode: int, parent_inode: int, depth: int, path: str, ext: str,
    uid: int, gid: int, size: int, fs_id: int, blocks: int, nlink: int,
    mode: str, atime: int, mtime: int, ctime: int,
    child_count: int, dir_total_size: int,
) -> str:
    """Format a single CSV line matching pwalk2 output."""
    return (
        f'{inode},{parent_inode},{depth},"{path}","{ext}",'
        f'{uid},{gid},{size},{fs_id},{blocks},{nlink},"{mode}",'
        f'{atime},{mtime},{ctime},{child_count},{dir_total_size}\n'
    )


def test_feather_from_csv_stream_basic(tmp_path: Path):
    """Write a small CSV buffer and verify the output Feather file."""
    lines = [
        _make_csv_line(1, 0, 0, "/data", "", 1000, 1000, 4096, 1, 8, 3, "040755",
                       1700000000, 1700000000, 1700000000, 2, 0),
        _make_csv_line(2, 1, 1, "/data/file1.txt", "txt", 1000, 1000, 1024, 1, 2, 1, "100644",
                       1700000000, 1700000000, 1700000000, -1, 0),
        _make_csv_line(3, 1, 1, "/data/file2.dat", "dat", 1000, 1000, 2048, 1, 4, 1, "100644",
                       1700000000, 1700000000, 1700000000, -1, 0),
    ]
    csv_bytes = "".join(lines).encode()
    stream = io.BytesIO(csv_bytes)

    output = str(tmp_path / "test.feather")
    feather_from_csv_stream(stream, output, do_agg=False)

    # Read back and verify
    df = pl.read_ipc(output)
    assert len(df) == 3
    assert set(df.columns) == set(COLUMN_NAMES)

    # Verify data integrity
    files = df.filter(pl.col("child_count") == -1)
    assert len(files) == 2
    assert files["size"].sum() == 1024 + 2048


def test_feather_from_csv_stream_with_agg(tmp_path: Path):
    """Verify sidecar .agg.parquet and .examples.parquet are produced."""
    lines = [
        _make_csv_line(1, 0, 0, "/data", "", 1000, 1000, 4096, 1, 8, 3, "040755",
                       1700000000, 1700000000, 1700000000, 2, 0),
        _make_csv_line(2, 1, 1, "/data/a.txt", "txt", 1000, 1000, 5000, 1, 10, 1, "100644",
                       1700000000, 1700000000, 1700000000, -1, 0),
        _make_csv_line(3, 1, 1, "/data/b.txt", "txt", 1000, 1000, 3000, 1, 6, 1, "100644",
                       1700000000, 1700000000, 1700000000, -1, 0),
        _make_csv_line(4, 1, 1, "/data/c.dat", "dat", 1000, 1000, 10000, 1, 20, 1, "100644",
                       1700000000, 1700000000, 1700000000, -1, 0),
    ]
    csv_bytes = "".join(lines).encode()
    stream = io.BytesIO(csv_bytes)

    output = str(tmp_path / "test.feather")
    feather_from_csv_stream(stream, output, do_agg=True)

    # Verify sidecar files exist
    agg_path = str(tmp_path / "test.agg.parquet")
    examples_path = str(tmp_path / "test.examples.parquet")
    assert Path(agg_path).exists()
    assert Path(examples_path).exists()

    # Verify agg conservation
    agg = pl.read_parquet(agg_path)
    assert agg["total_size"].sum() == 5000 + 3000 + 10000
    assert agg["file_count"].sum() == 3

    # Verify examples
    examples = pl.read_parquet(examples_path)
    assert examples.height > 0
    assert set(examples.columns) == {"dir_path", "size_bin", "filename", "ext", "size"}


def test_feather_from_csv_stream_empty(tmp_path: Path):
    """An empty CSV should produce a valid but empty Feather file."""
    stream = io.BytesIO(b"")
    output = str(tmp_path / "empty.feather")
    feather_from_csv_stream(stream, output, do_agg=False)

    # Should be a valid IPC file with 0 rows
    df = pl.read_ipc(output)
    assert len(df) == 0


def test_feather_from_batches_atomic(tmp_path: Path):
    """A writer that dies mid-stream must leave no .feather (or .tmp) behind."""
    import pyarrow as pa
    from ducl.scan import SCHEMA, feather_from_batches

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


def test_pwalk2_command_sudo(tmp_path: Path, monkeypatch):
    """--sudo prefixes only the walker with `sudo --`; nothing else changes."""
    import ducl.scan as scan_mod

    fake = tmp_path / "pwalk2"
    fake.write_text("")
    monkeypatch.setattr(scan_mod.shutil, "which", lambda name: "/usr/bin/sudo")

    plain = pwalk2_command("/data", pwalk2_args=["--threads", "8"], binary=fake)
    assert plain == [str(fake), "--threads", "8", "/data"]

    elevated = pwalk2_command("/data", pwalk2_args=["--threads", "8"], sudo=True, binary=fake)
    assert elevated == ["sudo", "--"] + plain

    monkeypatch.setattr(scan_mod.shutil, "which", lambda name: None)
    with pytest.raises(FileNotFoundError, match="sudo"):
        pwalk2_command("/data", sudo=True, binary=fake)


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
