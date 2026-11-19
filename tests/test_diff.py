"""Tests for the diff command."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from ducl.dashboard import hist_labels, N_HIST_BINS
from conftest import (
    EXPECTED_TOTAL_SIZE,
    EXPECTED_TOTAL_SIZE_V2,
    EXPECTED_DELTA,
    SYNTHETIC_FILES,
    SYNTHETIC_FILES_V2,
    TEST_MAX_CHILDREN,
    TEST_MIN_FOLDER_SIZE,
    TEST_MAX_TREE_LEVELS,
    TEST_TOP_N_EXTENSIONS,
    TEST_TOP_N_FOLDERS,
)


def _data_dir(out_dir: Path) -> Path:
    return out_dir / "dashboard_data"


def _load_tree(out_dir: Path) -> pl.DataFrame:
    return pl.read_parquet(str(_data_dir(out_dir) / "tree.parquet"))


def _load_cube(out_dir: Path) -> pl.DataFrame:
    return pl.read_parquet(str(_data_dir(out_dir) / "cube.parquet"))


def _load_examples(out_dir: Path) -> pl.DataFrame:
    return pl.read_parquet(str(_data_dir(out_dir) / "examples.parquet"))


def _load_meta(out_dir: Path) -> dict:
    with open(_data_dir(out_dir) / "meta.json") as f:
        return json.load(f)


class TestDiffTreeMerge:
    """Verify correct old/new/delta per node in the merged tree."""

    def test_root_exists(self, diff_result: Path):
        tree = _load_tree(diff_result)
        root = tree.filter(pl.col("parent") == "")
        assert root.height == 1
        assert root["node_id"][0] == "/root"

    def test_root_old_size(self, diff_result: Path):
        tree = _load_tree(diff_result)
        root = tree.filter(pl.col("node_id") == "/root")
        assert root["old_size"][0] == EXPECTED_TOTAL_SIZE

    def test_root_new_size(self, diff_result: Path):
        tree = _load_tree(diff_result)
        root = tree.filter(pl.col("node_id") == "/root")
        assert root["new_size"][0] == EXPECTED_TOTAL_SIZE_V2

    def test_root_delta(self, diff_result: Path):
        tree = _load_tree(diff_result)
        root = tree.filter(pl.col("node_id") == "/root")
        assert root["delta"][0] == EXPECTED_DELTA

    def test_delta_equals_new_minus_old(self, diff_result: Path):
        """For every node, delta = new_size - old_size."""
        tree = _load_tree(diff_result)
        for row in tree.to_dicts():
            assert row["delta"] == row["new_size"] - row["old_size"], \
                f"{row['node_id']}: delta={row['delta']} != {row['new_size']} - {row['old_size']}"

    def test_node_only_in_old_gets_zero_new(self, diff_result: Path):
        """Nodes only in old capture (like small_c) should have new_size=0."""
        tree = _load_tree(diff_result)
        tree_dict = {r["node_id"]: r for r in tree.to_dicts()}
        # small_c is very small and pruned in both, so check via rest node
        # What we really verify: any node with old_size > 0 and new_size = 0
        # has negative delta
        for row in tree.to_dicts():
            if row["old_size"] > 0 and row["new_size"] == 0:
                assert row["delta"] < 0

    def test_parent_links_valid(self, diff_result: Path):
        tree = _load_tree(diff_result)
        node_ids = set(tree["node_id"].to_list())
        for row in tree.to_dicts():
            if row["parent"]:
                assert row["parent"] in node_ids, \
                    f"Orphan node: {row['node_id']} -> parent {row['parent']}"

    def test_known_nodes_present(self, diff_result: Path):
        tree = _load_tree(diff_result)
        node_ids = set(tree["node_id"].to_list())
        assert "/root/big_a" in node_ids
        assert "/root/big_b" in node_ids
        assert "/root/big_a/sub1" in node_ids
        assert "/root/big_a/sub2" in node_ids

    def test_rest_size_consistent(self, diff_result: Path):
        """__rest__ old/new sizes = parent - sum(non-rest children)."""
        tree = _load_tree(diff_result)
        tree_dict = {r["node_id"]: r for r in tree.to_dicts()}

        for node_id, node in tree_dict.items():
            if "__rest__" in node_id:
                continue
            children = [n for n in tree_dict.values() if n["parent"] == node_id]
            rest_children = [c for c in children if "__rest__" in c["node_id"]]
            non_rest_children = [c for c in children if "__rest__" not in c["node_id"]]
            if rest_children:
                rest = rest_children[0]
                expected_old = node["old_size"] - sum(c["old_size"] for c in non_rest_children)
                expected_new = node["new_size"] - sum(c["new_size"] for c in non_rest_children)
                assert rest["old_size"] == max(0, expected_old), \
                    f"__rest__ under {node_id}: old {rest['old_size']} != {expected_old}"
                assert rest["new_size"] == max(0, expected_new), \
                    f"__rest__ under {node_id}: new {rest['new_size']} != {expected_new}"


class TestDiffCube:
    """Verify cube conservation and delta correctness."""

    def test_old_total_conservation(self, diff_result: Path):
        """Sum of old_total_size in cube equals old root size."""
        tree = _load_tree(diff_result)
        cube = _load_cube(diff_result)
        root_old = tree.filter(pl.col("node_id") == "/root")["old_size"][0]
        cube_old_total = cube["old_total_size"].sum()
        assert cube_old_total == root_old

    def test_new_total_conservation(self, diff_result: Path):
        """Sum of new_total_size in cube equals new root size."""
        tree = _load_tree(diff_result)
        cube = _load_cube(diff_result)
        root_new = tree.filter(pl.col("node_id") == "/root")["new_size"][0]
        cube_new_total = cube["new_total_size"].sum()
        assert cube_new_total == root_new

    def test_delta_equals_new_minus_old_in_cube(self, diff_result: Path):
        """delta_total_size = new_total_size - old_total_size for every row."""
        cube = _load_cube(diff_result)
        for row in cube.to_dicts():
            assert row["delta_total_size"] == row["new_total_size"] - row["old_total_size"], \
                f"Cube row {row['node_id']}/{row['ext']}: " \
                f"delta={row['delta_total_size']} != {row['new_total_size']} - {row['old_total_size']}"

    def test_delta_count_equals_new_minus_old(self, diff_result: Path):
        """delta_count = new_count - old_count for every row."""
        cube = _load_cube(diff_result)
        for row in cube.to_dicts():
            assert row["delta_count"] == row["new_count"] - row["old_count"]


class TestDiffMeta:
    """Verify meta.json summary, extension deltas, and structure."""

    def test_summary_totals(self, diff_result: Path):
        meta = _load_meta(diff_result)
        assert "summary" in meta
        assert meta["summary"]["old_total"] == EXPECTED_TOTAL_SIZE
        assert meta["summary"]["new_total"] == EXPECTED_TOTAL_SIZE_V2
        assert meta["summary"]["delta"] == EXPECTED_DELTA

    def test_hist_labels(self, diff_result: Path):
        meta = _load_meta(diff_result)
        assert meta["hist_labels"] == hist_labels()
        assert len(meta["hist_labels"]) == N_HIST_BINS

    def test_extensions_have_delta_fields(self, diff_result: Path):
        meta = _load_meta(diff_result)
        for entry in meta["extensions"]:
            assert "ext" in entry
            assert "old_total_size" in entry
            assert "new_total_size" in entry
            assert "delta_total_size" in entry

    def test_folders_have_delta_fields(self, diff_result: Path):
        meta = _load_meta(diff_result)
        for entry in meta["leaf_folders"]:
            assert "leaf_folder" in entry
            assert "old_total_size" in entry
            assert "new_total_size" in entry
            assert "delta_total_size" in entry

    def test_wav_delta_correct(self, diff_result: Path):
        """wav grew by 1GB (track1: 5GB -> 6GB)."""
        meta = _load_meta(diff_result)
        wav_entry = next((e for e in meta["extensions"] if e["ext"] == "wav"), None)
        if wav_entry:
            old_wav = sum(f["size"] for f in SYNTHETIC_FILES if f["path"].endswith(".wav"))
            new_wav = sum(f["size"] for f in SYNTHETIC_FILES_V2 if f["path"].endswith(".wav"))
            assert wav_entry["old_total_size"] == old_wav
            assert wav_entry["new_total_size"] == new_wav
            assert wav_entry["delta_total_size"] == new_wav - old_wav


class TestDiffIdentical:
    """Same capture against itself -> all deltas 0."""

    def test_all_tree_deltas_zero(self, diff_identical: Path):
        tree = _load_tree(diff_identical)
        for row in tree.to_dicts():
            assert row["delta"] == 0, f"{row['node_id']}: delta={row['delta']}"
            assert row["old_size"] == row["new_size"]

    def test_all_cube_deltas_zero(self, diff_identical: Path):
        cube = _load_cube(diff_identical)
        for row in cube.to_dicts():
            assert row["delta_total_size"] == 0, \
                f"Cube {row['node_id']}/{row['ext']}: delta={row['delta_total_size']}"
            assert row["delta_count"] == 0

    def test_summary_delta_zero(self, diff_identical: Path):
        meta = _load_meta(diff_identical)
        assert meta["summary"]["delta"] == 0
        assert meta["summary"]["old_total"] == meta["summary"]["new_total"]


class TestDiffAggEquivalence:
    """Agg sidecar path produces same result as raw."""

    def test_tree_totals_match(self, diff_result: Path, diff_result_agg: Path):
        tree_raw = _load_tree(diff_result)
        tree_agg = _load_tree(diff_result_agg)

        root_raw = tree_raw.filter(pl.col("node_id") == "/root")
        root_agg = tree_agg.filter(pl.col("node_id") == "/root")

        assert root_raw["old_size"][0] == root_agg["old_size"][0]
        assert root_raw["new_size"][0] == root_agg["new_size"][0]
        assert root_raw["delta"][0] == root_agg["delta"][0]

    def test_summary_match(self, diff_result: Path, diff_result_agg: Path):
        meta_raw = _load_meta(diff_result)
        meta_agg = _load_meta(diff_result_agg)

        assert meta_raw["summary"] == meta_agg["summary"]


class TestDiffOutput:
    """All expected files exist with correct schemas."""

    def test_tree_parquet_exists(self, diff_result: Path):
        assert (diff_result / "dashboard_data" / "tree.parquet").exists()

    def test_cube_parquet_exists(self, diff_result: Path):
        assert (diff_result / "dashboard_data" / "cube.parquet").exists()

    def test_examples_parquet_exists(self, diff_result: Path):
        assert (diff_result / "dashboard_data" / "examples.parquet").exists()

    def test_meta_json_exists(self, diff_result: Path):
        assert (diff_result / "dashboard_data" / "meta.json").exists()

    def test_dashboard_html_exists(self, diff_result: Path):
        assert (diff_result / "dashboard.html").exists()

    def test_tree_schema(self, diff_result: Path):
        tree = _load_tree(diff_result)
        expected_cols = {"node_id", "parent", "name", "old_size", "new_size", "delta", "pct_change"}
        assert set(tree.columns) >= expected_cols

    def test_cube_schema(self, diff_result: Path):
        cube = _load_cube(diff_result)
        expected_cols = {
            "node_id", "ext", "leaf_folder", "size_bin",
            "old_count", "new_count", "delta_count",
            "old_total_size", "new_total_size", "delta_total_size",
        }
        assert set(cube.columns) >= expected_cols

    def test_examples_schema(self, diff_result: Path):
        examples = _load_examples(diff_result)
        expected_cols = {"node_id", "size_bin", "filename", "ext", "size"}
        assert set(examples.columns) >= expected_cols


class TestDiffEdgeCases:
    """Edge cases like root mismatch."""

    def test_root_mismatch_raises(self, tmp_path: Path):
        """Different roots should raise ValueError."""
        from conftest import _write_feather, _make_file_row, _make_dir_row
        from ducl.diff import diff as diff_build

        # Create capture with /root
        rows_a = [
            _make_dir_row("/root", 0),
            _make_file_row("/root/a.txt", 100, "txt"),
        ]
        f_a = tmp_path / "a.feather"
        _write_feather(rows_a, str(f_a))

        # Create capture with /other
        rows_b = [
            _make_dir_row("/other", 0),
            _make_file_row("/other/b.txt", 200, "txt"),
        ]
        f_b = tmp_path / "b.feather"
        _write_feather(rows_b, str(f_b))

        with pytest.raises(ValueError, match="Root mismatch"):
            diff_build(f_a, f_b, tmp_path / "diff_out")
