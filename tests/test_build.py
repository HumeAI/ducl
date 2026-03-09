"""Tests for the build command."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from ducl.dashboard import (
    HIST_EDGES,
    N_HIST_BINS,
    assign_size_bin,
    hist_labels,
)
from conftest import (
    EXPECTED_TOTAL_SIZE,
    SYNTHETIC_FILES,
    TEST_MAX_CHILDREN,
    TEST_MIN_FOLDER_SIZE,
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


class TestTreeStructure:
    """test_tree_structure: Verify correct nodes, parent links, root size."""

    def test_root_exists(self, build_result: Path):
        tree = _load_tree(build_result)
        root = tree.filter(pl.col("parent") == "")
        assert root.height == 1
        assert root["node_id"][0] == "/root"

    def test_root_size_equals_total(self, build_result: Path):
        tree = _load_tree(build_result)
        root_size = tree.filter(pl.col("node_id") == "/root")["size"][0]
        assert root_size == EXPECTED_TOTAL_SIZE

    def test_parent_links_valid(self, build_result: Path):
        tree = _load_tree(build_result)
        node_ids = set(tree["node_id"].to_list())
        for row in tree.to_dicts():
            if row["parent"]:
                assert row["parent"] in node_ids, f"Orphan node: {row['node_id']} -> parent {row['parent']}"

    def test_known_nodes_present(self, build_result: Path):
        tree = _load_tree(build_result)
        node_ids = set(tree["node_id"].to_list())
        # big_a and big_b should survive pruning (large enough)
        assert "/root/big_a" in node_ids
        assert "/root/big_b" in node_ids
        # sub1 and sub2 should survive (large enough)
        assert "/root/big_a/sub1" in node_ids
        assert "/root/big_a/sub2" in node_ids


class TestTreePruning:
    """test_tree_pruning: Small folders pruned into __rest__."""

    def test_small_c_pruned(self, build_result: Path):
        """small_c total is tiny (180 bytes), should be in __rest__."""
        tree = _load_tree(build_result)
        node_ids = set(tree["node_id"].to_list())
        assert "/root/small_c" not in node_ids

    def test_root_rest_exists(self, build_result: Path):
        """Root should have a __rest__ containing small_c and other small stuff."""
        tree = _load_tree(build_result)
        root_rest = tree.filter(pl.col("node_id") == "/root/__rest__")
        assert root_rest.height == 1
        assert root_rest["parent"][0] == "/root"

    def test_rest_size_equals_parent_minus_children(self, build_result: Path):
        """__rest__ size = parent - sum(non-rest children)."""
        tree = _load_tree(build_result)
        tree_dict = {r["node_id"]: r for r in tree.to_dicts()}

        for node_id, node in tree_dict.items():
            if "__rest__" in node_id:
                continue
            children = [n for n in tree_dict.values() if n["parent"] == node_id]
            rest_children = [c for c in children if "__rest__" in c["node_id"]]
            non_rest_children = [c for c in children if "__rest__" not in c["node_id"]]
            if rest_children:
                rest = rest_children[0]
                expected_rest = node["size"] - sum(c["size"] for c in non_rest_children)
                assert rest["size"] == expected_rest, \
                    f"__rest__ under {node_id}: {rest['size']} != {expected_rest}"

    def test_sub3_pruned_into_big_a_rest(self, build_result: Path):
        """sub3 is tiny (600 bytes), should be in big_a's __rest__."""
        tree = _load_tree(build_result)
        node_ids = set(tree["node_id"].to_list())
        # With max_children=3, sub3 might survive if under the limit
        # But its size (600 bytes) is above min_folder_size=1024? No, 600 < 1024
        # So sub3 should be pruned
        assert "/root/big_a/sub3" not in node_ids


class TestCubeIntegrity:
    """test_cube_integrity: Non-recursive cube -- each file counted once at deepest node.

    Invariant: cube_sum(X) == tree_size(X) - sum(tree_size(C) for kept child C of X).
    Total conservation: sum of all cube rows == root tree size.
    """

    def test_cube_sums_match_tree(self, build_result: Path):
        tree = _load_tree(build_result)
        cube = _load_cube(build_result)

        tree_dict = {r["node_id"]: r["size"] for r in tree.to_dicts()}

        # Build map: parent -> list of kept (non-rest) children
        children_of: dict[str, list[str]] = {}
        for r in tree.to_dicts():
            if r["parent"] and "__rest__" not in r["node_id"]:
                children_of.setdefault(r["parent"], []).append(r["node_id"])

        cube_sums = (
            cube.group_by("node_id")
            .agg(pl.col("total_size").sum().alias("cube_total"))
        )
        cube_dict = {r["node_id"]: r["cube_total"] for r in cube_sums.to_dicts()}

        for node_id, tree_size in tree_dict.items():
            if "__rest__" in node_id:
                continue
            children_total = sum(tree_dict[c] for c in children_of.get(node_id, []))
            expected = tree_size - children_total
            actual = cube_dict.get(node_id, 0)
            assert actual == expected, \
                f"{node_id}: cube={actual} != expected={expected} " \
                f"(tree={tree_size} - children={children_total})"

    def test_cube_total_conservation(self, build_result: Path):
        """Sum of all cube rows equals total file size (each file counted once)."""
        tree = _load_tree(build_result)
        cube = _load_cube(build_result)
        root_size = tree.filter(pl.col("node_id") == "/root")["size"][0]
        cube_total = cube["total_size"].sum()
        assert cube_total == root_size


class TestExtensionBreakdown:
    """test_extension_breakdown: Known files with known exts -> verify cube ext totals."""

    def test_wav_total_in_sub1(self, build_result: Path):
        cube = _load_cube(build_result)
        sub1_wav = cube.filter(
            (pl.col("node_id") == "/root/big_a/sub1") & (pl.col("ext") == "wav")
        )
        expected = 5_000_000_000 + 2_000_000_000  # track1 + track2
        actual = sub1_wav["total_size"].sum()
        assert actual == expected

    def test_tar_gz_recognized(self, build_result: Path):
        """tar.gz files should have ext='tar.gz', not 'gz'."""
        cube = _load_cube(build_result)
        tar_gz = cube.filter(pl.col("ext") == "tar.gz")
        assert tar_gz.height > 0, "No tar.gz entries found"
        gz_plain = cube.filter(pl.col("ext") == "gz")
        assert gz_plain.height == 0, "Should not have plain 'gz' ext"

    def test_no_ext_recognized(self, build_result: Path):
        """Files with empty ext should become '(no ext)'.

        '(no ext)' may not be in top-N extensions (-> null in cube),
        but it must appear in the full extension summary or as null in cube.
        """
        meta = _load_meta(build_result)
        all_exts = [e["ext"] for e in meta["extensions"]]
        cube = _load_cube(build_result)
        if "(no ext)" in all_exts:
            no_ext = cube.filter(pl.col("ext") == "(no ext)")
            assert no_ext.height > 0
        else:
            # (no ext) files are small, not in top-N -> they become null (other bucket)
            null_ext = cube.filter(pl.col("ext").is_null())
            assert null_ext.height > 0, "Expected null ext rows for 'other' bucket"


class TestFolderBreakdown:
    """test_folder_breakdown: Same for leaf_folder dimension."""

    def test_sub1_folder_in_cube(self, build_result: Path):
        """Files in sub1/ should have leaf_folder='sub1'."""
        cube = _load_cube(build_result)
        meta = _load_meta(build_result)
        top_folders = [f["leaf_folder"] for f in meta["leaf_folders"]]
        if "sub1" in top_folders:
            sub1_folder = cube.filter(
                (pl.col("node_id") == "/root/big_a/sub1") & (pl.col("leaf_folder") == "sub1")
            )
            assert sub1_folder["total_size"].sum() > 0

    def test_null_folder_means_other(self, build_result: Path):
        """Folders not in top-N have null leaf_folder."""
        cube = _load_cube(build_result)
        meta = _load_meta(build_result)
        top_folders = set(f["leaf_folder"] for f in meta["leaf_folders"])
        non_null = cube.filter(pl.col("leaf_folder").is_not_null())
        for row in non_null.to_dicts():
            assert row["leaf_folder"] in top_folders


class TestCrossFilter:
    """test_cross_filter: Query cube for specific (ext, folder) combinations."""

    def test_wav_in_sub1(self, build_result: Path):
        cube = _load_cube(build_result)
        meta = _load_meta(build_result)
        top_folders = [f["leaf_folder"] for f in meta["leaf_folders"]]
        if "sub1" in top_folders:
            result = cube.filter(
                (pl.col("node_id") == "/root/big_a/sub1")
                & (pl.col("ext") == "wav")
                & (pl.col("leaf_folder") == "sub1")
            )
            # track1.wav (5GB) + track2.wav (2GB) are in /root/big_a/sub1/
            expected_total = 5_000_000_000 + 2_000_000_000
            assert result["total_size"].sum() == expected_total


class TestHistogramBinning:
    """test_histogram_binning: Known file sizes -> verify correct bin assignments."""

    def test_bin_assignments(self):
        """Verify assign_size_bin produces correct bins for known sizes."""
        sizes = np.array([
            500,               # < 100KB -> bin 0
            200_000,           # 100KB-300KB -> bin 1
            5_000_000,         # 3MB-10MB -> bin 4
            50_000_000,        # 30MB-100MB -> bin 6
            500_000_000,       # 300MB-1GB -> bin 8
            15_000_000_000,    # > 10GB -> bin 11
        ], dtype=float)
        bins = assign_size_bin(sizes)
        assert bins[0] == 0   # 500 B
        assert bins[1] == 1   # 200 KB
        assert bins[2] == 4   # 5 MB
        assert bins[3] == 6   # 50 MB
        assert bins[4] == 8   # 500 MB
        assert bins[5] == 11  # 15 GB

    def test_cube_bin_counts_match(self, build_result: Path):
        """For a known leaf node, verify bin counts match hand-counted files."""
        cube = _load_cube(build_result)
        # /root/big_b has 5 files, check total count
        big_b_cube = cube.filter(pl.col("node_id") == "/root/big_b")
        total_files = big_b_cube["count"].sum()
        expected_files = len([f for f in SYNTHETIC_FILES if f["path"].startswith("/root/big_b/")])
        assert total_files == expected_files


class TestExamples:
    """test_examples: Each (node, bin) has <= 3 examples, sizes in bin range."""

    def test_max_examples_per_bin(self, build_result: Path):
        examples = _load_examples(build_result)
        counts = (
            examples.group_by(["node_id", "size_bin"])
            .agg(pl.col("filename").count().alias("n"))
        )
        assert counts["n"].max() <= 3

    def test_example_sizes_in_bin_range(self, build_result: Path):
        examples = _load_examples(build_result)
        for row in examples.to_dicts():
            b = row["size_bin"]
            sz = row["size"]
            if b == 0:
                assert sz < HIST_EDGES[0], f"Bin 0 example size {sz} >= {HIST_EDGES[0]}"
            elif b == N_HIST_BINS - 1:
                assert sz >= HIST_EDGES[-1], f"Bin {b} example size {sz} < {HIST_EDGES[-1]}"
            else:
                assert sz >= HIST_EDGES[b - 1], f"Bin {b} example size {sz} < {HIST_EDGES[b-1]}"
                assert sz < HIST_EDGES[b], f"Bin {b} example size {sz} >= {HIST_EDGES[b]}"

    def test_examples_have_real_filenames(self, build_result: Path):
        examples = _load_examples(build_result)
        for row in examples.to_dicts():
            assert row["filename"], "Empty filename in examples"
            assert "/" not in row["filename"], f"Filename contains slash: {row['filename']}"


class TestMeta:
    """test_meta: meta.json has correct hist_labels, extension/folder summaries."""

    def test_hist_labels_count(self, build_result: Path):
        meta = _load_meta(build_result)
        assert len(meta["hist_labels"]) == N_HIST_BINS

    def test_hist_labels_content(self, build_result: Path):
        meta = _load_meta(build_result)
        expected = hist_labels()
        assert meta["hist_labels"] == expected

    def test_extensions_present(self, build_result: Path):
        meta = _load_meta(build_result)
        assert len(meta["extensions"]) > 0
        for entry in meta["extensions"]:
            assert "ext" in entry
            assert "total_size" in entry
            assert "file_count" in entry

    def test_leaf_folders_present(self, build_result: Path):
        meta = _load_meta(build_result)
        assert len(meta["leaf_folders"]) > 0
        for entry in meta["leaf_folders"]:
            assert "leaf_folder" in entry
            assert "total_size" in entry
            assert "file_count" in entry

    def test_extension_totals(self, build_result: Path):
        """Sum of top-N extension totals should be <= total file size."""
        meta = _load_meta(build_result)
        ext_total = sum(e["total_size"] for e in meta["extensions"])
        assert ext_total <= EXPECTED_TOTAL_SIZE

    def test_wav_total_correct(self, build_result: Path):
        """Verify wav extension total matches known wav files."""
        meta = _load_meta(build_result)
        wav_entry = next((e for e in meta["extensions"] if e["ext"] == "wav"), None)
        if wav_entry:
            expected = sum(
                f["size"] for f in SYNTHETIC_FILES
                if f["path"].endswith(".wav")
            )
            assert wav_entry["total_size"] == expected
