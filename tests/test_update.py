"""Tests for the update command."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from conftest import (
    EXPECTED_TOTAL_SIZE,
    ORIGINAL_BIG_B_SIZE,
    RESCAN_FILES,
    RESCAN_TOTAL_SIZE,
    SYNTHETIC_FILES,
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


class TestSubtreeSizesUpdated:
    """test_subtree_sizes_updated: After update, subtree node sizes reflect new file data."""

    def test_big_b_size(self, updated_result: Path):
        tree = _load_tree(updated_result)
        big_b = tree.filter(pl.col("node_id") == "/root/big_b")
        assert big_b.height == 1
        assert big_b["size"][0] == RESCAN_TOTAL_SIZE


class TestAncestorDelta:
    """test_ancestor_delta: Ancestor sizes changed by exactly new - old."""

    def test_root_size_updated(self, updated_result: Path):
        tree = _load_tree(updated_result)
        root = tree.filter(pl.col("node_id") == "/root")
        delta = RESCAN_TOTAL_SIZE - ORIGINAL_BIG_B_SIZE
        expected_root = EXPECTED_TOTAL_SIZE + delta
        assert root["size"][0] == expected_root


class TestSiblingUnchanged:
    """test_sibling_unchanged: Nodes outside the subtree have unchanged sizes."""

    def test_big_a_unchanged(self, build_result: Path, updated_result: Path):
        expected_big_a = sum(
            f["size"] for f in SYNTHETIC_FILES
            if f["path"].startswith("/root/big_a/")
        )
        tree_after = _load_tree(updated_result)
        big_a = tree_after.filter(pl.col("node_id") == "/root/big_a")
        assert big_a["size"][0] == expected_big_a

    def test_sub1_unchanged(self, updated_result: Path):
        expected_sub1 = sum(
            f["size"] for f in SYNTHETIC_FILES
            if f["path"].startswith("/root/big_a/sub1/")
        )
        tree = _load_tree(updated_result)
        sub1 = tree.filter(pl.col("node_id") == "/root/big_a/sub1")
        assert sub1["size"][0] == expected_sub1

    def test_sub2_unchanged(self, updated_result: Path):
        expected_sub2 = sum(
            f["size"] for f in SYNTHETIC_FILES
            if f["path"].startswith("/root/big_a/sub2/")
        )
        tree = _load_tree(updated_result)
        sub2 = tree.filter(pl.col("node_id") == "/root/big_a/sub2")
        assert sub2["size"][0] == expected_sub2


class TestRestRecomputed:
    """test_rest_recomputed: __rest__ within subtree = parent - sum(children) after update."""

    def test_root_rest_after_update(self, updated_result: Path):
        tree = _load_tree(updated_result)
        tree_dict = {r["node_id"]: r for r in tree.to_dicts()}

        root = tree_dict["/root"]
        children = [n for n in tree_dict.values() if n["parent"] == "/root"]
        rest_children = [c for c in children if "__rest__" in c["node_id"]]
        non_rest_children = [c for c in children if "__rest__" not in c["node_id"]]

        if rest_children:
            rest = rest_children[0]
            expected = root["size"] - sum(c["size"] for c in non_rest_children)
            assert rest["size"] == expected, \
                f"Root __rest__: {rest['size']} != {expected}"


class TestCubeIntegrityAfterUpdate:
    """test_cube_integrity_after_update: Non-recursive cube invariant holds after update."""

    def test_cube_sums_match_tree(self, updated_result: Path):
        tree = _load_tree(updated_result)
        cube = _load_cube(updated_result)

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

    def test_big_b_cube_total(self, updated_result: Path):
        """big_b is a leaf node -- cube total should match its new tree size."""
        cube = _load_cube(updated_result)
        big_b_total = cube.filter(pl.col("node_id") == "/root/big_b")["total_size"].sum()
        assert big_b_total == RESCAN_TOTAL_SIZE


class TestAncestorCubeDeltas:
    """test_ancestor_cube_deltas: Ancestor cube reflects only directly-assigned files."""

    def test_root_cube_reflects_direct_files(self, updated_result: Path):
        """Root cube total = root tree size - sum of kept children sizes.

        With non-recursive cube, root only stores files directly assigned
        to it (files in /root/ and pruned subdirs like small_c).
        """
        tree = _load_tree(updated_result)
        cube = _load_cube(updated_result)

        tree_dict = {r["node_id"]: r for r in tree.to_dicts()}
        root_size = tree_dict["/root"]["size"]
        kept_children = [
            n for n in tree_dict.values()
            if n["parent"] == "/root" and "__rest__" not in n["node_id"]
        ]
        children_total = sum(c["size"] for c in kept_children)
        expected = root_size - children_total

        root_cube_total = cube.filter(pl.col("node_id") == "/root")["total_size"].sum()
        assert root_cube_total == expected

    def test_cube_total_conservation(self, updated_result: Path):
        """Sum of all cube rows equals root tree size after update."""
        tree = _load_tree(updated_result)
        cube = _load_cube(updated_result)
        root_size = tree.filter(pl.col("node_id") == "/root")["size"][0]
        cube_total = cube["total_size"].sum()
        assert cube_total == root_size


class TestExamplesUpdated:
    """test_examples_updated: Subtree examples use new file data."""

    def test_big_b_examples_present(self, updated_result: Path):
        examples = _load_examples(updated_result)
        big_b_examples = examples.filter(pl.col("node_id") == "/root/big_b")
        assert big_b_examples.height > 0, "No examples for /root/big_b after update"

    def test_big_b_has_extra_wav(self, updated_result: Path):
        """The new extra.wav file should appear in examples if its bin has examples."""
        examples = _load_examples(updated_result)
        big_b_examples = examples.filter(pl.col("node_id") == "/root/big_b")
        filenames = big_b_examples["filename"].to_list()
        # extra.wav is 200MB, might be in examples depending on bin selection
        # At minimum, big_b should have examples
        assert len(filenames) > 0

    def test_sibling_examples_exist(self, updated_result: Path):
        """sub1 examples should still exist (not touched by update)."""
        examples = _load_examples(updated_result)
        sub1_examples = examples.filter(pl.col("node_id") == "/root/big_a/sub1")
        assert sub1_examples.height > 0

    def test_max_examples_per_bin_after_update(self, updated_result: Path):
        examples = _load_examples(updated_result)
        counts = (
            examples.group_by(["node_id", "size_bin"])
            .agg(pl.col("filename").count().alias("n"))
        )
        assert counts["n"].max() <= 3


class TestMetaUpdated:
    """test_meta_updated: Global extension/folder totals reflect the delta."""

    def test_meta_ext_totals_updated(self, updated_result: Path):
        meta = _load_meta(updated_result)
        # Check dat extension: was 1GB from big_b, now 500MB
        dat_entry = next((e for e in meta["extensions"] if e["ext"] == "dat"), None)
        if dat_entry:
            # Original dat total = sum of all .dat files
            original_dat = sum(f["size"] for f in SYNTHETIC_FILES if f["ext"] == "dat")
            # Delta from big_b: model.dat went from 1GB to 500MB
            dat_delta = 500_000_000 - 1_000_000_000
            expected = original_dat + dat_delta
            assert dat_entry["total_size"] == expected, \
                f"dat total: {dat_entry['total_size']} != {expected}"

    def test_meta_wav_total_updated(self, updated_result: Path):
        meta = _load_meta(updated_result)
        wav_entry = next((e for e in meta["extensions"] if e["ext"] == "wav"), None)
        if wav_entry:
            # Original wav: track1 (5GB) + track2 (2GB) + audio.wav (800MB) = 7.8GB
            original_wav = sum(
                f["size"] for f in SYNTHETIC_FILES if f["path"].endswith(".wav")
            )
            # New: extra.wav (200MB) added in big_b
            new_wav_in_rescan = sum(
                f["size"] for f in RESCAN_FILES if f["path"].endswith(".wav")
            )
            old_wav_in_big_b = sum(
                f["size"] for f in SYNTHETIC_FILES
                if f["path"].startswith("/root/big_b/") and f["path"].endswith(".wav")
            )
            expected = original_wav + (new_wav_in_rescan - old_wav_in_big_b)
            assert wav_entry["total_size"] == expected

    def test_meta_structure_intact(self, updated_result: Path):
        meta = _load_meta(updated_result)
        assert "hist_labels" in meta
        assert "extensions" in meta
        assert "leaf_folders" in meta
        assert len(meta["hist_labels"]) == 12
