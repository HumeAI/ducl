"""Tests for agg sidecar fast path -- equivalence with raw build."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from conftest import EXPECTED_TOTAL_SIZE, SYNTHETIC_FILES


def _load_tree(out_dir: Path) -> pl.DataFrame:
    return pl.read_parquet(str(out_dir / "tree.parquet"))


def _load_cube(out_dir: Path) -> pl.DataFrame:
    return pl.read_parquet(str(out_dir / "cube.parquet"))


def _load_examples(out_dir: Path) -> pl.DataFrame:
    return pl.read_parquet(str(out_dir / "examples.parquet"))


def _load_meta(out_dir: Path) -> dict:
    with open(out_dir / "meta.json") as f:
        return json.load(f)


class TestTreeIdentical:
    """Tree structure from agg path must exactly match raw build."""

    def test_tree_identical(self, build_result: Path, build_result_agg: Path):
        raw = _load_tree(build_result).sort("node_id")
        agg = _load_tree(build_result_agg).sort("node_id")
        assert raw["node_id"].to_list() == agg["node_id"].to_list()
        assert raw["parent"].to_list() == agg["parent"].to_list()
        assert raw["name"].to_list() == agg["name"].to_list()
        assert raw["size"].to_list() == agg["size"].to_list()


class TestCubeTotalsMatch:
    """Cube total_size per node must match between raw and agg builds."""

    def test_cube_totals_match(self, build_result: Path, build_result_agg: Path):
        raw_cube = _load_cube(build_result)
        agg_cube = _load_cube(build_result_agg)

        raw_sums = (
            raw_cube.group_by("node_id")
            .agg(pl.col("total_size").sum().alias("total"))
            .sort("node_id")
        )
        agg_sums = (
            agg_cube.group_by("node_id")
            .agg(pl.col("total_size").sum().alias("total"))
            .sort("node_id")
        )

        assert raw_sums["node_id"].to_list() == agg_sums["node_id"].to_list()
        assert raw_sums["total"].to_list() == agg_sums["total"].to_list()


class TestCubeCountsMatch:
    """Cube file counts per node must match between raw and agg builds."""

    def test_cube_counts_match(self, build_result: Path, build_result_agg: Path):
        raw_cube = _load_cube(build_result)
        agg_cube = _load_cube(build_result_agg)

        raw_counts = (
            raw_cube.group_by("node_id")
            .agg(pl.col("count").sum().alias("total_count"))
            .sort("node_id")
        )
        agg_counts = (
            agg_cube.group_by("node_id")
            .agg(pl.col("count").sum().alias("total_count"))
            .sort("node_id")
        )

        assert raw_counts["node_id"].to_list() == agg_counts["node_id"].to_list()
        assert raw_counts["total_count"].to_list() == agg_counts["total_count"].to_list()


class TestMetaIdentical:
    """Meta.json must match between raw and agg builds."""

    def test_meta_identical(self, build_result: Path, build_result_agg: Path):
        raw_meta = _load_meta(build_result)
        agg_meta = _load_meta(build_result_agg)

        assert raw_meta["hist_labels"] == agg_meta["hist_labels"]

        # Extensions: same set and totals
        raw_exts = sorted(raw_meta["extensions"], key=lambda e: e["ext"])
        agg_exts = sorted(agg_meta["extensions"], key=lambda e: e["ext"])
        assert len(raw_exts) == len(agg_exts)
        for r, a in zip(raw_exts, agg_exts):
            assert r["ext"] == a["ext"]
            assert r["total_size"] == a["total_size"]
            assert r["file_count"] == a["file_count"]

        # Folders: same set and totals
        raw_flds = sorted(raw_meta["leaf_folders"], key=lambda f: f["leaf_folder"])
        agg_flds = sorted(agg_meta["leaf_folders"], key=lambda f: f["leaf_folder"])
        assert len(raw_flds) == len(agg_flds)
        for r, a in zip(raw_flds, agg_flds):
            assert r["leaf_folder"] == a["leaf_folder"]
            assert r["total_size"] == a["total_size"]
            assert r["file_count"] == a["file_count"]


class TestAggConservation:
    """Agg sidecar must conserve total size and file count."""

    def test_agg_total_size_conserved(self, build_result_agg: Path, synthetic_feather: Path):
        """Sum of agg total_size equals sum of raw file sizes."""
        base = str(synthetic_feather).removesuffix(".feather")
        agg = pl.read_parquet(base + ".agg.parquet")
        assert agg["total_size"].sum() == EXPECTED_TOTAL_SIZE

    def test_agg_file_count_conserved(self, build_result_agg: Path, synthetic_feather: Path):
        """Sum of agg file_count equals number of files."""
        base = str(synthetic_feather).removesuffix(".feather")
        agg = pl.read_parquet(base + ".agg.parquet")
        expected_count = len(SYNTHETIC_FILES)
        assert agg["file_count"].sum() == expected_count


class TestExamplesNodesMatch:
    """Same set of nodes should have examples in raw and agg builds."""

    def test_examples_nodes_match(self, build_result: Path, build_result_agg: Path):
        raw_ex = _load_examples(build_result)
        agg_ex = _load_examples(build_result_agg)

        raw_nodes = sorted(raw_ex["node_id"].unique().to_list())
        agg_nodes = sorted(agg_ex["node_id"].unique().to_list())
        assert raw_nodes == agg_nodes

    def test_examples_max_per_bin(self, build_result_agg: Path):
        """Max 3 examples per (node_id, size_bin) in agg build."""
        examples = _load_examples(build_result_agg)
        counts = (
            examples.group_by(["node_id", "size_bin"])
            .agg(pl.col("filename").count().alias("n"))
        )
        assert counts["n"].max() <= 3
