"""Tests for folder-name collapse (ducl.foldernorm)."""

from __future__ import annotations

import polars as pl

from ducl.foldernorm import DIGIT_RULE, build_mapping, normalize_columns


def test_build_mapping_capture_group_collapses():
    rules = [DIGIT_RULE, r"delivery-(.*)", r".+-batch-(.*)"]
    m = build_mapping(
        [
            "delivery-20260423",                       # digit run -> delivery-*
            "resharded-delivery-2026-03-05-AJCNrXqe",  # unanchored delivery rule
            "zh-batch-0",                              # keep prefix, fold number
            "room-RM09a04e-1779919627",               # fold epoch only
            "keep-me",                                 # unchanged -> absent
        ],
        rules,
    )
    assert m["delivery-20260423"] == "delivery-*"
    assert m["resharded-delivery-2026-03-05-AJCNrXqe"] == "resharded-delivery-*"
    assert m["zh-batch-0"] == "zh-batch-*"
    assert m["room-RM09a04e-1779919627"] == "room-RM09a04e-*"
    assert "keep-me" not in m  # no change -> not in mapping


def test_no_capture_group_is_noop():
    # Without a capture group nothing is collapsed.
    assert build_mapping(["delivery-20260423"], [r"\d{7,}"]) == {}


def test_normalize_columns_paths_and_components():
    df = pl.DataFrame({
        "path": [
            "/vol/delivery-20260423/en-batch-0/a",
            "/vol/delivery-20260501/en-batch-1/b",
            "/vol/keep/c",
        ],
        "leaf_folder": ["en-batch-0", "en-batch-1", "keep"],
    })
    out = normalize_columns(
        df, [DIGIT_RULE, r".+-batch-(.*)"],
        path_cols=("path",), component_cols=("leaf_folder",),
    )
    paths = out["path"].to_list()
    # The two deliveries collapse to the same normalized path.
    assert paths[0] == "/vol/delivery-*/en-batch-*/a"
    assert paths[1] == "/vol/delivery-*/en-batch-*/b"
    assert paths[2] == "/vol/keep/c"
    assert out["leaf_folder"].to_list() == ["en-batch-*", "en-batch-*", "keep"]


def test_normalize_columns_noop_without_rules():
    df = pl.DataFrame({"path": ["/a/b"], "leaf_folder": ["b"]})
    assert normalize_columns(df, None, path_cols=("path",)).equals(df)
    assert normalize_columns(df, [], path_cols=("path",)).equals(df)
