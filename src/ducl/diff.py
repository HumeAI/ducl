"""Disk-usage diff: compare two filesystem captures.

Produces a diff-specific dashboard showing where disk usage grew or shrank.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import polars as pl

from .dashboard import (
    MAX_TREE_LEVELS,
    MAX_CHILDREN,
    MIN_FOLDER_SIZE,
    TOP_N_EXTENSIONS,
    TOP_N_FOLDERS,
    EXAMPLES_PER_BIN,
    load_files,
    load_agg,
    load_example_candidates,
    detect_root,
    compute_top_extensions,
    compute_top_folders,
    build_tree,
    _assign_deepest_node,
    build_cube,
    build_examples,
    _build_examples_from_candidates,
    hist_labels,
)


def _load_capture(input_file: str | Path) -> tuple[pl.DataFrame, pl.DataFrame | None]:
    """Load a capture, auto-detecting agg sidecars.

    Returns (files_df, example_candidates_or_None).
    """
    base = str(input_file).removesuffix(".feather")
    agg_path = Path(base + ".agg.parquet")
    examples_path = Path(base + ".examples.parquet")
    use_agg = agg_path.exists() and examples_path.exists()

    if use_agg:
        files_df = load_agg(agg_path)
        example_candidates = load_example_candidates(examples_path)
    else:
        files_df = load_files(input_file)
        example_candidates = None

    return files_df, example_candidates


def _merge_top_lists(
    list_a: list[str],
    list_b: list[str],
    summary_a: pl.DataFrame,
    summary_b: pl.DataFrame,
    key_col: str,
    size_col: str,
    top_n: int,
) -> list[str]:
    """Merge two top-N lists by union, re-sort by combined total, cap at top_n."""
    union = set(list_a) | set(list_b)

    # Build combined size from both summaries
    combined: dict[str, int] = {}
    for df in (summary_a, summary_b):
        for row in df.to_dicts():
            k = row[key_col]
            if k in union:
                combined[k] = combined.get(k, 0) + row[size_col]

    sorted_list = sorted(combined.keys(), key=lambda k: combined[k], reverse=True)
    return sorted_list[:top_n]


def _merge_trees(
    old_nodes: list[dict],
    new_nodes: list[dict],
    old_kept: dict[str, int],
    new_kept: dict[str, int],
) -> list[dict]:
    """Merge two pruned trees via outer join on node_id.

    Returns list of dicts with: node_id, parent, name, old_size, new_size, delta, pct_change.
    """
    # Index both trees
    old_by_id = {n["id"]: n for n in old_nodes}
    new_by_id = {n["id"]: n for n in new_nodes}

    # Collect all non-__rest__ node_ids from both trees
    all_real_ids = set()
    for n in old_nodes:
        if "__rest__" not in n["id"]:
            all_real_ids.add(n["id"])
    for n in new_nodes:
        if "__rest__" not in n["id"]:
            all_real_ids.add(n["id"])

    # Build merged nodes (non-rest first)
    merged: list[dict] = []
    # Track children per parent for __rest__ computation
    children_of: dict[str, list[str]] = {}

    for nid in sorted(all_real_ids, key=lambda x: x.count("/")):
        old_n = old_by_id.get(nid)
        new_n = new_by_id.get(nid)

        # Parent/name from whichever tree has the node (prefer new)
        if new_n:
            parent = new_n["parent"]
            name = new_n["name"]
        else:
            parent = old_n["parent"]
            name = old_n["name"]

        old_size = old_kept.get(nid, 0) if old_n else 0
        new_size = new_kept.get(nid, 0) if new_n else 0
        # For root or nodes without kept_sizes entry, use the node dict size
        if old_n and nid not in old_kept:
            old_size = old_n["size"]
        if new_n and nid not in new_kept:
            new_size = new_n["size"]

        delta = new_size - old_size

        if old_size == 0:
            pct = float("inf") if new_size > 0 else 0.0
        else:
            pct = delta / old_size * 100

        merged.append({
            "node_id": nid,
            "parent": parent,
            "name": name,
            "old_size": old_size,
            "new_size": new_size,
            "delta": delta,
            "pct_change": pct,
        })

        if parent:
            children_of.setdefault(parent, []).append(nid)

    # Recompute __rest__ per parent
    merged_lookup = {m["node_id"]: m for m in merged}
    for parent_id, child_ids in children_of.items():
        if parent_id not in merged_lookup:
            continue
        parent_node = merged_lookup[parent_id]
        sum_children_old = sum(merged_lookup[c]["old_size"] for c in child_ids if c in merged_lookup)
        sum_children_new = sum(merged_lookup[c]["new_size"] for c in child_ids if c in merged_lookup)
        rest_old = max(0, parent_node["old_size"] - sum_children_old)
        rest_new = max(0, parent_node["new_size"] - sum_children_new)
        rest_delta = rest_new - rest_old
        rest_pct = (rest_delta / rest_old * 100) if rest_old != 0 else (float("inf") if rest_new > 0 else 0.0)

        if rest_old > 0 or rest_new > 0:
            merged.append({
                "node_id": parent_id + "/__rest__",
                "parent": parent_id,
                "name": "(other)",
                "old_size": rest_old,
                "new_size": rest_new,
                "delta": rest_delta,
                "pct_change": rest_pct,
            })

    return merged


_NULL_SENTINEL = "__NULL__"


def _merge_cubes(
    old_cube: pl.DataFrame,
    new_cube: pl.DataFrame,
) -> pl.DataFrame:
    """Full outer join two cubes, compute delta columns."""
    join_keys = ["node_id", "ext", "leaf_folder", "size_bin"]
    nullable_keys = ["ext", "leaf_folder"]

    # Replace nulls with sentinel so join matches null==null
    def fill_sentinels(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            pl.col(c).fill_null(_NULL_SENTINEL) for c in nullable_keys
        )

    def restore_nulls(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            pl.when(pl.col(c) == _NULL_SENTINEL).then(pl.lit(None)).otherwise(pl.col(c)).alias(c)
            for c in nullable_keys
        )

    old_renamed = fill_sentinels(old_cube).rename({
        "count": "old_count",
        "total_size": "old_total_size",
    })
    new_renamed = fill_sentinels(new_cube).rename({
        "count": "new_count",
        "total_size": "new_total_size",
    })

    # Full outer join -- use join with how="full" and coalesce keys
    merged = old_renamed.join(
        new_renamed,
        on=join_keys,
        how="full",
        coalesce=True,
    )

    # Fill nulls with 0 and compute deltas
    merged = merged.with_columns(
        pl.col("old_count").fill_null(0),
        pl.col("new_count").fill_null(0),
        pl.col("old_total_size").fill_null(0),
        pl.col("new_total_size").fill_null(0),
    ).with_columns(
        (pl.col("new_count") - pl.col("old_count")).alias("delta_count"),
        (pl.col("new_total_size") - pl.col("old_total_size")).alias("delta_total_size"),
    )

    # Restore null sentinels back to actual nulls
    merged = restore_nulls(merged)

    return merged.select([
        "node_id", "ext", "leaf_folder", "size_bin",
        "old_count", "new_count", "delta_count",
        "old_total_size", "new_total_size", "delta_total_size",
    ])


def _build_diff_meta(
    old_ext_df: pl.DataFrame,
    new_ext_df: pl.DataFrame,
    old_folder_df: pl.DataFrame,
    new_folder_df: pl.DataFrame,
    old_total: int,
    new_total: int,
) -> dict:
    """Build diff meta.json with extension/folder deltas and summary."""
    # Extension outer join
    old_ext = old_ext_df.rename({
        "total_size": "old_total_size",
        "file_count": "old_file_count",
    })
    new_ext = new_ext_df.rename({
        "total_size": "new_total_size",
        "file_count": "new_file_count",
    })
    ext_merged = old_ext.join(new_ext, on="ext", how="full", coalesce=True).with_columns(
        pl.col("old_total_size").fill_null(0),
        pl.col("new_total_size").fill_null(0),
        pl.col("old_file_count").fill_null(0),
        pl.col("new_file_count").fill_null(0),
    ).with_columns(
        (pl.col("new_total_size") - pl.col("old_total_size")).alias("delta_total_size"),
        (pl.col("new_file_count") - pl.col("old_file_count")).alias("delta_file_count"),
    ).sort("delta_total_size", descending=True)

    # Folder outer join
    old_fld = old_folder_df.rename({
        "total_size": "old_total_size",
        "file_count": "old_file_count",
    })
    new_fld = new_folder_df.rename({
        "total_size": "new_total_size",
        "file_count": "new_file_count",
    })
    fld_merged = old_fld.join(new_fld, on="leaf_folder", how="full", coalesce=True).with_columns(
        pl.col("old_total_size").fill_null(0),
        pl.col("new_total_size").fill_null(0),
        pl.col("old_file_count").fill_null(0),
        pl.col("new_file_count").fill_null(0),
    ).with_columns(
        (pl.col("new_total_size") - pl.col("old_total_size")).alias("delta_total_size"),
        (pl.col("new_file_count") - pl.col("old_file_count")).alias("delta_file_count"),
    ).sort("delta_total_size", descending=True)

    return {
        "hist_labels": hist_labels(),
        "extensions": ext_merged.to_dicts(),
        "leaf_folders": fld_merged.to_dicts(),
        "summary": {
            "old_total": old_total,
            "new_total": new_total,
            "delta": new_total - old_total,
        },
    }


def diff(
    old_file: str | Path,
    new_file: str | Path,
    output_dir: str | Path,
    *,
    max_tree_levels: int = MAX_TREE_LEVELS,
    max_children: int = MAX_CHILDREN,
    min_folder_size: int = MIN_FOLDER_SIZE,
    top_n_extensions: int = TOP_N_EXTENSIONS,
    top_n_folders: int = TOP_N_FOLDERS,
    examples_per_bin: int = EXAMPLES_PER_BIN,
) -> None:
    """Compare two captures and build a diff dashboard."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    def log(msg: str) -> None:
        print(f"[{time.time() - t0:.1f}s] {msg}", file=sys.stderr, flush=True)

    # 1. Load both captures
    log("Loading old capture...")
    old_df, old_candidates = _load_capture(old_file)
    use_agg_old = old_candidates is not None
    log(f"  {old_df.height:,} rows (agg={use_agg_old})")

    log("Loading new capture...")
    new_df, new_candidates = _load_capture(new_file)
    use_agg_new = new_candidates is not None
    log(f"  {new_df.height:,} rows (agg={use_agg_new})")

    # 2. Validate roots. detect_root infers the root as the longest common
    # path prefix, so a capture that happens to cover a single top-level
    # folder (e.g. an S3 key that saw one bucket) detects a deeper root than
    # a capture of the same namespace with more folders. When one root is an
    # ancestor of the other, build both trees from the ancestor: the subtree
    # the narrower capture never saw then shows up as added/removed.
    old_root, old_n_parts = detect_root(old_df)
    new_root, new_n_parts = detect_root(new_df)
    if old_root != new_root:
        if old_root.startswith(new_root + "/"):
            old_root, old_n_parts = new_root, new_n_parts
        elif new_root.startswith(old_root + "/"):
            new_root, new_n_parts = old_root, old_n_parts
        else:
            raise ValueError(
                f"Root mismatch: old={old_root!r}, new={new_root!r}. "
                f"Cannot diff captures with different root paths."
            )
    log(f"  root: {old_root!r}")

    # 3. Compute union top-N extensions and folders
    log("Computing union top extensions and folders...")
    old_ext_df, old_ext_list = compute_top_extensions(old_df, top_n_extensions)
    new_ext_df, new_ext_list = compute_top_extensions(new_df, top_n_extensions)
    merged_ext_list = _merge_top_lists(
        old_ext_list, new_ext_list,
        old_ext_df, new_ext_df,
        "ext", "total_size", top_n_extensions,
    )

    old_folder_df, old_folder_list = compute_top_folders(
        old_df, top_n_folders, path_is_dir=use_agg_old,
    )
    new_folder_df, new_folder_list = compute_top_folders(
        new_df, top_n_folders, path_is_dir=use_agg_new,
    )
    merged_folder_list = _merge_top_lists(
        old_folder_list, new_folder_list,
        old_folder_df, new_folder_df,
        "leaf_folder", "total_size", top_n_folders,
    )
    log(f"  {len(merged_ext_list)} extensions, {len(merged_folder_list)} folders")

    # 4. Build pruned trees independently
    log("Building pruned trees...")
    old_nodes, old_kept = build_tree(
        old_df, old_root, old_n_parts,
        max_tree_levels=max_tree_levels,
        max_children=max_children,
        min_folder_size=min_folder_size,
    )
    new_nodes, new_kept = build_tree(
        new_df, new_root, new_n_parts,
        max_tree_levels=max_tree_levels,
        max_children=max_children,
        min_folder_size=min_folder_size,
    )
    log(f"  old: {len(old_nodes)} nodes, new: {len(new_nodes)} nodes")

    # 5. Merge trees via outer join
    log("Merging trees...")
    merged_tree = _merge_trees(old_nodes, new_nodes, old_kept, new_kept)
    log(f"  {len(merged_tree)} merged nodes")

    # 6. Build cubes against merged node list
    # We need to assign files using the union of nodes from both trees
    # so both cubes share the same node set
    log("Building cubes...")
    # Collect all non-rest nodes from the merged tree for assignment
    merged_non_rest = [
        {"id": n["node_id"], "parent": n["parent"], "name": n["name"],
         "size": n["old_size"]}
        for n in merged_tree if "__rest__" not in n["node_id"]
    ]

    old_cube = build_cube(old_df, merged_non_rest, merged_ext_list, merged_folder_list)
    new_cube = build_cube(new_df, merged_non_rest, merged_ext_list, merged_folder_list)
    log(f"  old: {old_cube.height} cube rows, new: {new_cube.height} cube rows")

    # 7. Merge cubes
    log("Merging cubes...")
    diff_cube = _merge_cubes(old_cube, new_cube)
    log(f"  {diff_cube.height} merged cube rows")

    # 8. Build examples from new capture only (current state)
    log("Building examples...")
    # Use the merged node list for consistent node assignment
    if new_candidates is not None:
        examples_df = _build_examples_from_candidates(
            new_candidates, merged_non_rest, examples_per_bin=examples_per_bin,
        )
    else:
        examples_df = build_examples(
            new_df, merged_non_rest, examples_per_bin=examples_per_bin,
        )
    log(f"  {examples_df.height} example rows")

    # 9. Build diff meta
    log("Building diff meta...")
    old_total = int(old_df["size"].sum())
    new_total = int(new_df["size"].sum())
    meta = _build_diff_meta(
        old_ext_df, new_ext_df,
        old_folder_df, new_folder_df,
        old_total, new_total,
    )

    # 10. Export — single copy of data in dashboard_data/, HTML alongside
    log("Writing outputs...")

    tree_df = pl.DataFrame(merged_tree).cast({
        "old_size": pl.Int64,
        "new_size": pl.Int64,
        "delta": pl.Int64,
        "pct_change": pl.Float64,
    })

    data_dir = output_dir / "dashboard_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    tree_df.write_parquet(str(data_dir / "tree.parquet"), compression="snappy")
    diff_cube.write_parquet(str(data_dir / "cube.parquet"), compression="snappy")
    examples_df.write_parquet(str(data_dir / "examples.parquet"), compression="snappy")

    with open(data_dir / "meta.json", "w") as f:
        json.dump(meta, f, separators=(",", ":"))

    html_src = Path(__file__).with_name("diff_dashboard.html")
    if html_src.exists():
        shutil.copy2(html_src, output_dir / "dashboard.html")

    log("Done!")
