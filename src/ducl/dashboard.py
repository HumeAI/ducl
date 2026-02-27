"""Disk-usage dashboard: build & incremental update.

Outputs Parquet (tree, cube, examples) + meta.json + dashboard.html.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

from .schema import HIST_EDGES
from .ext import normalize_ext_expr

# ---------------------------------------------------------------------------
# Constants (overridable via kwargs in build/update for testing)
# ---------------------------------------------------------------------------
MAX_TREE_LEVELS = 7
MAX_CHILDREN = 20
MIN_FOLDER_SIZE = 1 * 1024**3  # 1 GB
TOP_N_EXTENSIONS = 50
TOP_N_FOLDERS = 300
EXAMPLES_PER_BIN = 3

N_HIST_BINS = len(HIST_EDGES) + 1  # 12


def load_files(path: str | Path) -> pl.DataFrame:
    """Read Feather, keep files only, normalize ext, add leaf_folder + n_components."""
    lf = pl.scan_ipc(str(path))
    return (
        lf.filter(pl.col("child_count") == -1)
        .select("path", "ext", "size")
        .with_columns(
            (pl.col("path").str.count_matches("/") + 1).cast(pl.Int32).alias("n_components"),
            pl.col("path").str.extract(r"([^/]+)/[^/]+$").alias("leaf_folder"),
            pl.lit(1).cast(pl.Int64).alias("file_count"),
        )
        .with_columns(normalize_ext_expr())
        .collect()
    )


def load_agg(path: str | Path) -> pl.DataFrame:
    """Read .agg.parquet sidecar, rename columns for pipeline compatibility.

    Returns a DataFrame with columns: path, ext, leaf_folder, size_bin,
    n_components, file_count, size -- compatible with the existing pipeline.
    """
    df = pl.read_parquet(str(path))
    return df.rename({"dir_path": "path", "total_size": "size"})


def load_example_candidates(path: str | Path) -> pl.DataFrame:
    """Read .examples.parquet sidecar.

    Returns a DataFrame with columns: dir_path, size_bin, filename, ext, size.
    """
    return pl.read_parquet(str(path))


def detect_root(files_df: pl.DataFrame) -> tuple[str, int]:
    """Detect root path from files -- shallowest directory by slash count.

    Returns (root_path, root_n_parts).
    """
    min_nc = files_df["n_components"].min()
    # Root path = prefix with fewest components from any file
    sample_path = files_df.filter(pl.col("n_components") == min_nc)["path"][0]
    # The root is the directory portion with min_nc - 1 components
    parts = sample_path.split("/")
    root_path = "/".join(parts[: min_nc - 1])
    root_n_parts = len(root_path.split("/"))
    return root_path, root_n_parts


def prefix_regex(n: int) -> str:
    """Regex to extract first n components of '/'-split path.

    e.g. n=4 on '/a/b/c/d/e' -> '/a/b/c'
    """
    return f"^((?:/[^/]*){{{n - 1}}})"


def assign_size_bin(sizes: np.ndarray) -> np.ndarray:
    """Assign file sizes to histogram bins 0..11 using np.digitize."""
    return np.digitize(sizes, HIST_EDGES).astype(np.int8)


def pick_examples(
    sizes: np.ndarray,
    filenames: np.ndarray,
    exts: np.ndarray,
    n: int = EXAMPLES_PER_BIN,
) -> list[dict]:
    """Pick up to n example files: smallest, largest, median."""
    if len(sizes) == 0:
        return []
    sorted_idx = np.argsort(sizes)
    picks = [sorted_idx[0], sorted_idx[-1]]
    if len(sorted_idx) > 2:
        picks.append(sorted_idx[len(sorted_idx) // 2])
    seen: set[int] = set()
    examples: list[dict] = []
    for i in picks:
        ii = int(i)
        if ii not in seen and len(examples) < n:
            seen.add(ii)
            examples.append({
                "filename": str(filenames[ii]),
                "ext": str(exts[ii]),
                "size": int(sizes[ii]),
            })
    return examples


def edge_label(v: float) -> str:
    """Human-readable size label for a histogram edge value."""
    if v < 1024:
        return f"{v:g}B"
    if v < 1024**2:
        return f"{v / 1024:g}KB"
    if v < 1024**3:
        return f"{v / 1024**2:g}MB"
    return f"{v / 1024**3:g}GB"


def hist_labels() -> list[str]:
    """Compute the 12 human-readable histogram bin labels."""
    labels = ["<" + edge_label(HIST_EDGES[0])]
    for i in range(len(HIST_EDGES) - 1):
        labels.append(edge_label(HIST_EDGES[i]) + "-" + edge_label(HIST_EDGES[i + 1]))
    labels.append(">" + edge_label(HIST_EDGES[-1]))
    return labels


def compute_top_extensions(
    files_df: pl.DataFrame, top_n: int = TOP_N_EXTENSIONS
) -> tuple[pl.DataFrame, list[str]]:
    """Group by ext, return (summary_df, top_ext_list)."""
    ext_summary = (
        files_df.group_by("ext")
        .agg(
            pl.col("size").sum().alias("total_size"),
            pl.col("file_count").sum().alias("file_count"),
        )
        .sort("total_size", descending=True)
    )
    top = ext_summary.head(top_n)
    return top, top["ext"].to_list()


def compute_top_folders(
    files_df: pl.DataFrame, top_n: int = TOP_N_FOLDERS, *, path_is_dir: bool = False
) -> tuple[pl.DataFrame, list[str]]:
    """Extract all directory-name components, rank by total size."""
    if path_is_dir:
        dir_expr = pl.col("path")
    else:
        dir_expr = pl.col("path").str.replace(r"/[^/]+$", "")

    df = (
        files_df.with_columns(dir_expr.str.split("/").alias("_parts"))
        .explode("_parts")
        .filter(pl.col("_parts") != "")
    )

    folder_summary = (
        df.group_by(pl.col("_parts").alias("leaf_folder"))
        .agg(
            pl.col("size").sum().alias("total_size"),
            pl.col("file_count").sum().alias("file_count"),
        )
        .sort("total_size", descending=True)
    )
    top = folder_summary.head(top_n)
    return top, top["leaf_folder"].to_list()


# ---------------------------------------------------------------------------
# Tree building (incremental level-by-level pruning)
# ---------------------------------------------------------------------------

def build_tree(
    files_df: pl.DataFrame,
    root_path: str,
    root_n_parts: int,
    *,
    max_tree_levels: int = MAX_TREE_LEVELS,
    max_children: int = MAX_CHILDREN,
    min_folder_size: int = MIN_FOLDER_SIZE,
) -> tuple[list[dict], dict[str, int]]:
    """Incremental level-by-level pruning -> (pruned_nodes, kept_sizes).

    Each node dict has keys: id, parent, name, size.
    kept_sizes maps node_id -> size for non-__rest__ nodes.
    """
    root_size = int(files_df["size"].sum())
    pruned_nodes: list[dict] = [
        {"id": root_path, "parent": "", "name": root_path.split("/")[-1], "size": root_size}
    ]
    kept_sizes: dict[str, int] = {root_path: root_size}
    current_level_paths: list[str] = [root_path]

    for level in range(1, max_tree_levels + 1):
        if not current_level_paths:
            break

        n_parts = root_n_parts + level
        level_col = f"_fl{level}"
        parent_col = f"_fp{level}"

        # Prefix pre-filter
        if len(current_level_paths) <= 500:
            prefix_pattern = "|".join(re.escape(p) + "/" for p in current_level_paths)
            prefix_filter_expr = pl.col("path").str.contains(f"^(?:{prefix_pattern})")
        else:
            prefix_filter_expr = pl.lit(True)

        level_df = (
            files_df.lazy()
            .filter(pl.col("n_components") > n_parts)
            .filter(prefix_filter_expr)
            .with_columns(
                pl.col("path").str.extract(prefix_regex(n_parts)).alias(level_col),
                pl.col("path").str.extract(prefix_regex(n_parts - 1)).alias(parent_col),
            )
            .filter(pl.col(parent_col).is_in(current_level_paths))
        )

        # Aggregate total sizes per folder
        folder_agg = (
            level_df.group_by([level_col, parent_col])
            .agg(pl.col("size").sum().alias("total_size"))
            .collect(engine="streaming")
        )

        if folder_agg.height == 0:
            break

        # Filter out where level_col == parent_col
        folder_agg = folder_agg.filter(pl.col(level_col) != pl.col(parent_col))
        if folder_agg.height == 0:
            break

        folder_agg = folder_agg.with_columns(
            pl.col(level_col).str.extract(r"([^/]+)$").alias("name"),
        )

        # Keep top max_children per parent, above min_folder_size
        top = (
            folder_agg.filter(pl.col("total_size") >= min_folder_size)
            .with_columns(
                pl.col("total_size")
                .rank("ordinal", descending=True)
                .over(parent_col)
                .alias("_rank")
            )
            .filter(pl.col("_rank") <= max_children)
            .drop("_rank")
            .sort([parent_col, "total_size"], descending=[False, True])
        )

        # Compute __rest__ for each parent with subfolders
        parents_with_subfolders = set(folder_agg[parent_col].to_list())
        top_agg = top.group_by(parent_col).agg(pl.col("total_size").sum().alias("top_total"))
        for parent_path in current_level_paths:
            if parent_path not in parents_with_subfolders:
                continue
            parent_size = kept_sizes.get(parent_path, 0)
            row = top_agg.filter(pl.col(parent_col) == parent_path)
            top_total = int(row["top_total"][0]) if row.height > 0 else 0
            rest_size = parent_size - top_total
            if rest_size > 0:
                pruned_nodes.append({
                    "id": parent_path + "/__rest__",
                    "parent": parent_path,
                    "name": "(other)",
                    "size": rest_size,
                })

        # Add kept folders
        new_paths: list[str] = []
        for row in top.iter_rows(named=True):
            path = row[level_col]
            sz = int(row["total_size"])
            pruned_nodes.append({
                "id": path,
                "parent": row[parent_col],
                "name": row["name"],
                "size": sz,
            })
            kept_sizes[path] = sz
            new_paths.append(path)

        current_level_paths = new_paths

    return pruned_nodes, kept_sizes


# ---------------------------------------------------------------------------
# Deepest-node assignment (shared by cube + examples)
# ---------------------------------------------------------------------------

def _assign_deepest_node(
    files_df: pl.DataFrame,
    pruned_nodes: list[dict],
) -> pl.DataFrame:
    """Assign each file to its deepest matching kept (non-__rest__) node.

    Uses level-by-level prefix extraction with coalesce (deepest-first) so
    that each file is assigned to exactly one node.  This is O(files x levels)
    vectorized string ops instead of O(files x nodes) serial scans.

    Returns *files_df* with an added ``node_id`` column (Utf8, nullable).
    """
    non_rest = [n for n in pruned_nodes if "__rest__" not in n["id"]]

    # Group kept nodes by depth (number of '/'-separated parts)
    nodes_by_depth: dict[int, list[str]] = {}
    for n in non_rest:
        d = len(n["id"].split("/"))
        nodes_by_depth.setdefault(d, []).append(n["id"])

    # Sort deepest-first so coalesce picks the deepest match
    depths = sorted(nodes_by_depth.keys(), reverse=True)

    lf = files_df.lazy()

    # Step 1: extract prefix at each depth level
    prefix_cols: list[str] = []
    prefix_exprs: list[pl.Expr] = []
    for d in depths:
        col_name = f"_pfx_{d}"
        prefix_cols.append(col_name)
        prefix_exprs.append(
            pl.when(pl.col("n_components") > d)
            .then(pl.col("path").str.extract(prefix_regex(d)))
            .otherwise(pl.lit(None))
            .alias(col_name)
        )
    lf = lf.with_columns(prefix_exprs)

    # Step 2: coalesce deepest-first -- first match wins
    coalesce_args: list[pl.Expr] = []
    for d, col_name in zip(depths, prefix_cols):
        node_set = nodes_by_depth[d]
        coalesce_args.append(
            pl.when(pl.col(col_name).is_in(node_set))
            .then(pl.col(col_name))
            .otherwise(pl.lit(None))
        )

    lf = lf.with_columns(
        pl.coalesce(coalesce_args).alias("node_id")
    ).drop(prefix_cols)

    return lf.collect(engine="streaming")


# ---------------------------------------------------------------------------
# Cube aggregation
# ---------------------------------------------------------------------------

_EMPTY_CUBE_SCHEMA = {
    "node_id": pl.Utf8,
    "ext": pl.Utf8,
    "leaf_folder": pl.Utf8,
    "size_bin": pl.Int8,
    "count": pl.Int64,
    "total_size": pl.Int64,
}


def build_cube(
    files_df: pl.DataFrame,
    pruned_nodes: list[dict],
    top_ext_list: list[str],
    top_folder_list: list[str],
) -> pl.DataFrame:
    """Build the cube.parquet DataFrame (non-recursive, deepest-node).

    Each file is assigned to its deepest matching kept node.  The cube
    stores one aggregation row per (node_id, ext, leaf_folder, size_bin).

    Invariant: ``SUM(total_size)`` over *all* node_ids == root tree size
    (total conservation -- each file counted exactly once).
    """
    # Reuse pre-computed assignment if available
    if "node_id" not in files_df.columns:
        df = _assign_deepest_node(files_df, pruned_nodes)
    else:
        df = files_df

    df = df.filter(pl.col("node_id").is_not_null())

    if df.height == 0:
        return pl.DataFrame(schema=_EMPTY_CUBE_SCHEMA)

    # Normalize ext/folder to top-N (null = other bucket)
    df = df.with_columns(
        pl.when(pl.col("ext").is_in(top_ext_list))
        .then(pl.col("ext"))
        .otherwise(pl.lit(None))
        .alias("ext"),
        pl.when(pl.col("leaf_folder").is_in(top_folder_list))
        .then(pl.col("leaf_folder"))
        .otherwise(pl.lit(None))
        .alias("leaf_folder"),
    )

    # Assign size bins (skip if already present, e.g. from agg sidecar)
    if "size_bin" not in df.columns:
        sizes_np = df["size"].to_numpy().astype(float)
        bins = assign_size_bin(sizes_np)
        df = df.with_columns(pl.lit(pl.Series("size_bin", bins)).alias("size_bin"))

    # Single group-by over all files
    result = (
        df.group_by(["node_id", "ext", "leaf_folder", "size_bin"])
        .agg(
            pl.col("file_count").sum().cast(pl.Int64).alias("count"),
            pl.col("size").sum().alias("total_size"),
        )
    )

    return result.cast({"count": pl.Int64, "total_size": pl.Int64})


# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------

_EMPTY_EXAMPLES_SCHEMA = {
    "node_id": pl.Utf8,
    "size_bin": pl.Int8,
    "filename": pl.Utf8,
    "ext": pl.Utf8,
    "size": pl.Int64,
}


def build_examples(
    files_df: pl.DataFrame,
    pruned_nodes: list[dict],
    *,
    examples_per_bin: int = EXAMPLES_PER_BIN,
) -> pl.DataFrame:
    """Build the examples.parquet DataFrame (non-recursive, deepest-node).

    Each file is assigned to its deepest matching kept node (same as cube).
    For each (node_id, size_bin), pick up to *examples_per_bin* examples.
    """
    # Reuse pre-computed assignment if available
    if "node_id" not in files_df.columns:
        df = _assign_deepest_node(files_df, pruned_nodes)
    else:
        df = files_df

    df = df.filter(pl.col("node_id").is_not_null())

    if df.height == 0:
        return pl.DataFrame(schema=_EMPTY_EXAMPLES_SCHEMA)

    # Assign size bins and extract filenames
    sizes_np = df["size"].to_numpy().astype(float)
    bins = assign_size_bin(sizes_np)
    df = df.with_columns(
        pl.lit(pl.Series("size_bin", bins)).alias("size_bin"),
        pl.col("path").str.extract(r"([^/]+)$").alias("filename"),
    )

    # Pick examples per (node_id, size_bin)
    rows: list[dict] = []
    for node_df in df.partition_by("node_id"):
        nid = node_df["node_id"][0]
        sizes_arr = node_df["size"].to_numpy().astype(float)
        bins_arr = node_df["size_bin"].to_numpy()
        filenames_arr = node_df["filename"].to_numpy()
        exts_arr = node_df["ext"].to_numpy()

        for b in range(N_HIST_BINS):
            mask = bins_arr == b
            if not mask.any():
                continue
            examples = pick_examples(
                sizes_arr[mask], filenames_arr[mask], exts_arr[mask],
                n=examples_per_bin,
            )
            for ex in examples:
                rows.append({
                    "node_id": nid,
                    "size_bin": int(b),
                    "filename": ex["filename"],
                    "ext": ex["ext"],
                    "size": ex["size"],
                })

    if not rows:
        return pl.DataFrame(schema=_EMPTY_EXAMPLES_SCHEMA)

    return pl.DataFrame(rows).cast({"size_bin": pl.Int8, "size": pl.Int64})


def _build_examples_from_candidates(
    candidates: pl.DataFrame,
    pruned_nodes: list[dict],
    *,
    examples_per_bin: int = EXAMPLES_PER_BIN,
) -> pl.DataFrame:
    """Build examples from pre-computed candidate rows (from .examples.parquet sidecar).

    The candidates DataFrame has columns: dir_path, size_bin, filename, ext, size.
    We add path (= dir_path) and n_components, run _assign_deepest_node on the
    ~5M candidate rows (instead of 100M full files), then pick 3 per (node_id, size_bin).
    """
    df = candidates.with_columns(
        pl.col("dir_path").alias("path"),
        (pl.col("dir_path").str.count_matches("/") + 2).cast(pl.Int32).alias("n_components"),
    )

    df = _assign_deepest_node(df, pruned_nodes)
    df = df.filter(pl.col("node_id").is_not_null())

    if df.height == 0:
        return pl.DataFrame(schema=_EMPTY_EXAMPLES_SCHEMA)

    # Pick examples per (node_id, size_bin)
    rows: list[dict] = []
    for node_df in df.partition_by("node_id"):
        nid = node_df["node_id"][0]
        sizes_arr = node_df["size"].to_numpy().astype(float)
        bins_arr = node_df["size_bin"].to_numpy()
        filenames_arr = node_df["filename"].to_numpy()
        exts_arr = node_df["ext"].to_numpy()

        for b in range(N_HIST_BINS):
            mask = bins_arr == b
            if not mask.any():
                continue
            examples = pick_examples(
                sizes_arr[mask], filenames_arr[mask], exts_arr[mask],
                n=examples_per_bin,
            )
            for ex in examples:
                rows.append({
                    "node_id": nid,
                    "size_bin": int(b),
                    "filename": ex["filename"],
                    "ext": ex["ext"],
                    "size": ex["size"],
                })

    if not rows:
        return pl.DataFrame(schema=_EMPTY_EXAMPLES_SCHEMA)

    return pl.DataFrame(rows).cast({"size_bin": pl.Int8, "size": pl.Int64})


# ---------------------------------------------------------------------------
# Dashboard JSON export (browser-friendly)
# ---------------------------------------------------------------------------

def _export_dashboard_data(
    output_dir: Path,
    tree_df: pl.DataFrame,
    cube_df: pl.DataFrame,
    examples_df: pl.DataFrame,
    meta: dict,
) -> None:
    """Export Parquet + meta.json + dashboard.html for the browser.

    Writes dashboard.html into *output_dir* and data files into
    ``output_dir/dashboard_data/`` so the HTML can find them at its
    default relative path.  Parquet files use Snappy compression so
    hyparquet can read them with zero plugins.
    """
    data_dir = output_dir / "dashboard_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Copy dashboard.html next to the data directory
    html_src = Path(__file__).with_name("dashboard.html")
    if html_src.exists():
        shutil.copy2(html_src, output_dir / "dashboard.html")

    # Parquet files (snappy for hyparquet browser compatibility)
    tree_df.write_parquet(str(data_dir / "tree.parquet"), compression="snappy")
    cube_df.write_parquet(str(data_dir / "cube.parquet"), compression="snappy")
    examples_df.write_parquet(str(data_dir / "examples.parquet"), compression="snappy")

    with open(data_dir / "meta.json", "w") as f:
        json.dump(meta, f, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Build command
# ---------------------------------------------------------------------------

def build(
    input_file: str | Path,
    output_dir: str | Path,
    *,
    max_tree_levels: int = MAX_TREE_LEVELS,
    max_children: int = MAX_CHILDREN,
    min_folder_size: int = MIN_FOLDER_SIZE,
    top_n_extensions: int = TOP_N_EXTENSIONS,
    top_n_folders: int = TOP_N_FOLDERS,
    examples_per_bin: int = EXAMPLES_PER_BIN,
) -> None:
    """Full build: Feather -> tree.parquet + cube.parquet + examples.parquet + meta.json."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    def log(msg: str) -> None:
        print(f"[{time.time() - t0:.1f}s] {msg}", file=sys.stderr, flush=True)

    # Auto-detect agg sidecar files
    base = str(input_file).removesuffix(".feather")
    agg_path = Path(base + ".agg.parquet")
    examples_candidates_path = Path(base + ".examples.parquet")
    use_agg = agg_path.exists() and examples_candidates_path.exists()

    # 1. Load files (full or agg)
    if use_agg:
        log("Loading agg sidecar (fast path)...")
        files_df = load_agg(agg_path)
        example_candidates = load_example_candidates(examples_candidates_path)
        log(f"  {files_df.height:,} agg rows")
    else:
        log("Loading files...")
        files_df = load_files(input_file)
        example_candidates = None
        log(f"  {files_df.height:,} files")

    root_path, root_n_parts = detect_root(files_df)
    log(f"  root: {root_path}")

    # 2. Compute global top-N
    log("Computing top extensions and folders...")
    top_ext_df, top_ext_list = compute_top_extensions(files_df, top_n_extensions)
    top_folder_df, top_folder_list = compute_top_folders(files_df, top_n_folders, path_is_dir=use_agg)
    log(f"  {len(top_ext_list)} extensions, {len(top_folder_list)} folders")

    # 3. Build pruned tree
    log("Building pruned tree...")
    pruned_nodes, _kept_sizes = build_tree(
        files_df,
        root_path,
        root_n_parts,
        max_tree_levels=max_tree_levels,
        max_children=max_children,
        min_folder_size=min_folder_size,
    )
    log(f"  {len(pruned_nodes)} nodes")

    # 4. Assign files to deepest matching kept node (shared by cube + examples)
    log("Assigning files to nodes...")
    assigned_df = _assign_deepest_node(files_df, pruned_nodes)
    log(f"  {assigned_df.filter(pl.col('node_id').is_not_null()).height:,} rows assigned")

    # 5. Cube aggregation
    log("Building cube...")
    cube_df = build_cube(assigned_df, pruned_nodes, top_ext_list, top_folder_list)
    log(f"  {cube_df.height} cube rows")

    # 6. Examples
    log("Building examples...")
    if example_candidates is not None:
        examples_df = _build_examples_from_candidates(
            example_candidates, pruned_nodes, examples_per_bin=examples_per_bin,
        )
    else:
        examples_df = build_examples(assigned_df, pruned_nodes, examples_per_bin=examples_per_bin)
    log(f"  {examples_df.height} example rows")

    # 7. Write outputs
    log("Writing outputs...")

    # tree.parquet
    tree_df = pl.DataFrame([
        {"node_id": n["id"], "parent": n["parent"], "name": n["name"], "size": n["size"]}
        for n in pruned_nodes
    ]).cast({"size": pl.Int64})
    tree_df.write_parquet(str(output_dir / "tree.parquet"))

    # cube.parquet
    cube_df.write_parquet(str(output_dir / "cube.parquet"))

    # examples.parquet
    examples_df.write_parquet(str(output_dir / "examples.parquet"))

    # meta.json
    meta = {
        "hist_labels": hist_labels(),
        "extensions": top_ext_df.to_dicts(),
        "leaf_folders": top_folder_df.to_dicts(),
    }
    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Dashboard HTML + JSON (for browser)
    _export_dashboard_data(output_dir, tree_df, cube_df, examples_df, meta)

    log("Done!")


# ---------------------------------------------------------------------------
# Update command
# ---------------------------------------------------------------------------

def update(
    dashboard_dir: str | Path,
    subtree_feather: str | Path,
    *,
    examples_per_bin: int = EXAMPLES_PER_BIN,
) -> None:
    """Incrementally update an existing dashboard after a subtree rescan."""
    dashboard_dir = Path(dashboard_dir)

    t0 = time.time()

    def log(msg: str) -> None:
        print(f"[{time.time() - t0:.1f}s] {msg}", file=sys.stderr, flush=True)

    # 1. Load existing data
    log("Loading existing dashboard...")
    tree_df = pl.read_parquet(str(dashboard_dir / "tree.parquet"))
    cube_df = pl.read_parquet(str(dashboard_dir / "cube.parquet"))
    examples_df = pl.read_parquet(str(dashboard_dir / "examples.parquet"))
    with open(dashboard_dir / "meta.json") as f:
        meta = json.load(f)

    tree_by_id = {row["node_id"]: row for row in tree_df.to_dicts()}
    children_of: dict[str, list[str]] = {}
    for row in tree_df.to_dicts():
        if row["parent"]:
            children_of.setdefault(row["parent"], []).append(row["node_id"])

    top_ext_list = [e["ext"] for e in meta["extensions"]]
    top_folder_list = [f["leaf_folder"] for f in meta["leaf_folders"]]

    # 2. Load subtree files
    log("Loading subtree files...")
    sub_files_df = load_files(subtree_feather)
    log(f"  {sub_files_df.height:,} files")

    # Infer prefix (shortest path = root dir entry)
    prefix_lf = pl.scan_ipc(str(subtree_feather))
    prefix = (
        prefix_lf.select(pl.col("path").min().alias("p")).collect()["p"][0]
    )
    if prefix not in tree_by_id:
        all_paths = pl.read_ipc(str(subtree_feather), columns=["path", "child_count"])
        dirs = all_paths.filter(pl.col("child_count") >= 0)
        if dirs.height > 0:
            prefix = dirs.sort(pl.col("path").str.count_matches("/")).head(1)["path"][0]
        else:
            sample = sub_files_df["path"][0]
            parts = sample.split("/")
            for i in range(len(parts) - 1, 0, -1):
                candidate = "/".join(parts[:i])
                if candidate in tree_by_id:
                    prefix = candidate
                    break

    log(f"  Subtree prefix: {prefix}")
    if prefix not in tree_by_id:
        print(f"ERROR: prefix {prefix!r} not found in tree", file=sys.stderr)
        sys.exit(1)

    # 3. Classify nodes
    subtree_ids = {nid for nid in tree_by_id if nid == prefix or nid.startswith(prefix + "/")}
    ancestor_ids: list[str] = []
    cur = tree_by_id[prefix]["parent"]
    while cur:
        ancestor_ids.append(cur)
        cur = tree_by_id[cur]["parent"]

    log(f"  {len(subtree_ids)} subtree nodes, {len(ancestor_ids)} ancestors")

    # 4. Snapshot old state
    old_subtree_root_size = tree_by_id[prefix]["size"]
    non_rest_subtree = {nid for nid in subtree_ids if "__rest__" not in nid}

    old_subtree_cube = cube_df.filter(pl.col("node_id").is_in(list(non_rest_subtree)))

    # 5. Delete old subtree rows from cube + examples
    cube_df = cube_df.filter(~pl.col("node_id").is_in(list(non_rest_subtree)))
    examples_df = examples_df.filter(~pl.col("node_id").is_in(list(non_rest_subtree)))

    # 6. Recompute cube + examples for subtree nodes
    subtree_pruned = [
        {"id": nid, "parent": tree_by_id[nid]["parent"], "name": tree_by_id[nid]["name"],
         "size": tree_by_id[nid]["size"]}
        for nid in tree_by_id if nid in subtree_ids
    ]
    new_sub_cube = build_cube(sub_files_df, subtree_pruned, top_ext_list, top_folder_list)
    new_sub_examples = build_examples(sub_files_df, subtree_pruned, examples_per_bin=examples_per_bin)

    cube_df = pl.concat([cube_df, new_sub_cube])
    examples_df = pl.concat([examples_df, new_sub_examples])

    # 8. Update tree sizes
    for nid in sorted(non_rest_subtree, key=lambda x: x.count("/")):
        total = int(sub_files_df.filter(
            pl.col("path").str.starts_with(nid + "/")
        )["size"].sum())
        tree_by_id[nid]["size"] = total

    new_subtree_root_size = tree_by_id[prefix]["size"]
    delta = new_subtree_root_size - old_subtree_root_size

    # Propagate delta to ancestors
    for aid in ancestor_ids:
        tree_by_id[aid]["size"] += delta

    # Fix __rest__ nodes within subtree
    for nid in subtree_ids:
        if "__rest__" in nid:
            continue
        children = children_of.get(nid, [])
        rest_child_id = None
        non_rest_sum = 0
        for cid in children:
            if "__rest__" in cid:
                rest_child_id = cid
            else:
                non_rest_sum += tree_by_id[cid]["size"]
        if rest_child_id is not None:
            rest_size = tree_by_id[nid]["size"] - non_rest_sum
            tree_by_id[rest_child_id]["size"] = max(0, rest_size)

    # Also fix __rest__ for ancestors
    for aid in ancestor_ids:
        children = children_of.get(aid, [])
        rest_child_id = None
        non_rest_sum = 0
        for cid in children:
            if "__rest__" in cid:
                rest_child_id = cid
            else:
                non_rest_sum += tree_by_id[cid]["size"]
        if rest_child_id is not None:
            rest_size = tree_by_id[aid]["size"] - non_rest_sum
            tree_by_id[rest_child_id]["size"] = max(0, rest_size)

    # 9. Patch meta.json totals
    old_subtree_ext_dict: dict[str, int] = {}
    for row in old_subtree_cube.to_dicts():
        ext = row["ext"]
        if ext is not None and ext in top_ext_list:
            old_subtree_ext_dict[ext] = old_subtree_ext_dict.get(ext, 0) + int(row["total_size"])

    new_prefix_ext = (
        sub_files_df.filter(pl.col("ext").is_in(top_ext_list))
        .group_by("ext")
        .agg(pl.col("size").sum().alias("total_size"))
    )
    new_prefix_ext_dict = {r["ext"]: int(r["total_size"]) for r in new_prefix_ext.to_dicts()}

    for entry in meta["extensions"]:
        ext = entry["ext"]
        d = new_prefix_ext_dict.get(ext, 0) - old_subtree_ext_dict.get(ext, 0)
        entry["total_size"] += d

    # Recompute folder tags from full cube node_ids
    node_totals = cube_df.group_by("node_id").agg(
        pl.col("total_size").sum().alias("total_size"),
        pl.col("count").sum().alias("file_count"),
    )
    folder_agg: dict[str, list[int]] = {}  # name -> [total_size, file_count]
    for row in node_totals.to_dicts():
        for part in row["node_id"].split("/"):
            if not part:
                continue
            if part not in folder_agg:
                folder_agg[part] = [0, 0]
            folder_agg[part][0] += row["total_size"]
            folder_agg[part][1] += row["file_count"]
    sorted_tags = sorted(folder_agg.items(), key=lambda x: -x[1][0])[:TOP_N_FOLDERS]
    meta["leaf_folders"] = [
        {"leaf_folder": name, "total_size": vals[0], "file_count": vals[1]}
        for name, vals in sorted_tags
    ]

    # 10. Write updated files
    log("Writing updated files...")

    tree_out = pl.DataFrame([
        {"node_id": nid, "parent": v["parent"], "name": v["name"], "size": v["size"]}
        for nid, v in tree_by_id.items()
    ]).cast({"size": pl.Int64})
    tree_out.write_parquet(str(dashboard_dir / "tree.parquet"))

    cube_df.write_parquet(str(dashboard_dir / "cube.parquet"))
    examples_df.write_parquet(str(dashboard_dir / "examples.parquet"))

    with open(dashboard_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Dashboard HTML + JSON (for browser)
    _export_dashboard_data(dashboard_dir, tree_out, cube_df, examples_df, meta)

    log("Done!")
