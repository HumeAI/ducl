# ducl architecture

`ducl` ("disk usage command line") scans large storage backends and turns them
into an interactive, browser-based treemap dashboard. The design splits cleanly
across three languages, each doing what it is best at:

- **C** — the hot metadata-listing loop (`pwalk2`): walk a POSIX tree and emit
  one CSV row per file/dir as fast as the filesystem allows.
- **Python** — orchestration, format conversion, aggregation, and the dashboard
  *build* (tree pruning, cube, examples). Glue and data engineering.
- **JavaScript** (in a single static HTML file) — the *frontend*: load the
  pre-aggregated Parquet in the browser and render the treemap + filters.

Everything funnels through one canonical on-disk format (a zstd Feather file
plus two Parquet sidecars), so every source and every consumer is decoupled.

```
                      ┌─────────────── sources ───────────────┐
  POSIX filesystem ──▶│ pwalk2 (C)  ── CSV ──▶ feather_from_   │
  (local / WekaFS)    │                        csv_stream (Py) │
                      │                                        │
  S3 / S3-compatible ─│ boto3 list_objects (Py) ───────────────│──▶  <name>.feather   (Arrow IPC, zstd)
                      │                                        │     <name>.agg.parquet     (sidecar)
  Modal volumes ──────│ Modal job: pwalk2 (C) on mounted vol,  │     <name>.examples.parquet (sidecar)
                      │ returns feather bytes (Py orchestrates) │
                      └────────────────────────────────────────┘
                                          │
                                          ▼
                         dashboard.build() — Python (Polars/PyArrow)
                         detect root ▸ top-N ext/folders ▸ prune tree
                         ▸ assign files to nodes ▸ cube ▸ examples
                                          │
                                          ▼
                         dashboard_data/{tree,cube,examples}.parquet + meta.json
                                          │
                                          ▼
                         dashboard.html — JavaScript (hyparquet + Plotly)
                         loads Parquet in-browser ▸ treemap + ext/folder chips
```

---

## 1. The C core: `pwalk2`

Location: `src/ducl/pwalk2/` (`pwalk2.c`, `pw2_worker.c`, `pw2_output.c`,
`pw2_uring.h`). Compiled to a single static-ish binary at install time by the
hatch build hook (`hatch_build.py`), shipped inside the platform wheel.

**Why C:** listing tens of millions of files is bound by metadata-syscall
latency, not by per-row computation. The walker has to keep a network
filesystem's metadata path saturated, which means many `statx`/`getdents64`
calls in flight at once with minimal overhead — exactly where C + raw syscalls
beat a managed runtime.

**What it does (all in C):**

- **Threaded work queue** (`pwalk2.c`): a pool of worker threads (default 64)
  pulls directories off a shared queue; subdirectories discovered during a walk
  are pushed back on. A watchdog thread interrupts workers stuck >30 s in
  `open()`/`getdents64()` on a flaky mount.
- **Pipelined listing + stat** (`pw2_worker.c`): each worker double-buffers —
  while `io_uring` runs a batch of `statx` calls for directory *A*, it issues
  `getdents64` for directory *B*. This overlaps the two dominant sources of
  network metadata latency. Memory is bounded to ~2×ring_depth entries per
  worker.
- **io_uring `statx`** (`pw2_uring.h`): a header-only, dependency-free wrapper
  around raw `io_uring` syscalls (`IORING_OP_STATX`, kernel 5.15+). If
  `io_uring_setup` fails (e.g. under gVisor in a Modal container), each worker
  **falls back to synchronous `statx`** automatically — slower but fully
  correct.
- **CSV emission** (`pw2_output.c`): formats each entry into the canonical
  17-column CSV and writes it to stdout with its own output buffering. The C
  side also computes the per-directory rollups `pw_fcount` (direct child count)
  and `pw_dirsum` (sum of direct children's sizes), so directory rows carry real
  totals for free.

**Interface:** `pwalk2 [--threads N] [--depth N] [--one-file-system] <dir>` →
headerless CSV on stdout, 17 columns:
`inode, parent_inode, depth, "path", "ext", uid, gid, size, dev, blocks, nlink,
"mode", atime, mtime, ctime, pw_fcount, pw_dirsum`.

C does **no** aggregation, extension normalization, tree building, or I/O to the
final format — it is purely "directory tree → CSV rows, as fast as possible".

---

## 2. The Python layer

All under `src/ducl/`. Python owns everything except the raw walk and the
browser rendering. Core libraries: **PyArrow** (Arrow IPC / Parquet),
**Polars** (all dataframe work), **NumPy** (histogram binning), **Click** (CLI),
**boto3** (S3), **modal** (Modal jobs).

### Canonical schema — `schema.py`
The single source of truth: the 17-column Arrow `SCHEMA`, the matching
`COLUMN_NAMES`, and the histogram bin edges (`HIST_EDGES`, 12 size buckets).
Convention used everywhere downstream: `child_count == -1` marks a file; `>= 0`
marks a directory (whose `dir_total_size` holds the byte total of its direct
children).

### Sources → canonical feather
Three scanners, all producing the *identical* schema so the rest of the system
doesn't care where data came from:

- **Filesystem** — `scan.py`. Spawns the `pwalk2` C binary and pipes its CSV
  stdout into `feather_from_csv_stream()`, which streams PyArrow `RecordBatch`es
  straight to a zstd-compressed Arrow IPC ("Feather") file. As batches flow by
  it also builds the aggregation sidecars (see below). This is the
  **C→Python boundary**: C produces CSV bytes, Python parses/writes/aggregates.
- **S3 / S3-compatible** — `s3scan.py`. No C at all: a multiprocessing pool of
  Python workers paginates `list_objects_v2` across discovered prefixes,
  converts object listings into the same `RecordBatch` shape, and synthesizes
  directory rows (since S3 has no real directories) with a `DirectoryTracker`.
  `scan_all_buckets()` loops every bucket and concatenates, prefixing each path
  with `/<bucket>`.
- **Modal volumes** — `modaljob.py`. Fans out **one Modal job per volume**; each
  job mounts its volume and runs `ducl scan` (i.e. the **C** `pwalk2`) on the
  POSIX mount, then returns the compact feather bytes directly to the parent.
  Python on the parent side strips the mount prefix and concatenates per-volume
  results into one capture. So here C runs *inside* Modal containers and Python
  orchestrates the fan-out and merge. (See `MEMORY` / module docstring for the
  Ubuntu-24.04 + glibc and cloudpickle-by-value details.)

### Aggregation sidecars — `agg.py`
For each scan, Python pre-aggregates while streaming so dashboards build in
milliseconds instead of re-reading the whole feather:

- `<name>.agg.parquet` — grouped by `(dir_path, ext, leaf_folder, size_bin,
  n_components)` with `file_count` + `total_size`.
- `<name>.examples.parquet` — the top-3 largest files per `(dir_path,
  size_bin)`, for the detail panel.

`ext.py` normalizes extensions here (lowercasing, compound `.tar.gz`, empty →
`(no ext)`). `agg.py` is also where memory is bounded via periodic `compact_*`
re-aggregation.

### Dashboard build — `dashboard.py`
The analytical heart, all Polars. `build()`:
1. Load the agg sidecar (fast path) or the full feather (slow path).
2. *(optional)* **Collapse folder names** — `foldernorm.py` applies
   user-supplied regexes whose capture groups become `*` (e.g.
   `delivery-(.*)` → `delivery-*`), folding date-stamped / numbered siblings
   into shared groups before aggregation.
3. `detect_root` — longest common path prefix.
4. Top-N extensions and folders.
5. **Prune the tree** level-by-level: keep the top `MAX_CHILDREN` folders per
   parent above `MIN_FOLDER_SIZE`, bucket the rest into a synthetic `(other)`
   (`__rest__`) node. Bounds the tree to a browsable size.
6. Assign every file to its single deepest kept node (vectorized prefix match).
7. Build the **cube**: one row per `(node_id, ext, leaf_folder, size_bin)` —
   conservation-safe (Σ cube == root size).
8. Emit `dashboard_data/{tree,cube,examples}.parquet` + `meta.json`, and copy
   the static `dashboard.html`.

`diff.py` does the same for two captures (outer-joined tree with deltas →
`diff_dashboard.html`). `query.py` is a `du`-like CLI over a feather. `update`
incrementally rebuilds one subtree.

### CLI — `cli.py`
Click entry point: `scan` (`--s3` / `--modal`, with no PATH = "all
buckets/volumes"; `--collapse REGEX` repeatable), `dashboard`, `diff`, `update`,
`query`, and a passthrough `pwalk2`.

---

## 3. The JavaScript frontend: `dashboard.html`

A single self-contained static file (no build step), copied next to the data on
every build. Opened over HTTP or `file://`.

**Why JS/browser:** the heavy aggregation already happened in Python, so the
frontend only loads small pre-cut Parquet and renders. Doing it client-side
means the dashboard is just static files — trivially served, shareable, no
backend.

**What it does (all in-browser JS):**
- Loads `tree/cube/examples.parquet` with **hyparquet** (Parquet reader, no
  server) and `meta.json`, with per-load cache-busting on http(s).
- Computes recursive node sizes and the extension/folder **chip** totals from
  the cube, honoring active include/exclude filters.
- Renders the treemap and histograms with **Plotly**; clicking chips/nodes
  re-filters live.

No data leaves the browser; there is no server-side component at render time.

---

## 4. Auxiliary scripts

- `scan-and-diff.sh` (bash) — the production cron-style driver: `srun` a WekaFS
  scan, diff against the previous capture, update `latest_*` symlinks.
- `prune-old-dashboards.py` (Python) — logarithmic retention: keep recent
  captures densely and older ones sparsely; dry-run by default, protects the
  newest and any `latest_*` target.

---

## Language boundary, at a glance

| Concern                                   | Language | Where |
|-------------------------------------------|----------|-------|
| Walk a POSIX tree, `statx`/`getdents64`   | **C**    | `pwalk2/` |
| Per-dir child count + size rollup         | **C**    | `pw2_output.c` |
| CSV → Feather, streaming write            | Python   | `scan.py` |
| S3 listing + synthetic dirs               | Python   | `s3scan.py` |
| Modal fan-out / mount / merge (runs C)    | Python   | `modaljob.py` |
| Extension normalization, size binning     | Python   | `ext.py`, `agg.py` |
| Pre-aggregation sidecars                   | Python   | `agg.py` |
| Tree pruning, cube, examples, folder collapse | Python | `dashboard.py`, `foldernorm.py` |
| Diff, query, incremental update            | Python   | `diff.py`, `query.py` |
| Load Parquet in-browser, treemap, filters  | JS       | `dashboard.html` |

**One-line summary:** C lists the filesystem as fast as the hardware allows and
hands Python raw CSV; Python converts, aggregates, and carves the data into a
compact pre-cut cube; the browser just loads that cube and draws it.
