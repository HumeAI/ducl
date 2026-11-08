# ducl — Disk Usage Command Line toolkit

Scan filesystems (or S3 buckets), build interactive dashboards, and query scan data — all from one CLI.

## Install

```bash
pip install ducl            # from PyPI
pip install ducl[s3]        # with S3 support (boto3)
pip install -e ".[dev]"     # development (editable + pytest)
```

On Linux the bundled **pwalk2** C binary is compiled automatically at install time.
If `zig` is available (`pip install ziglang` is enough) it is cross-compiled against
glibc 2.17, so the resulting wheel runs on any distro since CentOS 7 / Ubuntu 14.04.
Without zig the host `gcc` is used and the binary requires the build host's glibc or newer.

## Usage

```bash
# Scan a filesystem → Feather file + dashboard
ducl scan /mnt/data -o scan.feather

# Same, but let the walker read every directory: only the pwalk2 child runs
# under sudo; ducl and the files it writes stay under your user
ducl scan /mnt/data -o scan.feather --sudo

# Scan an S3 bucket
ducl scan my-bucket -o bucket.feather --s3

# Build dashboard from existing Feather
ducl dashboard scan.feather ./output/

# Incremental update after rescanning a subtree
ducl update ./output/ subtree.feather

# Query scan data
ducl query scan.feather --under /mnt/data/models/ --ext wav --top 10

# Run bundled pwalk2 directly (add --sudo to elevate it)
ducl pwalk2 /mnt/data --threads 64
```

## Development

```bash
cd v3
pip install -e ".[dev]"
pytest -v
```

## License

MIT
