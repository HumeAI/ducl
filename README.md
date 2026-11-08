# ducl — Disk Usage Command Line toolkit

Scan filesystems (or S3 buckets), build interactive dashboards, and query scan data — all from one CLI.

## Install

```bash
pip install ducl            # from PyPI
pip install ducl[s3]        # with S3 support (boto3)
pip install -e ".[dev]"     # development (editable + pytest)
```

ducl is pure Python. The filesystem walker is `bvm` from the
[quiver-tools](https://pypi.org/project/quiver-tools/) dependency, which ships prebuilt in
its manylinux wheel (Linux x86_64, glibc 2.28 or newer), so nothing is compiled at install time.

## Usage

```bash
# Scan a filesystem → Feather file + dashboard
ducl scan /mnt/data -o scan.feather

# Same, but let the walker read every directory: only the bvm walker runs
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
```

## Development

```bash
cd v3
pip install -e ".[dev]"
pytest -v
```

## License

MIT
