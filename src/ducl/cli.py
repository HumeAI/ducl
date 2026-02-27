"""ducl CLI — one entry point for disk-usage operations."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import click


@click.group()
@click.version_option(package_name="ducl")
def cli():
    """Disk Usage Command Line toolkit."""


@cli.command()
@click.argument("path")
@click.option("-o", "--output", required=True, help="Output .feather file path")
@click.option("--s3", "use_s3", is_flag=True, help="Treat PATH as an S3 bucket name")
@click.option("--no-agg", is_flag=True, help="Skip sidecar aggregation")
@click.option("--no-dashboard", is_flag=True, help="Skip automatic dashboard build")
@click.option("--block-size", type=int, default=256 << 20, help="CSV read block size (default 256 MiB)")
@click.option("-w", "--workers", type=int, default=32, help="S3 parallel listing threads")
@click.option("--discover-depth", type=int, default=2, help="S3 prefix discovery depth")
@click.option("--endpoint-url", help="S3-compatible endpoint")
@click.option("--profile", help="AWS profile name")
@click.option("--region", help="AWS region")
def scan(path, output, use_s3, no_agg, no_dashboard, block_size, workers,
         discover_depth, endpoint_url, profile, region):
    """Scan a filesystem or S3 bucket into a Feather file."""
    do_agg = not no_agg

    if use_s3:
        from .s3scan import scan_bucket
        scan_bucket(
            bucket=path,
            output=output,
            workers=workers,
            discover_depth=discover_depth,
            do_agg=do_agg,
            endpoint_url=endpoint_url,
            profile=profile,
            region=region,
        )
    else:
        from .scan import scan_filesystem
        scan_filesystem(path, output, do_agg=do_agg, block_size=block_size)

    if not no_dashboard and os.path.exists(output):
        out_dir = Path(output).parent / (Path(output).stem + "_dashboard")
        click.echo(f"Building dashboard in {out_dir} ...", err=True)
        from .dashboard import build
        build(output, out_dir)


@cli.command()
@click.argument("feather")
@click.argument("output_dir")
def dashboard(feather, output_dir):
    """Build a dashboard from a Feather scan file."""
    from .dashboard import build
    build(feather, output_dir)


@cli.command("update")
@click.argument("dashboard_dir")
@click.argument("subtree_feather")
def update_cmd(dashboard_dir, subtree_feather):
    """Incrementally update a dashboard after a subtree rescan."""
    from .dashboard import update
    update(dashboard_dir, subtree_feather)


@cli.command()
@click.argument("feather")
@click.option("--under", help="Path prefix filter")
@click.option("--name", help="Exact match on any path component")
@click.option("--name-glob", help="Glob pattern on any path component")
@click.option("--ext", multiple=True, help="Extension filter (repeatable)")
@click.option("--min-size", help="Minimum file size (K/M/G/T suffixes)")
@click.option("--max-size", help="Maximum file size (K/M/G/T suffixes)")
@click.option("--path-contains", help="Substring match on full path")
@click.option("--dirs", is_flag=True, help="Directories only (default: files only)")
@click.option("--top", type=int, help="Show N largest entries")
@click.option("--delete", is_flag=True, help="Delete matching files from disk")
def query(feather, **kwargs):
    """Query a .feather scan file with du-style output."""
    from .query import run_query
    ext_vals = list(kwargs.pop("ext")) or None
    run_query(feather, ext=ext_vals, **kwargs)


@cli.command(
    "pwalk2",
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    ),
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def pwalk2_cmd(args):
    """Run the bundled pwalk2 binary (all arguments forwarded)."""
    binary = Path(__file__).parent / "pwalk2" / "pwalk2"
    if not binary.exists():
        click.echo(
            "Error: pwalk2 binary not found. "
            "pwalk2 requires Linux and is compiled at install time.",
            err=True,
        )
        sys.exit(1)
    os.execvp(str(binary), [str(binary)] + list(args))
