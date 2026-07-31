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
@click.argument("path", required=False)
@click.option("-o", "--output", required=True, help="Output .feather file path")
@click.option("--s3", "use_s3", is_flag=True, help="Scan S3 (PATH = one bucket name, or omit for all buckets)")
@click.option("--modal", "use_modal", is_flag=True, help="Scan Modal volumes via pwalk2 jobs (PATH = one volume name, or omit for all)")
@click.option("--no-agg", is_flag=True, help="Skip sidecar aggregation")
@click.option("--no-dashboard", is_flag=True, help="Skip automatic dashboard build")
@click.option("--block-size", type=int, default=256 << 20, help="CSV read block size (default 256 MiB)")
@click.option("--scanner", type=click.Choice(["pwalk2", "quiver"]), default="pwalk2",
              help="Filesystem walker: bundled pwalk2 (CSV) or quiver's bvm (Arrow direct)")
@click.option("--bvm", "bvm_exe", help="quiver bvm binary (default: quiver's own)")
@click.option("-w", "--workers", type=int, default=None, help="S3 listing threads (default 32) / Modal concurrent streams (default 8)")
@click.option("--discover-depth", type=int, default=2, help="S3 prefix discovery depth")
@click.option("--endpoint-url", help="S3-compatible endpoint")
@click.option("--profile", help="AWS profile name")
@click.option("--region", help="AWS region")
@click.option("--environment", help="Modal environment name")
@click.option("--max-objects", type=int, default=0, help="Stop after N objects (benchmarking)")
@click.option("--collapse", multiple=True, metavar="REGEX",
              help="Collapse folder names: a regex whose capture groups are replaced "
                   "with '*' (the rest is kept). Repeatable. "
                   r"e.g. --collapse '(\d{7,})' --collapse '.+-batch-(.*)'")
def scan(path, output, use_s3, use_modal,
         no_agg, no_dashboard, block_size, scanner, bvm_exe, workers,
         discover_depth, endpoint_url, profile, region, environment, max_objects, collapse):
    """Scan a filesystem, S3 bucket, or Modal volume into a Feather file."""
    do_agg = not no_agg

    if use_s3 and use_modal:
        raise click.UsageError("--s3 and --modal are mutually exclusive")
    if not (use_modal or use_s3) and not path:
        raise click.UsageError(
            "PATH is required (omit it only with --modal/--s3 to scan all volumes/buckets)"
        )

    if use_s3:
        s3_workers = workers if workers is not None else 32
        if path:
            from .s3scan import scan_bucket
            scan_bucket(
                bucket=path,
                output=output,
                workers=s3_workers,
                discover_depth=discover_depth,
                do_agg=do_agg,
                endpoint_url=endpoint_url,
                profile=profile,
                region=region,
                max_objects=max_objects,
            )
        else:
            from .s3scan import scan_all_buckets
            scan_all_buckets(
                output=output,
                workers=s3_workers,
                discover_depth=discover_depth,
                do_agg=do_agg,
                endpoint_url=endpoint_url,
                profile=profile,
                region=region,
                max_objects=max_objects,
            )
    elif use_modal:
        if path:
            from .modaljob import scan_volume_job
            scan_volume_job(
                volume=path,
                output=output,
                environment=environment,
                do_agg=do_agg,
            )
        else:
            from .modaljob import scan_all_volumes_jobs
            scan_all_volumes_jobs(
                output=output,
                environment=environment,
                do_agg=do_agg,
            )
    elif scanner == "quiver":
        # quiver's bvm emits Arrow STAT batches straight off its walker pool, so this skips
        # pwalk2's CSV -> pyarrow.csv hop entirely. Same table: validated column-for-column
        # against pwalk2 on three trees (see quiverscan.to_ducl_frame).
        from .quiverscan import scan_filesystem_quiver
        scan_filesystem_quiver(path, output, do_agg=do_agg,
                               nworkers=workers or 128, bvm_exe=bvm_exe)
    else:
        from .scan import scan_filesystem
        scan_filesystem(path, output, do_agg=do_agg, block_size=block_size)

    if not no_dashboard and os.path.exists(output):
        out_dir = Path(output).parent / (Path(output).stem + "_dashboard")
        click.echo(f"Building dashboard in {out_dir} ...", err=True)
        from .dashboard import build
        build(output, out_dir, collapse_rules=list(collapse) or None)


@cli.command()
@click.argument("feather")
@click.argument("output_dir")
@click.option("--collapse", multiple=True, metavar="REGEX",
              help="Collapse folder names: a regex whose capture groups are replaced "
                   "with '*' (the rest is kept). Repeatable. "
                   r"e.g. --collapse '(\d{7,})' --collapse '.+-batch-(.*)'")
def dashboard(feather, output_dir, collapse):
    """Build a dashboard from a Feather scan file."""
    from .dashboard import build
    build(feather, output_dir, collapse_rules=list(collapse) or None)


@cli.command("update")
@click.argument("dashboard_dir")
@click.argument("subtree_feather")
def update_cmd(dashboard_dir, subtree_feather):
    """Incrementally update a dashboard after a subtree rescan."""
    from .dashboard import update
    update(dashboard_dir, subtree_feather)


@cli.command()
@click.argument("old_feather")
@click.argument("new_feather")
@click.option("-o", "--output", required=True, help="Output directory for diff dashboard")
def diff(old_feather, new_feather, output):
    """Compare two scan captures and build a diff dashboard."""
    from .diff import diff as diff_build
    diff_build(old_feather, new_feather, output)


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
