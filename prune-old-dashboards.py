#!/usr/bin/env python3
"""Prune old disk-usage captures with logarithmic retention.

Captures live in DATA_DIR as ``<prefix>-YYYY-MM-DD.feather`` (plus the
``.agg.parquet`` / ``.examples.parquet`` sidecars and the ``_dashboard`` /
``_dashboard_change`` directories).  This thins them out over time: everything
recent is kept, and older captures are kept ever more sparsely on a logarithmic
schedule -- one capture per exponentially growing age bucket.

Retention rule (per prefix group, newest first):
  * keep every capture younger than --keep-all-days;
  * for older captures, keep the newest one in each age bucket
    ``floor(log_base(age_days))`` and drop the rest;
  * always keep the single newest capture, and never touch a capture that a
    ``latest_*`` symlink points at.

By default only the dashboard directories are pruned (they rebuild from the
feather); pass --include-data to also delete the feather + sidecars (where the
real disk space is).

Dry-run by default -- it prints the plan and the space it would free.  Pass
--apply to actually delete.

Examples:
  ./prune-old-dashboards.py                 # show plan for all prefixes
  ./prune-old-dashboards.py --include-data  # plan incl. feathers/sidecars
  ./prune-old-dashboards.py --apply --include-data
  ./prune-old-dashboards.py --prefix mnt-weka --base 2 --keep-all-days 14
"""

from __future__ import annotations

import argparse
import math
import os
import re
import shutil
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

DATA_DIR = "/mnt/weka/logs/disk-usage"
CAPTURE_RE = re.compile(r"^(?P<prefix>.+)-(?P<date>\d{4}-\d{2}-\d{2})\.feather$")

# Per-capture artifacts, relative to a "<prefix>-<date>" base.
DATA_SUFFIXES = (".feather", ".agg.parquet", ".examples.parquet")
DASHBOARD_SUFFIXES = ("_dashboard", "_dashboard_change")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Logarithmically prune old disk-usage dashboards/captures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", default=DATA_DIR, help="Directory holding captures")
    p.add_argument("--prefix", action="append", default=None,
                   help="Only consider this prefix (repeatable); default: all")
    p.add_argument("--base", type=float, default=2.0,
                   help="Logarithm base for age buckets (larger = more aggressive)")
    p.add_argument("--keep-all-days", type=int, default=14,
                   help="Keep every capture at least this recent")
    p.add_argument("--include-data", action="store_true",
                   help="Also delete the feather + sidecars (not just dashboards)")
    p.add_argument("--apply", action="store_true",
                   help="Actually delete (default is a dry run)")
    return p.parse_args(argv)


def discover(data_dir: Path) -> dict[str, list[date]]:
    """Map prefix -> sorted (desc) list of capture dates found in *data_dir*."""
    groups: dict[str, list[date]] = defaultdict(list)
    for entry in data_dir.iterdir():
        m = CAPTURE_RE.match(entry.name)
        if not m:
            continue
        y, mo, d = (int(x) for x in m["date"].split("-"))
        groups[m["prefix"]].append(date(y, mo, d))
    for prefix in groups:
        groups[prefix].sort(reverse=True)
    return groups


def keep_dates(dates: list[date], today: date, base: float, keep_all_days: int) -> set[date]:
    """Return the subset of *dates* to KEEP under logarithmic retention."""
    keep: set[date] = set()
    seen_buckets: set[int] = set()
    for d in sorted(dates, reverse=True):
        age = (today - d).days
        if age <= keep_all_days:
            keep.add(d)
            continue
        bucket = int(math.log(max(age, 1), base))
        if bucket not in seen_buckets:
            seen_buckets.add(bucket)
            keep.add(d)
    if dates:
        keep.add(max(dates))  # always keep the newest, regardless of age
    return keep


def symlink_protected(data_dir: Path) -> set[Path]:
    """Resolve latest_* symlinks so their targets are never pruned."""
    protected: set[Path] = set()
    for link in data_dir.glob("latest_*"):
        if link.is_symlink():
            target = (data_dir / os.readlink(link)).resolve()
            protected.add(target)
    return protected


def artifact_paths(data_dir: Path, base: str, include_data: bool) -> list[Path]:
    """Existing artifact paths for a "<prefix>-<date>" base."""
    suffixes = DASHBOARD_SUFFIXES + (DATA_SUFFIXES if include_data else ())
    paths = [data_dir / f"{base}{s}" for s in suffixes]
    return [p for p in paths if p.exists() or p.is_symlink()]


def path_size(p: Path) -> int:
    if p.is_symlink() or p.is_file():
        try:
            return p.stat(follow_symlinks=False).st_size
        except OSError:
            return 0
    total = 0
    for root, _dirs, files in os.walk(p):
        for f in files:
            fp = Path(root) / f
            try:
                total += fp.stat(follow_symlinks=False).st_size
            except OSError:
                pass
    return total


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if abs(n) < 1024 or unit == "PB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n:.1f} PB"


def main(argv=None) -> int:
    args = parse_args(argv)
    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        print(f"error: {data_dir} is not a directory", file=sys.stderr)
        return 2

    today = date.today()
    groups = discover(data_dir)
    if args.prefix:
        groups = {k: v for k, v in groups.items() if k in set(args.prefix)}
    if not groups:
        print("No captures found.", file=sys.stderr)
        return 0

    protected = symlink_protected(data_dir)
    total_freed = 0
    to_delete: list[Path] = []

    for prefix in sorted(groups):
        dates = groups[prefix]
        keep = keep_dates(dates, today, args.base, args.keep_all_days)
        print(f"\n=== {prefix}  ({len(dates)} captures, keeping {len(keep)}) ===")
        for d in dates:
            base = f"{prefix}-{d.isoformat()}"
            age = (today - d).days
            kept = d in keep
            # Protect captures referenced by a latest_* symlink.
            arts = artifact_paths(data_dir, base, args.include_data)
            is_protected = any(a.resolve() in protected for a in arts)
            if is_protected and not kept:
                kept = True
                note = " (kept: latest_* symlink)"
            else:
                note = ""
            if kept:
                print(f"  KEEP   {base}  ({age}d){note}")
                continue
            sz = sum(path_size(a) for a in arts)
            total_freed += sz
            to_delete.extend(arts)
            print(f"  PRUNE  {base}  ({age}d)  -{human(sz)}")
            for a in arts:
                print(f"           - {a.name}")

    scope = "dashboards + data" if args.include_data else "dashboards only"
    print(f"\nScope: {scope}.  Would free {human(total_freed)} "
          f"across {len(to_delete)} path(s).")

    if not args.apply:
        print("Dry run -- nothing deleted.  Re-run with --apply to delete.")
        return 0

    if not to_delete:
        print("Nothing to delete.")
        return 0

    print("\nDeleting...")
    for p in to_delete:
        try:
            if p.is_symlink() or p.is_file():
                p.unlink()
            else:
                shutil.rmtree(p)
            print(f"  removed {p.name}")
        except OSError as exc:
            print(f"  failed to remove {p.name}: {exc}", file=sys.stderr)
    print(f"Done.  Freed ~{human(total_freed)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
