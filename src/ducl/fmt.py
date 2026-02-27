"""Size parsing and formatting utilities."""

from __future__ import annotations


def parse_size(s: str) -> int:
    """Parse a human size string like '1G', '100M', '512K' into bytes."""
    s = s.strip()
    suffixes = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
    if s and s[-1].upper() in suffixes:
        return int(float(s[:-1]) * suffixes[s[-1].upper()])
    return int(s)


def fmt_size(n: int) -> str:
    """Format byte count as human-readable string."""
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if abs(n) < 1024:
            return f"{n:,.2f} {unit}" if unit != "B" else f"{n:,} B"
        n /= 1024
    return f"{n:,.2f} EB"
