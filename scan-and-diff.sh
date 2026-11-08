#!/usr/bin/env bash
# Scan /mnt/weka, diff against the previous capture, and update "latest" symlinks.
#
# Usage:
#   scan-and-diff.sh [NEW_DATE] [OLD_DATE]
#
# NEW_DATE defaults to today (YYYY-MM-DD).
# OLD_DATE defaults to the most recent existing mnt-weka-*.feather (excluding NEW).
set -euo pipefail

DATA_DIR="/mnt/weka/logs/disk-usage"
NEW_DATE="${1:-$(date +%Y-%m-%d)}"
NEW_BASE="mnt-weka-${NEW_DATE}"

cd "$DATA_DIR"

# A complete Arrow IPC file ends with the footer + "ARROW1" magic. ducl now writes
# to .feather.tmp and renames on success, but older scans that died mid-write (OOM,
# preemption) left date-named .feather files with no footer, which polars rejects
# with "InvalidFooter" -- so a bare -f test is not enough.
is_complete_feather() {
    [[ -s "$1" ]] && [[ "$(tail -c 6 "$1")" == "ARROW1" ]]
}

if [[ -n "${2:-}" ]]; then
    OLD_BASE="mnt-weka-${2}"
    if ! is_complete_feather "${OLD_BASE}.feather"; then
        echo "${OLD_BASE}.feather is missing or truncated (no Arrow footer)" >&2
        exit 1
    fi
else
    # Strict date-only names: backups like mnt-weka-DATE-presparse.feather
    # must not be picked up as the previous capture. Skip truncated ones.
    OLD_BASE=""
    for f in $(ls -1 mnt-weka-*.feather 2>/dev/null \
        | grep -E '^mnt-weka-[0-9]{4}-[0-9]{2}-[0-9]{2}\.feather$' \
        | grep -v "^${NEW_BASE}\.feather$" \
        | sort -r); do
        if is_complete_feather "$f"; then
            OLD_BASE="${f%.feather}"
            break
        fi
        echo ">>> skipping truncated capture $f" >&2
    done
fi

if [[ -z "${OLD_BASE}" ]]; then
    echo "No complete previous capture found to diff against" >&2
    exit 1
fi

echo ">>> NEW: ${NEW_BASE}"
echo ">>> OLD: ${OLD_BASE}"

DUCL="$(command -v ducl)"
if [[ -z "$DUCL" ]]; then
    echo "ducl not on PATH" >&2
    exit 1
fi

# 1. Scan (writes .feather, .agg.parquet, .examples.parquet, and _dashboard/).
if ! is_complete_feather "${NEW_BASE}.feather"; then
    if [[ -e "${NEW_BASE}.feather" ]]; then
        echo ">>> ${NEW_BASE}.feather is truncated, moving aside and rescanning"
        mv "${NEW_BASE}.feather" "${NEW_BASE}-truncated.feather"
    fi
    srun --partition=cpu -c 24 --job-name interactive --pty \
        "$DUCL" scan --sudo -o "${NEW_BASE}.feather" /mnt/weka/
else
    echo ">>> ${NEW_BASE}.feather already exists, skipping scan"
fi

# 2. Diff dashboard.
srun -p cpu -c 24 \
  "$DUCL" diff -o "${NEW_BASE}_dashboard_change/" \
      "${OLD_BASE}.feather" "${NEW_BASE}.feather"

# 3. Update "latest" symlinks.
ln -sfn "${NEW_BASE}_dashboard"        latest_dashboard
ln -sfn "${NEW_BASE}_dashboard_change" latest_dashboard_change

echo "Done."
