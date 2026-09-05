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

if [[ -n "${2:-}" ]]; then
    OLD_BASE="mnt-weka-${2}"
else
    OLD_BASE=$(ls -1 mnt-weka-*.feather 2>/dev/null \
        | grep -v "^${NEW_BASE}\.feather$" \
        | sed 's/\.feather$//' \
        | sort -r | head -n1)
fi

if [[ -z "${OLD_BASE}" || ! -f "${OLD_BASE}.feather" ]]; then
    echo "No previous capture found to diff against" >&2
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
if [[ ! -f "${NEW_BASE}.feather" ]]; then
#    srun --partition=gpu -G 4 --job-name interactive --pty 
        sudo "$DUCL" scan -o "${NEW_BASE}.feather" /mnt/weka/
else
    echo ">>> ${NEW_BASE}.feather already exists, skipping scan"
fi

# 2. Diff dashboard.
"$DUCL" diff -o "${NEW_BASE}_dashboard_change/" \
    "${OLD_BASE}.feather" "${NEW_BASE}.feather"

# 3. Update "latest" symlinks.
ln -sfn "${NEW_BASE}_dashboard"        latest_dashboard
ln -sfn "${NEW_BASE}_dashboard_change" latest_dashboard_change

echo "Done."
