#!/usr/bin/env bash
# Extract ALL Arba Sicula issues for the Cipolla gift, one issue per process, then
# reuse the AS47 checkpoint for the training set.
#
# Each issue runs in its own cgroup scope capped at $MEM (default 12G), so an
# out-of-memory kill takes down that issue only; build_all checkpoints every finished
# issue under <out>/issues/, so re-running this script resumes from the missing ones.
#
#     bash experiments/extraction/run_gift.sh            # MEM=12G by default
#     MEM=8G bash experiments/extraction/run_gift.sh
#
# Outputs (all gitignored, PRIVATE: AS47 is still at the printer):
#   data/processed/as_full_gift/corpus.tsv    gift, one row per pair with provenance
#   data/processed/arbasicula_47/corpus.*     AS47 for the fine-tuning set
set -u
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
PY="$REPO/.venv/bin/python"
BUILD="$REPO/experiments/extraction/build_all.py"
ISSUES="$REPO/extract-text/as-issues"
GIFT="$REPO/data/processed/as_full_gift"
MEM="${MEM:-12G}"
export HF_HUB_OFFLINE=1          # LaBSE must already be in the cache
mkdir -p "$GIFT/issues"

for pdf in "$ISSUES"/*.pdf; do
    stem="$(basename "$pdf" .pdf)"
    if [ -f "$GIFT/issues/$stem.tsv" ]; then
        echo "== $stem: done, skipped"
        continue
    fi
    tmp="$(mktemp -d)"
    ln -s "$pdf" "$tmp/"
    echo "== $stem: $(date +%T), $(df -h / | awk 'NR==2 {print $4}') free on /"
    systemd-run --user --scope --quiet -p MemoryMax="$MEM" -p MemorySwapMax=0 \
        "$PY" "$BUILD" --issues "$tmp" --out "$tmp/out" >"$tmp/log" 2>&1
    rc=$?
    grep -E "cand|ERROR" "$tmp/log"
    if [ $rc -eq 0 ] && [ -f "$tmp/out/issues/$stem.tsv" ]; then
        mv "$tmp/out/issues/$stem.tsv" "$GIFT/issues/"
    else
        echo "   $stem FAILED (exit $rc, 137 = out of memory), log kept in $tmp/log"
        continue
    fi
    rm -rf "$tmp"
done

missing=$(for p in "$ISSUES"/*.pdf; do s=$(basename "$p" .pdf); [ -f "$GIFT/issues/$s.tsv" ] || echo "$s"; done)
if [ -n "$missing" ]; then
    echo "MISSING issues, re-run to resume:" $missing
    exit 1
fi

# every issue is checkpointed: this pass only assembles, it does not load LaBSE
"$PY" "$BUILD" --issues "$ISSUES" --out "$GIFT" | tee "$GIFT/build.log"

# AS47 for the training set: same checkpoint, assembled on its own
T47="$REPO/data/processed/arbasicula_47"
mkdir -p "$T47/issues"
cp "$GIFT/issues/as47.tsv" "$T47/issues/"
tmp="$(mktemp -d)"
ln -s "$ISSUES/as47.pdf" "$tmp/"
"$PY" "$BUILD" --issues "$tmp" --out "$T47" | tail -3
rm -rf "$tmp"
