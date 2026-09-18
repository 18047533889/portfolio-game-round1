#!/usr/bin/env bash
# Fetch the COS US-stock panel needed for the round-1 factor study.
# Resumable: already-downloaded files are skipped. Safe to re-run.
set -uo pipefail

BIN="$HOME/quantsociety/bin/clean-cos-ro"
ROOT="$HOME/pg"
BARS="$ROOT/cache/bars"
IDX="$ROOT/cache/idx"
mkdir -p "$BARS" "$IDX"

COSROOT="cos://qs-cold/clean_data/us_stock/massive_data"

if [ ! -s "$ROOT/bars.list" ]; then
  "$BIN" ls "$COSROOT/StockDailyBar/" 2>/dev/null \
    | grep -oE '[0-9]{4}-[0-9]{2}-[0-9]{2}\.parquet' | sort -u > "$ROOT/bars.list"
fi
if [ ! -s "$ROOT/idx.list" ]; then
  "$BIN" ls "$COSROOT/StockIndicesComponents/" 2>/dev/null \
    | grep -oE '[0-9]{4}-[0-9]{2}-[0-9]{2}\.parquet' | sort -u > "$ROOT/idx.list"
fi

fetch() {
  local kind="$1" file="$2" dest="$ROOT/cache/$1"
  [ -s "$dest/$file" ] && return 0
  "$BIN" cp "$COSROOT/$3/$file" "$dest/$file" >/dev/null 2>&1 || rm -f "$dest/$file"
}

export -f fetch
export BIN ROOT COSROOT

echo "bars planned: $(wc -l < "$ROOT/bars.list")  idx planned: $(wc -l < "$ROOT/idx.list")"

# indices first (cheap, small, unblocks the universe definition)
cat "$ROOT/idx.list" | xargs -P 12 -I{} bash -c 'fetch idx "$@" StockIndicesComponents' _ {}

# daily bars (about 5.8 GB total)
cat "$ROOT/bars.list" | xargs -P 12 -I{} bash -c 'fetch bars "$@" StockDailyBar' _ {}

echo "bars on disk: $(ls "$BARS" | wc -l)  idx on disk: $(ls "$IDX" | wc -l)"
du -sh "$ROOT/cache"
echo FETCH_DONE
