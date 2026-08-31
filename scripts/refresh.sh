#!/bin/bash
# Local FPL pipeline refresh. Runs entirely on this machine at no API cost.
#
#   ./scripts/refresh.sh quick   prices + brief. Seconds. Safe to run often.
#   ./scripts/refresh.sh full    also re-scrapes Understat and the FPL archive.
#                                Minutes, because Understat is rate limited to
#                                roughly one request per second. Weekly is enough;
#                                shot data for matches already played never changes.
set -euo pipefail

cd "$(dirname "$0")/.."
MODE="${1:-quick}"
mkdir -p logs
LOG="logs/refresh-$(date +%Y%m%d).log"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG"; }

run_step() {
  local label="$1" module="$2"
  if uv run python -m "$module" >>"$LOG" 2>&1; then
    log "$label ok"
  else
    log "$label FAILED (exit $?)"
    return 1
  fi
}

log "refresh started (mode=$MODE)"

# Prices first and unconditionally. It is the only step whose data cannot be
# reconstructed after the fact, so a later failure must not prevent it.
run_step prices fplopt.prices.track || true

if [ "$MODE" = "full" ]; then
  run_step data fplopt.data.build || true
fi

run_step brief fplopt.brief || true

log "refresh finished (mode=$MODE)"
