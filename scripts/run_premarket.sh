#!/usr/bin/env bash
# Unattended morning run. Fires from cron at 20:30 and 21:30 Singapore time on
# weekdays (08:30 ET in summer and winter); premarket.run --scheduled decides
# whether to run (NYSE session, pre-market window, once per session).
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BIN="${PREMARKET_CONDA_BIN:-$HOME/miniconda3/bin/conda}"
CONDA_ENV="${PREMARKET_CONDA_ENV:-premarket}"
LOG_FILE="$PROJECT_DIR/logs/cron.log"
LOCK_DIR="$PROJECT_DIR/logs/.run.lock"

mkdir -p "$PROJECT_DIR/logs"
cd "$PROJECT_DIR"

# One run at a time (mkdir is atomic).
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "[$(date '+%F %T')] another run holds $LOCK_DIR; exiting" >> "$LOG_FILE"
  exit 0
fi
trap 'rmdir "$LOCK_DIR"' EXIT

if [ -f ".env" ]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

{
  echo "============================================================"
  echo "[$(date '+%F %T %Z')] premarket run start"
} >> "$LOG_FILE"

status=0
"$CONDA_BIN" run -n "$CONDA_ENV" python -m premarket.run --scheduled "$@" >> "$LOG_FILE" 2>&1 || status=$?
echo "[$(date '+%F %T %Z')] premarket run exit $status" >> "$LOG_FILE"
exit $status
