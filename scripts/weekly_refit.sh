#!/usr/bin/env bash
# Sunday refit of the direction models (cron: Sunday 20:00 Singapore = Sunday morning ET).
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BIN="${PREMARKET_CONDA_BIN:-$HOME/miniconda3/bin/conda}"
mkdir -p "$PROJECT_DIR/logs"
cd "$PROJECT_DIR"
echo "[$(date '+%F %T %Z')] weekly refit start" >> logs/refit_cron.log
status=0
"$CONDA_BIN" run -n "${PREMARKET_CONDA_ENV:-premarket}" python jobs/weekly_refit.py >> logs/refit_cron.log 2>&1 || status=$?
echo "[$(date '+%F %T %Z')] weekly refit exit $status" >> logs/refit_cron.log
exit $status
