#!/bin/bash
# Restart the XPU runner service. Override any value via the environment.
#   XPU_RUNNER_API_KEY=      -> disable auth (empty value is respected)
REPO=${LEETGPU_CHALLENGES:-$RUNNER/../leetgpu-challenges}
RUNNER=${XPU_RUNNER_DIR:-$(cd "$(dirname "$0")/.." && pwd)}
PY=${XPU_RUNNER_PYTHON:-python3}
LOG=${XPU_RUNNER_LOG_FILE:-/tmp/xpu_service.log}

pgrep -f "[p]ython -m service" | xargs -r kill
sleep 2
cd "$RUNNER" || exit 1
setsid env XPU_RUNNER_API_KEY="${XPU_RUNNER_API_KEY-test-key}" \
  XPU_RUNNER_DEVICES="${XPU_RUNNER_DEVICES:-0}" \
  XPU_RUNNER_PORT="${XPU_RUNNER_PORT:-8080}" \
  LEETGPU_CHALLENGES="$REPO" \
  nohup "$PY" -m service > "$LOG" 2>&1 < /dev/null &
sleep 6
curl -s -m 5 "http://localhost:${XPU_RUNNER_PORT:-8080}/healthz"
echo
