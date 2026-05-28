#!/usr/bin/env bash

set -u

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/torch/venv3/pytorch/bin/python}"
START_OFFSET="${1:-0}"
END_OFFSET="${2:-168000}"
CURRENT_SAMPLE_SIZE="${3:-500}"
CURRENT_OFFSET="$START_OFFSET"
LOG_DIR="$PROJECT_ROOT/logs"
STATE_DIR="$PROJECT_ROOT/artifacts/resume_state"
LOG_FILE="$LOG_DIR/build_resume_${START_OFFSET}_${END_OFFSET}.log"
STATE_FILE="$STATE_DIR/build_resume_${START_OFFSET}_${END_OFFSET}.state"

mkdir -p "$LOG_DIR" "$STATE_DIR"

shrink_sample_size() {
  local value="$1"
  if [ "$value" -gt 250 ]; then echo 250; return; fi
  if [ "$value" -gt 125 ]; then echo 125; return; fi
  if [ "$value" -gt 60 ]; then echo 60; return; fi
  if [ "$value" -gt 30 ]; then echo 30; return; fi
  if [ "$value" -gt 15 ]; then echo 15; return; fi
  if [ "$value" -gt 8 ]; then echo 8; return; fi
  if [ "$value" -gt 4 ]; then echo 4; return; fi
  if [ "$value" -gt 2 ]; then echo 2; return; fi
  echo 1
}

echo "resume started start_offset=${START_OFFSET} end_offset=${END_OFFSET} sample_size=${CURRENT_SAMPLE_SIZE}" | tee -a "$LOG_FILE"

while [ "$CURRENT_OFFSET" -lt "$END_OFFSET" ]; do
  echo "running offset=${CURRENT_OFFSET} sample_size=${CURRENT_SAMPLE_SIZE}" | tee -a "$LOG_FILE"
  if "$PYTHON_BIN" "$PROJECT_ROOT/scripts/run_build.py" \
    --start-offset "$CURRENT_OFFSET" \
    --sample-size "$CURRENT_SAMPLE_SIZE" \
    --state-file "$STATE_FILE" >>"$LOG_FILE" 2>&1; then
    CURRENT_OFFSET="$("$PYTHON_BIN" - <<PY
import json
from pathlib import Path
state = json.loads(Path(r"$STATE_FILE").read_text(encoding="utf-8"))
print(int(state["current_offset"]))
PY
)"
    CURRENT_SAMPLE_SIZE="$("$PYTHON_BIN" - <<PY
import json
from pathlib import Path
state = json.loads(Path(r"$STATE_FILE").read_text(encoding="utf-8"))
print(int(state["current_sample_size"]))
PY
)"
    echo "success next_offset=${CURRENT_OFFSET} sample_size=${CURRENT_SAMPLE_SIZE}" | tee -a "$LOG_FILE"
    if [ "$CURRENT_OFFSET" -ge "$END_OFFSET" ]; then
      break
    fi
  else
    if [ "$CURRENT_SAMPLE_SIZE" -le 1 ]; then
      echo "failed at sample_size=1, aborting offset=${CURRENT_OFFSET}" | tee -a "$LOG_FILE"
      exit 1
    fi
    CURRENT_SAMPLE_SIZE="$(shrink_sample_size "$CURRENT_SAMPLE_SIZE")"
    echo "retry with smaller sample_size=${CURRENT_SAMPLE_SIZE} at offset=${CURRENT_OFFSET}" | tee -a "$LOG_FILE"
  fi
done

echo "resume finished current_offset=${CURRENT_OFFSET} end_offset=${END_OFFSET} sample_size=${CURRENT_SAMPLE_SIZE}" | tee -a "$LOG_FILE"
