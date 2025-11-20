#!/usr/bin/env bash
# Usage: ./run_simvla.sh <N>
# Runs i in {1,5,9,...} ≤ N  (i.e., 4k+1)
#
# Tunables (env):
#   TIME_LIMIT="10m"       # per-run timeout
#   MAX_ATTEMPTS=0         # 0 = unlimited retries per i
#   SLEEP_BETWEEN=3        # secs between retries
#   TIMEOUT_CMD=timeout    # or gtimeout on macOS
#   RESUME=0               # 1 to skip already successful runs
# Advanced (optional):
#   START=1 STEP=4         # pattern START, START+STEP, ... ≤ N

set -uo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <N>"
  exit 2
fi

N="$1"
TASK="Isaac-Kitchen-v1103-00"
SCRIPT_PATH="scripts/simvla/generate.py"
DATASET_DIR="./datasets/anubis/Isaac-Kitchen-v1103"

TIME_LIMIT="${TIME_LIMIT:-10m}"
SLEEP_BETWEEN="${SLEEP_BETWEEN:-3}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-0}"
RESUME="${RESUME:-0}"

START="${START:-2}"
STEP="${STEP:-2}"

TIMEOUT_CMD="${TIMEOUT_CMD:-timeout}"
if ! command -v "$TIMEOUT_CMD" >/dev/null 2>&1; then
  if command -v gtimeout >/dev/null 2>&1; then
    TIMEOUT_CMD="gtimeout"
  else
    echo "Error: neither 'timeout' nor 'gtimeout' found."
    exit 3
  fi
fi

mkdir -p "$DATASET_DIR" logs

run_one() {
  local I="$1"
  local DATASET_FILE="${DATASET_DIR}/${I}.hdf5"
  local attempt=1

  if [[ "$RESUME" = "1" && -s "$DATASET_FILE" ]]; then
    if grep -s -q "Recorded 1 successful demonstrations" logs/run_"${I}"_attempt*.log 2>/dev/null; then
      echo "✅ Skipping ${I}: already successful."
      return 0
    fi
  fi

  while : ; do
    local log="logs/run_${I}_attempt${attempt}.log"
    echo "=============================="
    echo "Index: ${I} | Attempt: ${attempt} | Time limit: ${TIME_LIMIT}"
    echo "Output: ${DATASET_FILE}"
    echo "Log:    ${log}"
    echo "=============================="

    rm -f "$DATASET_FILE"

    LIVESTREAM=2 \
    "$TIMEOUT_CMD" "$TIME_LIMIT" \
      ./isaaclab.sh -p "$SCRIPT_PATH" \
        --task "$TASK" \
        --enable_cameras \
        --num_envs 1 \
        --headless \
        --dataset_file "$DATASET_FILE" \
      2>&1 | tee "$log"
    local timeout_status=${PIPESTATUS[0]}

    local success_msg_found=0
    if grep -q "Recorded 1 successful demonstrations" "$log"; then
      success_msg_found=1
    fi

    if [[ $success_msg_found -eq 1 && -s "$DATASET_FILE" ]]; then
      echo "✅ Success for ${I}"
      return 0
    fi

    if [[ $timeout_status -eq 124 ]]; then
      echo "⏱️ ${I} timed out after $TIME_LIMIT; retrying…"
    else
      echo "❌ ${I} failed to record success; retrying…"
    fi

    if [[ "$MAX_ATTEMPTS" -gt 0 && "$attempt" -ge "$MAX_ATTEMPTS" ]]; then
      echo "Reached MAX_ATTEMPTS=${MAX_ATTEMPTS} at ${I}. Exiting."
      return 1
    fi

    attempt=$((attempt + 1))
    sleep "$SLEEP_BETWEEN"
  done
}

# Main loop: i = START, START+STEP, ... ≤ N  (defaults to 1,5,9,…)
for (( i=START; i<=N; i+=STEP )); do
  run_one "$i" || { echo "🚫 Aborting at index ${i}."; exit 1; }
done

echo "🎉 Completed indices {${START}, ${START}+${STEP}, …} ≤ ${N}."

