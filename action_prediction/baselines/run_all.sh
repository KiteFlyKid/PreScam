#!/usr/bin/env bash
# Run zero-shot scammer action prediction for all models × {Unlimited, Limited}.
#
# Usage (from action_prediction/):
#   bash baselines/run_all.sh
#   bash baselines/run_all.sh --dry-run      # print commands only

set -euo pipefail

MODELS=(
    "gpt-4o-mini"
    "gpt-5-chat"
    "claude-sonnet-4-5"
    "DeepSeek-V3.2"
    "Llama-3.3-70B-Instruct"
)
SETTINGS=("unlimited" "limited")

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

LOG_DIR="results/logs"
mkdir -p "$LOG_DIR"
TOTAL=0
FAILED=0

for MODEL in "${MODELS[@]}"; do
    for SETTING in "${SETTINGS[@]}"; do
        if [[ "$SETTING" == "limited" ]]; then FLAG="--limit-turns"; else FLAG="--no-limit-turns"; fi
        RUN_NAME="${MODEL}_${SETTING}"
        CMD="python baselines/run_baselines.py --model ${MODEL} ${FLAG}"
        echo "[${RUN_NAME}] ${CMD}"
        TOTAL=$((TOTAL + 1))
        if $DRY_RUN; then continue; fi
        if $CMD 2>&1 | tee "${LOG_DIR}/${RUN_NAME}.log"; then
            echo "[DONE] ${RUN_NAME}"
        else
            echo "[FAIL] ${RUN_NAME}"
            FAILED=$((FAILED + 1))
        fi
    done
done

echo "All done. Total: ${TOTAL}, Failed: ${FAILED}"
