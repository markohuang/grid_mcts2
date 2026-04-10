#!/bin/bash
# Launch two R12 sequential runs in parallel (one per GPU).
# Usage: ./launch_r12_sequential_pair.sh <base_ckpt> <labelA> <modeA> <bonusA> <extraA> <labelB> <modeB> <bonusB> <extraB>
# Example:
#   ./launch_r12_sequential_pair.sh outputs/XXXX/checkpoints/final.ckpt \
#     12A layer_delta 2.0 "" \
#     12B plan_cost 0.0 ""

set -e
BASE_CKPT=$1
LABEL_A=$2; MODE_A=$3; BONUS_A=$4; EXTRA_A=$5
LABEL_B=$6; MODE_B=$7; BONUS_B=$8; EXTRA_B=$9

if [ -z "$BASE_CKPT" ] || [ ! -f "$BASE_CKPT" ]; then
    echo "ERROR: Provide valid base checkpoint as first argument"
    exit 1
fi

echo "=== Launching R12 pair ==="
echo "  GPU0: ${LABEL_A} (${MODE_A}, bonus=${BONUS_A})"
echo "  GPU1: ${LABEL_B} (${MODE_B}, bonus=${BONUS_B})"
echo "  Base checkpoint: $BASE_CKPT"
echo ""

# Launch A on GPU 0
nohup bash experiments/run_r12_sequential.sh "$MODE_A" "$BONUS_A" "$EXTRA_A" "$LABEL_A" "$BASE_CKPT" 0 \
  > experiments/r12_${LABEL_A}.log 2>&1 &
PID_A=$!
echo "Started ${LABEL_A} on GPU 0 (PID $PID_A)"

# Launch B on GPU 1
nohup bash experiments/run_r12_sequential.sh "$MODE_B" "$BONUS_B" "$EXTRA_B" "$LABEL_B" "$BASE_CKPT" 1 \
  > experiments/r12_${LABEL_B}.log 2>&1 &
PID_B=$!
echo "Started ${LABEL_B} on GPU 1 (PID $PID_B)"

echo ""
echo "Monitor with: bash experiments/monitor_r12.sh"
echo "Or check logs: tail -f experiments/r12_${LABEL_A}.log experiments/r12_${LABEL_B}.log"
