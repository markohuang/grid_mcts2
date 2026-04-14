#!/bin/bash
# Launch 2x2 scaling diagnostic: (α × sims) with MCTS sampling fix.
# Chains runs on each GPU: small sim run first, then large sim run.
set -e

# GPU 0: E0 then E1
(
  bash experiments/run_r12_scaling_diag.sh E0 0.03 50 0
  bash experiments/run_r12_scaling_diag.sh E1 0.03 250 0
) > experiments/r12_scaling_GPU0.log 2>&1 &
PID0=$!
echo "GPU 0 chain (E0 → E1): PID $PID0"

# GPU 1: E2 then E3
(
  bash experiments/run_r12_scaling_diag.sh E2 0.3 50 1
  bash experiments/run_r12_scaling_diag.sh E3 0.3 250 1
) > experiments/r12_scaling_GPU1.log 2>&1 &
PID1=$!
echo "GPU 1 chain (E2 → E3): PID $PID1"

echo ""
echo "Monitor: tail -f experiments/r12_scaling_GPU{0,1}.log"
