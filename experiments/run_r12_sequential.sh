#!/bin/bash
# Round 12: Sequential Specialist v2
# Usage: ./run_r12_sequential.sh <reward_mode> <search_bonus> <extra_flags> <run_label> <start_ckpt> [gpu_id]
# Example: ./run_r12_sequential.sh layer_delta 2.0 "" 12A outputs/XXXX/checkpoints/final.ckpt 0

set -e
PYTHON="../grid_mcts2/.venv/bin/python"
REWARD_MODE=${1:-layer_delta}
SEARCH_BONUS=${2:-0.0}
EXTRA_FLAGS=${3:-""}
RUN_LABEL=${4:-12X}
CKPT=${5:-""}
GPU_ID=${6:-0}

if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
    echo "ERROR: Must provide valid starting checkpoint as 5th argument (got: $CKPT)"
    exit 1
fi

NUM_PHASES=20
EPOCHS_PER_PHASE=30
GAMES_PER_EPOCH=50

COMMON="
  --config.map_num=2
  --config.random_board=True
  --config.training.epochs=${EPOCHS_PER_PHASE}
  --config.training.num_selfplay=${GAMES_PER_EPOCH}
  --config.training.batch_size=128
  --config.training.seed=42
  --config.mcts.num_simulations=250
  --config.mcts.root_dirichlet_alpha=0.3
  --config.training.data_augmentation=True
  --config.env.reward_mode=${REWARD_MODE}
  --config.mcts.plan_cost_search_bonus_weight=${SEARCH_BONUS}
  --config.network.p_hsize=64
  --config.experiment.early_stopping_patience=15
  --config.experiment.study=round12_sequential_v2
  ${EXTRA_FLAGS}
"

LOGFILE="experiments/r12_${RUN_LABEL}.log"
echo "=== R12 Sequential: ${RUN_LABEL} (${REWARD_MODE}, bonus=${SEARCH_BONUS}, GPU=${GPU_ID}) ===" | tee "$LOGFILE"
echo "Starting checkpoint: $CKPT" | tee -a "$LOGFILE"
echo "Phases: $NUM_PHASES, epochs/phase: $EPOCHS_PER_PHASE, games/epoch: $GAMES_PER_EPOCH, sims: 250" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"

RUN_IDS=""

for phase in $(seq 1 $NUM_PHASES); do
    MAP_SEED=$((phase * 100))
    echo "--- Phase $phase/$NUM_PHASES (seed=$MAP_SEED) ---" | tee -a "$LOGFILE"

    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
    CUDA_VISIBLE_DEVICES=$GPU_ID \
    $PYTHON main.py \
      $COMMON \
      --config.random_board_seed=$MAP_SEED \
      --config.experiment.load_checkpoint=$CKPT \
      --config.experiment.hypothesis=${RUN_LABEL}_phase${phase} \
      --config.experiment.variant=phase${phase}_seed${MAP_SEED} \
      --config.experiment.parent_run=$(basename $(dirname $(dirname $CKPT))) \
      --config.experiment.tags=sequential,specialist,${RUN_LABEL},phase${phase} \
      --config.experiment.notes="R12 ${RUN_LABEL} phase $phase: ${REWARD_MODE}, bonus=${SEARCH_BONUS}, 250sims, p64, seed=$MAP_SEED"

    LATEST_RUN=$(ls -td outputs/*/ | head -1)
    RUN_ID=$(basename $LATEST_RUN)
    RUN_IDS="$RUN_IDS $RUN_ID"
    CKPT="${LATEST_RUN}checkpoints/final.ckpt"

    # Log phase summary
    SUMMARY=$(tail -1 "${LATEST_RUN}metrics.jsonl" | python3 -c "
import json, sys
d = json.loads(sys.stdin.readline())
print(f'  run={\"$RUN_ID\"}, ep{d[\"epoch\"]}, avg={d[\"avg_cost\"]:.1f}, best={d[\"best_cost\"]}, eval0={d.get(\"eval_map0_best_so_far\",\"?\")}, ent_l0={d.get(\"entropy_layer0\",0):.2f}, depth={d.get(\"avg_mcts_depth\",0):.1f}')
")
    echo "$SUMMARY" | tee -a "$LOGFILE"
    echo "" | tee -a "$LOGFILE"
done

echo "=== ${RUN_LABEL} complete ===" | tee -a "$LOGFILE"
echo "Run IDs: $RUN_IDS" | tee -a "$LOGFILE"
echo "Final checkpoint: $CKPT" | tee -a "$LOGFILE"
