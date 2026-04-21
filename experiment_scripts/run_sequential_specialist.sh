#!/bin/bash
# Round 10B: Sequential Specialist Training
# Train on one random map at a time, loading from the previous checkpoint.
# Tests whether the model builds transferable features across maps.
#
# Key metrics to extract afterward:
#   - Starting avg_cost per phase (epoch 1): does it decrease over phases?
#   - Epochs to convergence per phase: does it speed up?
#   - eval_map0_best_so_far: does map2 performance degrade (catastrophic forgetting)?

set -e

PYTHON="../grid_mcts2/.venv/bin/python"
BASE_DIR="outputs"

# Starting checkpoint: 10A specialist converged to cost=11 on map2
CKPT="outputs/f91ba571/checkpoints/final.ckpt"

# Number of sequential phases (1 map per phase)
NUM_PHASES=20
EPOCHS_PER_PHASE=30
GAMES_PER_EPOCH=50

# Shared config
COMMON_FLAGS="
  --config.map_num=2
  --config.random_board=True
  --config.training.epochs=${EPOCHS_PER_PHASE}
  --config.training.num_selfplay=${GAMES_PER_EPOCH}
  --config.training.batch_size=128
  --config.training.seed=42
  --config.mcts.root_dirichlet_alpha=0.3
  --config.training.data_augmentation=True
  --config.env.reward_mode=layer_delta
  --config.mcts.plan_cost_search_bonus_weight=2.0
  --config.experiment.early_stopping_patience=15
  --config.experiment.study=round10_sequential_specialist
"

echo "=== Sequential Specialist Training ==="
echo "Starting checkpoint: $CKPT"
echo "Phases: $NUM_PHASES, epochs/phase: $EPOCHS_PER_PHASE, games/epoch: $GAMES_PER_EPOCH"
echo ""

# Track all run IDs for analysis
RUN_IDS=""

for phase in $(seq 1 $NUM_PHASES); do
    MAP_SEED=$((phase * 100))
    echo "--- Phase $phase / $NUM_PHASES (map_seed=$MAP_SEED, checkpoint=$CKPT) ---"

    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    $PYTHON main.py \
      $COMMON_FLAGS \
      --config.random_board_seed=$MAP_SEED \
      --config.experiment.load_checkpoint=$CKPT \
      --config.experiment.hypothesis=sequential_phase${phase} \
      --config.experiment.variant=phase${phase}_seed${MAP_SEED} \
      --config.experiment.parent_run=$(basename $(dirname $(dirname $CKPT))) \
      --config.experiment.tags=sequential,specialist,phase${phase} \
      --config.experiment.notes="Sequential phase $phase: map_seed=$MAP_SEED, from $(basename $(dirname $(dirname $CKPT)))"

    # Find the latest run directory
    LATEST_RUN=$(ls -td $BASE_DIR/*/ | head -1)
    RUN_ID=$(basename $LATEST_RUN)
    RUN_IDS="$RUN_IDS $RUN_ID"

    # Update checkpoint for next phase
    CKPT="${LATEST_RUN}checkpoints/final.ckpt"

    # Print phase summary
    echo "Phase $phase complete: run=$RUN_ID"
    tail -1 "${LATEST_RUN}metrics.jsonl" | python3 -c "
import json, sys
d = json.loads(sys.stdin.readline())
print(f'  Final: ep{d[\"epoch\"]}, avg={d[\"avg_cost\"]}, best={d[\"best_cost\"]}, eval0_bf={d.get(\"eval_map0_best_so_far\")}, ent_l0={d.get(\"entropy_layer0\",0):.2f}')
"
    echo ""
done

echo "=== All phases complete ==="
echo "Run IDs: $RUN_IDS"
echo ""
echo "Analysis: check starting cost, convergence speed, and map2 retention across phases:"
echo "  python3 -c \""
echo "import json, os"
echo "for rid in '$RUN_IDS'.split():"
echo "    metrics = [json.loads(l) for l in open(f'outputs/{rid}/metrics.jsonl')]"
echo "    manifest = json.load(open(f'outputs/{rid}/manifest.json'))"
echo "    phase = manifest.get('variant', '?')"
echo "    ep1 = metrics[0] if metrics else {}"
echo "    last = metrics[-1] if metrics else {}"
echo "    print(f'{phase}: start_avg={ep1.get(\"avg_cost\")}, final_avg={last.get(\"avg_cost\")}, final_best={last.get(\"best_cost\")}, epochs={last.get(\"epoch\")}, eval0_bf={last.get(\"eval_map0_best_so_far\")}')"
echo "\""
