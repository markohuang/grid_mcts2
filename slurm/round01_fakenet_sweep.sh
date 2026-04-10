#!/bin/bash
#SBATCH --job-name=r01_selfplay
#SBATCH --account=def-CHANGEME
#SBATCH --array=0-15             # 16 tasks: 4 reward modes x 4 nodes each
#SBATCH --cpus-per-task=64
#SBATCH --mem=32G
#SBATCH --time=1:00:00           # FakeNet is fast — 1 hour is plenty
#SBATCH --output=slurm/logs/r01_%A_%a.out

# ---- Configuration ----
DATASET_DIR=${DATASET_DIR:-~/scratch/grid_mcts2/datasets/round01_fakenet}
GAMES_PER_NODE=500              # 500 games x 16 tasks = 8000 total
NUM_WORKERS=60
BATCH_SIZE=60
NUM_SIMS=50                     # FakeNet + low sims = fast validation
MAP_NUM=2                       # 5x5, 12q

# ---- Map task ID to reward mode ----
# Tasks 0-3:  plan_cost
# Tasks 4-7:  layer_delta
# Tasks 8-11: layer_completion
# Tasks 12-15: layer_delta + search_bonus
TASK_ID=$SLURM_ARRAY_TASK_ID
GROUP=$((TASK_ID / 4))
NODE_IN_GROUP=$((TASK_ID % 4))

case $GROUP in
    0) REWARD_MODE="plan_cost"; BONUS=0.0 ;;
    1) REWARD_MODE="layer_delta"; BONUS=0.0 ;;
    2) REWARD_MODE="layer_completion"; BONUS=0.0 ;;
    3) REWARD_MODE="layer_delta"; BONUS=2.0 ;;
esac

NODE_ID="${REWARD_MODE}_node${NODE_IN_GROUP}"
if [ "$GROUP" -eq 3 ]; then
    NODE_ID="bonus_node${NODE_IN_GROUP}"
fi

# ---- Environment ----
module load StdEnv/2023 python/3.11
source ~/projects/def-CHANGEME/venvs/grid_mcts2/bin/activate
cd ~/grid_mcts2_prior_learning

echo "=== Round 01: FakeNet Reward Mode Sweep ==="
echo "  SLURM_JOB_ID=$SLURM_JOB_ID, TASK=$TASK_ID"
echo "  reward_mode=$REWARD_MODE, search_bonus=$BONUS"
echo "  node_id=$NODE_ID, hostname=$(hostname)"

python selfplay_worker.py \
    --dataset_dir=$DATASET_DIR \
    --num_games=$GAMES_PER_NODE \
    --num_workers=$NUM_WORKERS \
    --batch_size=$BATCH_SIZE \
    --node_id=$NODE_ID \
    --notes="R01 FakeNet sweep, mode=$REWARD_MODE, bonus=$BONUS" \
    --config.map_num=$MAP_NUM \
    --config.use_fake=True \
    --config.mcts.num_simulations=$NUM_SIMS \
    --config.env.reward_mode=$REWARD_MODE \
    --config.mcts.plan_cost_search_bonus_weight=$BONUS
