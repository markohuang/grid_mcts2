#!/bin/bash
#SBATCH --job-name=mcts_selfplay_v3b
#SBATCH --account=rrg-aspuru
#SBATCH --array=0-19               # 20 jobs x 1000 games = 20k games
#SBATCH --cpus-per-task=64
#SBATCH --mem=32G
#SBATCH --time=01:30:00            # v2f ~246s/game at 10k sims; random maps shouldn't change much
#SBATCH --output=slurm/logs/selfplay_v3b_%A_%a.out

# v3b: first random-board wave with reward_mode=layer_delta. Paired with v3a (plan_cost).
#
# Base config (post-v2 adoption):
#   FakeNet, 10k sims, pb_c_base=19652, pb_c_init=1.25,
#   dirichlet=0.1 (v2h-adopted), tau=1.0 constant, random_board=True.
#
# Only difference from v3a: reward_mode=layer_delta (instead of plan_cost).
# On fixed map 2 at this config, layer_delta gave v2f's cost median=23 vs plan_cost
# v2j's 21. v3a/v3b test whether that ordering holds on random maps or flips.
DATASET_DIR=${DATASET_DIR:-/scratch/huang651/grid_mcts2/datasets/wave03b_rnd_layerdelta}
MAP_NUM=${MAP_NUM:-2}
GAMES_PER_NODE=${GAMES_PER_NODE:-1000}
NUM_WORKERS=62
BATCH_SIZE=62

# --- Environment ---
module load StdEnv/2023
module load python/3.11 scipy-stack
cd /home/huang651/grid_mcts2
source /home/huang651/grid_mcts2/.venv/bin/activate

NODE_ID="v3b_node${SLURM_ARRAY_TASK_ID}"

echo "=== Self-play v3b (random boards, layer_delta) ==="
echo "  SLURM_JOB_ID=$SLURM_JOB_ID, ARRAY_TASK=$SLURM_ARRAY_TASK_ID, host=$(hostname)"
echo "  random_board=True, reward_mode=layer_delta, 10k sims, dirichlet=0.1"

python selfplay_worker.py \
    --preset=hpc \
    --dataset_dir=$DATASET_DIR \
    --num_games=$GAMES_PER_NODE \
    --num_workers=$NUM_WORKERS \
    --batch_size=$BATCH_SIZE \
    --node_id=$NODE_ID \
    --slurm_job_id=$SLURM_JOB_ID \
    --slurm_array_task_id=$SLURM_ARRAY_TASK_ID \
    --notes="v3b random layer_delta: slurm $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID" \
    --config.use_fake=True \
    --config.map_num=$MAP_NUM \
    --config.random_board=True \
    --config.mcts.num_simulations=10000 \
    --config.env.reward_mode=layer_delta
