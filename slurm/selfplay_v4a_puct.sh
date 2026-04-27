#!/bin/bash
#SBATCH --job-name=mcts_v4a_puct
#SBATCH --account=rrg-aspuru
#SBATCH --array=0-39               # 40 jobs x 500 games = 20k games
#SBATCH --cpus-per-task=64
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/v4a_puct_%A_%a.out

# v4a — pUCT control paired with selfplay_v4a_gumbel.sh.
# Identical config (MAPS[5] 8x8, 10k sims, plan_cost, random_board, FakeNet) except
# gumbel.enabled=False (default). Launch both in the same session for a clean A/B.
DATASET_DIR=${DATASET_DIR:-/scratch/huang651/grid_mcts2/datasets/v4a_puct}
MAP_NUM=${MAP_NUM:-5}
GAMES_PER_NODE=${GAMES_PER_NODE:-500}
NUM_WORKERS=62
BATCH_SIZE=62
NUM_SIMS=${NUM_SIMS:-10000}

# --- Environment ---
module load StdEnv/2023
module load python/3.11 scipy-stack
cd /home/huang651/grid_mcts2
source /home/huang651/grid_mcts2/.venv/bin/activate

NODE_ID="v4a_puct_node${SLURM_ARRAY_TASK_ID}"

echo "=== v4a pUCT control (8x8, 10k sims) ==="
echo "  SLURM_JOB_ID=$SLURM_JOB_ID, ARRAY_TASK=$SLURM_ARRAY_TASK_ID, host=$(hostname)"
echo "  MAP_NUM=$MAP_NUM (8x8 20qb 6gpl 5lyrs), NUM_SIMS=$NUM_SIMS, gumbel.enabled=False"
echo "  games_per_node=$GAMES_PER_NODE, workers=$NUM_WORKERS"

python selfplay_worker.py \
    --preset=hpc \
    --dataset_dir=$DATASET_DIR \
    --num_games=$GAMES_PER_NODE \
    --num_workers=$NUM_WORKERS \
    --batch_size=$BATCH_SIZE \
    --node_id=$NODE_ID \
    --slurm_job_id=$SLURM_JOB_ID \
    --slurm_array_task_id=$SLURM_ARRAY_TASK_ID \
    --notes="v4a puct: 8x8 pUCT control paired with v4a_gumbel" \
    --config.use_fake=True \
    --config.map_num=$MAP_NUM \
    --config.random_board=True \
    --config.mcts.num_simulations=$NUM_SIMS \
    --config.env.reward_mode=plan_cost
