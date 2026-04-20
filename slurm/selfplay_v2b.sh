#!/bin/bash
#SBATCH --job-name=mcts_selfplay_v2b
#SBATCH --account=rrg-aspuru
#SBATCH --array=0-19              # 20 jobs x 1000 games = 20k games
#SBATCH --cpus-per-task=64
#SBATCH --mem=32G
#SBATCH --time=01:30:00           # 2000 sims = 2.5x v1 runtime (~40 min) + buffer
#SBATCH --output=slurm/logs/selfplay_v2b_%A_%a.out

# v2b: scale num_simulations 800 -> 2000. pb_c_base=500 now engages more:
#   N_parent=2000 -> pb_c = 1.25 + log(2501/500) = 1.25 + 1.61 = 2.86 (+129% over init)
# Tests whether deeper search improves discovered solutions on the same map.
# Everything else matches v1: random-init real network, map 2 (5x5 12qb), 20k games.
DATASET_DIR=${DATASET_DIR:-/project/rrg-aspuru/huang651/grid_mcts2/datasets/wave02b_2000sims}
MAP_NUM=${MAP_NUM:-2}
GAMES_PER_NODE=${GAMES_PER_NODE:-1000}
NUM_WORKERS=62
BATCH_SIZE=62

# --- Environment ---
module load StdEnv/2023
module load python/3.11 scipy-stack
cd /home/huang651/grid_mcts2
source /home/huang651/grid_mcts2/.venv/bin/activate

NODE_ID="v2b_node${SLURM_ARRAY_TASK_ID}"

echo "=== Self-play v2b (2000 sims) ==="
echo "  SLURM_JOB_ID=$SLURM_JOB_ID, ARRAY_TASK=$SLURM_ARRAY_TASK_ID, host=$(hostname)"
echo "  DATASET_DIR=$DATASET_DIR"
echo "  MAP_NUM=$MAP_NUM, GAMES_PER_NODE=$GAMES_PER_NODE"

python selfplay_worker.py \
    --preset=hpc \
    --dataset_dir=$DATASET_DIR \
    --num_games=$GAMES_PER_NODE \
    --num_workers=$NUM_WORKERS \
    --batch_size=$BATCH_SIZE \
    --node_id=$NODE_ID \
    --slurm_job_id=$SLURM_JOB_ID \
    --slurm_array_task_id=$SLURM_ARRAY_TASK_ID \
    --notes="v2b 2000sims: slurm $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID" \
    --config.map_num=$MAP_NUM \
    --config.mcts.num_simulations=2000
