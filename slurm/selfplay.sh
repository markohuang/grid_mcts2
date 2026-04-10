#!/bin/bash
#SBATCH --job-name=mcts_selfplay
#SBATCH --account=def-CHANGEME
#SBATCH --array=0-99              # 100 nodes; adjust as needed
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=slurm/logs/selfplay_%A_%a.out

# --- Configuration (edit these) ---
DATASET_DIR=${DATASET_DIR:-~/projects/def-CHANGEME/grid_mcts2/datasets/run01}
WEIGHTS_PATH=${WEIGHTS_PATH:-""}    # empty = random init
MAP_NUM=${MAP_NUM:-2}               # 2 = 5x5 12q
NUM_SIMS=${NUM_SIMS:-1000}
GAMES_PER_NODE=${GAMES_PER_NODE:-2000}
NUM_WORKERS=60                      # leave 4 cores for OS overhead
BATCH_SIZE=60                       # games per batch file

# --- Environment ---
module load StdEnv/2023 python/3.11
source ~/projects/def-CHANGEME/venvs/grid_mcts2/bin/activate

cd ~/grid_mcts2_prior_learning

NODE_ID="node${SLURM_ARRAY_TASK_ID}"

echo "=== Self-play worker ==="
echo "  SLURM_JOB_ID=$SLURM_JOB_ID"
echo "  SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "  hostname=$(hostname)"
echo "  DATASET_DIR=$DATASET_DIR"
echo "  WEIGHTS_PATH=$WEIGHTS_PATH"
echo "  MAP_NUM=$MAP_NUM, NUM_SIMS=$NUM_SIMS"
echo "  GAMES_PER_NODE=$GAMES_PER_NODE"

WEIGHTS_FLAG=""
if [ -n "$WEIGHTS_PATH" ]; then
    WEIGHTS_FLAG="--weights_path=$WEIGHTS_PATH"
fi

python selfplay_worker.py \
    --dataset_dir=$DATASET_DIR \
    --num_games=$GAMES_PER_NODE \
    --num_workers=$NUM_WORKERS \
    --batch_size=$BATCH_SIZE \
    --node_id=$NODE_ID \
    --notes="SLURM array job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID" \
    $WEIGHTS_FLAG \
    --config.map_num=$MAP_NUM \
    --config.mcts.num_simulations=$NUM_SIMS
