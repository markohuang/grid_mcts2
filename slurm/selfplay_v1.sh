#!/bin/bash
#SBATCH --job-name=mcts_selfplay
#SBATCH --account=rrg-aspuru
#SBATCH --array=0-19              # 20 jobs x 1000 games = 20k games per wave
#SBATCH --cpus-per-task=64
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/selfplay_%A_%a.out

# --- Configuration (edit these) ---
DATASET_DIR=${DATASET_DIR:-/scratch/huang651/grid_mcts2/datasets/wave01}
WEIGHTS_PATH=${WEIGHTS_PATH:-""}    # empty = random init / fakenet per preset
MAP_NUM=${MAP_NUM:-2}               # 2 = 5x5 12qb (see MAPS in config.py)
GAMES_PER_NODE=${GAMES_PER_NODE:-1000}
NUM_WORKERS=62                      # leave 2 cores for driver/OS overhead
BATCH_SIZE=62                       # games per batch file (one per parallel wave)
PRESET=${PRESET:-hpc}               # hpc=800 sims; override with PRESET=default for smoke

# --- Environment ---
module load StdEnv/2023
module load python/3.11 scipy-stack
cd /home/huang651/grid_mcts2
source /home/huang651/grid_mcts2/.venv/bin/activate

NODE_ID="node${SLURM_ARRAY_TASK_ID}"

echo "=== Self-play worker ==="
echo "  SLURM_JOB_ID=$SLURM_JOB_ID"
echo "  SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "  hostname=$(hostname)"
echo "  PRESET=$PRESET"
echo "  DATASET_DIR=$DATASET_DIR"
echo "  WEIGHTS_PATH=$WEIGHTS_PATH"
echo "  MAP_NUM=$MAP_NUM"
echo "  GAMES_PER_NODE=$GAMES_PER_NODE"

WEIGHTS_FLAG=""
if [ -n "$WEIGHTS_PATH" ]; then
    WEIGHTS_FLAG="--weights_path=$WEIGHTS_PATH"
fi

python selfplay_worker.py \
    --preset=$PRESET \
    --dataset_dir=$DATASET_DIR \
    --num_games=$GAMES_PER_NODE \
    --num_workers=$NUM_WORKERS \
    --batch_size=$BATCH_SIZE \
    --node_id=$NODE_ID \
    --slurm_job_id=$SLURM_JOB_ID \
    --slurm_array_task_id=$SLURM_ARRAY_TASK_ID \
    --notes="SLURM array job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID preset=$PRESET" \
    $WEIGHTS_FLAG \
    --config.map_num=$MAP_NUM
