#!/bin/bash
#SBATCH --job-name=mcts_selfplay_v2a
#SBATCH --account=rrg-aspuru
#SBATCH --array=0-19              # 20 jobs x 1000 games = 20k games
#SBATCH --cpus-per-task=64
#SBATCH --mem=32G
#SBATCH --time=00:15:00           # fakenet has no NN forward pass -> much faster than v1
#SBATCH --output=slurm/logs/selfplay_v2a_%A_%a.out

# v2a: FakeNet (uniform policy, zero value) vs v1's random-init real network.
# Controlled comparison for "does the random-init NN add any signal over uniform?"
# Everything else matches v1: 800 sims, map 2 (5x5 12qb), 20k games.
DATASET_DIR=${DATASET_DIR:-/scratch/huang651/grid_mcts2/datasets/wave02a_fakenet}
MAP_NUM=${MAP_NUM:-2}
GAMES_PER_NODE=${GAMES_PER_NODE:-1000}
NUM_WORKERS=62
BATCH_SIZE=62

# --- Environment ---
module load StdEnv/2023
module load python/3.11 scipy-stack
cd /home/huang651/grid_mcts2
source /home/huang651/grid_mcts2/.venv/bin/activate

NODE_ID="v2a_node${SLURM_ARRAY_TASK_ID}"

echo "=== Self-play v2a (FakeNet) ==="
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
    --notes="v2a fakenet control: slurm $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID" \
    --config.use_fake=True \
    --config.map_num=$MAP_NUM
