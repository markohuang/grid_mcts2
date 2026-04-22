#!/bin/bash
#SBATCH --job-name=mcts_selfplay_smoke
#SBATCH --account=rrg-aspuru
#SBATCH --array=0-4               # 5 jobs is enough to exercise multi-job parquet writes
#SBATCH --cpus-per-task=64
#SBATCH --mem=32G                 # smoke uses less RAM than production
#SBATCH --time=00:40:00           # ~30 min target + 10 min buffer
#SBATCH --output=slurm/logs/smoke_selfplay_%A_%a.out

# --- Smoke-test configuration ---
# Goal: verify slurm integration, concurrent parquet shard writes, lineage fields,
# diagnostics passthrough. NOT a real self-play wave. Dataset is scratch, discard after.
DATASET_DIR=${DATASET_DIR:-/scratch/huang651/grid_mcts2/datasets/smoke01}
WEIGHTS_PATH=${WEIGHTS_PATH:-""}    # empty = random-init real network (fakenet via --config.use_fake=True)
MAP_NUM=${MAP_NUM:-2}               # 2 = 5x5 12qb (smallest real map)
GAMES_PER_NODE=${GAMES_PER_NODE:-120}   # 2 games/worker across 62 workers
NUM_WORKERS=62
BATCH_SIZE=62
NUM_SIMS=${NUM_SIMS:-100}           # smoke: 100 sims, not 800

# --- Environment ---
module load StdEnv/2023
module load python/3.11 scipy-stack
cd /home/huang651/grid_mcts2
source /home/huang651/grid_mcts2/.venv/bin/activate

NODE_ID="smoke_node${SLURM_ARRAY_TASK_ID}"

echo "=== Self-play SMOKE test ==="
echo "  SLURM_JOB_ID=$SLURM_JOB_ID"
echo "  SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "  hostname=$(hostname)"
echo "  DATASET_DIR=$DATASET_DIR"
echo "  MAP_NUM=$MAP_NUM"
echo "  GAMES_PER_NODE=$GAMES_PER_NODE, NUM_SIMS=$NUM_SIMS"

WEIGHTS_FLAG=""
if [ -n "$WEIGHTS_PATH" ]; then
    WEIGHTS_FLAG="--weights_path=$WEIGHTS_PATH"
fi

# Smoke uses preset=hpc so the HPC code path is exercised, but overrides
# num_simulations via CLI to keep per-job runtime in the ~30min ballpark.
python selfplay_worker.py \
    --preset=hpc \
    --dataset_dir=$DATASET_DIR \
    --num_games=$GAMES_PER_NODE \
    --num_workers=$NUM_WORKERS \
    --batch_size=$BATCH_SIZE \
    --node_id=$NODE_ID \
    --slurm_job_id=$SLURM_JOB_ID \
    --slurm_array_task_id=$SLURM_ARRAY_TASK_ID \
    --notes="SMOKE test: slurm $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID, ${NUM_SIMS} sims" \
    $WEIGHTS_FLAG \
    --config.map_num=$MAP_NUM \
    --config.mcts.num_simulations=$NUM_SIMS
