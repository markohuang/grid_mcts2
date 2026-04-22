#!/bin/bash
#SBATCH --job-name=mcts_selfplay_v2c
#SBATCH --account=rrg-aspuru
#SBATCH --array=0-19              # 20 jobs x 1000 games = 20k games
#SBATCH --cpus-per-task=64
#SBATCH --mem=32G
#SBATCH --time=00:30:00           # same compute budget as v1
#SBATCH --output=slurm/logs/selfplay_v2c_%A_%a.out

# v2c: bump root_dirichlet_alpha 0.3 -> 0.5 (first step of the "TODO: sweep 0.1-1.0"
# in config.py). Larger alpha = less spiky Dirichlet noise = more even exploration
# across root children. Tests whether the wave01 entropy (median 2.24 of ~2.48 max)
# has room to broaden via root-noise alone, without touching pb_c_init or sim count.
# Everything else matches v1: 800 sims, random-init real network, map 2, 20k games.
DATASET_DIR=${DATASET_DIR:-/scratch/huang651/grid_mcts2/datasets/wave02c_explore}
MAP_NUM=${MAP_NUM:-2}
GAMES_PER_NODE=${GAMES_PER_NODE:-1000}
NUM_WORKERS=62
BATCH_SIZE=62

# --- Environment ---
module load StdEnv/2023
module load python/3.11 scipy-stack
cd /home/huang651/grid_mcts2
source /home/huang651/grid_mcts2/.venv/bin/activate

NODE_ID="v2c_node${SLURM_ARRAY_TASK_ID}"

echo "=== Self-play v2c (+ exploration) ==="
echo "  SLURM_JOB_ID=$SLURM_JOB_ID, ARRAY_TASK=$SLURM_ARRAY_TASK_ID, host=$(hostname)"
echo "  DATASET_DIR=$DATASET_DIR"
echo "  MAP_NUM=$MAP_NUM, GAMES_PER_NODE=$GAMES_PER_NODE"
echo "  root_dirichlet_alpha=0.5 (v1 was 0.3)"

python selfplay_worker.py \
    --preset=hpc \
    --dataset_dir=$DATASET_DIR \
    --num_games=$GAMES_PER_NODE \
    --num_workers=$NUM_WORKERS \
    --batch_size=$BATCH_SIZE \
    --node_id=$NODE_ID \
    --slurm_job_id=$SLURM_JOB_ID \
    --slurm_array_task_id=$SLURM_ARRAY_TASK_ID \
    --notes="v2c +explore alpha=0.5: slurm $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID" \
    --config.map_num=$MAP_NUM \
    --config.mcts.root_dirichlet_alpha=0.5
