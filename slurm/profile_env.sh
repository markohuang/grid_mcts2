#!/bin/bash
#SBATCH --job-name=profile_env
#SBATCH --account=rrg-aspuru
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/profile_env_%j.out

# Profile one game to break down where env time goes.
# Single worker, 2000 sims, real network (cycle_05 ckpt), fast backend.
# Outputs cProfile tables (cumtime + tottime) for env.step internals.

CKPT=${CKPT:-/scratch/huang651/grid_mcts2/pipelines/pipeline_v3a01/checkpoints/cycle_05.ckpt}
SIMS=${SIMS:-2000}
MAP_NUM=${MAP_NUM:-2}

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

python scripts/profile_env.py \
    --ckpt "$CKPT" \
    --sims "$SIMS" \
    --map_num "$MAP_NUM"
