#!/bin/bash
#SBATCH --job-name=v4a01_smoke_train
#SBATCH --account=rrg-aspuru
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/v4a01_smoke_train_%j.out

# v4a01 pipeline smoke — Step 1: pretrain on v4a_gumbel bootstrap data.
#
# Produces cycle_00.ckpt which the selfplay smoke (Step 2) will use.
# After this job, submit v4a01_smoke_selfplay.sh with CKPT pointing at cycle_00.ckpt.
#
# Interpretation:
#   Training time gives the pretrain budget for production (set --train_time accordingly).
#   Expect ~10-20 min for 1 epoch on 1.2M transitions (20k games × 60 steps) on A100.
#
# Output: /scratch/huang651/grid_mcts2/pipelines/pipeline_v4a01_smoke/checkpoints/cycle_00.ckpt

PIPELINE_DIR=/scratch/huang651/grid_mcts2/pipelines/pipeline_v4a01_smoke
BOOTSTRAP_DATA=/scratch/huang651/grid_mcts2/datasets/v4a_gumbel

set -e

module load StdEnv/2023
module load python/3.11 scipy-stack
cd /home/huang651/grid_mcts2
source /home/huang651/grid_mcts2/.venv/bin/activate

mkdir -p "$PIPELINE_DIR/checkpoints"

echo "=== v4a01_smoke: pretrain on v4a_gumbel bootstrap (8x8, Gumbel) ==="
echo "  host=$(hostname)  SLURM_JOB_ID=$SLURM_JOB_ID"
echo "  bootstrap=$BOOTSTRAP_DATA"

python train_offline.py \
    --dataset_dirs="$BOOTSTRAP_DATA" \
    --save_ckpt="$PIPELINE_DIR/checkpoints/cycle_00.ckpt" \
    --run_id=v4a01_smoke \
    --epoch=0 \
    --train_epochs=1 \
    --preset=hpc \
    --config.map_num=5 \
    --config.random_board=True \
    --config.env.reward_mode=plan_cost \
    --config.training.buffer_size=200000

echo "=== pretrain done ==="
echo "  ckpt: $PIPELINE_DIR/checkpoints/cycle_00.ckpt"
echo "  next: sbatch --export=ALL,CKPT=$PIPELINE_DIR/checkpoints/cycle_00.ckpt slurm/v4a01_smoke_selfplay.sh"
