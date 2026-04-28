#!/bin/bash
#SBATCH --job-name=profile_env_gpu
#SBATCH --account=rrg-aspuru
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/profile_env_gpu_%j.out

# GPU counterpart to profile_env.sh.
# Identical config (map=2, sims=2000, fast backend, batch=32, vl=1.0, cycle_05 ckpt)
# except the NN runs on CUDA. Env always runs on CPU.
#
# Compare wall time and cProfile breakdown against M6 (CPU, job 59945684, 64.7s)
# to measure whether NN-on-GPU helps after Opt-1/2/3 raised NN fraction to 43.7%.
#
# M6 CPU reference:
#   Wall: 64.7s   NN cumtime: 28.3s (43.7%)   env.step cumtime: 21.7s (33.6%)
#   Amdahl limit for GPU: 1/(1-0.437) = 1.78x  (if GPU NN were free)

CKPT=${CKPT:-/scratch/huang651/grid_mcts2/pipelines/pipeline_v3a01/checkpoints/cycle_05.ckpt}
SIMS=${SIMS:-2000}
MAP_NUM=${MAP_NUM:-2}

module load StdEnv/2023
module load python/3.11 scipy-stack cuda cudnn
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

echo "profile_env_gpu  host=$(hostname)"
nvidia-smi -L || true

python scripts/profile_env.py \
    --ckpt "$CKPT" \
    --sims "$SIMS" \
    --map_num "$MAP_NUM" \
    --device cuda
