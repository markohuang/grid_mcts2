#!/bin/bash
#SBATCH --job-name=profile_gumbel
#SBATCH --account=rrg-aspuru
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/profile_gumbel_%j.out

# Profile one game with Gumbel search on an 8x8 map to understand the
# time breakdown before implementing fast_gumbel batching.
#
# Key questions:
#   1. What fraction is NN vs env.step vs _gumbel_non_root_select?
#   2. What is the terminal fraction at 2000 sims on 8x8?
#   3. How much of _gumbel_improved_policy / _gumbel_completed_q shows up?
#
# Compare with M6 pUCT/CPU (64.7s, map=2) and M3 pUCT/Gumbel-FakeNet (835s/game).

CKPT=${CKPT:-/scratch/huang651/grid_mcts2/pipelines/pipeline_v4a01_smoke/checkpoints/cycle_00.ckpt}
SIMS=${SIMS:-2000}
MAP_NUM=${MAP_NUM:-5}   # 8x8, 20 qubits, m=45

cd "$SLURM_SUBMIT_DIR"
module load StdEnv/2023
module load python/3.11 scipy-stack
source .venv/bin/activate

python scripts/profile_env.py \
    --ckpt "$CKPT" \
    --sims "$SIMS" \
    --map_num "$MAP_NUM" \
    --gumbel \
    --n_parity 0
