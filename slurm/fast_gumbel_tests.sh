#!/bin/bash
#SBATCH --job-name=fast_gumbel_tests
#SBATCH --account=rrg-aspuru
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/fast_gumbel_tests_%j.out

# Parity tests for the fast_gumbel backend:
#   - self-parity: classic Gumbel, fast_gumbel(batch=1,vl=0), fast_gumbel(batch=8,vl=1.0)
#   - cross-parity: classic Gumbel == fast_gumbel(batch=1,vl=0) byte-identical
#
# All tests use FakeNet except two real-net cases at sims=10 (cheap). CPU-only.

module load StdEnv/2023
module load python/3.11 scipy-stack
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

echo "=== fast_gumbel parity tests ==="
echo "host=$(hostname)  SLURM_JOB_ID=$SLURM_JOB_ID"

python -m pytest fast_mcts/test_gumbel_parity.py -v --tb=short

echo "=== done ==="
