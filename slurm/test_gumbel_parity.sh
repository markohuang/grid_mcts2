#!/bin/bash
#SBATCH --job-name=test_gumbel_parity
#SBATCH --account=rrg-aspuru
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/test_gumbel_parity_%j.out

# Parity tests for the new fast_gumbel backend (commit 8dd7ec5 — fast gumbel gpu).
#
# Verifies the byte-identical parity contract:
#   - classic Gumbel self-parity across maps, sim counts, fake/real net
#   - fast_gumbel self-parity at batch=1/vl=0 and batch=8/vl=1.0
#   - cross-parity: fast_gumbel(batch=1, vl=0) == classic Gumbel
#
# CPU-only — these tests don't exercise CUDA. 22 cases total. The slow ones are
# the real-net cases (map=0 sims=10) and the larger sim scans (sims=75).
# Expected runtime: 30-90 minutes on a single core; pytest runs sequentially.

module load StdEnv/2023
module load python/3.11 scipy-stack
cd /home/huang651/grid_mcts2
source /home/huang651/grid_mcts2/.venv/bin/activate

echo "=== fast_gumbel parity tests ==="
echo "  SLURM_JOB_ID=$SLURM_JOB_ID  host=$(hostname)"
echo "  commit=$(git rev-parse --short HEAD)  branch=$(git rev-parse --abbrev-ref HEAD)"
echo

# Force single-threaded torch so runs are deterministic across hosts.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python -m pytest fast_mcts/test_gumbel_parity.py -v --tb=short
EXIT=$?

echo
echo "=== parity tests done (exit=$EXIT) ==="
exit $EXIT
