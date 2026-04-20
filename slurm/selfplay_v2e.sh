#!/bin/bash
#SBATCH --job-name=mcts_selfplay_v2e
#SBATCH --account=rrg-aspuru
#SBATCH --array=0-19              # 20 jobs x 1000 games = 20k games (parity with prior waves)
#SBATCH --cpus-per-task=64
#SBATCH --mem=32G
#SBATCH --time=01:00:00           # 10k sims FakeNet: ~130s/game * 16 games/worker = ~35min worker, +buffer
#SBATCH --output=slurm/logs/selfplay_v2e_%A_%a.out

# v2e: 10k simulations with FakeNet. Goal: drive mcts_boundary_frac toward 1.0
# so MCTS becomes a near-unbiased Monte Carlo estimator of true returns, bypassing
# the untrained value head entirely. At depth approaching the 24-step episode length,
# the relationship between root_value and true game cost should cleanly go negative
# (currently +0.03 in v2d, the "bias-removed but still shallow" ceiling).
#
# Inherits v2d's directional fixes: reward_mode=layer_delta, temperature_init=1.0.
# Uses FakeNet since v2a proved FakeNet ~= random-init NN for data generation.
#
# Expected runtime (extrapolated from v2a 10.5s/game at 800 sims):
#   ~130s/game at 10k sims (12.5x sim count) => ~35 min/worker at 16 games/worker
#   --time=01:00:00 gives plenty of buffer for tree-growth overhead non-linearity.
DATASET_DIR=${DATASET_DIR:-/project/rrg-aspuru/huang651/grid_mcts2/datasets/wave02e_deepfull}
MAP_NUM=${MAP_NUM:-2}
GAMES_PER_NODE=${GAMES_PER_NODE:-1000}
NUM_WORKERS=62
BATCH_SIZE=62

# --- Environment ---
module load StdEnv/2023
module load python/3.11 scipy-stack
cd /home/huang651/grid_mcts2
source /home/huang651/grid_mcts2/.venv/bin/activate

NODE_ID="v2e_node${SLURM_ARRAY_TASK_ID}"

echo "=== Self-play v2e (10k sims, FakeNet, full-depth target) ==="
echo "  SLURM_JOB_ID=$SLURM_JOB_ID, ARRAY_TASK=$SLURM_ARRAY_TASK_ID, host=$(hostname)"
echo "  DATASET_DIR=$DATASET_DIR"
echo "  MAP_NUM=$MAP_NUM, GAMES_PER_NODE=$GAMES_PER_NODE"
echo "  Targets: boundary_frac > 0.95, corr(root_value, cost) < 0"

python selfplay_worker.py \
    --preset=hpc \
    --dataset_dir=$DATASET_DIR \
    --num_games=$GAMES_PER_NODE \
    --num_workers=$NUM_WORKERS \
    --batch_size=$BATCH_SIZE \
    --node_id=$NODE_ID \
    --slurm_job_id=$SLURM_JOB_ID \
    --slurm_array_task_id=$SLURM_ARRAY_TASK_ID \
    --notes="v2e 10k sims FakeNet: slurm $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID" \
    --config.use_fake=True \
    --config.map_num=$MAP_NUM \
    --config.mcts.num_simulations=10000 \
    --config.mcts.temperature_init=1.0 \
    --config.env.reward_mode=layer_delta
