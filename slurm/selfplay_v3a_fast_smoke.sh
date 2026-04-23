#!/bin/bash
#SBATCH --job-name=v3a_fast_smoke
#SBATCH --account=rrg-aspuru
#SBATCH --array=0
#SBATCH --cpus-per-task=64
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=slurm/logs/v3a_fast_smoke_%A_%a.out

# Wallclock smoke: fast_mcts (backend=fast, batch=32, vl=1.0) vs v3a01 baseline.
# Purpose: verify that per-game time drops with identical num_simulations=10000.
# Parameters are held fixed to v3a01 cycle 5 (trained prior, 5x5 random boards).
# Interpretation vs v3a01 cycle_05 baseline of 537s/game:
#   < 200s   -> ~2.5x+ wallclock speedup; proceed with v3a_fast pipeline series
#   200-400s -> marginal speedup; still useful but investigate nn_batch_size tuning
#   > 400s   -> fast_mcts overhead dominates at this sim count; hold off
#
# After run: compare avg_game_time_s in logs/<node>_summary.json to 537s.
# Cost quality check: avg cost should be within ~1 unit of v3a01 c5 avg (12.99).
DATASET_DIR=${DATASET_DIR:-/scratch/huang651/grid_mcts2/datasets/v3a_fast_smoke}
WEIGHTS_PATH=${WEIGHTS_PATH:-/scratch/huang651/grid_mcts2/pipelines/pipeline_v3a01/checkpoints/cycle_05.ckpt}
GAMES_PER_NODE=62
NUM_WORKERS=62
BATCH_SIZE=62
NUM_SIMS=10000
MAP_NUM=2

module load StdEnv/2023
module load python/3.11 scipy-stack
cd /home/huang651/grid_mcts2
source /home/huang651/grid_mcts2/.venv/bin/activate

NODE_ID="v3a_fast_smoke_node${SLURM_ARRAY_TASK_ID}"

echo "=== v3a_fast_smoke: fast_mcts wallclock calibration ==="
echo "  SLURM_JOB_ID=$SLURM_JOB_ID  host=$(hostname)"
echo "  backend=fast  nn_batch_size=32  virtual_loss=1.0  num_sims=$NUM_SIMS"
echo "  weights=$WEIGHTS_PATH"
echo "  baseline: v3a01 cycle_05 = 537s/game at same config with backend=classic"

python selfplay_worker.py \
    --preset=hpc \
    --dataset_dir="$DATASET_DIR" \
    --num_games=$GAMES_PER_NODE \
    --num_workers=$NUM_WORKERS \
    --batch_size=$BATCH_SIZE \
    --weights_path="$WEIGHTS_PATH" \
    --node_id="$NODE_ID" \
    --slurm_job_id="$SLURM_JOB_ID" \
    --slurm_array_task_id="$SLURM_ARRAY_TASK_ID" \
    --notes="v3a_fast_smoke: wallclock vs v3a01 c5 baseline" \
    --config.map_num=$MAP_NUM \
    --config.random_board=True \
    --config.mcts.num_simulations=$NUM_SIMS \
    --config.env.reward_mode=plan_cost \
    --config.mcts.root_dirichlet_alpha=0.1 \
    --config.mcts.backend=fast
