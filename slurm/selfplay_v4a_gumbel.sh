#!/bin/bash
#SBATCH --job-name=mcts_v4a_gumbel
#SBATCH --account=rrg-aspuru
#SBATCH --array=0-39               # 40 jobs x 500 games = 20k games
#SBATCH --cpus-per-task=64
#SBATCH --mem=96G                  # v4 smoke at 96G had no OOM (62 workers, 8x8, 10k sims)
#SBATCH --time=04:00:00            # Projected ~810s/game (v4 pUCT smoke 1687s at 20k → 844s at
                                   # 10k; Gumbel fixed-σ ≈ 96% of pUCT wallclock at 5x5).
                                   # 8 games/worker × 844s = 1.9h; 4h is 2x safety margin.
#SBATCH --output=slurm/logs/v4a_gumbel_%A_%a.out

# v4a — Gumbel AlphaZero on 8x8 MAPS[5] (first production wave at this scale).
#
# Config: fixed-σ Gumbel (c_visit=5.0, no max_N — see docs/selfplay/selfplay_gumbel.md §entropy
# calibration), MAPS[5] 8x8_20qb_6gpl_5lyrs, 10k sims, plan_cost, random_board, FakeNet.
#
# num_samples_m auto-derives to board_size - num_qubits + 1 = 64 - 20 + 1 = 45.
# Each of the 45 candidates gets floor(10000 / (45 * 6)) ≈ 37 sims in halving phase 1.
#
# Sizing note: v4 pUCT smoke (20k sims, 8x8) was 1687s/game median. At 10k sims,
# expect ~844s/game. 500 games/job, 62 workers → 8 games/worker × 844s ≈ 1.9h.
#
# This is the first Gumbel run at 8x8 scale. No paired pUCT control yet;
# treat this wave as baseline-establishment, not A/B comparison.
DATASET_DIR=${DATASET_DIR:-/scratch/huang651/grid_mcts2/datasets/v4a_gumbel}
MAP_NUM=${MAP_NUM:-5}
GAMES_PER_NODE=${GAMES_PER_NODE:-500}
NUM_WORKERS=62
BATCH_SIZE=62
NUM_SIMS=${NUM_SIMS:-10000}

# --- Environment ---
module load StdEnv/2023
module load python/3.11 scipy-stack
cd /home/huang651/grid_mcts2
source /home/huang651/grid_mcts2/.venv/bin/activate

NODE_ID="v4a_gumbel_node${SLURM_ARRAY_TASK_ID}"

echo "=== v4a Gumbel AlphaZero (8x8, 10k sims, fixed-sigma) ==="
echo "  SLURM_JOB_ID=$SLURM_JOB_ID, ARRAY_TASK=$SLURM_ARRAY_TASK_ID, host=$(hostname)"
echo "  MAP_NUM=$MAP_NUM (8x8 20qb 6gpl 5lyrs), NUM_SIMS=$NUM_SIMS, gumbel.enabled=True"
echo "  games_per_node=$GAMES_PER_NODE, workers=$NUM_WORKERS"

python selfplay_worker.py \
    --preset=hpc \
    --dataset_dir=$DATASET_DIR \
    --num_games=$GAMES_PER_NODE \
    --num_workers=$NUM_WORKERS \
    --batch_size=$BATCH_SIZE \
    --node_id=$NODE_ID \
    --slurm_job_id=$SLURM_JOB_ID \
    --slurm_array_task_id=$SLURM_ARRAY_TASK_ID \
    --notes="v4a gumbel: 8x8 first wave, fixed-sigma c_visit=5, 10k sims, plan_cost" \
    --config.use_fake=True \
    --config.map_num=$MAP_NUM \
    --config.random_board=True \
    --config.mcts.num_simulations=$NUM_SIMS \
    --config.env.reward_mode=plan_cost \
    --config.mcts.gumbel.enabled=True
