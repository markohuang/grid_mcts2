#!/bin/bash
#SBATCH --job-name=mcts_selfplay_v4_smoke
#SBATCH --account=rrg-aspuru
#SBATCH --array=0                  # single job — pure calibration
#SBATCH --cpus-per-task=64
#SBATCH --mem=96G                  # 5x5 at 10k sims used ~32G; 8x8 at 20k OOM'd on 32G (job 59728979).
                                   # Bumped to 96G: ~1.5GB/worker headroom at 62 workers. If still OOMs,
                                   # reduce NUM_WORKERS to 30 rather than bump further.
#SBATCH --time=02:00:00            # 2h ceiling; abort early if per-game runs long
#SBATCH --output=slurm/logs/selfplay_v4_smoke_%A_%a.out

# v4 SMOKE test. Goal: measure per-game wallclock on 8x8 at 20k sims with plan_cost.
# Single job, 62 games (1 per worker). Results determine v4a sizing.
#
# Interpretation guide:
#   game_time_s median < 600s  -> v4a at 1000 games/job x 20 jobs with --time=06:00:00 is safe
#   game_time_s median 600-1200s -> cut v4a to 500 games/job or drop to 15k sims
#   game_time_s median > 1800s  -> stop, re-scope sim count before any production run
#
# Config: v3a's winning parameters, scaled-up map.
#   FakeNet, random_board=True, plan_cost, 20k sims, MAPS[5] (8x8 20qb 6gpl 5lyrs).
#   pb_c_base=19652, dirichlet=0.1, tau=1 constant (all config defaults).
DATASET_DIR=${DATASET_DIR:-/scratch/huang651/grid_mcts2/datasets/wave04_smoke}
GAMES_PER_NODE=${GAMES_PER_NODE:-62}       # 1 game per worker, 62 workers
NUM_WORKERS=62
BATCH_SIZE=62
NUM_SIMS=${NUM_SIMS:-20000}
MAP_NUM=${MAP_NUM:-5}                       # MAPS[5] = 8x8 20qb 5lyr 6gpl

# --- Environment ---
module load StdEnv/2023
module load python/3.11 scipy-stack
cd /home/huang651/grid_mcts2
source /home/huang651/grid_mcts2/.venv/bin/activate

NODE_ID="v4smoke_node${SLURM_ARRAY_TASK_ID}"

echo "=== Self-play v4 SMOKE (calibration on 8x8) ==="
echo "  SLURM_JOB_ID=$SLURM_JOB_ID, host=$(hostname)"
echo "  MAP_NUM=$MAP_NUM (8x8 20qb 6gpl 5lyrs), NUM_SIMS=$NUM_SIMS, random_board=True"
echo "  games_per_node=$GAMES_PER_NODE (1 per worker)"

python selfplay_worker.py \
    --preset=hpc \
    --dataset_dir=$DATASET_DIR \
    --num_games=$GAMES_PER_NODE \
    --num_workers=$NUM_WORKERS \
    --batch_size=$BATCH_SIZE \
    --node_id=$NODE_ID \
    --slurm_job_id=$SLURM_JOB_ID \
    --slurm_array_task_id=$SLURM_ARRAY_TASK_ID \
    --notes="v4 smoke: 8x8 calibration, 20k sims, plan_cost" \
    --config.use_fake=True \
    --config.map_num=$MAP_NUM \
    --config.random_board=True \
    --config.mcts.num_simulations=$NUM_SIMS \
    --config.env.reward_mode=plan_cost
