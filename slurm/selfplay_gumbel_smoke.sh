#!/bin/bash
#SBATCH --job-name=mcts_gumbel_smoke
#SBATCH --account=rrg-aspuru
#SBATCH --array=0                  # single job — pure wallclock calibration
#SBATCH --cpus-per-task=64
#SBATCH --mem=32G                  # match v3a (5x5 at 10k sims used ~32G)
#SBATCH --time=01:30:00            # v3a: 176s/game; Gumbel may be faster (tree better concentrated)
                                   # or slower (per-sim work higher due to improved-policy recomputation).
                                   # 1.5h is v3a's budget; if this overruns we'll bump for the ablation wave.
#SBATCH --output=slurm/logs/gumbel_smoke_%A_%a.out

# Gumbel smoke: calibrate per-game wallclock at production settings before committing to the
# Phase 2 ablation wave. 1 job, 62 games (1 per worker), 10k sims, plan_cost, MAPS[2] random.
# Matches v3a config exactly EXCEPT --config.mcts.gumbel.enabled=True.
#
# Interpretation guide vs v3a's 176s/game:
#   game_time_s median < 250s  -> Gumbel is competitive on wallclock, launch Phase 2 at v3a shape.
#   game_time_s median 250-500s -> still launch Phase 2 but with --time=03:00:00 margin.
#   game_time_s median > 500s  -> stop, investigate bottleneck (improved-policy recomputation?)
#                                  before burning 20 jobs.
#
# See docs/gumbel_pczero_plan.md §10 for smoke history.
DATASET_DIR=${DATASET_DIR:-/scratch/huang651/grid_mcts2/datasets/gumbel_smoke}
GAMES_PER_NODE=${GAMES_PER_NODE:-62}
NUM_WORKERS=62
BATCH_SIZE=62
NUM_SIMS=${NUM_SIMS:-10000}
MAP_NUM=${MAP_NUM:-2}

# --- Environment ---
module load StdEnv/2023
module load python/3.11 scipy-stack
cd /home/huang651/grid_mcts2
source /home/huang651/grid_mcts2/.venv/bin/activate

NODE_ID="gumbel_smoke_node${SLURM_ARRAY_TASK_ID}"

echo "=== Gumbel smoke (Phase 1 wallclock calibration) ==="
echo "  SLURM_JOB_ID=$SLURM_JOB_ID, host=$(hostname)"
echo "  gumbel.enabled=True, MAP_NUM=$MAP_NUM (5x5 12qb 3lyrs), NUM_SIMS=$NUM_SIMS, random_board=True"
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
    --notes="gumbel smoke: 5x5 calibration at v3a config with gumbel on" \
    --config.use_fake=True \
    --config.map_num=$MAP_NUM \
    --config.random_board=True \
    --config.mcts.num_simulations=$NUM_SIMS \
    --config.env.reward_mode=plan_cost \
    --config.mcts.gumbel.enabled=True
