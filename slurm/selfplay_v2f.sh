#!/bin/bash
#SBATCH --job-name=mcts_selfplay_v2f
#SBATCH --account=rrg-aspuru
#SBATCH --array=0-19              # 20 jobs x 1000 games = 20k games
#SBATCH --cpus-per-task=64
#SBATCH --mem=32G
#SBATCH --time=01:30:00           # match v2e (10k sims) with extra buffer vs v2e's 1 timed-out job
#SBATCH --output=slurm/logs/selfplay_v2f_%A_%a.out

# v2f: same as v2e (FakeNet, 10k sims, layer_delta, tau=1) but REVERT pb_c_base
# and root_dirichlet_alpha to Round-04-era AlphaDev defaults. Tests whether this
# session's config changes (pb_c_base 19652->500, dirichlet 0.03->0.3) are a
# meaningful part of the remaining gap to Round 04's best cost=12 on map 2.
#
#   At 10k sims:
#     pb_c_base=500   -> pb_c = 1.25 + log(10501/500)   = 4.30  (+244% over init)
#     pb_c_base=19652 -> pb_c = 1.25 + log(29653/19652) = 1.66  (+33% over init)
#   That's 3x less late-sim exploration pressure = UCB is more Q-exploiting,
#   which should give deeper PVs and more discriminated visit distributions
#   (our current failure mode: uniform BFS, tiny depth-std, no top-K gap).
#
# Clean A/B: only the two suspect config knobs change from v2e.
DATASET_DIR=${DATASET_DIR:-/project/rrg-aspuru/huang651/grid_mcts2/datasets/wave02f_reverted}
MAP_NUM=${MAP_NUM:-2}
GAMES_PER_NODE=${GAMES_PER_NODE:-1000}
NUM_WORKERS=62
BATCH_SIZE=62

# --- Environment ---
module load StdEnv/2023
module load python/3.11 scipy-stack
cd /home/huang651/grid_mcts2
source /home/huang651/grid_mcts2/.venv/bin/activate

NODE_ID="v2f_node${SLURM_ARRAY_TASK_ID}"

echo "=== Self-play v2f (reverted pb_c_base + dirichlet to Round-04 defaults) ==="
echo "  SLURM_JOB_ID=$SLURM_JOB_ID, ARRAY_TASK=$SLURM_ARRAY_TASK_ID, host=$(hostname)"
echo "  DATASET_DIR=$DATASET_DIR"
echo "  MAP_NUM=$MAP_NUM, GAMES_PER_NODE=$GAMES_PER_NODE"
echo "  pb_c_base=19652 (was 500), root_dirichlet_alpha=0.03 (was 0.3)"

python selfplay_worker.py \
    --preset=hpc \
    --dataset_dir=$DATASET_DIR \
    --num_games=$GAMES_PER_NODE \
    --num_workers=$NUM_WORKERS \
    --batch_size=$BATCH_SIZE \
    --node_id=$NODE_ID \
    --slurm_job_id=$SLURM_JOB_ID \
    --slurm_array_task_id=$SLURM_ARRAY_TASK_ID \
    --notes="v2f reverted: slurm $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID" \
    --config.use_fake=True \
    --config.map_num=$MAP_NUM \
    --config.mcts.num_simulations=10000 \
    --config.mcts.temperature_init=1.0 \
    --config.env.reward_mode=layer_delta \
    --config.mcts.pb_c_base=19652 \
    --config.mcts.root_dirichlet_alpha=0.03
