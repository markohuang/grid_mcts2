#!/bin/bash
#SBATCH --job-name=mcts_selfplay_v2j
#SBATCH --account=rrg-aspuru
#SBATCH --array=0-4               # 5 nodes x 200 games = 1000 games total
#SBATCH --cpus-per-task=64
#SBATCH --mem=32G
#SBATCH --time=00:30:00           # v2f at 246s/game, 200 games x 246s / 62 workers ~13min
#SBATCH --output=slurm/logs/selfplay_v2j_%A_%a.out

# v2j: swap reward_mode layer_delta -> plan_cost on v2f's exact config. Tests whether v2d's
# reward-mode switch was load-bearing for the v2d->v2f gains (which also reverted pb_c_base
# and dirichlet).
#
# Base = v2f exactly except reward_mode=plan_cost:
#   FakeNet, fixed map 2, 10k sims, pb_c_base=19652, dirichlet=0.03, tau=1 constant.
#
# Interpretation:
#   v2j ~= v2f          : reward_mode wasn't doing load-bearing work at 10k sims + AlphaDev defaults.
#   v2j regresses       : layer_delta earns its keep; stick with it.
#   v2j improves        : plan_cost + AlphaDev pb_c_base is actually the right combo; revisit default.
#
# Caveat: result is specific to 10k sims / FakeNet / map 2. Doesn't generalize to trained-value
# regimes or random maps without further testing.
DATASET_DIR=${DATASET_DIR:-/scratch/huang651/grid_mcts2/datasets/wave02j_plancost}
MAP_NUM=${MAP_NUM:-2}
GAMES_PER_NODE=${GAMES_PER_NODE:-200}
NUM_WORKERS=62
BATCH_SIZE=62

# --- Environment ---
module load StdEnv/2023
module load python/3.11 scipy-stack
cd /home/huang651/grid_mcts2
source /home/huang651/grid_mcts2/.venv/bin/activate

NODE_ID="v2j_node${SLURM_ARRAY_TASK_ID}"

echo "=== Self-play v2j (plan_cost reward on v2f config) ==="
echo "  SLURM_JOB_ID=$SLURM_JOB_ID, ARRAY_TASK=$SLURM_ARRAY_TASK_ID, host=$(hostname)"
echo "  reward_mode=plan_cost (v2f was layer_delta), all other knobs = v2f baseline"

python selfplay_worker.py \
    --preset=hpc \
    --dataset_dir=$DATASET_DIR \
    --num_games=$GAMES_PER_NODE \
    --num_workers=$NUM_WORKERS \
    --batch_size=$BATCH_SIZE \
    --node_id=$NODE_ID \
    --slurm_job_id=$SLURM_JOB_ID \
    --slurm_array_task_id=$SLURM_ARRAY_TASK_ID \
    --notes="v2j plan_cost: slurm $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID" \
    --config.use_fake=True \
    --config.map_num=$MAP_NUM \
    --config.mcts.num_simulations=10000 \
    --config.env.reward_mode=plan_cost \
    --config.mcts.root_dirichlet_alpha=0.03
