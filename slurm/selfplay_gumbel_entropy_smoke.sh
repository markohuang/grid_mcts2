#!/bin/bash
#SBATCH --job-name=mcts_gumbel_entropy
#SBATCH --account=rrg-aspuru
#SBATCH --array=0
#SBATCH --cpus-per-task=64
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=slurm/logs/gumbel_entropy_smoke_%A_%a.out

# Entropy calibration smoke for the σ formula fix.
#
# Change: _gumbel_improved_policy now uses σ = c_visit * c_scale * q_norm (no max_N term).
# c_visit=5.0 (was 50.0) keeps σ on the same scale as FakeNet logits (~2-3).
#
# Gate (both must pass before Phase 3):
#   policy_entropy_median > 0.5   — training targets have meaningful diversity
#   cost_median          <= 14    — action selection quality preserved
#
# Compare against Phase 2 treatment: cost_median=13, policy_entropy=0.000.
DATASET_DIR=${DATASET_DIR:-/scratch/huang651/grid_mcts2/datasets/gumbel_entropy_smoke}
GAMES_PER_NODE=${GAMES_PER_NODE:-62}
NUM_WORKERS=62
BATCH_SIZE=62
NUM_SIMS=${NUM_SIMS:-10000}
MAP_NUM=${MAP_NUM:-2}

module load StdEnv/2023
module load python/3.11 scipy-stack
cd /home/huang651/grid_mcts2
source /home/huang651/grid_mcts2/.venv/bin/activate

NODE_ID="gumbel_entropy_smoke_node${SLURM_ARRAY_TASK_ID}"

echo "=== Gumbel entropy calibration smoke ==="
echo "  SLURM_JOB_ID=$SLURM_JOB_ID, host=$(hostname)"
echo "  sigma fix: c_visit * c_scale * q_norm (no max_N), c_visit=5.0"
echo "  MAP_NUM=$MAP_NUM, NUM_SIMS=$NUM_SIMS, games=$GAMES_PER_NODE"

python selfplay_worker.py \
    --preset=hpc \
    --dataset_dir=$DATASET_DIR \
    --num_games=$GAMES_PER_NODE \
    --num_workers=$NUM_WORKERS \
    --batch_size=$BATCH_SIZE \
    --node_id=$NODE_ID \
    --slurm_job_id=$SLURM_JOB_ID \
    --slurm_array_task_id=$SLURM_ARRAY_TASK_ID \
    --notes="gumbel entropy calibration: sigma fix (no max_N), c_visit=5.0" \
    --config.use_fake=True \
    --config.map_num=$MAP_NUM \
    --config.random_board=True \
    --config.mcts.num_simulations=$NUM_SIMS \
    --config.env.reward_mode=plan_cost \
    --config.mcts.gumbel.enabled=True \
    --config.mcts.gumbel.c_visit=5.0
