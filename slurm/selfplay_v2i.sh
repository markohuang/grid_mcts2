#!/bin/bash
#SBATCH --job-name=mcts_selfplay_v2i
#SBATCH --account=rrg-aspuru
#SBATCH --array=0-4               # 5 nodes x 200 games = 1000 games total
#SBATCH --cpus-per-task=64
#SBATCH --mem=32G
#SBATCH --time=01:00:00           # 20k sims ~550s/game, 200 games x 550s / 62 workers ~30min
#SBATCH --output=slurm/logs/selfplay_v2i_%A_%a.out

# v2i: crank sims from 10k to 20k on v2f's exact config. Single-knob "more compute" test.
# Expected: modest improvements in reached_terminal_frac (v2f=0.25 proxy via boundary_frac),
# possibly lower cost min. Does NOT test dirichlet=0.1 (will become default in v3).
#
# Base = v2f: FakeNet, fixed map 2, layer_delta, pb_c_base=19652, dirichlet=0.03, tau=1 constant.
#
# Smaller sample (1000 games vs 20000) is deliberate: 0% repeat rate already established at 20k,
# correlations have SE~0.03 at n=1000 (enough to distinguish meaningful effects), cost
# distribution shape resolves at 1-unit precision. Saves ~75% of compute.
DATASET_DIR=${DATASET_DIR:-/project/rrg-aspuru/huang651/grid_mcts2/datasets/wave02i_20ksims}
MAP_NUM=${MAP_NUM:-2}
GAMES_PER_NODE=${GAMES_PER_NODE:-200}
NUM_WORKERS=62
BATCH_SIZE=62

# --- Environment ---
module load StdEnv/2023
module load python/3.11 scipy-stack
cd /home/huang651/grid_mcts2
source /home/huang651/grid_mcts2/.venv/bin/activate

NODE_ID="v2i_node${SLURM_ARRAY_TASK_ID}"

echo "=== Self-play v2i (20k sims on v2f config) ==="
echo "  SLURM_JOB_ID=$SLURM_JOB_ID, ARRAY_TASK=$SLURM_ARRAY_TASK_ID, host=$(hostname)"
echo "  num_simulations=20000 (v2f was 10000), all other knobs = v2f baseline"

python selfplay_worker.py \
    --preset=hpc \
    --dataset_dir=$DATASET_DIR \
    --num_games=$GAMES_PER_NODE \
    --num_workers=$NUM_WORKERS \
    --batch_size=$BATCH_SIZE \
    --node_id=$NODE_ID \
    --slurm_job_id=$SLURM_JOB_ID \
    --slurm_array_task_id=$SLURM_ARRAY_TASK_ID \
    --notes="v2i 20k sims: slurm $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID" \
    --config.use_fake=True \
    --config.map_num=$MAP_NUM \
    --config.mcts.num_simulations=20000 \
    --config.env.reward_mode=layer_delta \
    --config.mcts.root_dirichlet_alpha=0.03
