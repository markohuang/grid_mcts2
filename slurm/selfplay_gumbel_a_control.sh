#!/bin/bash
#SBATCH --job-name=mcts_gumbel_a_ctrl
#SBATCH --account=rrg-aspuru
#SBATCH --array=0-19               # 20 jobs x 1000 games = 20k games (v3a parity)
#SBATCH --cpus-per-task=64
#SBATCH --mem=32G
#SBATCH --time=01:30:00            # v3a used 1:30h @ 176s/game. Adjust based on gumbel_smoke.
#SBATCH --output=slurm/logs/gumbel_a_control_%A_%a.out

# Gumbel Phase 2 — CONTROL wave.
# Identical to v3a config (pUCT, Dirichlet, softmax-visit target) but re-run on the SAME
# cluster conditions and codebase as the treatment wave so that any environmental drift
# affects both sides equally. Do NOT use v3a's original dataset — that was collected on an
# earlier commit. This control is the apples-to-apples baseline for the Gumbel treatment.
#
# Match: 5x5 MAPS[2] random_board, 10k sims, plan_cost, Dirichlet=0.1, pb_c_base=19652.
# Difference from v3a slurm: none — this is a re-run with the current HEAD on parallel clock.
#
# Pair: selfplay_gumbel_a_treatment.sh — launch both with sbatch in the same session.
DATASET_DIR=${DATASET_DIR:-/scratch/huang651/grid_mcts2/datasets/gumbel_a_control}
MAP_NUM=${MAP_NUM:-2}
GAMES_PER_NODE=${GAMES_PER_NODE:-1000}
NUM_WORKERS=62
BATCH_SIZE=62
NUM_SIMS=${NUM_SIMS:-10000}

# --- Environment ---
module load StdEnv/2023
module load python/3.11 scipy-stack
cd /home/huang651/grid_mcts2
source /home/huang651/grid_mcts2/.venv/bin/activate

NODE_ID="gumbel_a_ctrl_node${SLURM_ARRAY_TASK_ID}"

echo "=== Gumbel Phase 2 — CONTROL (pUCT, v3a config) ==="
echo "  SLURM_JOB_ID=$SLURM_JOB_ID, ARRAY_TASK=$SLURM_ARRAY_TASK_ID, host=$(hostname)"
echo "  gumbel.enabled=False (pUCT), MAP_NUM=$MAP_NUM, NUM_SIMS=$NUM_SIMS, plan_cost, random_board"

python selfplay_worker.py \
    --preset=hpc \
    --dataset_dir=$DATASET_DIR \
    --num_games=$GAMES_PER_NODE \
    --num_workers=$NUM_WORKERS \
    --batch_size=$BATCH_SIZE \
    --node_id=$NODE_ID \
    --slurm_job_id=$SLURM_JOB_ID \
    --slurm_array_task_id=$SLURM_ARRAY_TASK_ID \
    --notes="gumbel phase2 control: pUCT v3a-parity for Gumbel ablation" \
    --config.use_fake=True \
    --config.map_num=$MAP_NUM \
    --config.random_board=True \
    --config.mcts.num_simulations=$NUM_SIMS \
    --config.env.reward_mode=plan_cost
