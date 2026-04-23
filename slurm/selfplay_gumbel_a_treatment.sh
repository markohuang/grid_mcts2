#!/bin/bash
#SBATCH --job-name=mcts_gumbel_a_trmt
#SBATCH --account=rrg-aspuru
#SBATCH --array=0-19               # 20 jobs x 1000 games = 20k games (v3a parity)
#SBATCH --cpus-per-task=64
#SBATCH --mem=32G
#SBATCH --time=01:30:00            # Will bump if gumbel_smoke shows > 250s/game.
#SBATCH --output=slurm/logs/gumbel_a_treatment_%A_%a.out

# Gumbel Phase 2 — TREATMENT wave.
# Variant A from docs/gumbel_pczero_plan.md §3 ablation matrix. Identical to the paired
# control (selfplay_gumbel_a_control.sh) except --config.mcts.gumbel.enabled=True:
#   - Root: Gumbel-Top-m + sequential halving replaces Dirichlet + pUCT.
#   - Non-root: deterministic argmax [π'(a) - N(a)/(1+ΣN)] replaces pUCT.
#   - Policy target: guaranteed-improvement π' replaces softmax-of-visits.
# num_samples_m auto-derives to board_size - num_qubits + 1 = 14 on MAPS[2].
#
# Pair: selfplay_gumbel_a_control.sh — launch both with sbatch in the same session.
DATASET_DIR=${DATASET_DIR:-/scratch/huang651/grid_mcts2/datasets/gumbel_a_treatment}
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

NODE_ID="gumbel_a_trmt_node${SLURM_ARRAY_TASK_ID}"

echo "=== Gumbel Phase 2 — TREATMENT (Gumbel AlphaZero) ==="
echo "  SLURM_JOB_ID=$SLURM_JOB_ID, ARRAY_TASK=$SLURM_ARRAY_TASK_ID, host=$(hostname)"
echo "  gumbel.enabled=True, MAP_NUM=$MAP_NUM, NUM_SIMS=$NUM_SIMS, plan_cost, random_board"

python selfplay_worker.py \
    --preset=hpc \
    --dataset_dir=$DATASET_DIR \
    --num_games=$GAMES_PER_NODE \
    --num_workers=$NUM_WORKERS \
    --batch_size=$BATCH_SIZE \
    --node_id=$NODE_ID \
    --slurm_job_id=$SLURM_JOB_ID \
    --slurm_array_task_id=$SLURM_ARRAY_TASK_ID \
    --notes="gumbel phase2 treatment: Gumbel AlphaZero on v3a config (A variant)" \
    --config.use_fake=True \
    --config.map_num=$MAP_NUM \
    --config.random_board=True \
    --config.mcts.num_simulations=$NUM_SIMS \
    --config.env.reward_mode=plan_cost \
    --config.mcts.gumbel.enabled=True
