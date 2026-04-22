#!/bin/bash
#SBATCH --job-name=mcts_selfplay_v2h
#SBATCH --account=rrg-aspuru
#SBATCH --array=0-19
#SBATCH --cpus-per-task=64
#SBATCH --mem=32G
#SBATCH --time=02:00:00           # match v2f/v2g (10k sims -> ~246s/game)
#SBATCH --output=slurm/logs/selfplay_v2h_%A_%a.out

# v2h: middle-ground Dirichlet. Complements v2g (which moves pb_c_base). Forms a
# factorial with v2f (baseline = both AlphaDev defaults):
#   v2f: pb_c_base=19652, dirichlet=0.03   (baseline, cost median 23, 17.8% unique)
#   v2g: pb_c_base=5000,  dirichlet=0.03   (pb_c knob only)
#   v2h: pb_c_base=19652, dirichlet=0.1    (dirichlet knob only)
#
# Intuition for alpha:
#   alpha=0.03: Dirichlet(0.03) on 12 actions is near-one-hot. One action gets ~90%
#               of noise mass, rest near-zero. Different games boost different *single*
#               actions -- but only one per game.
#   alpha=0.1:  Noise mass spreads across 2-4 actions typically. Different games boost
#               different *sets* of 2-4 actions at root. More within-game action coverage
#               without washing out the prior.
#   alpha=0.3:  Noise spreads across most actions ~= uniform. v2e showed this washes out
#               the Q signal and regresses cost.
#
# At root with exploration_fraction=0.25: prior = 0.75 * pi_net + 0.25 * noise.
# v2h hypothesis: 0.1 gives more trajectory diversity than 0.03 without regressing cost
# like 0.3 did. Primary read: unique_triples_frac (target > 0.20).
#
# Includes reached_terminal_frac metric (added post-v2f).
DATASET_DIR=${DATASET_DIR:-/scratch/huang651/grid_mcts2/datasets/wave02h_dir0.1}
MAP_NUM=${MAP_NUM:-2}
GAMES_PER_NODE=${GAMES_PER_NODE:-1000}
NUM_WORKERS=62
BATCH_SIZE=62

# --- Environment ---
module load StdEnv/2023
module load python/3.11 scipy-stack
cd /home/huang651/grid_mcts2
source /home/huang651/grid_mcts2/.venv/bin/activate

NODE_ID="v2h_node${SLURM_ARRAY_TASK_ID}"

echo "=== Self-play v2h (middle-ground: dirichlet alpha=0.1) ==="
echo "  SLURM_JOB_ID=$SLURM_JOB_ID, ARRAY_TASK=$SLURM_ARRAY_TASK_ID, host=$(hostname)"
echo "  root_dirichlet_alpha=0.1 (v2f was 0.03), pb_c_base=19652 (same as v2f)"

python selfplay_worker.py \
    --preset=hpc \
    --dataset_dir=$DATASET_DIR \
    --num_games=$GAMES_PER_NODE \
    --num_workers=$NUM_WORKERS \
    --batch_size=$BATCH_SIZE \
    --node_id=$NODE_ID \
    --slurm_job_id=$SLURM_JOB_ID \
    --slurm_array_task_id=$SLURM_ARRAY_TASK_ID \
    --notes="v2h alpha=0.1: slurm $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID" \
    --config.use_fake=True \
    --config.map_num=$MAP_NUM \
    --config.mcts.num_simulations=10000 \
    --config.env.reward_mode=layer_delta \
    --config.mcts.root_dirichlet_alpha=0.1
