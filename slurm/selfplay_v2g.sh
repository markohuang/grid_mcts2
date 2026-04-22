#!/bin/bash
#SBATCH --job-name=mcts_selfplay_v2g
#SBATCH --account=rrg-aspuru
#SBATCH --array=0-19
#SBATCH --cpus-per-task=64
#SBATCH --mem=32G
#SBATCH --time=02:00:00           # v2f ran at 246s/game -> ~66min/worker. 2h for comfort.
#SBATCH --output=slurm/logs/selfplay_v2g_%A_%a.out

# v2g: middle-ground exploration. v2f (pb_c_base=19652) showed AlphaDev's default wins on
# cost AND diversity vs v2e (pb_c_base=500). But wave-level trajectory diversity is still
# only 17.8% unique triples out of 20k games -- each unique solution is replicated ~5x
# in training data. We want more landscape coverage without sacrificing v2f's cost gains.
#
# Single-knob test: pb_c_base 19652 -> 5000. Growth at 10k sims:
#   base=19652 -> pb_c = 1.25 + log(29653/19652) = 1.66  (+33%, v2f)
#   base=5000  -> pb_c = 1.25 + log(15001/5000)  = 2.35  (+88%, v2g)
#   base=500   -> pb_c = 1.25 + log(10501/500)   = 4.30  (+244%, v2e -- too much)
#
# Hypothesis: 5000 gives enough extra late-sim exploration to diversify the
# policy target while retaining the Q-exploitation that v2f got right. If diversity
# improves (unique_triples_frac > 0.20, trajs_at_min > 2) without cost regression
# (cost_median stays < 25), middle ground is the right choice. If cost regresses back
# toward v2e's 28, revert to 19652.
#
# NOTE: v2g includes the new reached_terminal_frac metric (added post-v2f).
# Dirichlet alpha stays at the (now-reverted) 0.03 default -- only one knob at a time.
DATASET_DIR=${DATASET_DIR:-/scratch/huang651/grid_mcts2/datasets/wave02g_pbc5000}
MAP_NUM=${MAP_NUM:-2}
GAMES_PER_NODE=${GAMES_PER_NODE:-1000}
NUM_WORKERS=62
BATCH_SIZE=62

# --- Environment ---
module load StdEnv/2023
module load python/3.11 scipy-stack
cd /home/huang651/grid_mcts2
source /home/huang651/grid_mcts2/.venv/bin/activate

NODE_ID="v2g_node${SLURM_ARRAY_TASK_ID}"

echo "=== Self-play v2g (middle-ground: pb_c_base=5000) ==="
echo "  SLURM_JOB_ID=$SLURM_JOB_ID, ARRAY_TASK=$SLURM_ARRAY_TASK_ID, host=$(hostname)"
echo "  pb_c_base=5000 (v2f was 19652), all other knobs = config defaults"

python selfplay_worker.py \
    --preset=hpc \
    --dataset_dir=$DATASET_DIR \
    --num_games=$GAMES_PER_NODE \
    --num_workers=$NUM_WORKERS \
    --batch_size=$BATCH_SIZE \
    --node_id=$NODE_ID \
    --slurm_job_id=$SLURM_JOB_ID \
    --slurm_array_task_id=$SLURM_ARRAY_TASK_ID \
    --notes="v2g pbc=5000: slurm $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID" \
    --config.use_fake=True \
    --config.map_num=$MAP_NUM \
    --config.mcts.num_simulations=10000 \
    --config.env.reward_mode=layer_delta \
    --config.mcts.pb_c_base=5000
