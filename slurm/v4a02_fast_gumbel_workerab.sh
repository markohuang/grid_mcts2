#!/bin/bash
#SBATCH --job-name=v4a02_fg_workerab
#SBATCH --account=rrg-aspuru
#SBATCH --array=0
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/v4a02_fg_workerab_%A_%a.out

# Worker-count A/B: does GPU dispatch contention explain the 1.87x speedup
# (vs 3.7x Amdahl ceiling) on v4a02_fast_gumbel_smoke?
#
# Same config as v4a02_fast_gumbel_smoke (same ckpt, sims, batch_size, vl,
# map, gumbel) — only num_workers and the per-batch grouping change.
#
# Control (already run, v4a02_fast_gumbel_smoke 60013190):
#   62 workers x 1 game x 1 batch  ->  game_t=1668s
#
# Treatment (this job):
#   16 workers x 4 games each x 4 batches  ->  ~1/4 the concurrent CUDA contexts
#   4 game_t samples (better statistics than the n=1 control batch)
#
# Hypothesis: with 16 concurrent contexts on the A100 instead of 62, each
# worker gets ~3.9x the GPU bandwidth share. If contention is the bottleneck,
# game_t drops from 1668s toward 800-1200s (closer to the 3.7x ceiling).
#
# Interpretation:
#   game_t < 1200s   -> contention dominated; production should use 16-32 workers
#                       and try nn_batch_size=64 next
#   game_t 1200-1500 -> partial; mixed bottleneck, both knobs help
#   game_t ~1668s    -> GPU is genuinely compute-saturated; only smaller model
#                       or fewer sims helps; stop chasing per-game time
#
# Time budget: 4 batches x ~1668s worst-case = 6700s = 1.85h. 3h ceiling is 1.6x
# margin if contention drops time, but caps the bad-case at the time limit.

CKPT=${CKPT:-/scratch/huang651/grid_mcts2/pipelines/pipeline_v4a01_smoke/checkpoints/cycle_00.ckpt}
DATASET_DIR=${DATASET_DIR:-/scratch/huang651/grid_mcts2/datasets/v4a02_fg_workerab_w16}
NUM_SIMS=${NUM_SIMS:-10000}
GAMES_PER_NODE=${GAMES_PER_NODE:-64}
NUM_WORKERS=${NUM_WORKERS:-16}
BATCH_SIZE=${BATCH_SIZE:-16}
MAP_NUM=${MAP_NUM:-5}
NN_BATCH_SIZE=${NN_BATCH_SIZE:-32}
VIRTUAL_LOSS=${VIRTUAL_LOSS:-1.0}

module load StdEnv/2023
module load python/3.11 scipy-stack cuda cudnn
cd /home/huang651/grid_mcts2
source /home/huang651/grid_mcts2/.venv/bin/activate

NODE_ID="v4a02_fg_workerab_node${SLURM_ARRAY_TASK_ID}"

echo "=== v4a02_fast_gumbel worker-count A/B (treatment: 16 workers) ==="
echo "  SLURM_JOB_ID=$SLURM_JOB_ID  host=$(hostname)  commit=$(git rev-parse --short HEAD)"
echo "  backend=fast_gumbel  selfplay_device=cuda  gumbel.enabled=True"
echo "  nn_batch_size=$NN_BATCH_SIZE  virtual_loss=$VIRTUAL_LOSS  num_sims=$NUM_SIMS"
echo "  games=$GAMES_PER_NODE  workers=$NUM_WORKERS  batch_size=$BATCH_SIZE  map=$MAP_NUM"
echo "  ckpt=$CKPT"
echo "  CONTROL (62 workers, v4a02_fast_gumbel_smoke 60013190): game_t=1668s"
nvidia-smi -L || true

mkdir -p "$DATASET_DIR"
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total \
    --format=csv -l 5 > "$DATASET_DIR/nvidia_smi.log" &
NSMI_PID=$!
echo "  nvidia-smi monitor PID=$NSMI_PID -> $DATASET_DIR/nvidia_smi.log"

python selfplay_worker.py \
    --preset=hpc \
    --dataset_dir="$DATASET_DIR" \
    --num_games=$GAMES_PER_NODE \
    --num_workers=$NUM_WORKERS \
    --batch_size=$BATCH_SIZE \
    --weights_path="$CKPT" \
    --node_id="$NODE_ID" \
    --slurm_job_id="$SLURM_JOB_ID" \
    --slurm_array_task_id="$SLURM_ARRAY_TASK_ID" \
    --notes="v4a02 workerab: 16w x 4g treatment vs 62w x 1g control" \
    --config.map_num=$MAP_NUM \
    --config.random_board=True \
    --config.mcts.num_simulations=$NUM_SIMS \
    --config.env.reward_mode=plan_cost \
    --config.mcts.root_dirichlet_alpha=0.1 \
    --config.mcts.gumbel.enabled=True \
    --config.mcts.backend=fast_gumbel \
    --config.mcts.nn_batch_size=$NN_BATCH_SIZE \
    --config.mcts.virtual_loss=$VIRTUAL_LOSS \
    --config.selfplay_device=cuda

EXIT=$?
kill $NSMI_PID 2>/dev/null || true

echo "=== nvidia-smi summary (last 5 lines) ==="
tail -5 "$DATASET_DIR/nvidia_smi.log" || true
echo "Full log: $DATASET_DIR/nvidia_smi.log"
echo "Check timing: cat $DATASET_DIR/logs/${NODE_ID}_summary.json | python3 -m json.tool"
echo "=== v4a02_fg_workerab done (exit=$EXIT) ==="
exit $EXIT
