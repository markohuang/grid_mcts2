#!/bin/bash
#SBATCH --job-name=v4a02_fast_gumbel_smoke
#SBATCH --account=rrg-aspuru
#SBATCH --array=0
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/v4a02_fast_gumbel_smoke_%A_%a.out

# v4a02 — fast_gumbel + GPU smoke on 8x8.
# First wallclock measurement of the new fast_gumbel backend (commit 8dd7ec5)
# combined with selfplay_device=cuda on MAPS[5]. Calibration for production v4 sizing.
#
# Baselines on 8x8 / 10k sims / Gumbel / cycle_00 ckpt:
#   v4a_gumbel (FakeNet, classic, CPU):     ~835s/game  (no NN)
#   v4a01_smoke_sp (trained, classic, GPU): 3112s/game  (NN=73% of game time)
#
# Expected with fast_gumbel + GPU:
#   nn_batch_size=32 amortizes NN dispatch across leaves; GPU forward stays similar.
#   On 5x5 the puct fast backend gave 2-3x at 10k sims; gumbel should land in the
#   same ballpark, projecting 1000-1800s/game on 8x8. Confirm here.
#
# Interpretation guide (median game_time_s):
#   < 1500s  -> production v4a02 at 500 games/job, --time=06:00:00
#   1500-2500 -> 250 games/job, consider larger nn_batch_size (64) or shrink to 5k sims
#   > 2500s  -> investigate effective batch fill / VL behavior; do not scale up
#
# Quality check: cost mean should be close to v4a01_smoke_sp (cost avg=29.5,
# min=25 over 62 games at 10k sims classic Gumbel).

CKPT=${CKPT:-/scratch/huang651/grid_mcts2/pipelines/pipeline_v4a01_smoke/checkpoints/cycle_00.ckpt}
DATASET_DIR=${DATASET_DIR:-/scratch/huang651/grid_mcts2/datasets/v4a02_fast_gumbel_smoke}
NUM_SIMS=${NUM_SIMS:-10000}
GAMES_PER_NODE=${GAMES_PER_NODE:-62}
NUM_WORKERS=${NUM_WORKERS:-62}
BATCH_SIZE=${BATCH_SIZE:-62}
MAP_NUM=${MAP_NUM:-5}
NN_BATCH_SIZE=${NN_BATCH_SIZE:-32}
VIRTUAL_LOSS=${VIRTUAL_LOSS:-1.0}

module load StdEnv/2023
module load python/3.11 scipy-stack cuda cudnn
cd /home/huang651/grid_mcts2
source /home/huang651/grid_mcts2/.venv/bin/activate

NODE_ID="v4a02_fast_gumbel_smoke_node${SLURM_ARRAY_TASK_ID}"

echo "=== v4a02_fast_gumbel_smoke: fast_gumbel+GPU calibration on 8x8 ==="
echo "  SLURM_JOB_ID=$SLURM_JOB_ID  host=$(hostname)  commit=$(git rev-parse --short HEAD)"
echo "  backend=fast_gumbel  selfplay_device=cuda  gumbel.enabled=True"
echo "  nn_batch_size=$NN_BATCH_SIZE  virtual_loss=$VIRTUAL_LOSS  num_sims=$NUM_SIMS"
echo "  games=$GAMES_PER_NODE  workers=$NUM_WORKERS  map=$MAP_NUM (8x8 20qb 6gpl 5lyrs)"
echo "  ckpt=$CKPT"
echo "  classic+GPU baseline (v4a01_smoke_sp): 3112s/game  cost avg=29.5  min=25"
nvidia-smi -L || true

# Background GPU utilization monitor.
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
    --notes="v4a02 smoke: fast_gumbel + cuda on 8x8 Gumbel 10k sims" \
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
echo "=== v4a02_fast_gumbel_smoke done (exit=$EXIT) ==="
exit $EXIT
