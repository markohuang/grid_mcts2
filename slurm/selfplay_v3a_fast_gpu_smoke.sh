#!/bin/bash
#SBATCH --job-name=v3a_fast_gpu_smoke
#SBATCH --account=rrg-aspuru
#SBATCH --array=0
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/v3a_fast_gpu_smoke_%A_%a.out

# GPU wallclock smoke: fast_mcts + model on GPU vs CPU baseline.
# Measures per-game time with num_workers=1 (sequential, clean timing) and
# optionally num_workers=4 (4 concurrent CUDA contexts for throughput).
#
# Baselines to beat:
#   v3a01 c5  classic/CPU 62 workers: 537s/game
#   v3a_fast_smoke fast/CPU 62 workers: TBD (run that first)
#
# Interpretation of avg_game_time_s vs CPU-fast baseline:
#   <  50% of CPU-fast -> GPU kernel amortization is significant, scale up NUM_WORKERS
#   50-80% of CPU-fast -> moderate GPU benefit; still worthwhile for pipeline use
#   > 80% of CPU-fast  -> GPU overhead dominates at this sim count; stick with CPU
#
# Note: num_workers=1 means sequential games (total wall = num_games * game_time).
# For throughput test set NUM_WORKERS=4; 4 CUDA contexts (~300MB each) fit A100 VRAM.
DATASET_DIR=${DATASET_DIR:-/scratch/huang651/grid_mcts2/datasets/v3a_fast_gpu_smoke}
WEIGHTS_PATH=${WEIGHTS_PATH:-/scratch/huang651/grid_mcts2/pipelines/pipeline_v3a01/checkpoints/cycle_05.ckpt}
NUM_GAMES=${NUM_GAMES:-10}
NUM_WORKERS=${NUM_WORKERS:-1}
NUM_SIMS=10000
MAP_NUM=2

module load StdEnv/2023
module load python/3.11 scipy-stack cuda cudnn
cd /home/huang651/grid_mcts2
source /home/huang651/grid_mcts2/.venv/bin/activate

NODE_ID="v3a_fast_gpu_smoke_node${SLURM_ARRAY_TASK_ID}"

echo "=== v3a_fast_gpu_smoke: GPU selfplay calibration ==="
echo "  SLURM_JOB_ID=$SLURM_JOB_ID  host=$(hostname)"
echo "  backend=fast  selfplay_device=cuda  nn_batch_size=32  virtual_loss=1.0"
echo "  num_workers=$NUM_WORKERS  num_games=$NUM_GAMES  num_sims=$NUM_SIMS"
echo "  weights=$WEIGHTS_PATH"
nvidia-smi -L || true

python selfplay_worker.py \
    --preset=hpc \
    --dataset_dir="$DATASET_DIR" \
    --num_games=$NUM_GAMES \
    --num_workers=$NUM_WORKERS \
    --batch_size=$NUM_WORKERS \
    --weights_path="$WEIGHTS_PATH" \
    --node_id="$NODE_ID" \
    --slurm_job_id="$SLURM_JOB_ID" \
    --slurm_array_task_id="$SLURM_ARRAY_TASK_ID" \
    --notes="v3a_fast_gpu_smoke: GPU vs CPU wallclock" \
    --config.map_num=$MAP_NUM \
    --config.random_board=True \
    --config.mcts.num_simulations=$NUM_SIMS \
    --config.env.reward_mode=plan_cost \
    --config.mcts.root_dirichlet_alpha=0.1 \
    --config.mcts.backend=fast \
    --config.selfplay_device=cuda
