#!/bin/bash
#SBATCH --job-name=v4a01_smoke_sp
#SBATCH --account=rrg-aspuru
#SBATCH --array=0
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/v4a01_smoke_sp_%A_%a.out

# v4a01 pipeline smoke — Step 2b: selfplay timing calibration with trained checkpoint,
# fast backend + GPU.
#
# On 8×8, NN is 73% of game time (FakeNet=835s → trained=3112s; delta=2277s).
# Terminal fraction = 6% → 94% of sims call NN → GPU is beneficial (Amdahl limit 3.7×).
# This run measures actual speedup with backend=fast + cuda and calibrates production sizing.
#
# First run (classic/CPU, 3112s/game) told us baseline. Expected improvement:
#   fast backend alone: ~2× (batching amortizes Python dispatch)
#   + GPU NN forward: ~2-3× on top (NN is 73% of time, not 10% like 5×5)
#   Combined target: 500-1000s/game — run this to get the real number.
#
# Interpretation guide:
#   game_time_s median < 1200s  -> 500 games/job (8 games/worker), --selfplay_time=06:00:00
#   game_time_s median 1200-2000s -> 250 games/job (4 games/worker), --selfplay_time=06:00:00
#   game_time_s median > 2000s  -> cut sims to 5k or reduce games_per_task further
#
# Run after v4a01_smoke.sh completes:
#   sbatch --export=ALL,CKPT=/scratch/.../pipeline_v4a01_smoke/checkpoints/cycle_00.ckpt \
#          slurm/v4a01_smoke_selfplay.sh
#
# Or set CKPT as default below:
CKPT=${CKPT:-/scratch/huang651/grid_mcts2/pipelines/pipeline_v4a01_smoke/checkpoints/cycle_00.ckpt}
DATASET_DIR=/scratch/huang651/grid_mcts2/datasets/v4a01_smoke_selfplay_fast_gpu
NUM_SIMS=10000
GAMES_PER_NODE=62
NUM_WORKERS=62
BATCH_SIZE=62
MAP_NUM=5

module load StdEnv/2023
module load python/3.11 scipy-stack
cd /home/huang651/grid_mcts2
source /home/huang651/grid_mcts2/.venv/bin/activate

NODE_ID="v4a01_smoke_sp_node${SLURM_ARRAY_TASK_ID}"

echo "=== v4a01_smoke_selfplay: classic+GPU timing on 8x8 with Gumbel ==="
echo "  host=$(hostname)  SLURM_JOB_ID=$SLURM_JOB_ID"
echo "  ckpt=$CKPT  num_sims=$NUM_SIMS  games=$GAMES_PER_NODE  workers=$NUM_WORKERS"
echo "  FakeNet baseline (v4a_gumbel wave): 835s/game"
echo "  Classic/CPU result (cycle_00): 3112s/game  (NN=73% of time, terminal_frac=6%)"
echo "  Expected with fast+GPU: 500-1200s/game (3.7x Amdahl limit; actual depends on kernel overhead)"

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
    --notes="v4a01_smoke: classic+GPU timing calibration (8x8, Gumbel, 10k sims)" \
    --config.map_num=$MAP_NUM \
    --config.random_board=True \
    --config.mcts.num_simulations=$NUM_SIMS \
    --config.env.reward_mode=plan_cost \
    --config.mcts.root_dirichlet_alpha=0.1 \
    --config.mcts.gumbel.enabled=True \
    --config.selfplay_device=cuda

echo "=== selfplay done ==="
echo "Check timing: cat $DATASET_DIR/logs/${NODE_ID}_summary.json | python3 -m json.tool"
