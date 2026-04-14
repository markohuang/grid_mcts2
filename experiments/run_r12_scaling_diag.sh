#!/bin/bash
# R12 scaling diagnostic: 2x2 grid of (dirichlet_alpha × num_simulations)
# All with the new fixed MCTS sampling (softmax(log N / τ)).
# Usage: ./run_r12_scaling_diag.sh <label> <alpha> <sims> <gpu_id>
set -e

PYTHON="../grid_mcts2/.venv/bin/python"
LABEL=$1
ALPHA=$2
SIMS=$3
GPU_ID=$4

echo "=== R12 Scaling Diag ${LABEL} (α=${ALPHA}, sims=${SIMS}, GPU=${GPU_ID}) ==="

OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 \
CUDA_VISIBLE_DEVICES=$GPU_ID \
$PYTHON main.py \
  --config.map_num=2 \
  --config.training.epochs=10 --config.training.num_selfplay=50 \
  --config.training.batch_size=128 --config.training.seed=42 \
  --config.mcts.num_simulations=$SIMS \
  --config.mcts.root_dirichlet_alpha=$ALPHA \
  --config.mcts.root_exploration_fraction=0.25 \
  --config.mcts.temperature_decay_steps=1000 \
  --config.mcts.temperature_final=0.25 \
  --config.training.data_augmentation=True \
  --config.env.reward_mode=layer_delta \
  --config.mcts.plan_cost_search_bonus_weight=2.0 \
  --config.network.p_hsize=64 \
  --config.experiment.early_stopping_patience=50 \
  --config.experiment.study=round12_scaling_diag \
  --config.experiment.hypothesis=scaling_${LABEL} \
  --config.experiment.variant=${LABEL}_a${ALPHA}_s${SIMS} \
  --config.experiment.tags=diagnostic,scaling,fixed-sampling,${LABEL} \
  --config.experiment.notes="R12 scaling diag ${LABEL}: α=${ALPHA}, sims=${SIMS}, with MCTS sampling fix"
