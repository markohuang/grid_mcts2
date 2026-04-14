#!/bin/bash
# R12 exploration diagnostic: 3 short 10-epoch runs with varied exploration constants.
# Usage: ./run_r12_diag.sh <label> <dirichlet_alpha> <temp_decay> <temp_final> <root_frac> <gpu_id>
set -e

PYTHON="../grid_mcts2/.venv/bin/python"
LABEL=$1
ALPHA=$2
DECAY=$3
TEMP_FINAL=$4
FRAC=$5
GPU_ID=$6

echo "=== R12 Diagnostic ${LABEL} (α=${ALPHA}, decay=${DECAY}, temp_final=${TEMP_FINAL}, frac=${FRAC}, GPU=${GPU_ID}) ==="

OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 \
CUDA_VISIBLE_DEVICES=$GPU_ID \
$PYTHON main.py \
  --config.map_num=2 \
  --config.training.epochs=10 --config.training.num_selfplay=50 \
  --config.training.batch_size=128 --config.training.seed=42 \
  --config.mcts.num_simulations=250 \
  --config.mcts.root_dirichlet_alpha=$ALPHA \
  --config.mcts.root_exploration_fraction=$FRAC \
  --config.mcts.temperature_decay_steps=$DECAY \
  --config.mcts.temperature_final=$TEMP_FINAL \
  --config.training.data_augmentation=True \
  --config.env.reward_mode=layer_delta \
  --config.mcts.plan_cost_search_bonus_weight=2.0 \
  --config.network.p_hsize=64 \
  --config.experiment.early_stopping_patience=50 \
  --config.experiment.study=round12_exploration_diag \
  --config.experiment.hypothesis=diag_${LABEL} \
  --config.experiment.variant=${LABEL}_a${ALPHA}_d${DECAY}_tf${TEMP_FINAL}_f${FRAC} \
  --config.experiment.tags=diagnostic,exploration,${LABEL} \
  --config.experiment.notes="R12 diag ${LABEL}: α=${ALPHA}, decay=${DECAY}, temp_final=${TEMP_FINAL}, frac=${FRAC}"
