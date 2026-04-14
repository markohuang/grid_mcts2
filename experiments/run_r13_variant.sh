#!/bin/bash
# R13 search-bonus redesign ablation.
# Usage: ./run_r13_variant.sh <label> <reward_mode> <bonus_mode> <beta> <gpu_id>
set -e

PYTHON="../grid_mcts2/.venv/bin/python"
LABEL=$1
REWARD_MODE=$2
BONUS_MODE=$3
BETA=$4
GPU_ID=$5

echo "=== R13 ${LABEL} (rm=${REWARD_MODE}, mode=${BONUS_MODE}, beta=${BETA}, GPU=${GPU_ID}) ==="

OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 \
CUDA_VISIBLE_DEVICES=$GPU_ID \
$PYTHON main.py \
  --config.map_num=2 \
  --config.training.epochs=15 --config.training.num_selfplay=50 \
  --config.training.batch_size=128 --config.training.seed=42 \
  --config.mcts.num_simulations=250 \
  --config.mcts.root_dirichlet_alpha=0.3 \
  --config.mcts.root_exploration_fraction=0.25 \
  --config.mcts.temperature_decay_steps=1000 \
  --config.mcts.temperature_final=0.25 \
  --config.training.data_augmentation=True \
  --config.env.reward_mode=$REWARD_MODE \
  --config.mcts.plan_cost_search_bonus_mode=$BONUS_MODE \
  --config.mcts.plan_cost_search_bonus_weight=$BETA \
  --config.network.p_hsize=64 \
  --config.experiment.early_stopping_patience=50 \
  --config.experiment.study=round13_search_bonus_redesign \
  --config.experiment.hypothesis=${LABEL} \
  --config.experiment.variant=${LABEL} \
  --config.experiment.tags=r13,search_bonus,${BONUS_MODE} \
  --config.experiment.notes="R13 ${LABEL}: rm=${REWARD_MODE}, mode=${BONUS_MODE}, beta=${BETA}"
