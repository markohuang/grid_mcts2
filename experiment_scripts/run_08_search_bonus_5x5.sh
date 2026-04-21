#!/usr/bin/env bash
set -euo pipefail

PYTHON="../grid_mcts2/.venv/bin/python"

# Control: layer_delta on fixed 5x5 map
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
"$PYTHON" main.py \
  --config.map_num=2 \
  --config.training.epochs=40 --config.training.num_selfplay=20 \
  --config.training.batch_size=128 --config.training.seed=42 \
  --config.mcts.root_dirichlet_alpha=0.3 \
  --config.training.data_augmentation=True \
  --config.experiment.early_stopping_patience=40 \
  --config.env.reward_mode=layer_delta \
  --config.experiment.study=round08_search_bonus_5x5 \
  --config.experiment.hypothesis=H1 \
  --config.experiment.variant=layer_delta_fixed_control \
  --config.experiment.tags=reward,search-bonus,control,fixed-5x5 \
  --config.experiment.notes="fixed 5x5 control for MCTS-only plan_cost bonus"

# Treatment: same setup, add MCTS-only plan_cost bonus
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
"$PYTHON" main.py \
  --config.map_num=2 \
  --config.training.epochs=40 --config.training.num_selfplay=20 \
  --config.training.batch_size=128 --config.training.seed=42 \
  --config.mcts.root_dirichlet_alpha=0.3 \
  --config.training.data_augmentation=True \
  --config.experiment.early_stopping_patience=40 \
  --config.env.reward_mode=layer_delta \
  --config.mcts.plan_cost_search_bonus_weight=2.0 \
  --config.experiment.study=round08_search_bonus_5x5 \
  --config.experiment.hypothesis=H1 \
  --config.experiment.variant=layer_delta_fixed_search_bonus_w2 \
  --config.experiment.tags=reward,search-bonus,treatment,fixed-5x5 \
  --config.experiment.notes="fixed 5x5 treatment with MCTS-only plan_cost bonus weight=2.0"
