#!/usr/bin/env bash
set -euo pipefail

PYTHON="../grid_mcts2/.venv/bin/python"

run_round08() {
  local seed="$1"
  local reward_mode="$2"
  local hypothesis="$3"
  local variant="$4"
  local tags="$5"
  local extra_args=("${@:6}")

  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  "$PYTHON" main.py \
    --config.map_num=2 --config.random_board=True \
    --config.training.epochs=60 --config.training.num_selfplay=20 \
    --config.training.batch_size=128 --config.training.seed="$seed" \
    --config.mcts.root_dirichlet_alpha=0.3 \
    --config.training.data_augmentation=True \
    --config.experiment.curriculum_maps=3 \
    --config.experiment.curriculum_initial_phase=3 \
    --config.experiment.early_stopping_patience=60 \
    --config.experiment.study=round08_reward_error_modes \
    --config.experiment.hypothesis="$hypothesis" \
    --config.experiment.variant="$variant" \
    --config.experiment.tags="$tags" \
    --config.experiment.notes="round08 $variant seed=$seed" \
    --config.env.reward_mode="$reward_mode" \
    "${extra_args[@]}"
}

# Seed 42
run_round08 42 plan_cost H1 plan_cost_g "reward,generalist,control"
run_round08 42 layer_delta H2 layer_delta_g "reward,generalist,treatment"

# Seed 123
run_round08 123 plan_cost H1 plan_cost_g_s123 "reward,generalist,control,seed123"
run_round08 123 layer_delta H2 layer_delta_g_s123 "reward,generalist,treatment,seed123"
