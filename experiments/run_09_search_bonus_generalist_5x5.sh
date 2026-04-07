#!/usr/bin/env bash
set -euo pipefail

PYTHON="../grid_mcts2/.venv/bin/python"

if [ "$#" -eq 0 ]; then
  set -- 42
fi

run_round09() {
  local seed="$1"
  local reward_mode="$2"
  local hypothesis="$3"
  local variant="$4"
  local tags="$5"
  local parent_run="$6"
  local notes="$7"
  local extra_args=("${@:8}")

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
    --config.experiment.study=round09_search_bonus_generalist_5x5 \
    --config.experiment.hypothesis="$hypothesis" \
    --config.experiment.variant="$variant" \
    --config.experiment.tags="$tags" \
    --config.experiment.parent_run="$parent_run" \
    --config.experiment.notes="$notes" \
    --config.env.reward_mode="$reward_mode" \
    "${extra_args[@]}"
}

for seed in "$@"; do
  suffix=""
  if [ "$seed" != "42" ]; then
    suffix="_s${seed}"
  fi

  run_round09 \
    "$seed" \
    plan_cost \
    H2 \
    "plan_cost_g_control${suffix}" \
    "reward,search-bonus,generalist,control,plan-cost" \
    e9968d5c \
    "round09 5x5 generalist baseline with plan_cost seed=${seed}"

  run_round09 \
    "$seed" \
    layer_delta \
    H1 \
    "layer_delta_g_control${suffix}" \
    "reward,search-bonus,generalist,control,layer-delta" \
    153cf066 \
    "round09 5x5 generalist baseline with layer_delta seed=${seed}"

  run_round09 \
    "$seed" \
    layer_delta \
    H2 \
    "layer_delta_g_search_bonus_w2${suffix}" \
    "reward,search-bonus,generalist,treatment,layer-delta" \
    66559890 \
    "round09 5x5 generalist treatment with MCTS-only plan_cost bonus weight=2.0 seed=${seed}" \
    --config.mcts.plan_cost_search_bonus_weight=2.0
done
