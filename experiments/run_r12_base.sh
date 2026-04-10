#!/bin/bash
# Round 12 Base: Fresh specialist with new arch + 250 sims + p_hsize=64
# This produces the starting checkpoint for all sequential runs.
set -e

PYTHON="../grid_mcts2/.venv/bin/python"

echo "=== R12 Base Specialist (250 sims, p_hsize=64) ==="

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
CUDA_VISIBLE_DEVICES=0 \
$PYTHON main.py \
  --config.map_num=2 \
  --config.training.epochs=200 --config.training.num_selfplay=50 \
  --config.training.batch_size=128 --config.training.seed=42 \
  --config.mcts.num_simulations=250 \
  --config.mcts.root_dirichlet_alpha=0.3 \
  --config.training.data_augmentation=True \
  --config.env.reward_mode=layer_delta \
  --config.mcts.plan_cost_search_bonus_weight=2.0 \
  --config.network.p_hsize=64 \
  --config.experiment.early_stopping_patience=100 \
  --config.experiment.study=round12_sequential_v2 \
  --config.experiment.hypothesis=base_specialist \
  --config.experiment.variant=fixed_250s_p64 \
  --config.experiment.tags=specialist,base,250sims,p64 \
  --config.experiment.notes="Fresh specialist: new arch, p_hsize=64, 250 sims, layer_delta+bonus"
