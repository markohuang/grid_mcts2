#!/bin/bash
# Round 05: Breaking the avg_cost plateau
# Run from repo root: bash experiments/05_avg_cost_plateau.sh

PYTHON=".venv/bin/python"
SHARED="--config.map_num=1 \
  --config.mcts.root_dirichlet_alpha=0.3 \
  --config.mcts.root_exploration_fraction=0.5 \
  --config.network.value_min=-25.0 \
  --config.network.value_max=25.0 \
  --config.training.num_selfplay=40 \
  --config.training.num_parallel_games=8 \
  --config.training.epochs=30 \
  --config.experiment.early_stopping_patience=20"

echo "=== Phase A: Address entropy collapse ==="

echo "--- Exp 1: Baseline (Round 04 scale-up config) ---"
$PYTHON main.py $SHARED \
  --config.mcts.num_simulations=50 \
  --config.training.batch_size=256 \
  --config.training.training_steps=400

echo "--- Exp 2: Policy target temp τ=2.0 ---"
$PYTHON main.py $SHARED \
  --config.mcts.num_simulations=50 \
  --config.training.batch_size=256 \
  --config.training.training_steps=400 \
  --config.training.policy_target_temperature=2.0

echo "--- Exp 3: Policy target temp τ=4.0 ---"
$PYTHON main.py $SHARED \
  --config.mcts.num_simulations=50 \
  --config.training.batch_size=256 \
  --config.training.training_steps=400 \
  --config.training.policy_target_temperature=4.0

echo "--- Exp 4: Policy entropy weight β=0.01 ---"
$PYTHON main.py $SHARED \
  --config.mcts.num_simulations=50 \
  --config.training.batch_size=256 \
  --config.training.training_steps=400 \
  --config.training.policy_entropy_weight=0.01

echo "=== Phase B: Address signal dilution ==="

echo "--- Exp 5: Best Phase A + buffer_size=10000 ---"
$PYTHON main.py $SHARED \
  --config.mcts.num_simulations=50 \
  --config.training.batch_size=256 \
  --config.training.training_steps=400 \
  --config.training.buffer_size=10000

echo "--- Exp 6: Best Phase A + lr=0.0001 ---"
$PYTHON main.py $SHARED \
  --config.mcts.num_simulations=50 \
  --config.training.batch_size=256 \
  --config.training.training_steps=400 \
  --config.training.lr=0.0001

echo "--- Exp 7: Best Phase A + 100 sims ---"
$PYTHON main.py $SHARED \
  --config.mcts.num_simulations=100 \
  --config.training.batch_size=256 \
  --config.training.training_steps=400

echo "=== Done ==="
