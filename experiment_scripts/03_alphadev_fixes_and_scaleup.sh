#!/bin/bash
set -euo pipefail

# Round 03: AlphaDev Alignment Fixes + Scale-Up
# Run from repo root: bash experiments/03_alphadev_fixes_and_scaleup.sh
#
# Phases:
#   A (Exps 1-3): Sanity checks, sequential, ~10-30 min each
#   B (Exps 4-6): Scale-up, parallel, ~15-60 min each

SHARED="--config.map_num=1 \
  --config.mcts.root_dirichlet_alpha=0.3 \
  --config.mcts.root_exploration_fraction=0.5 \
  --config.training.epochs=30 \
  --config.experiment.early_stopping_patience=15"

echo "=========================================="
echo "Phase A: Sanity checks (sequential)"
echo "=========================================="

echo ""
echo "--- Exp 1: Baseline with fixes ---"
.venv/bin/python main.py $SHARED \
  --config.mcts.num_simulations=25 \
  --config.training.num_selfplay=10 \
  --config.training.batch_size=64 \
  --config.training.training_steps=100

echo ""
echo "--- Exp 2: gate_only reward ---"
.venv/bin/python main.py $SHARED \
  --config.env.reward_mode=gate_only \
  --config.mcts.num_simulations=25 \
  --config.training.num_selfplay=10 \
  --config.training.batch_size=64 \
  --config.training.training_steps=100

echo ""
echo "--- Exp 3: gate_only + AlphaDev weights ---"
.venv/bin/python main.py $SHARED \
  --config.env.reward_mode=gate_only \
  --config.mcts.num_simulations=25 \
  --config.training.num_selfplay=10 \
  --config.training.batch_size=64 \
  --config.training.training_steps=100 \
  --config.network.correctness_weight=2.0 \
  --config.network.latency_weight=0.5

echo "=========================================="
echo "Phase B: Scale-up (parallel)"
echo "=========================================="

echo ""
echo "--- Exp 4: 100 sims ---"
.venv/bin/python main.py $SHARED \
  --config.mcts.num_simulations=100 \
  --config.training.num_selfplay=10 \
  --config.training.batch_size=64 \
  --config.training.training_steps=100

echo ""
echo "--- Exp 5: Parallel + scale ---"
.venv/bin/python main.py $SHARED \
  --config.mcts.num_simulations=50 \
  --config.training.num_selfplay=40 \
  --config.training.num_parallel_games=8 \
  --config.training.batch_size=256 \
  --config.training.training_steps=400 \
  --config.training.buffer_size=5000

echo ""
echo "--- Exp 6: Full scale (gate_only + 100 sims + parallel + weights) ---"
.venv/bin/python main.py $SHARED \
  --config.env.reward_mode=gate_only \
  --config.mcts.num_simulations=100 \
  --config.training.num_selfplay=40 \
  --config.training.num_parallel_games=8 \
  --config.training.batch_size=256 \
  --config.training.training_steps=400 \
  --config.training.buffer_size=5000 \
  --config.network.correctness_weight=2.0 \
  --config.network.latency_weight=0.5

echo ""
echo "=========================================="
echo "All experiments complete. Run: python analyze.py"
echo "=========================================="
