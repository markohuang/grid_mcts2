#!/bin/bash
set -euo pipefail

# Round 04: Value Pretraining + Critical Bug Fixes
# Run from repo root: bash experiments/04_value_pretraining.sh

SHARED="--config.map_num=1 \
  --config.mcts.root_dirichlet_alpha=0.3 \
  --config.mcts.root_exploration_fraction=0.5 \
  --config.training.epochs=30 \
  --config.experiment.early_stopping_patience=15 \
  --config.network.value_min=-25.0 \
  --config.network.value_max=25.0"

SMALL="--config.mcts.num_simulations=25 \
  --config.training.num_selfplay=10 \
  --config.training.batch_size=64 \
  --config.training.training_steps=100"

echo "=========================================="
echo "Phase A: Isolate effects (sequential)"
echo "=========================================="

echo ""
echo "--- Exp 1: Value range fix only ---"
.venv/bin/python main.py $SHARED $SMALL

echo ""
echo "--- Exp 2: + lw>cw ---"
.venv/bin/python main.py $SHARED $SMALL \
  --config.network.correctness_weight=0.5 \
  --config.network.latency_weight=2.0

echo ""
echo "--- Exp 3: + pretrain ---"
.venv/bin/python main.py $SHARED $SMALL \
  --config.network.correctness_weight=0.5 \
  --config.network.latency_weight=2.0 \
  --config.training.pretrain_value_steps=500

echo ""
echo "--- Exp 4: + aux loss ---"
.venv/bin/python main.py $SHARED $SMALL \
  --config.network.correctness_weight=0.5 \
  --config.network.latency_weight=2.0 \
  --config.training.pretrain_value_steps=500 \
  --config.training.aux_value_weight=0.5

echo "=========================================="
echo "Phase B: Scale (using best from Phase A)"
echo "=========================================="

echo ""
echo "--- Exp 5: Scaled ---"
.venv/bin/python main.py $SHARED \
  --config.mcts.num_simulations=50 \
  --config.training.num_selfplay=40 \
  --config.training.num_parallel_games=8 \
  --config.training.batch_size=256 \
  --config.training.training_steps=400 \
  --config.training.buffer_size=5000 \
  --config.network.correctness_weight=0.5 \
  --config.network.latency_weight=2.0 \
  --config.training.pretrain_value_steps=500 \
  --config.training.aux_value_weight=0.5

echo ""
echo "--- Exp 6: + gate_only ---"
.venv/bin/python main.py $SHARED \
  --config.env.reward_mode=gate_only \
  --config.mcts.num_simulations=50 \
  --config.training.num_selfplay=40 \
  --config.training.num_parallel_games=8 \
  --config.training.batch_size=256 \
  --config.training.training_steps=400 \
  --config.training.buffer_size=5000 \
  --config.network.correctness_weight=0.5 \
  --config.network.latency_weight=2.0 \
  --config.training.pretrain_value_steps=500 \
  --config.training.aux_value_weight=0.5

echo ""
echo "=========================================="
echo "All experiments complete. Run: python analyze.py"
echo "=========================================="
