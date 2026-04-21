#!/bin/bash
# Round 02: Addressing policy entropy collapse on Map 1 (4x4, 8 qubits, 3 identical tasks)
# Cost bounds: lb=6, ub=24
# Action space: 129
#
# Hypotheses:
#   H1: Policy target temperature (τ>1) keeps training targets softer, preventing entropy collapse
#   H2: Entropy bonus in loss (β>0) directly penalizes low-entropy policies
#   H3: More self-play games per epoch provide enough data diversity to slow convergence
#   H4: Combining target temperature + entropy bonus yields better results than either alone
#   H5: These fixes transfer from Map 0 to Map 1 (harder map with larger action space)
#
# Key metrics:
#   - best_cost, avg_cost, completion_rate  (solution quality)
#   - avg_policy_entropy                    (primary: does entropy stay above 0?)
#   - avg_root_value                        (value learning)

set -e
PYTHON=.venv/bin/python

# Shared settings
MAP=1
SIMS=25
EPOCHS=20
SELFPLAY=10
BATCH=64
STEPS=100
PATIENCE=20
EXPLORE_ALPHA=0.3
EXPLORE_FRAC=0.5

COMMON="--config.map_num=$MAP \
  --config.mcts.num_simulations=$SIMS \
  --config.mcts.root_dirichlet_alpha=$EXPLORE_ALPHA \
  --config.mcts.root_exploration_fraction=$EXPLORE_FRAC \
  --config.training.epochs=$EPOCHS \
  --config.training.num_selfplay=$SELFPLAY \
  --config.training.batch_size=$BATCH \
  --config.training.training_steps=$STEPS \
  --config.experiment.early_stopping_patience=$PATIENCE"

echo "=========================================="
echo "  Round 02: Entropy Collapse Fixes"
echo "  Map 1: 4x4, 8q, 3 identical tasks"
echo "=========================================="

echo ""
echo "[1/7] Baseline (no fixes, high exploration only)"
$PYTHON main.py $COMMON \
  2>&1 | grep -E "(Run:|>>|Best solution|complete|Trainable)"

echo ""
echo "[2/7] Policy target temperature τ=2.0 (H1)"
$PYTHON main.py $COMMON \
  --config.training.policy_target_temperature=2.0 \
  2>&1 | grep -E "(Run:|>>|Best solution|complete|Trainable)"

echo ""
echo "[3/7] Policy target temperature τ=4.0 (H1)"
$PYTHON main.py $COMMON \
  --config.training.policy_target_temperature=4.0 \
  2>&1 | grep -E "(Run:|>>|Best solution|complete|Trainable)"

echo ""
echo "[4/7] Entropy bonus β=0.01 (H2)"
$PYTHON main.py $COMMON \
  --config.training.policy_entropy_weight=0.01 \
  2>&1 | grep -E "(Run:|>>|Best solution|complete|Trainable)"

echo ""
echo "[5/7] Entropy bonus β=0.05 (H2)"
$PYTHON main.py $COMMON \
  --config.training.policy_entropy_weight=0.05 \
  2>&1 | grep -E "(Run:|>>|Best solution|complete|Trainable)"

echo ""
echo "[6/7] More self-play: 25 games/epoch (H3)"
$PYTHON main.py $COMMON \
  --config.training.num_selfplay=25 \
  2>&1 | grep -E "(Run:|>>|Best solution|complete|Trainable)"

echo ""
echo "[7/7] Combined: τ=2.0 + β=0.01 (H4)"
$PYTHON main.py $COMMON \
  --config.training.policy_target_temperature=2.0 \
  --config.training.policy_entropy_weight=0.01 \
  2>&1 | grep -E "(Run:|>>|Best solution|complete|Trainable)"

echo ""
echo "=========================================="
echo "  Round 02 complete!"
echo "  Results: cat outputs/run_registry.jsonl"
echo "=========================================="
