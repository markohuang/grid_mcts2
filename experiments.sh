#!/bin/bash
# Sanity check experiments on Map 0 (2x6, 9 qubits, 3 tasks)
# Cost bounds: lb=6 (gate-only best), ub=22 (gate-only worst)
# Total time estimate: ~30-40 minutes
#
# Hypotheses:
#   H1: MCTS search alone provides value (more sims = better, even with random policy)
#   H2: Learned policy/value improves over random (trained net beats FakeNet)
#   H3: Exploration-exploitation balance matters (Dirichlet noise tuning)
#   H4: Network capacity affects learning quality
#   H5: Value estimates improve with training (root values become calibrated)
#
# Key metrics per run (in outputs/<run_id>/metrics.jsonl):
#   - best_cost, avg_cost, completion_rate  (solution quality)
#   - avg_root_value                        (H5: value calibration)
#   - avg_policy_entropy                    (H3: search focus)
#   - train_policy, train_correctness, train_latency  (H4: loss components)

set -e
PYTHON=.venv/bin/python

# Common small-experiment settings
SELFPLAY=5
BATCH=32
PATIENCE=20  # effectively disabled for short runs

echo "=========================================="
echo "  H1: Does MCTS search help? (no learning)"
echo "=========================================="

echo "[1/8] FakeNet baseline, 10 sims"
$PYTHON main.py \
  --config.use_fake=True \
  --config.mcts.num_simulations=10 \
  --config.training.epochs=1 \
  --config.training.num_selfplay=30 \
  --config.experiment.early_stopping_patience=$PATIENCE \
  2>&1 | grep -E "(Run:|>>|best_cost|complete)"

echo ""
echo "[2/8] FakeNet baseline, 50 sims"
$PYTHON main.py \
  --config.use_fake=True \
  --config.mcts.num_simulations=50 \
  --config.training.epochs=1 \
  --config.training.num_selfplay=30 \
  --config.experiment.early_stopping_patience=$PATIENCE \
  2>&1 | grep -E "(Run:|>>|best_cost|complete)"

echo ""
echo "=========================================="
echo "  H2: Does learning improve over random?"
echo "=========================================="

echo "[3/8] Learning with 10 sims, 15 epochs"
$PYTHON main.py \
  --config.mcts.num_simulations=10 \
  --config.training.epochs=15 \
  --config.training.num_selfplay=$SELFPLAY \
  --config.training.batch_size=$BATCH \
  --config.training.training_steps=50 \
  --config.experiment.early_stopping_patience=$PATIENCE \
  2>&1 | grep -E "(Run:|>>|loss:|Best solution|complete)"

echo ""
echo "[4/8] Learning with 25 sims, 10 epochs"
$PYTHON main.py \
  --config.mcts.num_simulations=25 \
  --config.training.epochs=10 \
  --config.training.num_selfplay=$SELFPLAY \
  --config.training.batch_size=$BATCH \
  --config.training.training_steps=50 \
  --config.experiment.early_stopping_patience=$PATIENCE \
  2>&1 | grep -E "(Run:|>>|loss:|Best solution|complete)"

echo ""
echo "=========================================="
echo "  H3: Exploration-exploitation balance"
echo "=========================================="

echo "[5/8] High exploration (alpha=0.3, frac=0.5)"
$PYTHON main.py \
  --config.mcts.num_simulations=10 \
  --config.mcts.root_dirichlet_alpha=0.3 \
  --config.mcts.root_exploration_fraction=0.5 \
  --config.training.epochs=15 \
  --config.training.num_selfplay=$SELFPLAY \
  --config.training.batch_size=$BATCH \
  --config.training.training_steps=50 \
  --config.experiment.early_stopping_patience=$PATIENCE \
  2>&1 | grep -E "(Run:|>>|loss:|Best solution|complete)"

echo ""
echo "[6/8] Low exploration (alpha=0.003, frac=0.1)"
$PYTHON main.py \
  --config.mcts.num_simulations=10 \
  --config.mcts.root_dirichlet_alpha=0.003 \
  --config.mcts.root_exploration_fraction=0.1 \
  --config.training.epochs=15 \
  --config.training.num_selfplay=$SELFPLAY \
  --config.training.batch_size=$BATCH \
  --config.training.training_steps=50 \
  --config.experiment.early_stopping_patience=$PATIENCE \
  2>&1 | grep -E "(Run:|>>|loss:|Best solution|complete)"

echo ""
echo "=========================================="
echo "  H4: Network capacity"
echo "=========================================="

echo "[7/8] Small network (v_hsize=32, p_hsize=16)"
$PYTHON main.py \
  --config.mcts.num_simulations=10 \
  --config.network.v_hsize=32 \
  --config.network.p_hsize=16 \
  --config.training.epochs=15 \
  --config.training.num_selfplay=$SELFPLAY \
  --config.training.batch_size=$BATCH \
  --config.training.training_steps=50 \
  --config.experiment.early_stopping_patience=$PATIENCE \
  2>&1 | grep -E "(Run:|>>|loss:|Best solution|complete|Trainable)"

echo ""
echo "[8/8] Large network (v_hsize=128, p_hsize=64)"
$PYTHON main.py \
  --config.mcts.num_simulations=10 \
  --config.network.v_hsize=128 \
  --config.network.p_hsize=64 \
  --config.training.epochs=15 \
  --config.training.num_selfplay=$SELFPLAY \
  --config.training.batch_size=$BATCH \
  --config.training.training_steps=50 \
  --config.experiment.early_stopping_patience=$PATIENCE \
  2>&1 | grep -E "(Run:|>>|loss:|Best solution|complete|Trainable)"

echo ""
echo "=========================================="
echo "  All experiments complete!"
echo "  Results: cat outputs/run_registry.jsonl"
echo "=========================================="
