#!/bin/bash
# Round 04: Layer-Level MDP Validation
# Run from project root: bash experiments.sh
#
# Baselines:
#   Map 0 (2x6, 9q): no-reconfig=16, random=~27, lb=6
#   Map 1 (4x4, 8q): no-reconfig=18, random=~33, lb=6
#
# Hypotheses:
#   H1: MCTS with 12-action space finds below-baseline costs even without learning
#   H2: Learning improves costs over epochs
#   H3: More simulations help more than before (smaller branching factor)
#   H4: Policy entropy does NOT collapse (no attractor)

set -e
PYTHON="../grid_mcts2/.venv/bin/python"

echo "=== Round 04: Layer-Level MDP Validation ==="
echo ""

# 4.1: MCTS-only baselines (FakeNet, no learning)
echo "--- 4.1: MCTS-only baselines ---"

echo "[4.1a] Map 0, FakeNet, 50 sims, 100 games"
$PYTHON main.py --config.use_fake=True --config.training.epochs=1 \
    --config.training.num_selfplay=100 --config.mcts.num_simulations=50 \
    2>&1 | grep -E "(Run:|>>|Best|complete)"

echo ""
echo "[4.1b] Map 1, FakeNet, 50 sims, 100 games"
$PYTHON main.py --config.use_fake=True --config.training.epochs=1 \
    --config.training.num_selfplay=100 --config.mcts.num_simulations=50 \
    --config.map_num=1 \
    2>&1 | grep -E "(Run:|>>|Best|complete)"

echo ""
echo "[4.1c] Map 0, FakeNet, 200 sims, 50 games"
$PYTHON main.py --config.use_fake=True --config.training.epochs=1 \
    --config.training.num_selfplay=50 --config.mcts.num_simulations=200 \
    2>&1 | grep -E "(Run:|>>|Best|complete)"

echo ""

# 4.2: Learning baselines
echo "--- 4.2: Learning baselines ---"

echo "[4.2a] Map 0, 20 epochs, 20 games/epoch, 50 sims"
$PYTHON main.py --config.training.epochs=20 --config.training.num_selfplay=20 \
    --config.mcts.num_simulations=50 \
    2>&1 | grep -E "(Run:|>>|loss:|Best|complete)"

echo ""
echo "[4.2b] Map 1, 20 epochs, 20 games/epoch, 50 sims"
$PYTHON main.py --config.training.epochs=20 --config.training.num_selfplay=20 \
    --config.mcts.num_simulations=50 --config.map_num=1 \
    2>&1 | grep -E "(Run:|>>|loss:|Best|complete)"

echo ""

# 4.3: Simulation count comparison (Map 1)
echo "--- 4.3: Simulation count comparison (Map 1) ---"

for SIMS in 10 25 50 100; do
    echo "[4.3] Map 1, 15 epochs, ${SIMS} sims"
    $PYTHON main.py --config.training.epochs=15 --config.training.num_selfplay=20 \
        --config.mcts.num_simulations=$SIMS --config.map_num=1 \
        2>&1 | grep -E "(Run:|>>|loss:|Best|complete)"
    echo ""
done

echo "=== Round 04 complete ==="
echo "Check results: cat outputs/run_registry.jsonl"
