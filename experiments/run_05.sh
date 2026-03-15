#!/bin/bash
# Round 05: Specialist vs Generalist on 8x8
set -e
PYTHON="../grid_mcts2/.venv/bin/python"

echo "=== Round 05: Specialist vs Generalist ==="
echo "Started: $(date)"
echo ""

echo "[5.1] Specialist: Map 3 only"
$PYTHON main.py --config.map_num=3 \
    --config.training.epochs=30 --config.training.num_selfplay=20 \
    --config.mcts.num_simulations=50 --config.training.buffer_size=5000 \
    2>&1 | grep -E "(Run:|>>|loss:|Best|complete)"
echo ""

echo "[5.2] Generalist: random 8x8 boards"
$PYTHON main.py --config.map_num=3 --config.random_board=True \
    --config.training.epochs=30 --config.training.num_selfplay=20 \
    --config.mcts.num_simulations=50 --config.training.buffer_size=5000 \
    2>&1 | grep -E "(Run:|>>|loss:|Best|complete)"
echo ""

echo "=== Round 05 complete ==="
echo "Finished: $(date)"
