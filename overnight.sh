#!/bin/bash
# Round 04b: Overnight scaling experiments
# Estimated time: 6-10 hours on A6000
set -e
PYTHON="../grid_mcts2/.venv/bin/python"

echo "=== Round 04b: Overnight Scaling ==="
echo "Started: $(date)"
echo ""

# 4b.1a: Map 0 long run — can we break the plateau at cost=11?
echo "[4b.1a] Map 0, 50 sims, 50 epochs"
$PYTHON main.py --config.map_num=0 \
    --config.training.epochs=50 --config.training.num_selfplay=20 \
    --config.mcts.num_simulations=50 \
    2>&1 | grep -E "(Run:|>>|loss:|Best|complete)"
echo ""

# 4b.1b: Map 0 deep search — 200 sims for the sliding puzzle
echo "[4b.1b] Map 0, 200 sims, 50 epochs"
$PYTHON main.py --config.map_num=0 \
    --config.training.epochs=50 --config.training.num_selfplay=20 \
    --config.mcts.num_simulations=200 \
    2>&1 | grep -E "(Run:|>>|loss:|Best|complete)"
echo ""

# 4b.2a: Map 1 final push — hunting for cost=6
echo "[4b.2a] Map 1, 100 sims, 50 epochs"
$PYTHON main.py --config.map_num=1 \
    --config.training.epochs=50 --config.training.num_selfplay=20 \
    --config.mcts.num_simulations=100 \
    2>&1 | grep -E "(Run:|>>|loss:|Best|complete)"
echo ""

# 4b.3a: Map 2 first attempt
echo "[4b.3a] Map 2, 50 sims, 30 epochs"
$PYTHON main.py --config.map_num=2 \
    --config.training.epochs=30 --config.training.num_selfplay=20 \
    --config.mcts.num_simulations=50 \
    2>&1 | grep -E "(Run:|>>|loss:|Best|complete)"
echo ""

# 4b.3b: Map 2 more search
echo "[4b.3b] Map 2, 100 sims, 30 epochs"
$PYTHON main.py --config.map_num=2 \
    --config.training.epochs=30 --config.training.num_selfplay=20 \
    --config.mcts.num_simulations=100 \
    2>&1 | grep -E "(Run:|>>|loss:|Best|complete)"
echo ""

# 4b.4a: 25 vs 50 sim anomaly — 25 sims extended
echo "[4b.4a] Map 1, 25 sims, 30 epochs (anomaly check)"
$PYTHON main.py --config.map_num=1 \
    --config.training.epochs=30 --config.training.num_selfplay=20 \
    --config.mcts.num_simulations=25 \
    2>&1 | grep -E "(Run:|>>|loss:|Best|complete)"
echo ""

# 4b.4b: 25 vs 50 sim anomaly — 50 sims extended
echo "[4b.4b] Map 1, 50 sims, 30 epochs (anomaly check)"
$PYTHON main.py --config.map_num=1 \
    --config.training.epochs=30 --config.training.num_selfplay=20 \
    --config.mcts.num_simulations=50 \
    2>&1 | grep -E "(Run:|>>|loss:|Best|complete)"
echo ""

echo "=== Round 04b complete ==="
echo "Finished: $(date)"
echo "Results: cat outputs/run_registry.jsonl"
