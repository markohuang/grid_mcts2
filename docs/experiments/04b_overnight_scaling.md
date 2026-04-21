# Round 04b: Overnight Scaling Experiments

## Findings from Round 04

| Map | Best | Avg | Sims | Epochs | No-reconfig | LB |
|-----|------|-----|------|--------|-------------|-----|
| 0 | 11 | 13.3 | 50 | 20 | 16 | 6 |
| 1 | **7** | **7.0** | 50 | 20 | 18 | 6 |
| 1 | **7** | **7.0** | 100 | 15 | 18 | 6 |

Key observations:
- Map 1 is essentially solved (avg=7.0, lb=6)
- Map 0 plateaus at cost=11 with gap to lb=6 — but board has only 3 empty cells (branching=4)
- 25 sims + learning (avg=8.0) outperforms 50 sims + learning (avg=12.1) on Map 1 at 15 epochs — an anomaly
- Map 2 has not been tested yet

## Hypotheses

**H1: Map 0 plateau is due to tight board, not insufficient training.**
The 2x6 board with 9 atoms has only 3 empty cells. Moving one atom requires a vacant spot, creating a sliding-puzzle dynamic where multi-step plans are needed just to free up space. 50 sims with branching=4 gives ~12 visits per action — decent coverage. The bottleneck may be the planning horizon, not the search width.

**H2: More epochs will break the Map 0 plateau.**
The learning curve on Map 0 was still improving at epoch 20 (avg went 21.9→13.3). With 50+ epochs, the value network may learn the sliding-puzzle cost structure well enough to guide MCTS deeper.

**H3: Higher sims help Map 0 more than Map 1.**
Map 0's tighter board creates deeper dependencies between placements. More sims → deeper search → better plans for the sliding puzzle.

**H4: Map 2 (5x5, 12q) is tractable.**
Board has 13 empty cells (branching=14), similar to Map 1's 8 empty. Episode length is 24 (same as Map 1). Cost structure is similar. The system should find good solutions with 50-100 sims.

**H5: The 25-sim anomaly on Map 1 is a training dynamics effect.**
With 50 sims, each game takes longer but the policy targets are sharper (more concentrated visit distributions). This may cause the policy to overfit to narrow solutions early, reducing exploration. 25 sims produces softer targets that allow broader exploration.

## Experiments

### 4b.1: Map 0 long runs (break the plateau?)

```bash
# [4b.1a] Map 0, 50 sims, 50 epochs — more training time
../grid_mcts2/.venv/bin/python main.py --config.map_num=0 \
    --config.training.epochs=50 --config.training.num_selfplay=20 \
    --config.mcts.num_simulations=50

# [4b.1b] Map 0, 200 sims, 50 epochs — deeper search
../grid_mcts2/.venv/bin/python main.py --config.map_num=0 \
    --config.training.epochs=50 --config.training.num_selfplay=20 \
    --config.mcts.num_simulations=200

# [4b.1c] Map 0, 50 sims, 50 epochs, more games — bigger replay buffer
../grid_mcts2/.venv/bin/python main.py --config.map_num=0 \
    --config.training.epochs=50 --config.training.num_selfplay=50 \
    --config.mcts.num_simulations=50 --config.training.buffer_size=5000
```

### 4b.2: Map 1 final push (can we hit lb=6?)

```bash
# [4b.2a] Map 1, 100 sims, 50 epochs — can we reach cost=6?
../grid_mcts2/.venv/bin/python main.py --config.map_num=1 \
    --config.training.epochs=50 --config.training.num_selfplay=20 \
    --config.mcts.num_simulations=100
```

### 4b.3: Map 2 first attempt

```bash
# [4b.3a] Map 2, 50 sims, 30 epochs — first attempt
../grid_mcts2/.venv/bin/python main.py --config.map_num=2 \
    --config.training.epochs=30 --config.training.num_selfplay=20 \
    --config.mcts.num_simulations=50

# [4b.3b] Map 2, 100 sims, 30 epochs — more search
../grid_mcts2/.venv/bin/python main.py --config.map_num=2 \
    --config.training.epochs=30 --config.training.num_selfplay=20 \
    --config.mcts.num_simulations=100
```

### 4b.4: 25 vs 50 sim anomaly investigation (Map 1)

```bash
# [4b.4a] Map 1, 25 sims, 30 epochs — does 25 sims stay ahead?
../grid_mcts2/.venv/bin/python main.py --config.map_num=1 \
    --config.training.epochs=30 --config.training.num_selfplay=20 \
    --config.mcts.num_simulations=25

# [4b.4b] Map 1, 50 sims, 30 epochs — does 50 sims catch up with more time?
../grid_mcts2/.venv/bin/python main.py --config.map_num=1 \
    --config.training.epochs=30 --config.training.num_selfplay=20 \
    --config.mcts.num_simulations=50
```

## Success criteria

- [ ] Map 0 best_cost < 11 (any improvement)
- [ ] Map 0 best_cost < 9 (significant breakthrough)
- [ ] Map 1 best_cost = 6 (lower bound — perfect solution)
- [ ] Map 2 best_cost < 18 (beats no-reconfig baseline)
- [ ] Map 2 best_cost < 12 (competitive with Maps 0/1)
- [ ] 25 vs 50 sim anomaly resolved
