# Round 05: Specialist vs Generalist on 8x8

## Question

Does training on diverse random boards improve or hurt performance on a specific fixed board?

## Setup

**Benchmark**: Map 3 (8x8, 20q, 5 layers, seed=42)
- No-reconfig cost: 62
- Lower bound: 10 (2 × 5 layers)
- FakeNet baseline: ~105-110

**Specialist**: trained only on Map 3 (fixed board)
**Generalist**: trained on random 8x8 boards (20q, 5 layers, different seed each game)

Both use identical hyperparameters, only the training distribution differs.

## Experiments

### 5.1: Specialist baseline

```bash
../grid_mcts2/.venv/bin/python main.py --config.map_num=3 \
    --config.training.epochs=30 --config.training.num_selfplay=20 \
    --config.mcts.num_simulations=50 --config.training.buffer_size=5000
```

### 5.2: Generalist

```bash
../grid_mcts2/.venv/bin/python main.py --config.map_num=3 --config.random_board=True \
    --config.training.epochs=30 --config.training.num_selfplay=20 \
    --config.mcts.num_simulations=50 --config.training.buffer_size=5000
```

### 5.3: Evaluate generalist on fixed Map 3

After generalist training completes, evaluate its checkpoint on Map 3
(the generalist sees random boards during training but is evaluated on the fixed one).

## Hypotheses

**H1**: Specialist converges faster on Map 3 (overfits to the specific instance).
**H2**: Generalist may achieve comparable or better final cost (learns transferable structure).
**H3**: Generalist has more stable training (less variance across epochs, less overfitting).

## Success criteria

- [ ] Both runs complete and converge
- [ ] Specialist best_cost < 60 on Map 3
- [ ] Generalist best_cost < 70 on Map 3 (competitive despite diversity)
- [ ] Compare learning curves: specialist fast start vs generalist steady improvement
