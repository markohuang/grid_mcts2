# Round 04: Layer-Level MDP Validation

## Context

The environment was restructured from the old MDP (129 actions, budget-bounded, GATE_ACTION attractor) to a **layer-level per-qubit placement** MDP (board_size actions, deterministic length, no gate action).

This round validates the new MDP: can MCTS + learning find good solutions now that the attractor is eliminated?

## Baselines

| Map | Board | No-reconfig cost | Random play cost | Lower bound | Action space | Episode length |
|-----|-------|------------------|------------------|-------------|--------------|----------------|
| 0 | 2x6, 9q | 16 | ~27 | 6 | 12 | 22 |
| 1 | 4x4, 8q | 18 | ~33 | 6 | 16 | 24 |
| 2 | 5x5, 12q | 18 | ~?? | 6 | 25 | 24 |

**No-reconfig cost** = cost of executing all layers without moving any atoms (all no-ops). This is the "do nothing" baseline — any cost below this means the agent found useful reconfigurations.

## Hypotheses

1. **MCTS alone (no learning)** should find solutions below the no-reconfig baseline within a few games, because the branching factor is small enough for 50 sims to explore meaningfully.
2. **MCTS + learning** should improve costs over epochs as the network learns to guide search.
3. **Policy entropy should NOT collapse** — there's no attractor, so the agent must actually explore to find good placements.
4. **More simulations should help more** than in the old MDP — each sim explores one of ~12 options (useful) vs one of ~129 (mostly noise).

## Experiments

### 4.1: MCTS-only baseline (no learning, FakeNet)

Purpose: establish what pure MCTS can achieve with the new branching factor.

```bash
# Map 0 — 50 sims, 100 games
../grid_mcts2/.venv/bin/python main.py --config.use_fake=True --config.training.epochs=1 --config.training.num_selfplay=100 --config.mcts.num_simulations=50

# Map 1 — 50 sims, 100 games
../grid_mcts2/.venv/bin/python main.py --config.use_fake=True --config.training.epochs=1 --config.training.num_selfplay=100 --config.mcts.num_simulations=50 --config.map_num=1

# Map 0 — 200 sims, 50 games (more search depth)
../grid_mcts2/.venv/bin/python main.py --config.use_fake=True --config.training.epochs=1 --config.training.num_selfplay=50 --config.mcts.num_simulations=200
```

Key metrics: best_cost, avg_cost, avg_policy_entropy.

### 4.2: Learning baseline (small training)

Purpose: verify the learning loop works end-to-end and cost improves over epochs.

```bash
# Map 0 — 20 epochs, 20 games/epoch, 50 sims
../grid_mcts2/.venv/bin/python main.py --config.training.epochs=20 --config.training.num_selfplay=20 --config.mcts.num_simulations=50

# Map 1 — same
../grid_mcts2/.venv/bin/python main.py --config.training.epochs=20 --config.training.num_selfplay=20 --config.mcts.num_simulations=50 --config.map_num=1
```

Key metrics: best_cost trend over epochs, loss convergence, policy entropy over time.

### 4.3: Simulation count comparison

Purpose: quantify how much more effective simulations are with the smaller branching factor.

```bash
# Map 1: 10, 25, 50, 100 sims
../grid_mcts2/.venv/bin/python main.py --config.training.epochs=15 --config.training.num_selfplay=20 --config.mcts.num_simulations=10
../grid_mcts2/.venv/bin/python main.py --config.training.epochs=15 --config.training.num_selfplay=20 --config.mcts.num_simulations=25
../grid_mcts2/.venv/bin/python main.py --config.training.epochs=15 --config.training.num_selfplay=20 --config.mcts.num_simulations=50
../grid_mcts2/.venv/bin/python main.py --config.training.epochs=15 --config.training.num_selfplay=20 --config.mcts.num_simulations=100
```

### 4.4: Temperature schedule sweep

Purpose: test whether the new linear decay (2.0 → 0.25 over 1000 steps) is appropriate.

```bash
# High exploration throughout
../grid_mcts2/.venv/bin/python main.py --config.training.epochs=15 --config.mcts.temperature_init=2.0 --config.mcts.temperature_final=1.0 --config.mcts.temperature_decay_steps=2000

# Fast convergence
../grid_mcts2/.venv/bin/python main.py --config.training.epochs=15 --config.mcts.temperature_init=1.0 --config.mcts.temperature_final=0.1 --config.mcts.temperature_decay_steps=500
```

## Success criteria

- [ ] **Sanity**: FakeNet runs at 100% completion, costs > 0, costs decrease with more sims
- [ ] **Learning works**: best_cost improves over epochs (loss decreases)
- [ ] **Beat no-reconfig**: best_cost < 16 on Map 0, < 18 on Map 1
- [ ] **No entropy collapse**: policy entropy stays > 0.3 through training
- [ ] **Cost below old best**: best_cost < 13 on either map would be a breakthrough

## Analysis script

```bash
# Quick analysis of a run
../grid_mcts2/.venv/bin/python -c "
import json, sys
run_dir = sys.argv[1]
with open(f'{run_dir}/metrics.jsonl') as f:
    metrics = [json.loads(l) for l in f]
for m in metrics:
    parts = [f'epoch={m[\"epoch\"]}']
    if 'best_cost' in m: parts.append(f'best={m[\"best_cost\"]}')
    if 'avg_cost' in m: parts.append(f'avg={m[\"avg_cost\"]:.1f}')
    if 'avg_policy_entropy' in m: parts.append(f'entropy={m[\"avg_policy_entropy\"]:.2f}')
    if 'train_total' in m: parts.append(f'loss={m[\"train_total\"]:.3f}')
    print(', '.join(parts))
" outputs/<run_id>
```
