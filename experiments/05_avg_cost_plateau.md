# Round 05: Breaking the avg_cost Plateau

**Date:** 2026-03-13
**Map:** 1 (4x4, 8q, 3 identical layers, 4 gates each)
**Cost bounds:** lb=6, ub=24. Action space: 129.

## Context

Round 04 fixed the value range bug ([-10,10] → [-25,25]), enabling the first real learning. Scale-up runs reach best_cost=13 (1 move, 18→13) but **avg_cost plateaus at ~22** — the policy doesn't learn to reproduce the best solutions.

### Diagnosis from afdfeb26 (best scale-up run)

| Metric | Epoch 1 | Epoch 10 | Epoch 19 |
|--------|---------|----------|----------|
| avg_cost | 25.7 | 21.4 | 22.1 |
| best_cost | 17 | 18 | 13 |
| root_value | -1.6 | 16.2 | 19.7 |
| entropy | 1.09 | 0.11 | 0.04 |
| gate_frac | 9% | 45% | 45% |

Three problems:

1. **Policy entropy collapse (again).** Entropy drops from 1.09 to 0.04 by epoch 17. The policy is near-deterministic, locked into a fixed move pattern that includes ~45% moves — but not the RIGHT moves. Unlike Round 01-02 collapse (to 100% execute), this time it collapses to a mixed strategy that happens to include random moves.

2. **Value head overconfidence.** Root value climbs from -1.6 to 19.7, but actual game outcomes stay at avg_cost=22. The value head thinks the position is worth 20 when the policy actually achieves 22. This reduces MCTS exploration — high confidence means less branching.

3. **Rare signal dilution.** Cost=13 games appear in ~15% of epochs (best_cost=13 in 7/19 epochs). With 40 games/epoch, that's maybe 1-3 optimal games. Training on 40 games where 37 are mediocre and 3 are good → the policy mostly learns the mediocre strategy.

### Why avg_cost is 22 (not 18 or 13)

Cost=22 corresponds to games that make 2-4 RANDOM moves before executing. The policy learned "do some moves" (correct) but not "do THIS specific move" (not learned). With 128 possible move actions and near-zero entropy, the policy picks a specific suboptimal move and repeats it.

Cost=13 requires a SPECIFIC qubit relocation (e.g., q6 → (3,0)). This is 1 action out of 128 moves. The probability of the policy converging to exactly this action is low without stronger signal.

## Hypotheses

- **H1 — Policy target temperature prevents entropy collapse.** Softening MCTS visit count targets with τ>1 keeps the policy from locking into a single move. Predicted: entropy stays above 0.1 throughout training.

- **H2 — Larger replay buffer preserves rare good games.** Current buffer_size=1000. With 40 games * ~5 steps = 200 samples/epoch, a cost=13 game's transitions get flushed within ~5 epochs. With buffer_size=10000, they persist longer and get sampled more. Predicted: avg_cost decreases more steadily.

- **H3 — More training steps extract more signal per epoch.** Current 400 training steps with batch_size=256 = 102k samples seen per epoch. With the buffer at ~2000, each sample is seen ~50x. More steps or larger buffer may help the policy learn rare events. But may also overfit.

- **H4 — Lower learning rate prevents overshooting.** Current lr=0.0002. The value head overshoots (root_val=20 when avg_cost=22). Lower lr may allow more gradual learning. Predicted: slower but more stable improvement.

## Experiments

### Shared parameters

```
--config.map_num=1
--config.mcts.root_dirichlet_alpha=0.3
--config.mcts.root_exploration_fraction=0.5
--config.network.value_min=-25.0
--config.network.value_max=25.0
--config.training.num_selfplay=40
--config.training.num_parallel_games=8
--config.training.epochs=30
--config.experiment.early_stopping_patience=20
```

### Phase A: Address entropy collapse

| # | Experiment | Key change |
|---|-----------|------------|
| 1 | Baseline (Round 04 Exp 5 config) | 50 sims, batch=256, steps=400 |
| 2 | + policy target temp τ=2.0 | Softer training targets |
| 3 | + policy target temp τ=4.0 | Even softer targets |
| 4 | + policy entropy weight β=0.01 | Direct entropy bonus in loss |

### Phase B: Address signal dilution

| # | Experiment | Key change |
|---|-----------|------------|
| 5 | Best Phase A + buffer_size=10000 | Preserve rare good games longer |
| 6 | Best Phase A + lr=0.0001 | Slower, more stable learning |
| 7 | Best Phase A + 100 sims | More search budget to find good moves |

## What to look for

1. **entropy trajectory** — does it stay above 0.1? Does τ prevent collapse?
2. **avg_cost trend** — does it decrease steadily or plateau?
3. **root_value vs avg_cost gap** — is the value head calibrated?
4. **best_cost timing** — does cost=13 appear earlier with more sims?
5. **Per-epoch cost distribution** — are we getting more cost<18 games or just the same rare lucky ones?
