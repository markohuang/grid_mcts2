# Round 02: Addressing Policy Entropy Collapse

**Date:** 2026-03-12
**Map:** 1 (4x4 board, 8 qubits, 3 identical task layers, 4 gates each)
**Cost bounds:** lb=6, ub=24. Action space: 129.

## Goal

Address the policy entropy collapse discovered in Round 01. The network converges to a near-deterministic policy within 5 epochs, limiting MCTS exploration and preventing discovery of better solutions. Test three orthogonal interventions on a harder map (Map 1) to see which keeps entropy alive while maintaining or improving solution quality.

## Hypotheses

- **H1 — Policy target temperature helps.** Applying temperature τ>1 to visit count distributions before using as training targets should produce softer targets, slowing the peaked-policy feedback loop. Predicted: higher τ = higher entropy at epoch 20, possibly at the cost of slower convergence.
- **H2 — Entropy bonus helps.** Adding `- β * H(π)` to the loss directly penalizes low-entropy policies. Predicted: nonzero β maintains entropy > 0, but too-high β may hurt solution quality.
- **H3 — More self-play games help.** More diverse data per epoch slows the policy's convergence to a single mode. Predicted: 25 games/epoch > 10 games/epoch in entropy and cost.
- **H4 — Combining fixes is best.** Target temperature + entropy bonus should compose well. Predicted: combined > either alone.
- **H5 — Fixes transfer to harder maps.** These interventions should work on Map 1 (larger action space, requires actual reconfiguration) not just Map 0.

## Setup

All experiments use high exploration from Round 01 findings (α=0.3, frac=0.5).

| Parameter | Value |
|-----------|-------|
| `map_num` | 1 |
| `mcts.num_simulations` | 25 |
| `mcts.root_dirichlet_alpha` | 0.3 |
| `mcts.root_exploration_fraction` | 0.5 |
| `training.epochs` | 20 |
| `training.num_selfplay` | 10 |
| `training.batch_size` | 64 |
| `training.training_steps` | 100 |
| `experiment.early_stopping_patience` | 20 |

## Experiments

### Exp 1 — Baseline (high exploration only)

```bash
.venv/bin/python main.py \
  --config.map_num=1 \
  --config.mcts.num_simulations=25 \
  --config.mcts.root_dirichlet_alpha=0.3 \
  --config.mcts.root_exploration_fraction=0.5 \
  --config.training.epochs=20 \
  --config.training.num_selfplay=10 \
  --config.training.batch_size=64 \
  --config.training.training_steps=100 \
  --config.experiment.early_stopping_patience=20
```

### Exp 2 — Target temperature τ=2.0

```bash
# Same as Exp 1, plus:
  --config.training.policy_target_temperature=2.0
```

### Exp 3 — Target temperature τ=4.0

```bash
# Same as Exp 1, plus:
  --config.training.policy_target_temperature=4.0
```

### Exp 4 — Entropy bonus β=0.01

```bash
# Same as Exp 1, plus:
  --config.training.policy_entropy_weight=0.01
```

### Exp 5 — Entropy bonus β=0.05

```bash
# Same as Exp 1, plus:
  --config.training.policy_entropy_weight=0.05
```

### Exp 6 — More self-play: 25 games/epoch

```bash
# Same as Exp 1, plus:
  --config.training.num_selfplay=25
```

### Exp 7 — Combined: τ=2.0 + β=0.01

```bash
# Same as Exp 1, plus:
  --config.training.policy_target_temperature=2.0 \
  --config.training.policy_entropy_weight=0.01
```

## Results

| # | Experiment | best_cost | avg_cost (final) | completion (final) | avg_steps (final) | pi_entropy (final) | entropy @ epoch 5 |
|---|-----------|-----------|-----------------|-------------------|-------------------|-------------------|-------------------|
| 1 | Baseline | **13** | 18.0 | 100% | 3.0 | 0.00 | 0.08 |
| 2 | τ=2.0 | 17 | 18.0 | 100% | 3.0 | 0.02 | 0.03 |
| 3 | τ=4.0 | **15** | 18.0 | 100% | 3.0 | 1.46 | 1.53 |
| 4 | β=0.01 | 18 | 18.0 | 100% | 3.0 | 0.00 | 0.01 |
| 5 | β=0.05 | **15** | 18.0 | 100% | 3.0 | 0.00 | 0.06 |
| 6 | 25 games | 17 | 20.0 | 100% | 5.0 | 0.00 | 0.02 |
| 7 | τ=2.0+β=0.01 | 18 | 18.0 | 100% | 3.0 | 0.02 | 0.04 |

Run IDs: 8d77b799, ea259be0, c3582133, 180a8b51, 2702626b, ac830768, 359f56aa.

### Convergence timeline (epochs to 100% completion)

| Exp | Epochs to 100% | Notes |
|-----|-----------------|-------|
| 1 Baseline | 7 | Then locks to avg_cost=18 |
| 2 τ=2.0 | 6 | Fastest convergence |
| 3 τ=4.0 | 7 | But unstable — drops to 70-90% in middle epochs |
| 4 β=0.01 | 6 | Fast convergence |
| 5 β=0.05 | 7 | |
| 6 25 games | 7 | Never fully locks — avg_steps stays ~5, avg_cost ~20 |
| 7 τ=2.0+β=0.01 | 6 | Same as τ=2.0 alone |

## Analysis

### H1: Policy target temperature — Partially confirmed

τ=2.0 (Exp 2) maintains entropy at 0.02 instead of 0.00 — a small improvement. It didn't help cost (best=17 vs baseline best=13).

τ=4.0 (Exp 3) successfully maintains high entropy (~1.46-1.50 throughout). However, convergence is much slower and less stable — completion fluctuates between 70-100% even late in training. It found cost=15 but avg_cost stayed high. **Too much softening prevents the network from committing to good actions.**

Verdict: τ=2.0 has negligible effect. τ=4.0 keeps entropy alive but slows learning too much. There may be a sweet spot around τ=2.5-3.0 but the tradeoff is clear: higher entropy = slower convergence.

### H2: Entropy bonus — Not confirmed

β=0.01 (Exp 4): entropy still collapses to 0.00 by epoch 6. The bonus is too weak to overcome the peaked-target feedback loop. best_cost=18 (worst of all experiments).

β=0.05 (Exp 5): also collapses to 0.00 by epoch 7. Found cost=15 early (epoch 5) but couldn't sustain exploration. The entropy bonus operates on the *network's* output, but the *training targets* are already peaked — so the gradient from the cross-entropy loss dominates the entropy bonus.

Verdict: entropy bonus alone doesn't work because the training targets are the root cause. The bonus fights the loss signal rather than addressing the source of peaked targets.

### H3: More self-play — Mixed

25 games/epoch (Exp 6) showed interesting behavior: it never locked into the deterministic avg_steps=3.0 pattern. avg_steps stayed around 4.5-6 and avg_cost around 19-21 even at epoch 20. Entropy still collapses to 0.00, but the policy learned a *different* deterministic strategy — one that occasionally tries a reconfig move.

best_cost=17, worse than baseline (13). More data diversity didn't help with the entropy problem or with finding better solutions.

Verdict: more self-play helps with learning stability but doesn't address entropy collapse. It produces different convergence dynamics but not better solutions.

### H4: Combining fixes — Not confirmed

τ=2.0 + β=0.01 (Exp 7) performed essentially the same as τ=2.0 alone: entropy at 0.02, best_cost=18. The combination didn't produce a synergy.

### H5: Transfer to harder map — Partially confirmed

All interventions worked mechanically on Map 1 (no crashes, completion achieved). The entropy fixes showed the same directional effects as expected. But the *underlying problem is deeper than entropy collapse*:

**The real finding: all runs converge to cost=18 (avg_cost=18, avg_steps=3.0).**

Cost=18 = 3 layers × 6 per layer (2 × 3 gate parallel groups). This means the network learns to just execute gate layers immediately without any reconfiguration — exactly the same pattern as Map 0. The cost=13 found in the baseline (Exp 1, epoch 5) was a lucky early find before the policy collapsed, not a learned strategy.

### Unexpected findings

**The "execute immediately" attractor.** On both maps, the network converges to the trivial strategy of just pressing "execute gates" 3 times (avg_steps=3.0). This gives a valid but suboptimal solution. Reconfiguration moves that would reduce gate parallelism cost are never explored once the policy locks in, because:
1. The immediate reward from executing gates is positive (completes the episode)
2. Reconfig moves have zero or negative immediate reward (no cost reduction until gates execute)
3. The value network would need to predict that reconfig moves lead to *future* cost reductions — a multi-step credit assignment problem

**Entropy fixes miss the real issue.** The problem isn't just that the policy entropy is low — it's that the reward signal doesn't incentivize multi-step planning through reconfiguration. Even τ=4.0, which maintains high entropy, converges to avg_cost=18.0. The policy explores more actions but still settles on "just execute."

**Early lucky finds.** Baseline found cost=13 at epoch 5 (80% completion) and β=0.05 found cost=15 at epoch 5 — both before the policy fully converged. These come from Dirichlet noise + incomplete learning, not from a good policy.

## Next steps

The entropy collapse fixes are insufficient on their own. The deeper issue is **reward shaping for multi-step reconfiguration planning**. Options:

1. **Longer MCTS lookahead** — more simulations so search can discover that reconfig moves pay off later. Current 25 sims may be too few for 129-action space.
2. **Reward shaping for reconfig moves** — give a small positive reward for moves that reduce future gate parallelism cost, even before executing.
3. **Curriculum learning** — start with 1 task layer (where the optimal strategy is simpler), then increase.
4. **Higher Dirichlet noise + τ=3.0** — combine moderate target softening with aggressive noise to keep finding lucky solutions and learning from them.
5. **Investigate cost=13 solution** — what reconfig moves did it use? This could inform reward shaping.
