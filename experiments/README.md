# Experiments

Experiment log for AlphaZero-style MCTS training on neutral atom reconfiguration.

Each round is a self-contained document with hypotheses, exact commands, results, and analysis.
Run outputs live in `outputs/<run_id>/` with per-epoch `metrics.jsonl` and `config.json`.

## Rounds

| Round | Date | Focus | Key finding |
|-------|------|-------|-------------|
| [01](01_sanity_checks.md) | 2026-03-12 | Baseline sanity checks | MCTS + learning both work; policy entropy collapses to 0 prematurely |
| [02](02_entropy_collapse.md) | 2026-03-12 | Entropy collapse fixes on Map 1 | Target temp and entropy bonus don't solve the real issue — the "execute immediately" attractor |
| [03](03_alphadev_fixes_and_scaleup.md) | 2026-03-13 | AlphaDev alignment fixes + scale-up | Value range [-10,10] clips all targets; no learning beyond trivial |
| [04](04_value_pretraining.md) | 2026-03-13 | Value pretraining + critical bug fixes | _In progress_ |

## Known bottlenecks

1. **"Execute immediately" attractor.** On all maps tested, the network converges to the trivial strategy of just executing gate layers without reconfiguration (avg_steps=3.0, cost=18 on Map 1). Reconfig moves have zero/negative immediate reward, so the network never learns that they can reduce future gate parallelism cost. This is a multi-step credit assignment problem.

2. **Policy entropy collapse** (secondary to #1). The network converges to a near-deterministic policy within 5-10 epochs. Policy target temperature (τ=4.0) can keep entropy alive but doesn't improve costs — the policy explores but still converges to "just execute." Entropy bonus (β≤0.05) is too weak to fight peaked training targets.

## Proposed next experiments (not yet run)

1. **More MCTS simulations** — 25 sims may be too few for 129-action space. Try 50-100 sims.
2. **Reward shaping for reconfig** — positive reward for moves that reduce future gate cost.
3. **Curriculum learning** — start with 1 task layer, then increase.
4. **Investigate cost=13 solution** — understand what reconfiguration was found to inform approach.

## Best results so far

| Map | Best cost | Lower bound | Config | Run ID |
|-----|-----------|-------------|--------|--------|
| 0 (2x6, 9q) | 13 | 6 | high explore (α=0.3, frac=0.5), 10 sims, 15 epochs | 55f9f5d5 |
| 1 (4x4, 8q) | 13 | 6 | high explore, 25 sims, 20 epochs (lucky early find) | 8d77b799 |
