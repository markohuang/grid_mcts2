# Experiments

Experiment log for AlphaZero-style MCTS training on neutral atom reconfiguration.

Each round is a self-contained document with hypotheses, exact commands, results, and analysis.
Run outputs live in `outputs/<run_id>/` with per-epoch `metrics.jsonl` and `config.json`.

## Rounds

| Round | Date | Focus | Key finding |
|-------|------|-------|-------------|
| [01](01_sanity_checks.md) | 2026-03-12 | Baseline sanity checks | MCTS + learning both work; policy entropy collapses to 0 prematurely |

## Known bottlenecks

1. **Policy entropy collapse** — the network converges to a near-deterministic policy within 5 epochs across all learning runs. Root cause: few simulations produce peaked visit count distributions → peaked training targets → peaked policy → even more peaked visits (positive feedback loop). Only Dirichlet noise at root provides residual diversity.

## Proposed next experiments (not yet run)

These address the entropy collapse bottleneck:

1. **Policy target temperature** — apply temperature to MCTS visit count distribution before using as training target: `softmax(log(visits) / τ)`. Directly breaks the feedback loop.
2. **Entropy bonus in policy loss** — add `- β * H(π)` to penalize low-entropy policies. Range to test: β = 0.01 to 0.1.
3. **More self-play games per epoch** — current default is 5 games/epoch. More games = more diverse replay buffer = slower policy convergence.

These are orthogonal and composable. Test individually first, then combine winners.

## Best results so far

| Map | Best cost | Lower bound | Config | Run ID |
|-----|-----------|-------------|--------|--------|
| 0 (2x6, 9q) | 13 | 6 | high explore (α=0.3, frac=0.5), 10 sims, 15 epochs | 55f9f5d5 |
