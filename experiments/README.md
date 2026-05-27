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
| [04](04_value_pretraining.md) | 2026-03-13 | Value pretraining + critical bug fixes | Value range [-10,10]→[-25,25] was THE fix; avg_cost < 18 first time |
| [05](05_avg_cost_plateau.md) | 2026-03-13 | Breaking the avg_cost plateau | *planned — not yet run* |

## Known bottlenecks

1. **avg_cost plateau.** With the value range fix, MCTS *search* finds cost=13 solutions (best_cost), but the *policy* doesn't learn to reproduce them (avg_cost ~22). The best solutions come from lucky MCTS rollouts, not from the policy consistently choosing good moves. The policy is exploring (gate_frac ~40%) but not converging toward the optimal reconfig sequence.

2. **Sparse cost landscape on Map 1.** Random boards produce only 3 distinct cost values: 12 (19%), 18 (56%), 24 (26%). This makes value learning and pretraining difficult — there's no smooth gradient to follow. Moves that change cost are rare and dramatic (18→13 from a single qubit relocation).

3. **Credit assignment for multi-step reconfigs.** Even with correct value ranges, the network struggles to learn that specific multi-step move sequences reduce cost. Most individual moves have zero or negative reward; only specific qubit relocations produce the 18→13 jump.

## Proposed next experiments (not yet run)

1. **Test on Maps 0 and 2** — verify the value range fix generalizes beyond Map 1.
2. **Random baseline (FakeNet)** — run FakeNet with many games to confirm the network adds value over pure random MCTS exploration.
3. **Policy distillation / imitation** — train on the best solutions found so far as expert trajectories, then continue self-play from that policy.
4. **Action space pruning** — restrict moves to gate-relevant qubits or neighboring cells to reduce branching factor.
5. **Curriculum learning** — start with 1 task layer, then increase.
6. **Longer training** — the learning curve in Round 04 Exp 1 (25 sims) hadn't plateaued at epoch 30. Scale-up runs plateau earlier at avg~22, suggesting more sims help search but not policy learning.

## Best results so far

| Map | Best cost | Lower bound | Config | Run ID |
|-----|-----------|-------------|--------|--------|
| 0 (2x6, 9q) | 13 | 6 | high explore (α=0.3, frac=0.5), 10 sims, 15 epochs | 55f9f5d5 |
| 1 (4x4, 8q) | 13 | 6 | value fix [-25,25], 50 sims, 40 games, 8 parallel, 19 epochs | afdfeb26 |
