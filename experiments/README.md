# Experiments

Experiment log for AlphaZero-style MCTS training on neutral atom reconfiguration.

Each round is a self-contained document with hypotheses, exact commands, results, and analysis.
Run outputs live in `outputs/<run_id>/` with per-epoch `metrics.jsonl` and `config.json`.

## Rounds

| Round | Date | Focus | Key finding |
|-------|------|-------|-------------|
| [01](01_sanity_checks.md) | 2026-03-12 | Baseline sanity checks | MCTS + learning both work; policy entropy collapses to 0 prematurely |
| [02](02_entropy_collapse.md) | 2026-03-12 | Entropy collapse fixes on Map 1 | Target temp and entropy bonus don't solve the real issue — the "execute immediately" attractor |
| [03](03_alphadev_fixes_and_scaleup.md) | 2026-03-13 | AlphaDev alignment fixes + scale-up | Bootstrap, two-hot, value bins fixes |
| **04** (planned) | TBD | **Layer-level MDP validation** | First experiments with new per-qubit placement MDP |

## Architecture change: Layer-Level MDP

Between Round 03 and Round 04, the environment was restructured:

- **Old MDP**: action = GATE_ACTION or move(qubit, cell), 129-action space, budget-bounded
- **New MDP**: action = cell index for current qubit, 12-16 action space, deterministic episode length

Key properties:
- 100% completion rate by construction (no budget, no gate action to choose)
- Branching factor ~12 vs ~129
- "Execute immediately" attractor eliminated (no gate action exists)
- Episodes are deterministic length: `sum(k_t)` where `k_t` = relevant atoms per layer

Baseline costs with random play (FakeNet):
- Map 0: ~25-29 (no-reconfig baseline: 16, lower bound: 6)
- Map 1: ~32-33 (no-reconfig baseline: 18, lower bound: 6)

## Previous best results (old MDP, for reference)

| Map | Best cost | Lower bound | Notes |
|-----|-----------|-------------|-------|
| 0 (2x6, 9q) | 13 | 6 | Old MDP, 129-action space |
| 1 (4x4, 8q) | 13 | 6 | Old MDP, lucky early find |

These are not directly comparable to the new MDP results since the cost computation and action encoding changed.
