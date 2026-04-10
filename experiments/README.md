# Experiments

Experiment log for AlphaZero-style MCTS training on neutral atom reconfiguration.

Each round is a self-contained document with hypotheses, exact commands, results, and analysis.
Run outputs live in `outputs/<run_id>/` with per-epoch `metrics.jsonl` and `config.json`.

Tracking support:

- `TRACKING_SYSTEM.md` — workflow for linking hypotheses to runs
- `core_mcts_tracker.md` — live backlog / decision ledger
- `templates/core_mcts_round_template.md` — new round template

## Rounds

| Round | Date | Focus | Key finding |
|-------|------|-------|-------------|
| [01](01_sanity_checks.md) | 2026-03-12 | Baseline sanity checks | MCTS + learning both work; policy entropy collapses to 0 prematurely |
| [02](02_entropy_collapse.md) | 2026-03-12 | Entropy collapse fixes on Map 1 | Target temp and entropy bonus don't solve the real issue — the "execute immediately" attractor |
| [03](03_alphadev_fixes_and_scaleup.md) | 2026-03-13 | AlphaDev alignment fixes + scale-up | Bootstrap, two-hot, value bins fixes |
| [04](04_layer_mdp_validation.md) | 2026-03-13 | Layer-level MDP validation | New MDP works: Map 1 best=7 (lb=6), Map 0 best=11, 100% completion |
| [04b](04b_overnight_scaling.md) | 2026-03-14 | Overnight scaling on all maps | Map 2 (5x5) first tested; Map 1 near-solved; Map 0 limited by 3 empty cells |
| [05](05_specialist_vs_generalist.md) | 2026-03-14 | Specialist vs generalist on 8x8 | Does training on diverse random boards help or hurt fixed-map performance? |
| [06](06_curriculum_augmentation_reward_modes.md) | 2026-03-23 | Curriculum + augmentation + reward modes + exploration | map2=**11** (sims=200); alpha=0.3 best generalization (map1=13); Option G too slow from pretrained ckpt |
| [07](07_generalist_and_reward_modes.md) | 2026-03-24 | Generalist training + reward mode shootout from scratch | `plan_cost` stayed stronger than unbiased alternatives on generalist, while `layer_delta` looked best on fixed-map specialist training |
| [08](08_reward_signal_error_modes.md) | 2026-03-31 | Reward-signal error modes before generalist scale-up | Pure `layer_delta` fixed the local signal story but did not beat `plan_cost` on matched generalist seed42; missing long-horizon search structure became the leading diagnosis |
| [09](09_search_bonus_generalist_5x5.md) | 2026-04-02 | 5x5 generalist follow-up with MCTS-only search bonus | Planned: test whether `layer_delta` plus `plan_cost` search guidance can keep the cleaner value target and recover generalist search depth |
| [10](10_convergence_and_scaling.md) | 2026-04-02 | Convergence studies and scaling indicators | Long runs (500ep), more games/ep (50-100), NN capacity test (139K vs 418K), fixed + generalist with `layer_delta + search_bonus=2.0` |
| [11](11_architecture_validation.md) | 2026-04-09 | Architecture validation (Transformer + cross-positional features) | Benchmark new arch against R10 results on 3 reward modes; test if generalist plateau breaks |

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
