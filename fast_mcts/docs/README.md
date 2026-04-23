# fast_mcts — progress log

Efficient batched MCTS backend. Non-invasive alternative to `neutral_atoms.mcts`:
same `Game` / `Network` / `NeutralAtomsEnv` interfaces, same replay-buffer
output format. Existing `neutral_atoms/` is untouched; `backend='fast'` flips
control flow only.

## Reading order

1. [`philosophy.md`](philosophy.md) — AlphaDev + lc0 alignment rules we enforce
2. [`tests.md`](tests.md) — 4-layer correctness coverage + how to run
3. [`phases.md`](phases.md) — per-phase design, bench numbers, bugs, next steps
4. [`gumbel_integration.md`](gumbel_integration.md) — design for `fast_gumbel` backend (implement after classic Gumbel A/B validates)

## Status

| Phase | Scope | Status | Speedup (real net, map=2, 100 sims, CPU) |
|-------|-------|--------|------------------------------------------|
| 0 | Bench harness + classic baseline + test scaffolding | done | 1.00× (baseline) |
| 1 | Leaf-gather + virtual loss (lc0 BackendComputation analog) | done | **5.35×** |
| 2 | NN transposition cache (FETCHED_IMMEDIATELY short-circuit) | **done** | **1.50× deterministic / 1.08× noise-on** (on top of Phase 1) |
| 3 | Vec env — drop per-sim `env.clone()` | planned | — |
| 4 | Flat-array tree (parent/child idx tensors) | planned | — |

## Contract

For **any** knob combination, fast backend must:
1. Produce a completed Game (`tasks_done == num_tasks`, `len(history) == episode_length`).
2. Satisfy tree structural invariants (visit sums, legal children, priors sum=1).
3. Not systematically degrade search quality vs classic on real net (stat-parity test).

For the **safe knob set** (`nn_batch_size=1, virtual_loss=0, prior_mix_weight=0`),
fast must be **byte-identical** to classic. That's the parity contract.

## Test suite

| File | Scope | Runtime |
|------|-------|---------|
| `test_parity.py` | Self-parity (classic + fast) + cross-backend byte-identity | ~10s |
| `test_invariants.py` | Tree + full-game structural invariants, both backends | ~5s |
| `test_snapshot.py` | Regression hashes for classic at 6 fixed scenarios | ~4s |
| `test_stat_parity.py` | Paired mean-cost check, real net | ~90s |
| **total pytest fast_mcts/** | 48 tests | **1m47s** |

Quick dev loop (no stat parity): 44 tests, 19s.
