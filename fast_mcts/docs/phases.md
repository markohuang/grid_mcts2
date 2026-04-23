# Phases — per-phase log

Per-phase: scope, design decisions, bench numbers, bugs caught, next steps.

---

## Phase 0 — bench harness + test scaffolding

**Goal**: profile classic MCTS to identify biggest lever. Build test
infrastructure before touching control flow.

### Delivered

- `fast_mcts/backends.py` — pluggable backend registry; `classic` wraps
  `neutral_atoms.mcts.play_game` unchanged.
- `fast_mcts/bench.py` — monkey-patch timing harness. Hooks onto
  `NeutralAtomsEnv.clone`, `step`, `get_features`, `legal_actions`;
  `Network.inference`; `mcts._expand_node`, `_backpropagate`, `_select_child`,
  `run_mcts`. Reverts patches on exit.
- `fast_mcts/_common.py` — shared test scaffolding (`seed_all`, `build_cfg`,
  `build_env_spec`, `make_network`, `play_once`, `game_snapshot`,
  `assert_snapshot_equal`).
- `fast_mcts/test_parity.py` — self-parity across 10 config branches
  (multi-map, fake+real, noise on/off, temperature decay, prior-mix, sim scan).
- `fast_mcts/test_invariants.py` — tree + full-game structural invariants.
- `fast_mcts/test_snapshot.py` — 6-scenario regression hashes.
- `fast_mcts/baseline.json` — bench numbers for later comparison.

### Classic baseline (CPU, 1 thread)

| Config | Wall | g/s | NN % | env.step % |
|--------|------|-----|------|------------|
| map=2, fake, 25 sims, 2 games | 0.44s | 4.55 | 11.9% | 38.7% |
| map=2, real, 25 sims, 2 games | 1.90s | 1.05 | **73.2%** | 12.1% |
| map=2, fake, 100 sims, 3 games | 2.81s | 1.07 | 10.0% | **45.7%** |
| map=2, real, 100 sims, 3 games | **18.63s** | 0.27 | **70.5%** | 14.8% |

**Verdict**: with real NN at 100 sims, **10,643 NN calls at batch=1, 1.23 ms
each, 70.5% of wall**. Leaf batching is the highest-leverage target.

Everything else < 5%: `mcts._select_child` 5.0%, `env.legal_actions` 2.9%,
`env.get_features` 1.8%, `env.clone` 1.0%, `mcts._expand_node` 0.8%,
`mcts._backpropagate` 0.3%.

---

## Phase 1 — leaf-gather + virtual loss (lc0 BackendComputation analog)

**Goal**: collapse the 70.5% NN bottleneck by stacking K leaves into one
forward pass. Preserve AlphaDev semantics exactly at safe knobs.

### Delivered

- `fast_mcts/nn_backend.py::NNBackend` — `add` / `compute_blocking` / `get` /
  `reset` / `pending_size`. Two paths:
  - `cap == 1` or `use_fake` → per-slot `Network.inference(obs, aslist=True)`.
    Byte-exact equivalent to classic.
  - `cap > 1` and real net → `nnet(stacked_features, current_qubit=stacked_idx)`
    once, split outputs. ~1e-7 drift vs per-sample, not bit-exact (used for
    production, not parity).
- `fast_mcts/search.py::run_mcts_batched` — gather up to `nn_batch_size`
  leaves with virtual loss applied to each path, one `compute_blocking`,
  revert VL + expand + backup each path in submission order.
- `fast_mcts/search.py::play_game` — matches classic `play_game` signature.
  Registered as backend `fast` in `backends.py`.

### Config knobs (read from `cfg.mcts` with `getattr` fallback — no edit to
`neutral_atoms/config.py`)

- `nn_batch_size: int = 1` — gather cap
- `virtual_loss: float = 0.0` — subtracted from value_sum on in-flight paths

### Parity contract

`nn_batch_size=1, virtual_loss=0, prior_mix_weight=0` → **byte-identical** to
classic. Enforced by `test_cross_classic_vs_fast_defaults` across 6 combos.

Fast asserts `prior_mix_weight == 0` (Phase 1 doesn't batch the prior-mix
lookahead — use classic if you need it).

### Bugs caught during development

#### Bug 1: gather loop doesn't yield on terminal leaves

**Symptom**: `test_cross_classic_vs_fast_defaults[0-10-False]` failed. All
10 sims in one `run_mcts` call descended before any backup → UCB state
stale → every sim picked the same child.

**Root cause**: terminal leaves don't call `backend.add`, so
`backend.pending_size()` stayed 0, so the gather loop condition
`backend.pending_size() < nn_batch_size` kept firing.

**Fix** (`search.py:115`): count all pending leaves (terminal + NN-backed)
toward batch cap: `while len(pending) < nn_batch_size and ...`.

#### Bug 2: terminal leaf `_expand_node` not called

**Symptom**: after fix 1, still diverged. Step 20 visit counts matched but
root `value_sum` differed by exactly 16.0.

**Root cause**: classic calls `_expand_node(node, legal=[], ...)` even for
terminals. `_expand_node` early-returns on empty `actions` **but not before
`node.reward = reward`**. That reward is load-bearing in `_backpropagate`'s
`value = node.reward + discount * value` — without it, leaves propagate 0
instead of e.g. −3 on the way up, shifting sibling Q values and flipping
UCB on the next sim.

**Fix** (`search.py:158`): call `_expand_node(req.leaf, [], None,
req.leaf_reward, ...)` in the terminal branch of commit.

After fix: all 6 cross-backend parity combos pass byte-identical.

### Bench results (CPU, 1 thread, map=2, 100 sims, 3 games, real net)

| Config | Wall | g/s | NN calls | NN % | vs classic |
|--------|------|-----|----------|------|-----------|
| classic | 11.40s | 0.26 | 6411 | 70.7% | 1.00× |
| fast, batch=1, vl=0 (parity) | 11.39s | 0.26 | 6284 | 69.8% | 1.00× (≡ classic) |
| fast, batch=16, vl=1.0 | 2.36s | 1.27 | 347 | 19.7% | **4.83×** |
| fast, batch=32, vl=1.5 | 2.13s | 1.41 | 148 | 9.0% | **5.35×** |

NN calls dropped 43× at batch=32 (6411 → 148). `env.step` + `env.legal_actions`
are now the dominant buckets (45% combined) — Phase 3 targets.

### Statistical parity (real net)

Paired design, 20 games per seed, noise on.

| Scenario | classic mean | fast mean | diff | z |
|----------|--------------|-----------|------|---|
| map0, sims=25, batch=8, vl=1.0 | 26.70 | 27.25 | +0.55 | 0.74 |
| map0, sims=25, batch=16, vl=1.0 | 26.70 | 25.55 | −1.15 | −1.37 |
| map0, sims=25, batch=16, vl=2.0 | 26.70 | 25.15 | −1.55 | −1.62 |

All within noise floor. With informative policy priors, batched descent does
not degrade quality. Fake net shows measurable degradation at the same knobs
(z=3-5) — inherent VL/uniform-prior interaction, not a bug.

### Non-goals (for Phase 1)

- Prior-mix lookahead batching — deferred to Phase 1b.
- Cross-game batching (multiple games share one `NNBackend`) — deferred.
  In practice we still get strong batching within a single game because
  each `run_mcts` call has num_simulations leaves to gather.
- GPU tuning — all bench was CPU. Expect 2-4× further speedup on GPU from
  kernel-launch amortization.

### Files touched

New under `fast_mcts/`:
- `nn_backend.py` — 105 lines
- `search.py` — 190 lines
- `backends.py` — updated to register `fast`
- `test_parity.py` — added cross-backend + fast self-parity
- `test_invariants.py` — added fast batched scenarios
- `test_stat_parity.py` — new
- `baseline.json` — appended `phase1_results`
- `docs/` — this directory

Zero lines modified outside `fast_mcts/`.

---

## Phase 2 — transposition cache (planned)

Keyed by `(board.tobytes(), tasks_done, current_atom_idx)`. Cache hit →
skip GPU entirely (`FETCHED_IMMEDIATELY` analog). LRU cap ~200k entries.
Expected win: 20-40% on repeated states in long games.

---

## Phase 3 — vec env (planned)

Kill per-sim `env.clone()`. Either:
- Batched env: stack boards / positions across N sims, vectorized step.
- Or: undo-stack on single env (`step` + `reverse_step`) — avoid clone entirely.

Expected to collapse the `env.step` (28-38%) + `env.clone` (4%) + `env.legal_actions`
(12%) buckets that dominate after Phase 1.

---

## Phase 4 — flat-array tree (planned)

Replace `Node` + dict-children with preallocated arrays (`N[node]`,
`W[node]`, `P[node, action]`, `child_idx[node, action]`, `R[node]`).
UCB vectorized per-node across actions. Removes Python object churn.

---

## Phase 5 — native backend (optional, probably not needed)

Drop NN into TorchScript / `torch.compile`. Lower inference latency.

Full-C++ (bind lc0 `search/classic` via pybind11 + custom GameState) —
only if Phase 1-4 exhausted.
