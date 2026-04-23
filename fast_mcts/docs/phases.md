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

## Phase 2 — transposition cache (scouting complete, ready to build)

### Scouting: cache_estimator.py

Before building the cache, we measured **cache-hit-rate upper bound** by
instrumenting `NNBackend.add`: record `(features.tobytes(), current_qubit)`
per request, count unique keys vs total. Since the NN is deterministic in
eval mode, any duplicate key is a guaranteed cache hit — tells us exactly
what a cache would save.

Run: `python -m fast_mcts.cache_estimator --sweep` (~60s).

### Results (fake net unless noted)

| scenario | total req | unique | hit% | within-game | cross-game | projected × |
|----------|-----------|--------|------|-------------|------------|-------------|
| m0 25sim 1game | 496 | 342 | 31.0% | 31.0% | 0.0% | 1.28× |
| m0 25sim 3games | 1488 | 342 | 77.0% | 31.0% | 46.0% | 2.17× |
| m0 25sim 10games | 4960 | 342 | 93.1% | 31.0% | 62.1% | 2.87× |
| m0 100sim 1game | 1874 | 1163 | 37.9% | 37.9% | 0.0% | 1.36× |
| m0 100sim 10games | 18740 | 1163 | 93.8% | 37.9% | 55.9% | 2.91× |
| m2 25sim 1game | 543 | 479 | 11.8% | 11.8% | 0.0% | 1.09× |
| m2 25sim 10games | 5430 | 479 | 91.2% | 11.8% | 79.4% | 2.76× |
| m2 100sim 1game | 2060 | 1712 | 16.9% | 16.9% | 0.0% | 1.13× |
| m2 100sim 3games | 6180 | 1712 | 72.3% | 16.9% | 55.4% | 2.02× |
| m2 100sim 10games | 20600 | 1712 | 91.7% | 16.9% | 74.8% | 2.79× |
| m2 200sim 10games | 40560 | 3292 | 91.9% | 18.8% | 73.0% | 2.80× |
| m2 100sim **batch=32 vl=1.5** | 1720 | 342 | **80.1%** | 0.6% | 79.5% | 2.28× |
| real-net m0 25sim 3games | 1485 | 332 | 77.6% | 32.9% | 44.7% | 2.19× |

Projected speedup assumes NN = 70% of wall (real-net CPU baseline). Cache
overhead ~1 µs per hit vs ~1.3 ms per NN call → negligible.

### Findings

1. **Within-game hits scale with sims × map complexity.** Small map (m0) has
   31-38% within-game; medium map (m2) has 12-19%. More sims per move →
   slightly more tree overlap, but not dominant.

2. **Cross-game hits dominate at ≥3 games** (46-80%). Fixed-map games
   re-explore the same early-layer states every game. By game 10 the
   state space is saturated (342 unique on m0, 1712-3292 on m2).

3. **Virtual loss destroys within-game hits.** batch=32/vl=1.5 drops
   within-game from 16.9% → 0.6% — VL's whole job is to force different
   paths, and it works. **Cross-game hits unaffected** (79.5%), and that's
   what matters in a training run with many self-play games.

4. **Random-board mode (`v4`) is the worst case.** Fresh map each game →
   no cross-game reuse → only within-game (≤20%) → 1.1-1.2× speedup.
   Probably not worth building the cache just for this.

5. **Fixed-map self-play (eval, specialist training, iterative refinement)
   is the best case.** 2-3× speedup on top of Phase 1.

6. **Memory cheap.** Worst case we saw (m2 200sim, 10 games) = 3292
   unique × ~900 bytes/entry = 3 MB. 8×8 maps could reach 100k entries
   (~90 MB) — still trivial. LRU cap at 50k-200k safe.

### Phase 2 design

**Two-layer cache**, both wrapping `NNBackend`:

- **L1 (per-run_mcts)**: dict reset each `compute_blocking`, handles the
  collision dedup we already do. Marginal — the within-game rate is 1-38%
  depending on map; mostly covered by L2 anyway.
- **L2 (persistent)**: LRU (`collections.OrderedDict`) keyed by
  `(map_id, features.tobytes(), current_qubit)`. Lives across games of
  the same map. Invalidated when `cfg.random_board = True` (each game =
  new map).

**Alternative — per-map cache handle**: trainer / self-play driver creates
one `NNCache(map_id, cap=50_000)`, passes into `play_game`. Fast backend
reads it. Lifetime explicit, no global state. Preferred.

**Integration point**: subclass `NNBackend` or compose:

```python
class CachedNNBackend(NNBackend):
    def __init__(self, network, cache, *, cap):
        super().__init__(network, cap=cap)
        self._cache = cache   # or None to disable
    def add(self, features, current_qubit):
        if self._cache is not None:
            key = (features.tobytes(), int(current_qubit))
            hit = self._cache.get(key)
            if hit is not None:
                slot = super().add(features, current_qubit)
                self._slots_out[slot] = hit   # FETCHED_IMMEDIATELY
                return slot
        return super().add(features, current_qubit)
    def compute_blocking(self):
        # collect keys for newly-computed slots, insert into cache after.
        ...
```

**Key choice**: `features.tobytes()` keeps it simple and exact. A smarter
version keys on `(board, tasks_done)` alone and caches the full per-qubit
policy tensor, slicing `current_qubit` on lookup — would raise hit rate
further (value head output is qubit-independent). Defer to Phase 2b.

### Parity contract for Phase 2

- Cache on / off must produce byte-identical output at `batch=1, vl=0`.
  NN is deterministic → cache hit = re-evaluation, same bytes.
- Parametrize `test_cross_classic_vs_fast_defaults` over `use_cache ∈
  {True, False}` — new scenarios, same assertion.
- Cache can never see `prior_mix_weight > 0` (phase 1 already rejects).
- Gumbel path (`_gumbel_simulate` calls `network.inference` directly)
  bypasses our cache in Phase 2. Cache only helps pUCT path. Note in docs.

### Go/no-go

**Go.** Expected 2-3× on fixed-map training workloads (which is most of
our real training). 1.1-1.3× on random-board. Low implementation risk
(well-understood LRU + dict). Low memory.

### Implementation (delivered)

- `fast_mcts/nn_cache.py` — `NNCache` LRU (OrderedDict, `cap`, hits/misses/
  insertions/evictions, `make_feature_key`).
- `fast_mcts/nn_backend.py` — `cache=None` kwarg on `NNBackend`. On `add`:
  lookup, pre-fill slot on hit (`cache_hits` counter), queue + register
  key on miss. On `compute_blocking`: skip pre-filled slots, insert new
  results into cache.
- `fast_mcts/search.py` — `cache` kwarg plumbed through `play_game` and
  `run_mcts_batched`. Root inference uses same cache via `_root_inference`
  helper. Default `None` = Phase 1 behaviour unchanged.
- `fast_mcts/test_cache.py` — 18 tests, 3 layers:
  - LRU semantics (7 tests): get/put/miss, eviction, MRU promotion, clear,
    overwrite, feature-key stability.
  - Backend integration (2): hit byte-equal to miss, no-cache matches Phase 1.
  - Full-game (9): byte-parity cache-on vs cache-off (4 combos), cache-on
    vs classic (2 combos), invariants with cache at batch=8 vl=1, cross-game
    persistence, cap enforcement.

### Bench (map=2, sims=100, batch=32, vl=1.5, real net, CPU, 10 games)

| mode | cache | hit rate | wall | g/s | speedup |
|------|-------|----------|------|-----|---------|
| deterministic | off | — | 7.29s | 1.37 | 1.00× |
| deterministic | cap=50k | **90.6%** | 4.86s | 2.06 | **1.50×** |
| noise (production) | off | — | 7.10s | 1.41 | 1.00× |
| noise (production) | cap=50k | 11.6% | 6.57s | 1.52 | 1.08× |

### Key finding: cache value depends on Dirichlet noise

Deterministic self-play (eval mode, no noise) reaches 90.6% hit rate
(matches estimator's 91.7%) → **1.5× speedup on top of Phase 1's 5.35×,
cumulative 8× vs classic**.

Production self-play (Dirichlet on at root) gets only 11.6% hit rate.
Root noise is different every game → root priors differ → trees explore
different branches → leaf states rarely repeat across games. Cross-game
reuse (the dominant 79% from estimator) assumed shared descent — noise
breaks that assumption.

**Implications:**
- **Eval / ablation / specialist training with noise_off** → full 1.5×
  cache win. Bake into `eval_checkpoint.py` / `pipeline_eval.py`.
- **Production self-play with noise** → 5-12% wall win. Still net positive
  (cache overhead ~1 µs / hit, NN saves ~1.3 ms), but not transformative.
- **Within-game hits preserved** even with noise, because a single tree's
  UCB descent produces some collisions. That's the 5-12% we do see.

### Parity status: ALL GREEN

- 62/62 fast_mcts tests pass (44 Phase 1 + 18 Phase 2)
- Cross-backend byte-parity (classic ≡ fast cache=on) at safe knobs: 2/2 combos
- Cache-on vs cache-off byte-parity: 4/4 combos
- Invariants + full-game completion under cache: all pass
- Runtime: 21.77s for quick suite (was 16.35s pre-cache) +5s for 18 new tests
- Runtime w/ stat-parity: ~1m47s (unchanged)

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
