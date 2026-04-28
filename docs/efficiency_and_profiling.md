# Efficiency and Profiling

Tracks all timing experiments: what was measured, under what conditions, and what it revealed.

---

## Measurement log

### M1 — Phase 1 bench (fast_mcts/bench.py)

**Date:** pre-smoke  
**Script:** `fast_mcts/bench.py`  
**Network:** FakeNet (uniform random priors, CPU)  
**Config:** map=2 (5×5, 12q), sims=100, classic vs fast (batch=1/8/32/64, vl=0/1)  
**Hardware:** CPU (login node / local)

FakeNet has a flat prior — every simulation must query the NN (no trained concentration toward terminals). This is the best case for NN-batching.

| backend | batch | vl | speedup vs classic |
|---|---|---|---|
| classic | — | — | 1.0× |
| fast | 1 | 0.0 | ~1.0× (parity) |
| fast | 32 | 1.0 | **5.35×** |

**What it revealed:** NN batching gives large speedup when NN is the bottleneck (FakeNet, low sims). This is the theoretical ceiling for Phase 1.

---

### M2 — Smoke A: CPU fast selfplay (job 59789765)

**Date:** cycle during fast_mcts integration  
**Script:** `slurm/selfplay_v3a_fast_smoke.sh`  
**Network:** real (v3a01 cycle_05 checkpoint, 5 cycles of training)  
**Config:** map=2, sims=10000, backend=fast, batch=32, vl=1.0, selfplay_device=cpu  
**Hardware:** CPU node, 62 workers (one game each)  
**Baseline:** v3a01 cycle_05 classic/CPU at 537s/game

| metric | value |
|---|---|
| avg_game_time_s | **262s** (2.05× vs 537s classic) |
| avg_cost | 12.82 (vs 12.99 classic — quality maintained) |
| terminal_frac | **84.7%** |
| nn_eff_batch | **5.5 / 32** |
| NN % of game time | **10.6%** |
| env.step % of game time | **89.4%** |

**What it revealed:** With a trained prior, 84.7% of simulations hit terminal nodes and never call the NN. The effective batch fills to only 5.5/32. Phase 1 gives 2× (real), not 5× (FakeNet bench), because the bottleneck shifted from NN to env.step.

---

### M3 — Smoke B: GPU fast selfplay (job 59789771)

**Date:** same session  
**Script:** `slurm/selfplay_v3a_fast_gpu_smoke.sh`  
**Network:** real (v3a01 cycle_05)  
**Config:** map=2, sims=10000, backend=fast, batch=32, vl=1.0, selfplay_device=cuda  
**Hardware:** 1 GPU worker (A100), sequential — clean per-game timing

| metric | value |
|---|---|
| avg_game_time_s | ~363s (0.72× vs CPU fast — **slower**) |
| GPU util (nvidia-smi) | **1%** |
| nn_eff_batch | ~5.5 (same as CPU — terminal_frac unchanged) |
| speedup vs classic | ~1.48× (worse than CPU fast) |

**What it revealed:** GPU selfplay is counterproductive in this regime. NN is 10.6% of game time. Amdahl limit = 1/(0.894 + 0.106/∞) = **1.12×** even with instantaneous GPU compute. Actual result is 0.72× because kernel launch overhead (~50-200 µs/call) exceeds compute savings when batches average only 5.5 tensors.

Key insight: "GPU selfplay" means the NN forward runs on GPU; env.step always runs on CPU. GPU only touches the 10.6% fraction, so it cannot break through the Amdahl ceiling.

---

### M4 — cProfile breakdown (job 59793348)

**Date:** 2026-04-27  
**Script:** `scripts/profile_env.py`  
**Network:** real (v3a01 cycle_05)  
**Config:** map=2, sims=2000, backend=fast, batch=32, vl=1.0, selfplay_device=cpu  
**Hardware:** CPU node (2 cpus, 8G), 1 game  
**Log:** `slurm/logs/profile_env_59793348.out`

Wall time: **150s** for 1 game, 2000 sims, 24 moves.

#### Time breakdown (cumtime)

| component | cumtime | % total | ncalls |
|---|---|---|---|
| `_select_child` (all UCB) | 50.8s | **33.9%** | 403,775 |
| — `_ucb_score` math | 20.7s | 13.8% | 5,652,850 |
| — ml_collections config overhead | 18.0s | **12.0%** | 17,976,188 |
| `env.step` (total) | 56.6s | **37.7%** | 403,799 |
| — `_compute_remaining_cost` | 39.1s | **26.1%** | 807,598 |
| — — `_compute_layer_cost_fast` | 36.2s | 24.1% | 1,344,244 |
| — — `count_groups_fast` (Numba) | 18.0s | 12.0% | 1,671,961 |
| NN forward (`compute_blocking`) | 34.1s | **22.7%** | 1,512 |
| — `torch._transformer_encoder_layer_fwd` | 29.4s | 19.6% | 6,420 |
| `env.clone` | 1.1s | **<1%** | 48,000 |

Note: at 2000 sims, terminal_frac is lower than at 10k sims, so NN is 22.7% here vs 10.6% in Smoke A. The UCB and `_compute_remaining_cost` fractions are proportionally larger at lower sim counts.

#### Key findings

**1. env.clone is negligible (<1%).**  
Prior assumption ("env.step/clone = 89.4%") attributed all non-NN time to env. The clone itself costs almost nothing. The actual costs are inside `_descend`: UCB selection + env.step.

**2. UCB selection (33.9%) splits into two fixable parts:**

- **ml_collections `ConfigDict` attribute lookups (12% of total).**  
  `_ucb_score` is called 5.65M times per game. Each call reads `config.pb_c_base`, `config.pb_c_init`, `config.discount` from `ml_collections.ConfigDict`, which routes through `__getattr__` → `__getitem__` with `isinstance` checks. 17.9M such lookups cost 18s.  
  **Fix:** Extract the three constants once in `_select_child` before the child iteration loop, pass as plain Python floats to `_ucb_score`. **Applied in `neutral_atoms/mcts.py` (2026-04-27).**

- **`_ucb_score` Python loop (13.8%).**  
  The `max(... for action, child in node.children.items())` genexpr iterates per child in pure Python. Vectorizing over all children with numpy/torch would convert this to a single batched op.

**3. `_compute_remaining_cost` (26.1%) is called exactly twice per `env.step` (before + after in plan_cost mode).**  
With 2000 sims and a trained prior, MCTS revisits the same tree nodes many times. The same board state appears as input to `_compute_remaining_cost` on nearly every simulation that passes through that node.  
See §Cache hit rate analysis below.

**4. `count_groups_fast` (Numba) is 12% of total but individually cheap (~10 µs/call).**  
The cost accumulates from 1.67M calls (one per layer per remaining_cost evaluation). Eliminating redundant `_compute_remaining_cost` calls via cache eliminates most of these.

---

## Cache hit rate analysis

**Context:** potential memoization of `_compute_remaining_cost` by `(board_state_hash, tasks_done)` → `int`.

`_compute_remaining_cost` is called on `sim_env`, which is a fresh clone of root replaying the path from scratch each simulation. Two simulations that follow the same action prefix up to depth d will have identical board states at depth d — the board state is deterministic given the action sequence.

**MCTS visit structure at 2000 sims, map=2:**
- Profile shows 48,000 `_descend` calls across 24 moves → 2000 descents per move.
- Average descent depth: 403,799 steps / 48,000 descents ≈ **8.4 steps/descent**.
- Nodes expanded per move: ≈ 2000 (one new leaf per simulation).
- Total `_compute_remaining_cost` calls per move: 807,598 / 24 ≈ 33,650.
- Unique (board_state, tasks_done) pairs per move: ≈ 2000 × average_depth = small relative to 33,650 calls.

**Estimated hit rate:** With a trained prior, MCTS concentrates visits heavily on a few paths. The top path may receive 30-40% of all 2000 simulations. Each unique state along that path is visited ~600-800 times. Average visits per unique state across the whole tree: 33,650 calls / ~2,000 unique states ≈ **17× average reuse**.

Expected cache hit rate: **(unique_states - 1) / total_calls ≈ 94–99%**, depending on concentration.

**Cost of hashing:** The board state is `atom_positions` (shape `(12, 2)`, int16 on CPU). Hashing a 24-element tuple: < 1 µs. Total hashing overhead at 807,598 lookups: < 1s. Net saving vs 39.1s of `_compute_remaining_cost`: **~38s**.

**Caveat:** Cache must be scoped to one `run_mcts_batched` call (one move). The root board state changes each move, invalidating all entries. A per-call dict (cleared in `backend.reset()` or passed as a kwarg) is the right scope.

**Verdict:** Very high hit rate expected; implementation is low-risk. Priority after config-overhead fix.

---

### M5 — cProfile after Opt-1 + Opt-2 (job 59944155)

**Date:** 2026-04-27  
**Script:** `scripts/profile_env.py`  
**Network:** real (v3a01 cycle_05)  
**Config:** map=2, sims=2000, backend=fast, batch=32, vl=1.0  
**Hardware:** CPU node (2 cpus, 8G), 1 game  
**Log:** `slurm/logs/profile_env_59944155.out`  
**Changes since M4:** Opt-1 (config constant extraction) + Opt-2 (`_compute_remaining_cost` memoization)

Wall time: **82.3s** (was 150s — **45% reduction**).  
Parity: 5/5 games byte-identical vs vanilla (no-cache) formulation. Cache hit rate: **99.5%**.

#### Time breakdown (cumtime)

| component | cumtime | % total | ncalls | vs M4 |
|---|---|---|---|---|
| `env.step` (total) | 22.3s | **27.1%** | 403,799 | was 37.7% |
| — `_compute_remaining_cost` | 6.7s | 8.2% | 807,598 | was 26.1% (↓ 69%) |
| — — numpy key compute | ~3.3s | ~4% | 807,598 | new overhead |
| `_select_child` (all UCB) | 17.7s | **21.5%** | 403,775 | was 33.9% |
| — `_ucb_score` (genexpr + math) | 9.6s | 11.7% | 5,652,850 | was 13.8% |
| — ml_collections config overhead | 2.0s | 2.5% | 1,273,602 | was 12.0% (↓ 91%) |
| — `max()` builtin | 1.4s | 1.7% | 868,514 | new visible |
| NN forward (`compute_blocking`) | 33.9s | **41.2%** | 1,512 | was 22.7% |
| — `torch._transformer_encoder_layer_fwd` | 29.3s | 35.6% | 6,420 | was 19.6% |
| `env.clone` | ~1.1s | ~1.3% | 48,000 | unchanged |

Note: NN fraction grew from 22.7% → 41.2% not because NN got slower, but because env.step and UCB overhead were cut nearly in half. Amdahl limit for GPU acceleration improved from 1.12× to **1.55×**.

#### Key findings from M5

**Opt-1 effect (config extraction):** ml_collections overhead dropped from 18.0s → 2.0s (91% reduction). The residual 2.0s is from `_ucb_score` itself still receiving `pb_c_base`/`pb_c_init`/`discount` as function arguments (their reads are now just local variable accesses, not ConfigDict lookups). This matches the 12% → 2.5% proportional drop.

**Opt-2 effect (cost cache):** `_compute_remaining_cost` dropped from 39.1s → 6.7s cumtime. The savings come entirely from cache hits (99.5% hit rate) avoiding `_compute_layer_cost_fast` calls. New cost introduced: ~3.3s for numpy key computation (`atom_positions.numpy().tobytes()` + phase_moves tuple) at every call including hits. This is the primary remaining overhead inside `_compute_remaining_cost`.

**Remaining UCB cost (17.7s):** After config extraction, `_select_child` still consumes 21.5% of total via:
- Python genexpr iterating over children: 5.65M `_ucb_score` calls × ~1.7 µs each = 9.6s
- `max()` builtin over the genexpr stream: 868k calls × ~1.6 µs = 1.4s
- `math.log` + `math.sqrt` computed once per child (not once per node): 5.65M × 2 calls each = 1.0s + 0.7s

**Next optimization:** Inline UCB into `_select_child` as an argmax loop — compute `pb_c_factor` once per node, inline `child.value_sum / cv` and `(q - mn) / rng` to eliminate 5.65M function calls + 868k `max()` calls. Estimated saving: **~9s**, bringing wall time toward ~73s.

---

### M6 — cProfile after Opt-1 + Opt-2 + Opt-3 (job 59945684)

**Date:** 2026-04-27  
**Script:** `scripts/profile_env.py`  
**Network:** real (v3a01 cycle_05)  
**Config:** map=2, sims=2000, backend=fast, batch=32, vl=1.0  
**Hardware:** CPU node (2 cpus, 8G), 1 game  
**Log:** `slurm/logs/profile_env_59945684.out`  
**Changes since M5:** Opt-3 (UCB inlining into argmax loop)

Wall time: **64.7s** (M4: 150s → M5: 82.3s → M6: 64.7s — **57% total reduction**).  
Parity: 5/5 games byte-identical vs vanilla. Cache hit rate: 99.5%.  
Total function calls: **26.95M** (M4: 128M → M5: 51.5M → M6: 27.0M — 79% reduction).

#### Time breakdown (cumtime)

| component | cumtime | % total | ncalls | vs M5 |
|---|---|---|---|---|
| NN forward (`compute_blocking`) | 28.3s | **43.7%** | 1,512 | was 33.9s |
| — `torch._transformer_encoder_layer_fwd` | 24.3s | 37.6% | 6,420 | was 29.3s |
| `env.step` (total) | 21.7s | **33.6%** | 403,799 | was 22.3s |
| — `_compute_remaining_cost` | 6.5s | 10.0% | 807,598 | was 6.7s (unchanged) |
| — — numpy key overhead | ~2.5s | ~3.9% | 807,598 | new; irreducible with current key design |
| `_select_child` (all UCB) | 6.8s | **10.5%** | 403,775 | was 17.7s (↓ 62%) |
| — `_select_child` tottime | 4.8s | 7.4% | 403,775 | inline argmax loop |
| — ml_collections (3 reads/call) | 1.5s | 2.3% | 1,273,602 | residual config reads |
| `env.clone` | 1.1s | 1.7% | 48,000 | unchanged |

Note: `_ucb_score`, `max()` builtin, `math.log`/`math.sqrt` as separate entries — all gone from profile. `_select_child` 62% cumtime reduction confirms the loop overhead was the dominant UCB cost.

#### Key findings from M6

**Opt-3 effect (UCB inlining):** `_select_child` cumtime dropped from 17.7s → 6.8s. Savings breakdown:
- Eliminated 5.65M `_ucb_score` Python function calls (~2.8s frame overhead)
- Eliminated 868k `max()` builtin calls over genexpr stream (~1.4s)
- Eliminated 5.65M `math.log` / 5.65M `math.sqrt` per-child calls — replaced by 1 log + 1 sqrt per node (~0.7s + 1.0s)
- Eliminated 1.4M `Node.value()` + 955k `normalize()` method calls (~0.9s)

**Remaining UCB cost (6.8s):** `_select_child` still accounts for 10.5% because:
- The argmax loop body itself (403k × avg 14 children = 5.6M iterations of simple Python) costs ~4.8s
- 3 ml_collections config reads per `_select_child` call (pb_c_base, pb_c_init, discount) = 1.21M reads at ~1.2 µs each = ~1.5s

**NN is now the dominant cost (43.7%):** The transformer forward is irreducible on CPU. This is the Amdahl limit: Amdahl ceiling = 1 / (0.437 + 0.563/∞) = **2.29×** if all non-NN work were free. Actual achievable GPU speedup is lower since not all non-NN work is zero.

**env.step tottime (12.8s)** is now the second-largest component. It includes:
- Two `_compute_remaining_cost` cache lookups per step (key computation ~3 µs × 807k = ~2.5s)
- Board tensor updates, `current_qubit` property, reward arithmetic

**What's left to optimize:**
1. Extract `pb_c_base/pb_c_init/discount` once per `run_mcts`/`run_mcts_batched` call rather than per `_select_child` → saves ~1.5s (2.3% of wall)
2. Faster cache key: `atom_positions.data_ptr()` hash + int fields instead of `numpy().tobytes()` → would reduce the 2.5s key overhead, but requires care (contiguous memory guarantee)
3. GPU selfplay: with NN now at 43.7%, GPU acceleration gives up to 1.54× speedup (vs 1.12× before optimizations), and actual measured gain would be better than 0.72× seen in M3

---

### M7 — GPU profile, pUCT/fast (job 59949469)

**Date:** 2026-04-28  
**Script:** `scripts/profile_env.py --device cuda`  
**Config:** map=2, sims=2000, backend=fast, batch=32, vl=1.0, device=cuda  
**Hardware:** A100-SXM4-40GB (GPU node)  
**Log:** `slurm/logs/profile_env_gpu_59949469.out`

Wall time: **33.5s** (M6 CPU: 64.7s → **1.93× GPU speedup**). Parity: 5/5.

| component | M7 GPU | M6 CPU | delta |
|---|---|---|---|
| NN forward (`compute_blocking`) | 3.6s (10.7%) | 28.3s (43.7%) | −87% |
| `torch._transformer_encoder_layer_fwd` | 1.13s tottime | 24.3s tottime | −95% |
| `env.step` cumtime | 17.4s (51.9%) | 21.7s (33.6%) | −20% |
| `_select_child` cumtime | 5.8s (17.2%) | 6.8s (10.5%) | −15% |
| `_compute_remaining_cost` cumtime | 5.4s (16.1%) | 6.5s (10.0%) | −17% |

Amdahl prediction was 1.78× (if GPU NN free); actual 1.93× because A100 is ~20× faster than CPU for this batch size, and non-NN time also dropped slightly (faster CPU on GPU node). New bottleneck: `env.step` at 51.9%.

**Conclusion:** GPU selfplay is now worthwhile at 2000 sims (1.93×). At 10k sims with trained prior, NN is ~22% → GPU probably still counterproductive (need fresh smoke test to confirm).

---

### M-G1 — Gumbel profile, classic/CPU (job 60005348)

**Date:** 2026-04-28  
**Script:** `scripts/profile_env.py --gumbel --map_num 5`  
**Config:** map=5 (8×8, 20q), sims=2000, backend=classic, Gumbel enabled, cycle_00 ckpt  
**Hardware:** CPU node (2 cpus, 8G)  
**Log:** `slurm/logs/profile_gumbel_60005348.out`

Wall time: **1370.4s** — 21× slower than M6 pUCT on 5×5. 60 moves (vs 24 for 5×5).

| component | cumtime | % total | ncalls |
|---|---|---|---|
| `network.inference` (total NN) | 1232.6s | **89.9%** | 113,007 |
| `torch._transformer_encoder_layer_fwd` | 1037.6s tottime | **75.7%** | 678,042 |
| `module.__getattr__` (PyTorch traversal) | 30.8s tottime | 2.2% | 47,914,968 |
| `env.step` | 49.4s | 3.6% | 337,255 |
| `_gumbel_non_root_select` | 32.5s | 2.4% | 218,995 |
| `_gumbel_improved_policy` | 25.9s | 1.9% | 219,055 |
| `_compute_remaining_cost` | 20.9s | 1.5% | 674,510 |

**Terminal fraction: ~6%** (113,007 NN calls / 118,200 sims = 95.6% call NN). This is the exact opposite of 5×5 pUCT at 10k sims (84.7% terminal). On 8×8 with early training, the tree is shallow and most sims reach unexplored leaves — every sim needs NN.

**Root cause of NN dominance:** 113,007 **sequential** inference calls (one per sim, no batching). Compare pUCT M6: 1,070 **batched** calls (836 batched + 234 single). Each call pays 424 `module.__getattr__` PyTorch traversal ops → 47.9M total = 30.8s of pure Python overhead before the forward even runs.

**Amdahl limit:** NN = 89.9% → limit = 1/(1−0.899) = **9.9×** for GPU, or for batching alone.

**fast_gumbel batching (batch=32) expected impact:**
- NN calls: 113,007 → ~3,532 (32× fewer)  
- `module.__getattr__` overhead: 30.8s → ~1s  
- Transformer kernel: each call handles 32 samples; A100 processes these in similar wall-time to 1 sample → saves 31/32 × 1037.6s ≈ **1005s**  
- Non-NN work unchanged: ~137s  
- **Projected total: ~170s — 8× speedup CPU-only; with GPU likely 15-20×**

**Optimizations already applied (Gumbel-specific, 2026-04-28):**
- `_shared_cost_cache` enabled in `_gumbel_plan` — same Opt-2 benefit, was completely absent before
- `c_visit`, `c_scale`, `discount` extracted at `_gumbel_plan` entry, threaded through all callers — eliminates nested ConfigDict lookups in the sim loop

These save ~20s (1.5% from cost cache + 2% from config reads). Marginal vs NN dominance; fast_gumbel batching is the real lever.

---

## Optimization status

### pUCT / fast backend (map=2, 5×5)

| fix | target | expected gain | actual gain | status |
|---|---|---|---|---|
| Extract config constants in `_select_child` | UCB config overhead (12%) | ~12% wall | combined: 45% (M4→M5: 150→82s) | **Done** |
| Memoize `_compute_remaining_cost` | plan_cost reward (26%) | ~20-25% wall | combined with above | **Done** |
| Inline UCB into `_select_child` argmax loop | UCB loop overhead (21.5%) | ~8-10% wall | **21% (M5→M6: 82→65s)** | **Done** |
| GPU selfplay at 2000 sims | NN 43.7% | up to 1.78× | **1.93× (M7: 65→34s)** | **Done** |
| Extract config scalars above `_descend` loop | ml_collections residual (2.3%) | ~1.5s | — | planned |
| Faster cache key (avoid numpy().tobytes()) | key computation overhead (~3.9%) | ~2s | — | planned |
| Phase 3: vec env / undo-stack | env.step bulk | 3-5× | — | deferred |

### Gumbel / classic backend (map=5, 8×8)

| fix | target | expected gain | status |
|---|---|---|---|
| `_shared_cost_cache` in `_gumbel_plan` | cost cache absent from Gumbel | ~20s (1.5% of 1370s) | **Done (2026-04-28)** |
| Extract `c_visit/c_scale/discount` scalars | ml_collections in sim loop | ~27s (2%) | **Done (2026-04-28)** |
| **fast_gumbel: batch NN calls within each phase** | NN 89.9% (1233s of 1370s) | **~8× CPU, ~15-20× CPU+GPU** | **next** |
| GPU selfplay (after fast_gumbel) | NN after batching ~?% | up to 9.9× from M-G1 | planned |

#### Rationale for UCB inlining (Opt-3)

After Opt-1+2 the profile shows `_select_child` still at 17.7s (21.5%). The remaining cost breaks down as:
1. **5.65M `_ucb_score` Python function calls** — CPython frame creation overhead alone ~0.5 µs/call = 2.8s.
2. **Recomputing `pb_c_factor = log(…) * sqrt(…)` per child** — this quantity is constant for all children of the same parent; it should be computed once per `_select_child` call.
3. **`max()` over a generator stream** — 868k calls × ~1.6 µs = 1.4s, replaceable by an explicit argmax loop.
4. **`child.value()` method call** — 1.41M calls at 0.44s, replaceable by `child.value_sum / cv` inline.
5. **`min_max_stats.normalize()` method call** — replaceable by `(q - mn) / rng` inline.

The inlined version computes the same values with zero function call overhead and hoists the constant `pb_c_factor` out of the per-child loop. `_ucb_score` is retained unchanged as a standalone function for unit tests.

After Opt-1+2+3, expected total speedup vs M4 (150s): **~1.05× additional on 82.3s → ~73-75s**.
