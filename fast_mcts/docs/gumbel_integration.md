# Gumbel + fast_mcts — integration design

Status: **plan only, no code**. Implement after Gumbel A/B on classic lands
and results indicate go.

Purpose: acceleration path for `_gumbel_plan` equivalent to what Phase 1
did for pUCT. Targets the same NN bottleneck (70% wall at large sims) by
leaf-gathering within each sequential-halving phase.

## Classic Gumbel structure (from `neutral_atoms/mcts.py`)

```
_gumbel_plan(config, root, env, network):
    seed root (visit=1, value_sum=V)
    compute gumbel noise g(a) ~ Gumbel(0,1) per legal action
    candidates = top-m by g(a) + logit(a)        # Gumbel-Top-m
    for phase in range(num_phases = ceil(log2(m))):
        n_per_cand = max(1, total_sims // (num_phases * m_phase))
        for cand in candidates:
            for _ in range(n_per_cand):
                _gumbel_simulate(root, cand, env, network)
        halving: candidates ← top-m/2 by g(a) + logit(a) + σ(completedQ(a))
    winner = argmax_a [g + logit + σ(completedQ)] over survivors
    return winner

_gumbel_simulate(root, root_action, env, network):
    clone env, force first step = root_action
    descend via _gumbel_non_root_select (deterministic argmax)
    NN call at leaf, expand, backup
```

**Serial barriers**: between phases (halving uses final Q stats from that
phase). **Within a phase**, `m_phase × n_per_cand` simulations are independent
and batchable.

## Mapping to fast_mcts primitives

| Gumbel component | fast_mcts analog | change needed |
|------------------|------------------|---------------|
| `_gumbel_simulate` inner loop | `_descend` (search.py) | reuse `sim_env.step` descend, swap selector |
| `_gumbel_non_root_select` | replaces `_select_child` during descent | unchanged, just dispatch by mode |
| Leaf NN call | `NNBackend.add` + `compute_blocking` | same as pUCT |
| `_expand_node`, `_backpropagate` (with `min_max_stats=None`) | reused verbatim | no change |
| Virtual-loss on descent | **implicit**: incrementing `visit_count` already lowers `π'(a) - N(a)/(1+ΣN)` score → natural steering | no extra `virtual_loss` constant needed |
| Cache (Phase 2) | `NNCache` passed through | no change — cache keys on features alone |
| Telemetry buckets | `_gumbel_plan` already computes `_mcts_*` | port verbatim |

## New files

- `fast_mcts/gumbel_search.py` — `run_gumbel_batched(cfg, root, env, network,
  cache=None, nn_batch_size=32)` and top-level `play_game_gumbel`.
- `fast_mcts/backends.py` — register `fast_gumbel` backend.
- `fast_mcts/test_gumbel_parity.py` — cross-backend byte-parity vs
  classic `_gumbel_plan` at safe knobs.

Zero touch to `neutral_atoms/mcts.py` Gumbel code — we consume it via
`from neutral_atoms.mcts import _gumbel_non_root_select, _gumbel_completed_q,
_gumbel_improved_policy, _gumbel_halving_scores, Node, _expand_node,
_backpropagate`.

## Dispatch

Option A (cleanest): new backend name.
```python
config.mcts.backend = 'fast_gumbel'  # activates fast Gumbel
config.mcts.gumbel.enabled = True     # still required for telemetry / target_policy
```

Option B (auto-detect): `backend='fast'` + `gumbel.enabled=True` →
dispatch to `play_game_gumbel` internally. Minimizes config churn but
couples two orthogonal knobs.

**Recommend Option A** for explicit contract. One extra line in config,
zero surprise.

## Batching algorithm sketch

```
def run_gumbel_batched(cfg, root, env, network, cache=None):
    backend = NNBackend(network, cap=cfg.mcts.nn_batch_size, cache=cache)
    gumbel_noise, priors_logit = init_gumbel(root, cfg)
    candidates = gumbel_topm(root, cfg, gumbel_noise, priors_logit)

    for phase in range(num_phases):
        m_phase = len(candidates)
        n_per_cand = max(1, total_sims // (num_phases * m_phase))
        tasks = [(cand, s) for cand in candidates for s in range(n_per_cand)]

        # Gather + dispatch in chunks of nn_batch_size.
        i = 0
        while i < len(tasks):
            pending = []
            while len(pending) < cfg.mcts.nn_batch_size and i < len(tasks):
                cand, _ = tasks[i]; i += 1
                sim_env = env.clone()
                # Forced root action + deterministic descent via _gumbel_non_root_select.
                path, leaf, leaf_reward, terminal = _gumbel_descend(
                    root, cand, sim_env, cfg)
                # "Virtual loss" for Gumbel = pre-incrementing visit counts
                # (descent score depends on them). Classic doesn't do this
                # because sims are sequential; we must.
                _apply_visit_loss(path)
                if terminal:
                    pending.append(_LeafReq(path, leaf, leaf_reward,
                                            sim_env, [], slot=None, terminal=True))
                else:
                    slot = backend.add(sim_env.get_features(),
                                       sim_env.current_qubit)
                    pending.append(_LeafReq(path, leaf, leaf_reward, sim_env,
                                            sim_env.legal_actions(),
                                            slot=slot, terminal=False))
            backend.compute_blocking()
            for req in pending:
                _revert_visit_loss(req.path)
                if req.terminal:
                    _expand_node(req.leaf, [], None, req.leaf_reward,
                                 sim_env=req.sim_env, config=cfg)
                    _backpropagate(req.path, 0.0, cfg.discount, None)
                else:
                    out = backend.get(req.slot)
                    _expand_node(req.leaf, req.legal, out, req.leaf_reward,
                                 sim_env=req.sim_env, config=cfg)
                    _backpropagate(req.path, out.value, cfg.discount, None)
            backend.reset()

        # Halving (serial barrier — needs fresh stats)
        if m_phase > 1:
            candidates = halving(root, candidates, cfg,
                                 gumbel_noise, priors_logit)

    winner = pick_final(...)
    root._gumbel_policy = _gumbel_improved_policy(root, cfg)
    return winner
```

### Visit-loss: important subtlety

`_gumbel_non_root_select` picks `argmax [π'(a) - N(a)/(1+ΣN)]`. With
sequential sims (classic), N(a) updates after each backup, naturally
spreading sims. With batched sims, N(a) stays stale within a batch → all
sims in the same phase pick the same child repeatedly.

**Fix**: apply `visit_count += 1` on descent (same as pUCT virtual loss
but without the value_sum adjustment, since non-root selection doesn't
use Q directly — it uses `π'` which is recomputed from Q+visits each call).
Revert on commit, then do real backup.

Net effect: within-batch sims explore distinct children; across batches
backups update for real. Identical outcome to sequential Gumbel in
expectation; per-seed trajectories differ.

### Parity contract for `fast_gumbel`

- `nn_batch_size=1` → sequential, no visit-loss → **byte-identical** to
  classic `_gumbel_plan`. Tested via `test_cross_gumbel_parity` with
  6 combos mirroring pUCT parity tests.
- `nn_batch_size>1` → breaks byte-parity by design. Stat parity via
  paired Gumbel runs (same seeds, classic vs fast_gumbel, compare mean
  cost). Reuse `test_stat_parity.py` scaffolding.

## Cache reuse (Phase 2)

No changes. `NNCache` keys on `(features.tobytes(), current_qubit)` —
agnostic to which search algorithm queried. Cross-game hits especially
valuable in Gumbel because:
- Gumbel has **fewer total NN calls** than pUCT (sims spread across phases),
  so each saved call is proportionally higher value.
- Gumbel's root forced-action pattern makes early-phase leaves more
  similar across games (root + one fixed action from Top-m) → higher
  overlap.

## Expected speedup

Per-phase batch size = `min(m_phase × n_per_cand, nn_batch_size)`.

Example: 400 sims total, m=16, num_phases=4, n_per_cand=400/(4×16)≈6.
Per phase: 16×6 = 96 sims. At nn_batch_size=32 → 3 batches per phase.
Final phase: 1 × 96 = 96 sims, still batches of 32.

Rough napkin: 4× on NN wall (vs sequential) → if NN = 70% of classic
Gumbel wall, saves 52.5% → **2.1× end-to-end** vs classic Gumbel.
Add cache (deterministic eval): another ~1.5× → **3.1× cumulative**.

In production self-play (noise on), less per estimator pattern.

## Risks + open questions

| risk | notes |
|------|-------|
| **Visit-loss interaction with halving scores** | `_gumbel_halving_scores(root, ...)` uses `completedQ` which includes `c.value()` = `value_sum / visit_count`. If we apply visit-loss and don't also adjust value_sum (pUCT does), Q gets underestimated for in-flight paths. For non-root this may be OK (selector uses π' not Q directly), but for halving it matters. **Test**: same halving result batch=1 vs batch>1 on deterministic runs; mean Q drift across seeds. |
| **Num_samples_m default** | Config sets `m = board_size - num_qubits + 1`. Small maps: m=4-8. Fewer phases, less batchable parallelism. Large maps (8×8, m=44): 6 phases, lots of parallelism. Speedup scales with map size. |
| **First-phase all same `cand` → redundant NN calls** | If we batch [cand1, cand1, cand1, cand2, cand2, ...], first two clones reach identical states pre-descent. Cache catches these. Without cache, we dedup via `id(leaf_node)` collision logic from Phase 1. |
| **Cache hit rate on Gumbel trees** | Unknown until measured. Build `cache_estimator.py --gumbel` scenario before committing to Phase 2 integration strategy. |
| **Gumbel still uses `network.inference` direct** | `_gumbel_simulate` in classic calls `network.inference` — that bypass must be replaced with `backend.add` in our port. Nothing breaks; just a discipline point. |

## Order of work (when Gumbel validates)

1. **Estimator pass**: extend `cache_estimator.py` to support Gumbel; measure
   hit-rate pattern specifically for Gumbel. 2 hours.
2. **Port**: `gumbel_search.py` with batching + visit-loss. 1 day.
3. **Parity tests**: byte-parity at batch=1, stat-parity at batch=32.
   Half day.
4. **Bench**: compare classic_gumbel vs fast_gumbel vs fast_gumbel+cache.
   Update `docs/phases.md` with actual numbers.
5. **Selfplay wiring**: `config.mcts.backend = 'fast_gumbel'` dispatch in
   `selfplay_worker.py` (already dispatches by backend name → 0-line change).

Total estimate: ~2 days if gumbel wins the A/B.

## Anti-goals

- Do **not** modify `_gumbel_plan` in `neutral_atoms/mcts.py`. Classic
  path must remain untouched so ongoing Gumbel experiments keep their
  reference baseline.
- Do **not** share a single `run_gumbel_batched` with pUCT's `run_mcts_batched`.
  Different selector, different stopping criterion, different telemetry
  attribution. Clean fork, reuse of primitives only.
- No partial builds. Either the whole Gumbel port lands with parity tests
  green, or nothing ships. This keeps the `backend='fast_gumbel'` flag
  trustworthy.
