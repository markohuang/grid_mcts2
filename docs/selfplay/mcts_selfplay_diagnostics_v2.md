# MCTS Self-Play Diagnostics & Scaling Notes — v2

**Supersedes:** [`mcts_selfplay_diagnostics.md`](mcts_selfplay_diagnostics.md) (v1).

**Why a v2.** The v1 doc was written against a snapshot of `mcts.py` / `game.py` / `trainer.py` that was stale relative to `HEAD`. Four of the five issues v1 flagged as blockers were already fixed on branch `prior_learning` at commits `6d40fd0`, `a130767`, and `75a8498`. See §0 for the diff against v1 and §1.3 for the issues that *remain*. v1 is kept in the tree as a changelog — read v2 for the current-state analysis.

Scope: the MCTS + self-play infrastructure under `neutral_atoms/` (`mcts.py`, `game.py`, `trainer.py`, `selfplay_worker.py`). Out of scope: `atom-viz/`, `iterative_refinement/`, `soft_constraint/`.

This doc has three jobs:

1. A critical read of the **current** self-play pipeline (HEAD `75a8498`) — what's solid, what's load-bearing, and what will bite us as we scale.
2. A hyperparameter health checklist: which diagnostics already exist in the current HEAD, which are missing, and what each one *means* for setting exploration/exploitation constants, simulation budget, temperature, and priors.
3. A concrete logging and infrastructure proposal so future runs surface enough signal to tune MCTS without re-instrumenting from scratch, and so the self-play dataset is ready for actor-critic training.

---

## 0. Corrections to v1

v1 flagged five "scaling blockers". Four are already resolved on HEAD — v1 was wrong (or at least out of date) about all four. The fifth is real and persists. v1 also **missed** two bigger items: no batched inference, no tree reuse. Details:

| v1 claim | Actual status on HEAD `75a8498` |
|---|---|
| **`Game.from_dict` rebuilds observations `O(T²)`** | **Fixed** in `6d40fd0`. `from_dict` now caches each intermediate observation inside the replay loop (plus one terminal cache), and the ad-hoc `game.cache_observation()` in `trainer.load_dataset_into_buffer` has been removed. |
| **Only 2/7 MCTS diagnostics serialized** | **Fixed** in `6d40fd0`. `Game.to_dict`/`from_dict` now round-trip all seven `_mcts_*` arrays. |
| **`_softmax_sample` uses `softmax(N/τ)`, not `N^(1/τ)`** | **Fixed** in `a130767`. Now `softmax(log(N+ε)/τ)`, matching the policy target in `Game.store_search_statistics`, with an explicit `τ == 0` argmax branch. |
| **`plan_cost_search_bonus_weight` is a one-shot UCB addend on visited edges only** | **Obsolete** in `75a8498`. Replaced by `mcts.prior_mix_weight`, which probes *every* legal action once at expansion time on a cloned env and folds `β · plan_cost_delta / cost_ub` into the log-prior. Config knob space collapsed to three self-documenting options (`layer_delta`, `plan_cost`, `layer_delta + prior_mix`) with a fail-fast assertion forbidding `plan_cost + prior_mix`. |
| **`_backpropagate` updates `min_max_stats` from the running mean `node.value()`** | **Still present.** Line 222 of `mcts.py`. Real latent issue, see §1.3(A). |
| *(v1 did not mention)* | **No batched leaf inference / virtual loss.** Biggest throughput ceiling. §1.3(C). |
| *(v1 did not mention)* | **No tree reuse across decisions.** Wastes ~30–50% of per-episode search work. §1.3(C). |
| *(v1 did not mention)* | **`prior_mix_weight` clones env once per legal action on every expansion** — ~2–2.5× wall-clock multiplier. §1.3(E). |

---

## 1. Critical read of the current MCTS/self-play infrastructure

### 1.1 What the tree currently does

`mcts.py:play_game` runs `num_simulations` rollouts per decision. Each simulation:

- Clones the env (`env.clone()`) and descends via `_ucb_score` until it reaches an unexpanded node.
- Queries the network, expands, and backpropagates the leaf value along the path.
- Single-player, no negation (`_backpropagate` just adds `value`).
- `discount=1.0` and `known_bounds=(-6, 6)` are baked in via `MinMaxStats` normalization of `child.reward + discount * child.value()`.

`_expand_node` now takes `sim_env` and the mcts config. When `prior_mix_weight > 0`, it **clones the env once per legal action** and applies a one-step probe (`probe = sim_env.clone(); probe.step(a)`), then adds `β · plan_cost_delta / cost_ub` to that action's log-prior. The heuristic rides on the standard `pb_c · √ΣN / (1+N)` UCB envelope and distills into `π_θ` through visit counts. This is the right place for heuristic injection — R13 showed the old UCB-addend form collapsed self-play entropy and underperformed plain `layer_delta`, while non-decaying variants matched, indicating the benefit belongs in the prior, not in UCB.

`_select_action` picks from root visits using `_softmax_sample(visit_counts, τ)` with `τ = temperature_override or get_temperature(network.training_steps(), config)`. The formula is now the standard `π ∝ N^(1/τ)` (`softmax(log(N+ε)/τ)`). When `deterministic=True` or `τ == 0`, it falls back to argmax.

`Node` is back to its minimal form: `visit_count, prior, value_sum, children, reward`. No `search_bonus`.

### 1.2 Observation & feature pipeline

`Game.from_dict` now replays the full history once, caching the pre-step observation before each `env.step(action)` plus one final post-terminal cache. `observation_cache` therefore has `len(history) + 1` entries after reload, with the last entry being the terminal observation. `game_to_tensordict`'s bootstrap lookup `make_observation(min(i + td_steps, len(history)))` lands on a valid cache entry at the boundary.

`Game.make_observation` has three branches: live (`-1`), cached, and env-replay fallback. With the v2-era `from_dict` populating the cache, **reloaded games never hit the env-replay fallback**. This is the big O(T²) → O(T) win from `6d40fd0`. Don't break the invariant: any new path that constructs a `Game` via `__new__` must populate `observation_cache` if it expects `make_observation(i)` with `i ≥ 0` to be cheap.

### 1.3 Remaining issues — root causes

These are the live problems worth tracking.

#### (A) `_backpropagate` normalizes UCB using the running *mean* of each node

```python
def _backpropagate(search_path, value, discount, min_max_stats):
    for node in reversed(search_path):
        node.value_sum += value
        node.visit_count += 1
        min_max_stats.update(node.value())        # running mean, not Q-target
        value = node.reward + discount * value
```

AlphaZero's `min_max_stats` is meant to bracket the Q-target so UCB's value term occupies `[0,1]`. Updating from `node.value()` (the cumulative mean) rather than from `node.reward + discount * value` (the Q-target `_ucb_score` actually compares) means the normalizer tracks a slightly compressed range. Under dense positive rewards this narrows the normalization window early in each search and flattens the value term, making MCTS look prior-dominated for the first ~10 simulations. Latent, not a blocker, but the most likely suspect when "search isn't exploiting" despite a reasonable `pb_c_init`. One-line fix; re-run a calibration epoch on each active reward mode afterwards.

#### (B) Temperature is driven by `network.training_steps()`

`play_game` reads the learner's global step counter at call time. In **distributed self-play**, a worker launched with a stale checkpoint exposes that checkpoint's `training_steps`, so workers run with whatever effective τ the underlying checkpoint believed. At `temperature_decay_steps=1000` and `training_steps=200/epoch`, decay finishes at epoch 5 of local training — but HPC workers' effective τ is determined by whichever checkpoint they loaded, which may lag the active learner arbitrarily.

**Effect.** Temperature sweeps done locally don't transfer cleanly to HPC runs, and there is currently no log of the *effective* τ that was actually used on a given game — it has to be reconstructed from `weight_gen` lineage, which is itself lossy (see D).

#### (C) No batched inference, no tree reuse

Two standard AlphaZero optimizations are absent. Both become load-bearing the moment we want a large-scale actor-critic database.

- **Batched leaf evaluation with virtual loss.** Simulations run strictly sequentially: each iteration does one `network.inference()` call on a single leaf. The ~100 inferences/sec single-core ceiling noted in `docs/narval_hpc_migration.md` is what falls out of this. Leaf parallelization — collect a batch of leaves, apply virtual loss so siblings don't collapse to the same path, run one batched forward pass, then backprop — is typically a 10–30× throughput win on CPU and much more on GPU.
- **Tree reuse across decisions.** `play_game` builds a fresh root every step and discards the subtree, including the visits accumulated under the chosen child. On 30-step 5×5 episodes at `num_simulations=1000` that is a ~30–50% waste in search work, depending on how concentrated the policy is.

Neither is required to ship the layer-level MDP. Both are the biggest compute multipliers on the table for scaling self-play.

#### (D) Dataset schema is near-ready but missing three things for AC training

`data.py:INDEX_SCHEMA` already captures `cost, steps, num_moves, noop_frac, move_distance, policy_entropy, root_value, mcts_depth, game_time_s, num_simulations, weight_gen, reward_mode, prior_mix_weight` — a solid seed. Gaps:

1. **`weight_gen` is just the filename basename.** For off-policy corrections in actor-critic training we want a monotonic generation counter (or the training-step count of the checkpoint that produced the game). The worker already loads a checkpoint dict; it just needs to surface that field.
2. **Only three of the seven per-step MCTS diagnostics reach the parquet index.** `Game.to_dict` now stores all seven, and the raw `.pt` batches carry them, but the index aggregates only `policy_entropy`, `root_value`, `mcts_depth`. The most useful for hyperparameter health — `reward_sum_std`, `boundary_reach_frac`, `sign_changes` — are invisible to DuckDB queries until they're added to `INDEX_SCHEMA` and the `_game_metrics` reducer in `selfplay_worker.py`.
3. **No step-level on-disk store.** Everything is per-game; `load_dataset_into_buffer` expands into step-level `TensorDict` rows at load time. For per-step shuffling, disk granularity is still game-level. A "flattened" writer — rows of `(obs_features, action, reward, root_value, policy_target, weight_gen, game_id, step_id)` — would let us do proper step-level sampling directly off disk. Not urgent today; needed once dataset crosses ~1M games.

#### (E) `prior_mix_weight` expansion cost is nontrivial

Every call to `_expand_node` with `prior_mix_weight > 0` does one `sim_env.clone()` + `.step()` **per legal action**. On 5×5 with ~15 legal actions, that is 15 extra env clones per expansion, and there are `num_simulations + 1` expansions per decision. At `num_simulations=1000` that is ~15,000 additional clone+step calls per decision — on top of the 1000 simulations' own clones. Expect roughly a 2×–2.5× wall-clock cost when turning `prior_mix_weight` on.

This is tolerable but has to be budgeted: experiments that compare `layer_delta + prior_mix` against `plan_cost` should control for **wall-clock**, not simulation count. A single `num_simulations=500` run with `prior_mix_weight > 0` is roughly compute-equivalent to `num_simulations=1000` without.

### 1.4 Things that are quietly load-bearing

Not bugs, but non-obvious and will confuse future instances:

- `Game.terminal()` returns true if `legal_actions()` is empty. The layer-level MDP always has the current cell legal, so this is dead in practice — it will not short-circuit episodes with incomplete tasks.
- `MinMaxStats` is re-created *per decision* (per root), initialized from `known_bounds`. Setting `known_bounds` wider than the empirical value range compresses the value term in UCB toward zero during the early simulations of each decision. The defaults `[-6, 6]` are fine for `layer_delta` but over-wide for `plan_cost` on 5×5, where per-step rewards are typically in `[-3, 3]`.
- `Network.inference` runs under `torch.no_grad()` but relies on callers having set the net to `.eval()` mode. `trainer.run_selfplay` and `_play_single_game` both do this — don't break the invariant in new entry points.
- Parallel self-play workers each recreate the Network from `state_dict` *per game* (`_play_single_game`). At high throughput the repeated `load_state_dict` becomes visible; minor but measurable.
- `from_dict`'s cache-during-replay means `observation_cache` has length `len(history)+1`, terminal entry last. `game_to_tensordict`'s bootstrap lookup is correct by construction at the boundary.

### 1.5 Scaling blockers (ordered by severity)

1. **No batched inference / virtual loss** — throughput ceiling. Hardest fix, biggest lever.
2. **No tree reuse across decisions** — wastes ~30–50% of the search budget per episode.
3. **Weight generation lineage is a filename, not an integer** — blocks off-policy corrections in AC training. Small fix; must land before the first serious AC run.
4. **Step-level dataset store** — needed once dataset crosses ~1M games. Not urgent today.
5. **`min_max_stats.update(node.value())` bug** — subtle, easy fix, measurable effect on prior-vs-value balance. Fix before any new `pb_c_init` sweep.
6. **Effective τ not logged per game** — required so HPC-worker temperature schedules are auditable.

---

## 2. Hyperparameter health checklist

All "already logged" assumes HEAD `75a8498`, where `Game.to_dict` serializes all seven `_mcts_*` arrays. "Missing" means not present in either the raw game dict or the parquet index.

### 2.1 `num_simulations` — is the search budget enough?

**What it controls.** Branching explored per decision. The policy target is the visit distribution, so too few simulations means noisy targets and the network is trained on its own prior.

**Healthy signs**

- `mcts_depths` (per step) should be *well below* the remaining-episode length. On 5×5 / 3 layers, average depth in 5–10 at `num_simulations=50` is typical.
- Root visit counts concentrated but not collapsed: top-1 visit fraction around 0.3–0.7 is the sweet spot.
- `avg_policy_entropy` should decline smoothly over epochs, not crash.

**Unhealthy signs**

- `mcts_depths` saturating near remaining-episode length → tree is fully expanded; extra sims are wasted. Either reduce sims *or* increase branching via Dirichlet noise.
- `avg_root_value` drifting far from empirically observed return → value bins miscalibrated (`network.value_min/max`) *or* leaf signal not propagating (too few sims, or §1.3(A) squashing the value term).

**Missing diagnostics worth adding**

- Fraction of sims that expanded a *new* node vs. re-traversing existing paths.
- Root visit-count top-1 fraction (cheapest single number).

### 2.2 `pb_c_init` / `pb_c_base`

**What they control.** `pb_c = log((N+base+1)/base) + init`, then `× √ΣN / (N_child+1) × prior`. At our scales (`num_simulations ≤ 1000`), the `log` term barely moves, so **`pb_c_init` is the dominant knob**. `pb_c_base=19652` is the AlphaZero chess value and is essentially decorative here.

**Healthy signs**

- `mcts_reward_sum_std` (now serialized) non-trivially positive — sibling simulations explore genuinely different return outcomes.
- Top child's prior tracks its visit fraction within ~2× by mid-training.

**Unhealthy signs**

- Fully concentrated visits with tiny `mcts_reward_sum_std` → raise `pb_c_init` (useful range here: 1.0–2.5).
- High `mcts_reward_sum_std` plus unstable best-action across consecutive epochs → lower `pb_c_init`; the tree is thrashing.

**Missing diagnostic**

- Per-decision prior-dominated vs value-dominated fraction: at each selection step, does `argmax(prior_score)` equal `argmax(value_score)`? Cheapest single number for exploration/exploitation balance.

### 2.3 Dirichlet noise (`root_dirichlet_alpha`, `root_exploration_fraction`)

`alpha=0.03` is the Go value; with branching ~15 here, that draw is very spiky. `root_exploration_fraction=0.25` means 25% of prior mass is redistributed through that spiky draw.

**Unhealthy signs.** Identical game cost every time on an under-trained network → policy only explores Dirichlet-spiky actions. For our branching factor, `alpha ∈ [0.1, 0.3]` is more appropriate than 0.03.

**Recommendation.** Sweep `alpha ∈ {0.03, 0.1, 0.25}` in any big data-generation run; log per-map best-cost variance across repeated games.

### 2.4 Temperature

Now `softmax(log N / τ)` (standard form). `temperature_decay_steps=1000` with `training_steps=200/epoch` finishes decay at epoch 5 of local training. On distributed workers, effective τ is determined by which checkpoint they loaded (§1.3 B).

**Healthy signs**

- `avg_policy_entropy` declining monotonically over epochs, not crashing.
- Best-cost improvement continuing *past* `temperature_decay_steps`; if improvement stops exactly at decay end, decay was too fast.

**Missing diagnostic**

- Effective τ per game. Record inside `_select_action`, stash on root, serialize via `Game.to_dict`.

### 2.5 Reward / value bounds

- `known_bounds=(-6, 6)` only matters through `MinMaxStats` normalization (running — narrower bounds widen themselves, wider bounds stay wide and compress the value term). §1.3(A) interacts with this.
- `value_min/value_max/num_bins` on the network side must cover the actual return distribution. If you change `reward_scale`, revisit these bounds or CE loss will clip.

**Cheap diagnostic worth adding.** At end of each search, log `min_max_stats.maximum - min_max_stats.minimum`. If the window is always ~ `known_bounds`'s initial width, bounds are too wide for the current reward mode.

### 2.6 `prior_mix_weight` (β)

**What it controls.** The mixing strength between the learned policy and the one-step plan-cost lookahead. `β · plan_cost_delta / cost_ub` is added to log-prior. With `cost_ub ≈ 30` on 5×5 and typical `plan_cost_delta ∈ [-2, 2]`, the per-action log-prior shift is `~β · 0.07`: `β=1` is negligible, `β=10` is a ~2× multiplicative nudge per action.

**Healthy signs**

- Policy entropy at `β > 0` lower than at `β = 0` early in training (the heuristic is doing work), then rising toward the `β = 0` level as the learned policy catches up (distillation is happening).
- Per-decision compute cost roughly 2–2.5× the `β = 0` version at equal `num_simulations`.

**Unhealthy signs**

- Entropy permanently lower than the `β = 0` control → mixture is dominating the learned policy instead of distilling. Lower β.
- Entropy never differs from the `β = 0` control → β too small. Raise it.

Always compare against `β = 0` at fixed **wall-clock**, not fixed sims (§1.3 E).

### 2.7 Summary table

| Knob | Primary signal | Secondary signal | Missing data |
|---|---|---|---|
| `num_simulations` | `mcts_depths` mean | `avg_policy_entropy`, best-cost trend | expanded-frac-of-sims |
| `pb_c_init` | top-1 visit fraction | `mcts_reward_sum_std` (serialized) | prior-vs-value argmax divergence |
| Dirichlet α/frac | across-game cost variance | per-epoch best-cost stability | per-game root argmax histogram |
| Temperature schedule | `avg_policy_entropy` curve | best-cost improvement past decay | **effective τ per game** |
| Value bounds | `avg_root_value` vs empirical | normalization width at end-of-search | logged explicitly per decision |
| `prior_mix_weight` | entropy vs β=0 control | wall-clock-normalized best-cost | β/cost_ub effective shift logged |

---

## 3. Recommendations for the scaling push

Ordered by effort, all in service of building a self-play database that is useful for actor-critic training.

### 3.1 Cheap (no schema break)

- **Log effective τ, top-1 visit fraction, and new-expansion fraction per decision.** Stash on `root` in `_select_action` / `run_mcts`; add three arrays to `Game.to_dict`.
- **Fix `min_max_stats.update(node.value())` → update on the Q-target `node.reward + discount * value` that `_ucb_score` actually compares.** One-line change; re-run a calibration epoch on each active reward mode afterwards.
- **Record `min_max_stats.maximum - min_max_stats.minimum`** at end of each search on the root, per reward mode, as a `known_bounds` sanity check.

### 3.2 Moderate (one schema migration)

- **Extend parquet `INDEX_SCHEMA`** with: `mcts_reward_std_mean`, `expanded_frac_mean`, `effective_temp_mean`, `top1_visit_frac_mean`, `training_step_of_weights`. Version-bump the schema, backfill existing shards on next ingest.
- **Replace `weight_gen` (basename) with an integer training-step counter** pulled from the checkpoint dict the worker already loads. Keep the filename as a secondary string for human debugging.

### 3.3 Structural (needed for AC training at scale)

- **Batched leaf evaluation + virtual loss in `run_mcts`.** Collect `B` leaves before calling `network.inference` once on the batch; apply a virtual-loss penalty when adding a leaf to the in-flight set so its siblings stay attractive. Single largest throughput multiplier for the entire pipeline and unblocks GPU self-play.
- **Tree reuse across `play_game` decisions.** After action selection: `root = root.children[action]; root.parent = None`; preserves accumulated visits under the chosen subtree.
- **Step-level on-disk schema.** Write each step as a row `(obs_features, current_qubit, action, reward, root_value, policy_target, weight_gen_int, game_id, step_id)` with Parquet partitioning by `map_class / weight_gen_bucket`. Removes the per-game replay cost entirely and is the natural input to a DataLoader for AC training.

---

## 4. Open questions for the next sweep

- Is `min_max_stats` the reason our empirical `pb_c_init=1.25` feels too exploitative at low sim counts? Fix §3.1 first, then rerun the calibration.
- At what `num_simulations` does 5×5 hit diminishing returns once `mcts_depths` is trustworthy? A run at sims ∈ {50, 200, 500, 1000} under `prior_mix_weight=0` should answer this directly via depth saturation.
- Is `pb_c_init=1.25` still right once Dirichlet α is tuned for our small branching factor? These two knobs interact — sweep jointly, not independently.
- What is the right `β` for `prior_mix_weight` at fixed wall-clock? R13 used a specific value, but cost comparisons at fixed sims are misleading (§1.3 E). The landing point may move when rescored on wall-clock.
- How much of worker throughput is `env.clone()` vs. `network.inference()`? A profiling pass on a single worker before building batched MCTS would tell us whether leaf parallelization is on the critical path or whether the env clone is.

None of these are blocking — they are the backlog for the next calibration round once the §3.1/§3.2 logging changes land.
