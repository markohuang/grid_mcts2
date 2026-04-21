# MCTS Self-Play Diagnostics & Scaling Notes

Scope: the MCTS + self-play infrastructure under `neutral_atoms/` (`mcts.py`, `game.py`, `trainer.py`, `selfplay_worker.py`). Everything outside `neutral_atoms/` — `atom-viz/`, `iterative_refinement/`, `soft_constraint/` — is out of scope here.

This doc has three jobs:

1. A critical read of the current self-play pipeline — what's solid, what's load-bearing, and what will bite us as we scale.
2. A hyperparameter health checklist: which diagnostics already exist, which are missing, and what each one *means* for setting exploration/exploitation constants, simulation budget, temperature, and priors.
3. A concrete logging proposal so that future runs surface enough signal to tune MCTS without re-instrumenting from scratch.

---

## 1. Critical read of the MCTS/self-play infrastructure

### 1.1 What the tree actually does

`mcts.py:play_game` runs `num_simulations` rollouts per decision. Each simulation:

- Clones the env (`env.clone()`) and descends via `_ucb_score` until it hits an unexpanded node.
- Queries the network, expands, and backpropagates the leaf value along the path.
- **Single-player, no negation** (`_backpropagate` just adds `value`).
- `discount=1.0` and `known_bounds=(-6, 6)` are baked in via `MinMaxStats` normalization of `child.reward + discount * child.value()`.

The search is standard AlphaZero-Single-Agent with two non-standard bits:

1. **`plan_cost_search_bonus_weight`** — an optional per-step UCB additive bonus derived from the environment's `plan_cost_delta` info key, scaled by `sim_env.cost_ub`. Off by default (`0.0`). It lets MCTS see a *search-time* shaping signal even when the training reward mode is myopic (`layer_delta`). Important: this bonus is stored on the child (`node.search_bonus`) at *expansion* time of the simulation, so children visited once always use the reward observed on that one visit — it does not get re-averaged if the same edge is re-traversed.
2. **Temperature schedule is driven by `network.training_steps()`** — the step counter of the learner, *not* wall-clock games or environment moves. In distributed self-play this makes temperature global-to-training, which is fine when workers reload weights each batch but surprising if a worker is started with a stale checkpoint.

Action sampling uses `_softmax_sample` on **raw visit counts** with `temperature` as a softmax temperature. Classic AlphaZero uses `N^(1/τ)`; here it's `softmax(N/τ)`. These are very different at large `N`: the `N/τ` form collapses much faster, so `temperature_init=2.0` with `num_simulations=50` is nothing like τ=2.0 in the MuZero paper. **Document this explicitly in any experiment that tunes τ.**

### 1.2 Observation & feature pipeline

`Game.make_observation(i)` has three branches: (a) live observation at `-1`, (b) cached observation via `observation_cache`, (c) *replay the env* from scratch if the step is past the cache. Branch (c) is correct but expensive — and when we load games from the HPC dataset, `Game.from_dict` *replays the entire history* through a fresh env before `cache_observation` is ever called. Because `load_dataset_into_buffer` calls `cache_observation()` *once* after reconstruction, then calls `game_to_tensordict(game, td_steps)` which calls `make_observation(i)` for every step, **every reloaded game currently hits branch (c) for every step**. That's `O(T²)` per game in the env's step cost. At 30-step 5×5 this is tolerable; at 75-step 8×8 with 200k games it is not.

**Fix target:** make `game_to_tensordict` populate the observation cache in a single forward pass (or have `Game.from_dict` do a full replay with caching). This is the single biggest efficiency risk for dataset-driven training.

### 1.3 Per-game compute profile

Per decision, the CPU cost is dominated by `num_simulations` network inferences plus `num_simulations` env clones + steps. On the 5×5 map at `num_simulations=1000` the HPC plan estimates ~5 min/game/core (see `docs/narval_hpc_migration.md`). That's ~100 inferences/sec — i.e. the network is the bottleneck even with `torch.set_num_threads(1)`, which `_play_single_game` correctly sets.

Things to watch when scaling sims:

- **Tree depth is capped by episode length** (`sum(k_t)`, ~30 on 5×5, ~75 on 8×8-30q). Once `num_simulations` exceeds `(branching_factor)^depth / 2` or so, new sims just re-traverse the fully-expanded tree and repeat the same leaf value queries. The `root._mcts_avg_depth` diagnostic measures this — see §2.
- **Clone cost matters.** `env.clone()` is called per simulation. Any state added to `NeutralAtomsEnv` that is not cheap to `deepcopy` immediately inflates this linearly with `num_simulations`.

### 1.4 Replay / dataset path and actor-critic readiness

Two storage paths currently coexist:

| Path | Writer | Reader | Purpose |
|---|---|---|---|
| In-process TensorDict replay buffer | `trainer.save_game` | `trainer.fit` | On-box AlphaZero loop |
| Parquet-indexed game batches on disk | `selfplay_worker.run_worker` | `trainer.load_dataset_into_buffer` | HPC-generated offline dataset |

For building an **actor-critic training database**, the disk path is the one to invest in. Observations worth flagging:

- `data.py:INDEX_SCHEMA` already captures cost, policy entropy, root value, MCTS depth, sims, weight generation — this is a solid seed for a dataset-of-datasets. DuckDB-over-parquet (`query_index`) is the right call for filtered sampling.
- **But the stored game dict only has `history + rewards + child_visits + root_values + mcts_depths + mcts_reward_fracs`** (see `Game.to_dict`). The extra MCTS diagnostics (`reward_sum_means/stds`, `boundary_reach_fracs`, `sign_changes_means`) are **computed and attached to the root node but silently dropped at serialization time**. If you want them in the database, extend `Game.to_dict`/`from_dict` — they exist for free at self-play time.
- There is no on-disk schema for `advantages` or any actor-critic specific targets. Right now everything is computed at buffer-load time from rewards + root values via `Game.make_target`. For actor-critic we probably want to additionally persist: per-step value estimate, bootstrap target, and a stable step-level ID so we can shuffle at the step level without the `O(T²)` per-game replay cost.
- **Weight lineage is loose.** `weight_gen` is just `os.path.basename(weights_path)`. For off-policy corrections we will want a monotonic integer generation and/or the training-step count that produced the weights.

### 1.5 Things that are quietly load-bearing

These aren't bugs but are non-obvious and will cause confusion when scaling:

- `Game.terminal()` returns true if `legal_actions()` is empty, so an unreachable state (no empty cell + current cell is illegal for some reason) would end the episode without completing tasks. The layer-level MDP always has the current cell legal, so this is dead in practice — but worth knowing.
- `_backpropagate` updates `min_max_stats` from `node.value()` (the running mean), not `reward + discount * value`. Normalization therefore tracks visited-child means, which under dense positive rewards can make the normalization window very narrow early on and flatten UCB's value term. If you see "MCTS not exploiting" with aggressive reward scales, this is a likely cause.
- `plan_cost_search_bonus_weight > 0` sets `node.search_bonus` inside the simulation loop using *the last `result.info`*, meaning only the edge that was actually stepped on that simulation gets its bonus set. Unvisited siblings stay at 0.0 bonus. This is a one-shot guidance, not a proper shaping potential.
- `Network.inference` always runs with `torch.no_grad()` from the MCTS call sites but relies on the caller having put the net in eval mode. `trainer.run_selfplay` does `self.network.eval()`; the worker path's `_play_single_game` does too. Don't break this invariant in new entry points.

### 1.6 Scaling blockers (ordered by severity)

1. **`O(T²)` dataset re-loading** — fix `game_to_tensordict` for reloaded games (see §1.2).
2. **Diagnostic leakage at serialization** — extend `Game.to_dict` to include all `_mcts_*` diagnostics so the parquet index can store them.
3. **Temperature semantics** — the `softmax(N/τ)` formula is non-standard; make it explicit in docs and prefer explicit `temperature_override` for eval.
4. **`search_bonus` is ad-hoc** — either promote it to a real shaping potential (with a principled sign and scale) or remove it; right now it makes A/B reproducibility painful.
5. **No step-level ID / weight generation** — required for any off-policy correction in actor-critic training.

---

## 2. Hyperparameter health checklist

The goal of this section: given a self-play run, which already-logged signals should you look at *first* to decide whether your MCTS hyperparameters are reasonable? And which signals are missing that you should add before a big sweep?

### 2.1 `num_simulations` — is the search budget enough?

**What it controls:** branching explored per decision. The policy target is the *visit distribution*, so with too few simulations the targets are noisy and the network is effectively trained on its own prior.

**Healthy signs:**

- `root._mcts_avg_depth` (logged per step into `game.mcts_depths`) should be *well below* the episode remaining length. If average depth ≈ remaining length, the tree has fully expanded and extra sims are wasted. On 5×5, depth in the 5–10 range at `num_simulations=50` is typical; depth saturating near 25+ means you should either *reduce* sims or *increase* the branching via the Dirichlet noise.
- Visit counts at the root should be **concentrated but not collapsed**: the top action should have > 2× the second's count (meaningful preference) but the second shouldn't be zero (else no policy information beyond the argmax).
- The policy target entropy `avg_policy_entropy` (already logged in `selfplay_worker._game_metrics`) tracks this directly. Near-zero entropy means `temperature_init` is too low *or* `num_simulations` is too high for the current exploration mix.

**Unhealthy signs:**

- `mcts_avg_depth` saturating near the remaining-episode length → increase branching, not sims.
- `avg_root_value` drifting far from the empirically observed `correctness_value` target → either value bins are miscalibrated (`network.value_min/max`) or the search isn't propagating enough leaf signal (too few sims *or* too aggressive `min_max_stats` normalization).

**Missing diagnostics worth adding:**

- Root visit-count *Gini* or top-1 fraction — the `avg_policy_entropy` already in the worker covers this but isn't in `game.to_dict`.
- Fraction of `num_simulations` that expanded a genuinely new node (vs. re-traversing already expanded ones). Cheap to count in `run_mcts`.

### 2.2 `pb_c_init` / `pb_c_base` — exploration vs. exploitation

**What they control:** in `_ucb_score`, `pb_c = log((N+base+1)/base) + init`, then scaled by `√N_parent / (N_child+1) * prior`. At the scales we use (`num_simulations ≤ 1000`), the `log` term barely moves (`log(1050/19652) ≈ -2.9`, clamped up by `pb_c_init=1.25`), so **`pb_c_init` is the dominant knob**. `pb_c_base=19652` is the AlphaZero chess/shogi value — it's essentially unused here.

**Healthy signs:**

- `mcts_reward_sum_std / mcts_reward_sum_mean` (already computed per root but *not serialized*) should be non-trivial — sibling simulations exploring genuinely different outcomes. If std collapses to ~0 early, search is stuck in one branch.
- The top child's prior should track its visit fraction within ~2× by mid-training. Large divergence means priors are being overridden by value; small divergence means priors are dominating and the value signal is not informing the tree.

**Unhealthy signs:**

- Fully-concentrated visits with small `mcts_reward_sum_std` → raise `pb_c_init` (typical useful range here: 1.0–2.5).
- High `mcts_reward_sum_std` plus unstable best-action across consecutive epochs → lower `pb_c_init`, the tree is thrashing.

**Missing diagnostic worth adding:**

- A per-decision "prior-dominated vs value-dominated" ratio: fraction of edges where the argmax of `prior_score` differs from the argmax of `value_score`. This is the cleanest single number for diagnosing exploration/exploitation balance.

### 2.3 Dirichlet noise (`root_dirichlet_alpha`, `root_exploration_fraction`)

**What it controls:** how much the root prior is smeared per game. `alpha=0.03` is the Go value; with branching factor ~15 here (empty cells), the noise draw is very spiky — a few actions get most of the noise mass. With `root_exploration_fraction=0.25`, 25% of the prior mass gets redistributed via that spiky draw.

**Healthy signs:**

- Across games on the same map with the same network, the argmax action at the *root* should vary across games but converge over epochs. If argmax never varies, alpha is too low (no exploration in trajectory space).

**Unhealthy signs:**

- If you see games finding the same cost every time on an under-trained net, your policy is likely exploring only the Dirichlet-spiky actions. On our small branching factor, `alpha=0.1–0.3` is more appropriate than 0.03.

**Recommendation for scaling:** sweep `alpha ∈ {0.03, 0.1, 0.25}` as part of any big data-generation run; log per-map best-cost variance across repeated games, which is the cleanest signal.

### 2.4 Temperature (`temperature_init`, `temperature_final`, `temperature_decay_steps`)

**Watch out:** as noted in §1.1, `_softmax_sample` uses `softmax(visit_counts / τ)`, not `N^(1/τ)`. With `num_simulations=50` the effective action distribution at τ=2.0 is *already* nearly uniform, and at τ=0.25 it is essentially argmax. The decay window is in units of **global training steps**, not game steps or games. At `training_steps=200/epoch` and `temperature_decay_steps=1000`, temperature reaches its final value at epoch 5.

**Healthy signs:**

- `avg_policy_entropy` declining monotonically over epochs — not crashing in a single epoch.
- Best-cost improvement continuing after temperature decay completes; if improvement stops *exactly* at decay end, you've annealed too fast.

**Missing diagnostic worth adding:**

- Log the *effective* temperature actually used on each game (not just the config value), because `play_game` reads `network.training_steps()` at call time, which is stale in distributed workers.

### 2.5 Reward / value bounds (`known_bounds`, `value_min/max`, `num_bins`)

These aren't MCTS hyperparameters per se but they determine whether MCTS value normalization is sane.

- `known_bounds=(-6, 6)` only matters via `MinMaxStats.normalize`, which is *running* (updated from observed node values). Setting bounds narrower than actual observed range is harmless (they widen); wider than observed compresses the value term in UCB toward zero. If you log `min_max_stats.maximum - min_max_stats.minimum` at end-of-search, you can check this cheaply.
- `value_min/value_max/num_bins` in `network` have to cover the actual return distribution. `compute_solution_cost` values range 6–50 depending on map; normalized latency via `self.latency_reward = -move_dist/episode_length` is typically in `[-5, 0]`. **If you change `reward_scale`, revisit these bounds or the cross-entropy loss will clip.**

### 2.6 Summary table

| Knob | Primary signal | Secondary signal | Missing data |
|---|---|---|---|
| `num_simulations` | `mcts_depths` mean | `avg_policy_entropy`, best-cost trend | fraction-of-sims-that-expanded |
| `pb_c_init` | visit-count top-1 frac | `mcts_reward_sum_std` (not serialized) | prior-vs-value argmax divergence |
| Dirichlet `alpha`/`frac` | across-game cost variance | per-epoch best-cost stability | per-game root argmax histogram |
| Temperature schedule | `avg_policy_entropy` curve | best-cost improvement past decay | effective τ per game |
| Value bounds | `avg_root_value` vs empirical | normalization width at end-of-search | logged explicitly per batch |

---

## 3. Logging recommendations for the scaling push

If the near-term goal is a large dataset for actor-critic training, the logging changes that give the best leverage per effort are:

### 3.1 Cheap (session-scoped, no schema break)

- **Serialize already-computed MCTS diagnostics.** Extend `Game.to_dict` with `mcts_reward_sum_means`, `mcts_reward_sum_stds`, `mcts_reward_abs_sum_means`, `mcts_boundary_reach_fracs`, `mcts_sign_changes_means`. Mirror in `Game.from_dict`. This unblocks §2.2 and §2.5 diagnostics for free because they already exist per-root.
- **Log effective temperature** inside `_select_action` and stash on the root (e.g. `root._effective_temperature`). Surface via `Game.to_dict`.
- **Add `expanded_sims` counter** in `run_mcts` — increment whenever `not node.expanded()` at the end of a descent. Store `expanded_sims / num_simulations` on the root.

### 3.2 Moderate (schema change, one migration)

- **Extend parquet `INDEX_SCHEMA`** (`data.py`) with: `mcts_reward_std_mean`, `expanded_frac_mean`, `effective_temp_mean`, `top1_visit_frac_mean`. These are one `mean()` each over the game's per-step values. Cost: one column-add and a backfill script for existing shards (or a version bump).
- **Weight generation as monotonic int.** Store training-step count alongside the file name so downstream off-policy weighting is feasible. Small, but forward-compatible.

### 3.3 Structural (needed for actor-critic at scale)

- **Step-level storage option.** Add a writer path that flattens `game` into `(obs, action, reward, root_value, policy_target, weight_gen, step_id)` rows. This removes the per-game replay cost in `load_dataset_into_buffer` and lets you shuffle/sample at the step level — which is what actor-critic wants.
- **Fix `game_to_tensordict` re-loading**: make `Game.from_dict` populate `observation_cache` during the replay loop it already runs, so `make_observation(i)` stays on the cache branch. This is the single biggest throughput fix for dataset-driven training and blocks #3 from being useful on large datasets.
- **Record UCB statistics at decision time** (top-1 visit frac, prior-dominated frac, normalization window width) once per decision into a compact per-step array. These are exactly the signals §2 says are missing.

---

## 4. Open questions for the next sweep

- Does `plan_cost_search_bonus_weight` actually help in the layer-level MDP, given it is off by default and only set for visited edges? Experiments/08 touched this; needs a controlled rerun with the diagnostics from §3.1 to tell us what the search is actually doing.
- At what point does `num_simulations` hit diminishing returns on the 5×5 map? With `mcts_depths` serialized, a single run at sims ∈ {50, 200, 500, 1000} should answer this directly — the depth distribution is the best proxy.
- Is `pb_c_init=1.25` still right once Dirichlet α is tuned for our small branching factor? These two knobs interact and should be swept jointly, not independently.

None of these are blocking — they're the backlog for the next calibration round once the logging changes above land.
