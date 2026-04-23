# Tests — 4-layer correctness coverage

Each layer catches a different class of error. Together they pin down MCTS
behavior from primitive math up through distribution of outcomes.

## Layer 1 — semantic primitives reused, not reimplemented

**Where**: `fast_mcts/search.py` (import from `neutral_atoms.mcts`).

Fast backend re-imports `Node`, `MinMaxStats`, `_select_child`, `_expand_node`,
`_backpropagate`, `_add_exploration_noise`, `_select_action`, `get_temperature`.
**One source of truth** for UCB, backup, Dirichlet, visit-softmax. Any edit to
these formulas lands in classic and fast simultaneously — caught by existing
`neutral_atoms/test_env.py` and by layer 4 snapshots below.

## Layer 2 — byte-identical parity (`test_parity.py`)

**Contract**: `nn_batch_size=1, virtual_loss=0, prior_mix_weight=0,
deterministic=True, add_exploration_noise=False` → fast trajectory byte-equal
to classic.

```python
ga = play_once('classic', cfg, net, tasks, ip, seed=42, deterministic=True,
               add_exploration_noise=False)
gb = play_once('fast',    cfg, net, tasks, ip, seed=42, deterministic=True,
               add_exploration_noise=False)
assert_snapshot_equal(game_snapshot(ga), game_snapshot(gb))
```

`game_snapshot` captures `history + rewards + root_values + child_visits +
latency_reward + done + tasks_done`.

### Coverage matrix

| Test | Scope |
|------|-------|
| `test_classic_self_parity_maps_fake[0,1,2]` | Each map reproducible |
| `test_classic_self_parity_real_net_small` | Real network path reproducible |
| `test_classic_self_parity_with_noise` | Dirichlet RNG reproducible |
| `test_classic_self_parity_temperature_decay` | `get_temperature` + softmax sampling |
| `test_classic_self_parity_prior_mix` | Prior-mix lookahead branch |
| `test_classic_self_parity_sims_scan[5,25,75]` | Sim-count scan |
| `test_fast_self_parity_maps_fake[0,1,2]` | Fast backend self-reproducibility |
| `test_fast_self_parity_real_net_small` | Fast real-net self-reproducibility |
| `test_cross_classic_vs_fast_defaults[...]` | 6 combos, fast ≡ classic |

20 tests, ~10s wall.

## Layer 3 — structural invariants (`test_invariants.py`)

**Holds at any knob.** Virtual loss + batch > 1 break byte-parity by design,
but must never violate these:

### Per search tree (after one `run_mcts` call)

- `root.visit_count == 1 + num_simulations`
- `sum(child.visit_count for direct children) == num_simulations`
- `set(root.children.keys()) ⊆ set(env.legal_actions())`
- All priors ∈ [0, 1]; sum ≈ 1 (abs_tol=1e-5)

### Per full game

- `game.done == True`; `env.tasks_done == num_tasks`
- `len(game.history) == env.episode_length`
- All actions in [0, action_space)
- Every `child_visits[i]` row sums to 1 (policy target)
- `len(rewards) == len(root_values) == len(child_visits) == len(history)`

### Coverage matrix

| Test | Knobs |
|------|-------|
| `test_mcts_tree_invariants_fake[(0,10),(1,20),(2,25)]` | Tree @ classic, fake |
| `test_mcts_tree_invariants_real` | Tree @ classic, real |
| `test_mcts_tree_invariants_noise` | Tree @ classic, with Dirichlet |
| `test_mcts_tree_invariants_prior_mix` | Tree @ classic, prior-mix branch |
| `test_full_game_invariants[0,1,2][classic,fast]` | Full game, both backends |
| `test_full_game_invariants_real_net` | Real net, classic |
| `test_fast_full_game_invariants_batched[(1,0),(4,0),(8,1),(16,2)]` | Fast non-parity knobs |
| `test_fast_tree_invariants_batched` | Tree @ fast, batch=16 vl=1 |

18 tests, ~5s wall.

## Layer 4 — snapshot regression (`test_snapshot.py`)

**Where**: `fast_mcts/snapshots/*.json`. JSON-serialized `game_snapshot()` at
fixed seeds. Runs classic play_game at 6 scenarios (maps, sim counts, fake/real,
prior-mix). Any change to `mcts.py` / `env.py` / `network.py` / `rewards.py` /
`board.py` that alters output → test fails.

### Scenarios

1. `classic_map0_fake_25sims` — map 0, 25 sims, FakeNet
2. `classic_map1_fake_25sims`
3. `classic_map2_fake_25sims`
4. `classic_map2_fake_50sims` — sim-count variant
5. `classic_map0_real_10sims` — real network
6. `classic_map2_priormix_25sims` — prior-mix=0.5 branch

### Refresh workflow

```bash
UPDATE_SNAPSHOTS=1 ../grid_mcts2/.venv/bin/python -m fast_mcts.test_snapshot
git diff fast_mcts/snapshots/   # review intentional drift
git add fast_mcts/snapshots/    # commit alongside the code change
```

6 tests, ~4s wall.

## Layer 5 — statistical parity (`test_stat_parity.py`)

**Paired design**: same seed → classic vs fast, 20-30 games each, paired
cost differences. Exploration noise **on**. Assertion: `mean_diff` within
`abs_tol` (cost units) **or** within `z_tol * stderr` of zero.

### What it catches

- Virtual-loss too high → starves deep search → higher cost
- Gather loop yields too early → wastes sims on stale UCB
- Descent-order bugs that aren't caught by structural invariants

### Real net (strict: `abs_tol=1.5`, `z_tol=2.5`)

| Scenario | n | classic_mean | fast_mean | diff | z |
|----------|---|--------------|-----------|------|---|
| `real_map0_sims25_batch8_vl1` | 20 | 26.70 | 27.25 | +0.55 | 0.74 |
| `real_map0_sims25_batch16_vl1` | 20 | 26.70 | 25.55 | −1.15 | −1.37 |
| `real_map0_sims25_batch16_vl2` | 20 | 26.70 | 25.15 | −1.55 | −1.62 |

All within noise floor. With learned priors, batching does not degrade quality.

### Fake net smoke (`abs_tol=4.0`, `z_tol=5.0`)

| Scenario | n | classic_mean | fast_mean | diff | z |
|----------|---|--------------|-----------|------|---|
| `fake_map2_sims25_batch8_vl1_smoke` | 30 | 32.97 | 35.47 | +2.50 | 3.18 |

Known noisy: FakeNet uniform priors + batch staleness. **Real net is the
correctness gate.** Kept as smoke check to confirm nothing catastrophic.

4 tests, ~90s wall (dominated by real-net forward passes on CPU).

## Running

```bash
# quick (19s, 44 tests — skip stat parity)
../grid_mcts2/.venv/bin/python -m pytest fast_mcts/ --ignore=fast_mcts/test_stat_parity.py -v

# full (1m47s, 48 tests)
../grid_mcts2/.venv/bin/python -m pytest fast_mcts/ -v

# single scenario (dev loop)
../grid_mcts2/.venv/bin/python -m fast_mcts.test_parity
../grid_mcts2/.venv/bin/python -m fast_mcts.test_invariants
../grid_mcts2/.venv/bin/python -m fast_mcts.test_stat_parity
```

## Bug-catch history

| Bug | Layer caught | Fix |
|-----|--------------|-----|
| `_expand_node` not called on terminal leaves → `leaf.reward` stays 0 → wrong backup value → UCB divergence at step 20+ | Layer 2 cross-parity (classic vs fast, real net, map 0, 10 sims) | `search.py:158` — call `_expand_node(leaf, [], None, leaf_reward, ...)` on terminals |
| Gather loop on terminal leaves didn't yield → 10 sims descend w/o backup → UCB stale → all sims pick same action | Layer 2 cross-parity (same) | `search.py:115` — count terminal leaves toward batch cap |
