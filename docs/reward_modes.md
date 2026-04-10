# Reward Modes

Four reward modes for training. Each produces a reward signal at every `env.step()` that gets used by MCTS for Q-value estimation and by the value network as a training target.

For detailed mathematical derivations, telescoping proofs, and historical context, see [reward_analysis.md](reward_analysis.md).

---

## 1. plan_cost

**Config**: `config.env.reward_mode='plan_cost'`

Reward = change in total remaining cost from this placement, computed before auto-execute.

- Dense: every placement step produces a non-zero reward
- Cross-layer: remaining_cost includes all future layers, so moving an atom that helps a future layer is rewarded immediately
- Total episode reward = `do_nothing_cost - actual_cost` (shifted by a per-map constant)
- V_θ learns a shifted target: `do_nothing(s) - remaining_cost(s)`. Valid for single-map training but complicates multi-map because the shift differs per map

```python
# env.py step() — plan_cost reward
cost_before = self._compute_remaining_cost()    # all remaining layers from current positions
# ... apply atom placement ...
cost_after_move = self._compute_remaining_cost() # recompute after placement, BEFORE auto-execute
reward = (cost_before - cost_after_move) * self.reward_scale
# auto-execute happens AFTER reward computation — this prevents telescoping
```

`_compute_remaining_cost()` calls `compute_total_cost()`, which simulates executing all remaining layers from current atom positions. For the current layer it includes reconfig groups from moves made so far + gate groups. For future layers it includes only gate groups (no future reconfig — those depend on the policy and are unknown).

---

## 2. layer_delta

**Config**: `config.env.reward_mode='layer_delta'`

Reward = change in current layer's cost from this placement, with a correction at layer boundaries so the total is exact.

- Dense: every placement step produces a reward (within-layer cost change)
- Within-layer only: reward does not directly reflect impact on future layers
- Cross-layer signal comes indirectly: good positioning in layer L lowers `initial_cost(L+1)`, which affects the boundary correction in layer L+1
- Total episode reward = `-actual_cost` (exact, no shift)
- V_θ learns the true remaining cost (unbiased)

```python
# env.py step() — layer_delta reward

# At init / layer boundary: save the layer's initial cost
self._current_layer_cost = self._compute_layer_cost_fast()
self._layer_initial_cost = self._current_layer_cost

# During atom placements within a layer:
prev_layer_cost = self._current_layer_cost
# ... apply atom placement ...
curr_layer_cost = self._compute_layer_cost_fast()
reward = (prev_layer_cost - curr_layer_cost) * self.reward_scale
self._current_layer_cost = curr_layer_cost

# At layer completion (last atom placed, before auto-execute):
curr_layer_cost = self._compute_layer_cost_fast()
reward = ((prev_layer_cost - curr_layer_cost) - self._layer_initial_cost) * self.reward_scale
#          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^     ^^^^^^^^^^^^^^^^^^^^^^^^^
#          last placement's delta               correction: cancels do-nothing baseline
#
# Per-layer total: (initial - final) + (-initial) = -final = -actual_layer_cost
# Episode total:   sum over layers = -actual_cost
```

`_compute_layer_cost_fast()` computes cost for the current layer only: reconfig groups from moves so far + 2× gate groups from current positions. Cheaper than `_compute_remaining_cost()` since it doesn't iterate over future layers.

---

## 3. layer_delta + search bonus

**Config**: `config.env.reward_mode='layer_delta'` + `config.mcts.plan_cost_search_bonus_weight=2.0`

Uses layer_delta for the learning target (unbiased V_θ) but injects plan_cost information into MCTS search as a UCB bonus. This gives MCTS cross-layer guidance without contaminating the value target.

- Same training signal as layer_delta (V_θ learns true remaining cost)
- MCTS search gets cross-layer awareness via the bonus term
- Setting `plan_cost_search_bonus_weight > 0` auto-enables `track_plan_delta=True`, which computes `_compute_remaining_cost()` at every sim step

```python
# mcts.py — search bonus computation during MCTS simulation
if use_plan_bonus:
    # plan_cost_delta = remaining_cost_before - remaining_cost_after for this sim step
    # Normalized by cost_ub so it's in [0, 1] range
    node.search_bonus = result.info['plan_cost_delta'] / max(sim_env.cost_ub, 1)

# mcts.py — UCB score with search bonus
def _ucb_score(config, parent, child, min_max_stats):
    prior_score = pb_c * child.prior                    # policy prior
    value_score = min_max_stats.normalize(               # value estimate
        child.reward + config.discount * child.value()
    )
    search_bonus = config.plan_cost_search_bonus_weight * child.search_bonus
    return prior_score + value_score + search_bonus      # bonus biases search, not learning
```

The bonus only affects which nodes MCTS explores — it does not appear in the reward, value target, or policy target. The value network trains on layer_delta rewards. The policy network trains on MCTS visit counts (which are influenced by the bonus, providing indirect cross-layer signal to the policy).

**Overhead**: `_compute_remaining_cost()` is called at every MCTS simulation step (~1.5× slower per sim than plain layer_delta).

---

## 4. layer_completion

**Config**: `config.env.reward_mode='layer_completion'` + `config.training.td_steps=10`

Reward = zero during placements, negative actual layer cost at layer boundaries. The simplest formulation: no approximations, no baselines, no deltas.

- Sparse: reward only at layer boundaries (every ~8 steps on map2)
- Total episode reward = `-actual_cost` (exact)
- V_θ learns true remaining cost (unbiased)
- Requires `td_steps >= max_atoms_per_layer` so the n-step return reaches the layer boundary reward

```python
# env.py step() — layer_completion reward

# During atom placements within a layer:
reward = 0.0  # no signal until layer completes

# At layer completion (all relevant atoms placed):
if self.current_atom_idx >= len(self.relevant_atoms):
    reward = -self._compute_layer_cost_fast() * self.reward_scale
    #         ^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #         actual cost incurred by executing this layer
    #         = reconfig_groups + 2 * gate_groups
```

**Why td_steps matters**: With default `td_steps=5` and 8 atoms per layer, a placement at step 0 of a layer has its 5-step return cover steps 0–4, missing the layer completion reward at step 7. The value network must fill in the gap. With `td_steps=10`, every step's n-step return includes the full layer completion — direct, unbiased MC signal.

**Cold-start problem**: From scratch, V_θ is random, so all intermediate steps see zero reward and a random bootstrap value. MCTS can't discriminate actions. Previously failed from scratch (R07 7G: best=23 after 300 epochs). Being retested with the new transformer architecture (R11 11D).

---

## Summary

| Mode | Config | Dense? | Cross-layer? | Total reward | V_θ target |
|------|--------|--------|-------------|--------------|------------|
| plan_cost | `reward_mode=plan_cost` | Yes | Yes (direct) | shifted | shifted |
| layer_delta | `reward_mode=layer_delta` | Yes | Indirect | exact | unbiased |
| layer_delta + bonus | `layer_delta` + `search_bonus=2.0` | Yes | Yes (search only) | exact | unbiased |
| layer_completion | `layer_completion` + `td_steps≥8` | Sparse | Via V_θ | exact | unbiased |

### Best results per mode (as of R11)

| Mode | Best specialist | Best generalist | Notes |
|------|----------------|----------------|-------|
| plan_cost | avg=12.0 | avg=17.5 | R06/R09, old architecture |
| layer_delta | avg=12.2 | avg=19.2 | R07/R08 |
| layer_delta + bonus | **avg=11.0 (100%@11)** | avg=15.2 | R10, old arch. R11 retesting with new arch |
| layer_completion | best=23 (failed) | — | R07. R11 retesting with new arch |
