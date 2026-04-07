# Reward Design Analysis

## The Original Reward (v1.5 from grid_mcts)

From `/home/marko/grid_mcts/grid_mcts/env.py`, `correctness_reward()`:

```python
def correctness_reward(self) -> float:
    # Step 1: Compute gate execution groups for ALL tasks (not just current layer)
    # by padding remaining actions with gate-actions (i.e., "what if we executed
    # all remaining layers right now from current positions?")
    gate_execs = self.parallel_move_executions(self.tasks, self.actions)
    # gate_execs is a list of lists: gate_execs[layer] = list of parallel groups

    # Step 2: Count total gate execution groups across ALL layers
    curr_gate_execs = sum([len(moves) for moves in gate_execs])
    # e.g., if layers need [3, 2, 3] groups → curr_gate_execs = 8

    # Step 3: Compute correctness as distance from worst case
    cost_ub = len(list(chain.from_iterable(self.tasks)))
    # cost_ub = total number of gates (worst case: each gate in its own group)

    # For v1.5: add entropy shaping
    max_entropy = sum(map(entropy, [len(x)*[1] for x in self.tasks]))
    curr_entropy = sum(map(entropy, [[y.count() for y in x] + ...]))
    cost_ub += max_entropy
    correctness = cost_ub - curr_gate_execs - curr_entropy

    # Step 4: Normalize to [0, 1]
    max_correctness = cost_ub - cost_lb  # cost_lb = num_tasks (best: 1 group/layer)
    correctness /= max_correctness

    # Step 5: DELTA reward (current minus previous)
    reward = correctness_reward_weight * (correctness - self.previous_correctness)
    self.previous_correctness = correctness
    return reward
```

**Key point**: `parallel_move_executions` pads the action sequence with gate-actions to simulate executing ALL remaining layers. It asks: "given the moves made so far, what would the total gate execution cost be if we finished everything right now?" This is computed across ALL layers, not just the current one.

The reward is the **delta** (change from previous step). This is conceptually identical to:
```
reward = prev_all_layers_cost - curr_all_layers_cost
```
(Just rescaled and with entropy shaping.)

**NOTE**: The v1.5 reward only counts gate execution groups — it does NOT include reconfig groups in the cost. The `num_rcfg_executions()` method exists but is only used in v2.

## Option A: All-Layers Cost Delta (our first translation)

Our `compute_total_cost` is a more complete version that includes reconfig cost:

```python
def compute_total_cost(board, atom_positions, tasks, tasks_done, current_phase_moves):
    total_groups = 0
    sim_board = board.clone()
    sim_positions = atom_positions.clone()

    # Iterate over ALL remaining layers (tasks_done..num_tasks)
    for layer_idx in range(tasks_done, len(tasks)):

        # For the current layer: count reconfig groups from moves made so far
        if layer_idx == tasks_done and len(current_phase_moves) > 0:
            reconfig_moves = torch.stack(current_phase_moves)
            total_groups += count_groups(reconfig_moves, canonicalize=False)
            # Apply reconfig to get updated positions for gate cost
            sim_board, sim_positions = apply_moves_batch(sim_board, sim_positions, reconfig_moves)

        # For every remaining layer: count gate execution groups from current positions
        gate_moves = gates_to_moves(tasks[layer_idx], sim_positions)
        if len(gate_moves) > 0:
            total_groups += 2 * count_groups(gate_moves, canonicalize=True)
            # Note: positions don't change between layers in this simulation
            # (gate execution brings atoms together then returns them)

    return total_groups
```

Reward: `prev_total_cost - curr_total_cost` at each step.

This matches the spirit of the original v1.5 — it asks "how many parallel groups would we need to finish everything?" and rewards improvements.

### Why it telescopes

The reward at each step t is: `R_t = C_t - C_{t+1}` where `C_t = compute_total_cost(state_t)`.

Sum over the full episode:
```
R_0 + R_1 + ... + R_T = (C_0 - C_1) + (C_1 - C_2) + ... + (C_{T-1} - C_T)
                       = C_0 - C_T
```
This is a telescoping sum. Every intermediate term cancels.

At the terminal state (all tasks done), `tasks_done == num_tasks`, so `range(tasks_done, num_tasks)` is empty and `C_T = 0`.

Therefore: **Total reward = C_0 - 0 = C_0 = initial cost = CONSTANT** regardless of what actions the agent took.

### Why this breaks MCTS

In MCTS, the Q-value of an action is:
```
Q(s, a) = R(s, a) + V(s')     where s' = state after taking action a
```

With a perfect value function, `V(s') = expected future reward from s'`. Since total future reward from any state equals its `compute_total_cost`, we get `V(s') = C(s')`.

```
Q(s, a) = [C(s) - C(s')] + C(s') = C(s)
```

**Q(s, a) = C(s) for ALL actions a.** MCTS cannot distinguish between actions because every action has the same Q-value. The better the value network learns, the more the Q-values converge, and the worse MCTS performs.

### Does this match the original v1.5?

**Yes.** The original v1.5 uses `correctness - previous_correctness` which is the same delta pattern. The only differences are:
1. v1.5 only counts gate groups (no reconfig cost) — but the telescoping is identical
2. v1.5 adds entropy shaping — this adds a non-telescoping component, but it's small and was proven ineffective (Round 02)

The original system also suffered from this, but it was masked by:
- The GATE_ACTION creating a "when to execute" decision that added genuine variance
- The budget constraint causing some games to not complete, breaking the telescoping
- The value network being inaccurate enough that Q-values didn't fully converge

## Option B: Current-Layer-Only Cost Delta

```python
reward_t = prev_layer_cost - curr_layer_cost   # within-layer delta at each placement step
```

`initial_layer_cost(L)` = `_compute_layer_cost_fast()` at the start of layer L = **gate cost only** (no reconfig yet, since no moves have been made): `2 * count_groups(gate_moves from current atom_positions)`. For L > 0, this depends on where the agent positioned atoms during previous layers.

### Bug in the original implementation (now fixed)

The original code computed the layer_delta reward **after** auto-execute (after `tasks_done++` and `current_phase_moves.clear()`). This means the reward at the last placement of layer L was comparing layer L's partial cost to **layer L+1's initial gate cost** — a cross-layer comparison that caused the total to telescope:

```
total = Σ_L (initial_cost(L) - initial_cost(L+1)) = initial_cost(0) = CONSTANT
```

This is the same pathology as Option A. The within-layer intermediate rewards were non-constant (providing some useful signal), but the total was a constant that couldn't distinguish games with different solution costs.

### Corrected Option B: sum = −actual_solution_cost

The fix adds a correction term at each layer boundary (computed **before** `tasks_done++`):

```python
# At layer L completion (before auto-execute state changes):
curr_layer_cost = _compute_layer_cost_fast()           # still in layer L
step_reward = prev_layer_cost - curr_layer_cost         # last placement's delta
correction  = -_layer_initial_cost                      # cancel the "free" do-nothing baseline
reward = (step_reward + correction) * reward_scale

# Then: tasks_done++, current_phase_moves.clear()
# Save initial cost for next layer:
_layer_initial_cost = _compute_layer_cost_fast()        # layer L+1's initial gate cost
```

Per-layer total:
```
Σ_t(within-layer deltas) + correction
= (initial_cost(L) - final_cost(L)) + (-initial_cost(L))
= -final_cost(L)  ✓
```

Grand total: `-Σ_L final_cost(L) = -actual_solution_cost  ✓`

The dense within-layer deltas distribute the signal across each placement step (useful for credit assignment). The boundary correction ensures the total is exact. The value function V(s_t) now converges to the true expected remaining actual cost under π.

### What it still misses

The within-layer deltas use the do-nothing baseline for **unplaced atoms** in the current layer — e.g. placing atom A's contribution is measured assuming atoms B, C, ... stay put. This is an approximation for within-layer credit assignment, not a bias on the total. The total is exact.

Cross-layer incentives come entirely from `initial_cost(L)` for future layers: good positioning in layer L lowers `initial_cost(L+1)`, which reduces the reward at layer L+1's boundary. This cross-layer signal is indirect but present.

## Option C: Layer-Completion Reward

```
reward = 0 during atom placements
reward = -(reconfig_groups + 2 * gate_groups) when layer auto-executes
```

### Properties

- Total reward = -actual_solution_cost (varies with actions ✓)
- No telescoping (reward is only given at discrete events ✓)
- Cross-layer signal comes through the VALUE NETWORK, not the reward:
  - V(state) must predict: "what is the expected -(remaining cost from here?)"
  - "Remaining cost" includes the current layer being placed AND all future layers
  - So V must learn: "these atom positions are good for the current layer AND future layers"
  - This is the cross-layer reasoning we need

### Trade-off and how to make it usable

Reward is sparse — only at layer boundaries (every ~8 steps on Map 2). Steps that don't reach a boundary get reward=0 and must rely on the value network at the leaf. Cold-starting Option C from scratch fails because the value network has no useful signal initially.

**Solution: set `td_steps ≥ max_atoms_per_layer`.**

`max_atoms_per_layer` is available in `config.env.max_atoms_per_layer` (auto-computed by `set_derived_config`). For Map 2: 8 atoms per layer → set `td_steps=10` (or higher).

With td_steps=10, **every** placement step's n-step target spans the full layer and includes the exact layer completion reward:
```
target(step t in layer L) = 0 + 0 + ... + (-actual_layer_cost)   [from n-step sum]
                           + γ^k * V_θ(s_at_start_of_layer_L+1)  [cross-layer bootstrap]
```

Within-layer: entirely unbiased MC signal (no do-nothing assumption anywhere). Cross-layer: bootstrapped via V_θ, which improves over time as it's trained on unbiased within-layer returns.

**This is the recommended configuration for Option C.** Start from a pretrained specialist checkpoint (so V_θ has a reasonable cross-layer estimate) and set `td_steps` ≥ `max_atoms_per_layer`.

```bash
python main.py --config.env.reward_mode=layer_completion \
               --config.training.td_steps=10 \
               --config.experiment.load_checkpoint=<specialist_ckpt>
```

### What `actual_layer_cost` means

To be precise, when layer t auto-executes, the reward would be:
```python
if self.current_atom_idx >= len(self.relevant_atoms):
    # Layer just completed
    layer_cost = 0
    if self.current_phase_moves:
        layer_cost += count_groups(torch.stack(self.current_phase_moves), canonicalize=False)
    gate_moves = gates_to_moves(self.tasks[self.tasks_done], self.atom_positions)
    if len(gate_moves) > 0:
        layer_cost += 2 * count_groups(gate_moves, canonicalize=True)
    reward = -layer_cost
    # then: tasks_done++, current_phase_moves = []
```

This is the ACTUAL cost of executing this specific layer — reconfig groups for the moves made during this layer + gate groups from the final positions. It's NOT the remaining-cost-of-everything. It's the concrete cost incurred by this layer's execution.

Total reward across the episode = `Σ_layers -layer_cost = -actual_solution_cost`.

## Summary (after Option B fix)

| Property | Option A (all-layers delta) | Option B corrected | Option C (layer-completion) |
|---|---|---|---|
| Total = −actual_cost? | No (constant) | **Yes** | **Yes** |
| Cross-layer signal? | Yes (useless, telescoping) | Indirect (via initial_cost(L+1)) | Via value network |
| Dense signal? | Every step | Every step | Sparse (every ~8 steps) |
| Value function learns | do-nothing baseline → Q collapse | True remaining actual cost | True remaining actual cost |
| Requires | — | — | td_steps ≥ max_atoms_per_layer |
| Cold-start friendly? | Yes | Yes | No (needs pretrained V or long td_steps) |

## Option D: Raw Remaining Cost as Reward

```
reward_t = -compute_total_cost(state_after_action) / episode_length
```

At every step, the agent receives the negative remaining cost (scaled). No delta, no layer-completion trigger.

### Why it doesn't telescope

```
Total reward = Σ_t -C(s_t) / L

This is NOT a telescoping sum. It's the integral of the cost curve.
Different trajectories have different cost curves → different totals.
```

### What it incentivizes

The agent is penalized at every step for the total remaining cost. An action that reduces C from 18 to 14 at step 1 saves penalty for all 23 remaining steps. This creates a strong incentive to **reduce cost early** and to make moves that benefit **all remaining layers** (since C includes all of them).

### Cross-layer signal

C(s) = compute_total_cost includes ALL remaining layers. Moving an atom to help layer 2 reduces C immediately, giving positive signal even if it slightly hurts the current layer. The all-layers view is the cross-layer signal we need.

### MCTS Q-values don't collapse

```
Q(s, a) = -C(s') / L + V_D(s')
V_D(s') = expected sum of -C(s_k) / L for remaining steps
```

V_D(s') depends on the trajectory from s', which depends on s'. Different actions → different s' → different V_D → Q-values differ. No cancellation.

Compare with Option A (delta):
```
Q(s, a) = [C(s) - C(s')] + V_A(s')
If V_A(s') = C(s'), then Q = C(s) - C(s') + C(s') = C(s) = same for all a
```

The key difference: Option A's reward and value CANCEL because they encode the same quantity with opposite signs. Option D's reward is -C(s') and value is the sum of future -C values — these don't cancel.

### Practical considerations

- **Scaling**: Raw rewards ≈ -18 per step, total ≈ -350. Need to scale by 1/episode_length and widen value bins (value_min ≈ -20).
- **compute_total_cost is a heuristic**: It assumes "execute everything from current positions" — no future reconfigurations. As the agent learns better reconfigurations, C(s) underestimates the remaining cost improvement possible. But it's still informative as a signal.
- **Computational cost**: compute_total_cost is called at every step (was previously optimized away by lazy evaluation). For Map 2 this is cheap (~0.3ms). For larger maps it could be a bottleneck.

### Comparison table

| Property | A (all-layers Δ) | B corrected | C (layer completion) | D (raw cost) | E (plan cost) |
|---|---|---|---|---|---|
| Total = −actual_cost? | No (constant) | **Yes** | **Yes** | No (offset) | **Yes** |
| Cross-layer? | Yes (useless) | Indirect | Via value net | Yes (direct) | Yes (direct) |
| Dense? | Every step | Every step | Every ~8 steps | Every step | Every step |
| MCTS distinguishes? | No (Q collapse) | Yes | Yes | Yes | Yes |
| Value function learns | Wrong (collapses) | True V^π | True V^π | Integral of cost | Biased V^π |
| Needs accurate V? | Fatally | Somewhat | Yes + td_steps≥k | Somewhat | Less (biased ok) |
| Status | Broken | **Fixed, to test** | **Recommended + td_steps** | Legacy | Current default |

## Option E: Plan Cost (remaining cost delta, reward computed before auto-execute)

```python
cost_before = _compute_remaining_cost()  # BEFORE the move
# apply move
cost_after_move = _compute_remaining_cost()  # AFTER move, BEFORE auto-execute
reward = cost_before - cost_after_move
# THEN auto-execute (tasks_done++, clear moves) — no reward for this
```

### How it works

Same delta as Option A (`remaining_cost_before - remaining_cost_after`), but the reward
is computed BEFORE the auto-execute state changes. The `tasks_done++` and
`current_phase_moves` clearing happen AFTER reward computation, so they don't
create the large artificial jump that causes telescoping in Option A.

### Why Option A telescopes but Option E doesn't

In Option A, the reward at auto-execute = `C(before_autoexec) - C(after_autoexec)`.
The auto-execute drops the completed layer from `compute_remaining_cost`, creating
a large positive reward (= the layer's full cost). This "free" reward makes the
total sum constant: `Σ rewards = C(initial) - C(terminal) = C(initial) - 0 = constant`.

In Option E, the reward at the auto-execute step is the impact of the LAST PLACEMENT
only. The `tasks_done++` and `current_phase_moves` clearing happen after, contributing
no reward. The "free" layer-drop reward is eliminated.

### Auto-execute rewards are nonzero

The last placement in a layer changes atom positions and adds a reconfig move,
which changes `_compute_remaining_cost()`. Empirically, 60% of auto-execute step
rewards are nonzero (range [-5, +2]). This is the move's actual impact on remaining
cost — genuine signal, not an artifact.

### Total reward

```
total = Σ (cost_before_move - cost_after_move)  [only move impacts, no layer-drop jumps]
      = initial_heuristic - actual_solution_cost  [varies with actions]
```

### Why Q-values don't collapse

In Option A: the total future reward from any state s = `C(s)` regardless of policy
(mathematical identity from telescoping). So V*(s) = C(s), and Q = R + V = C(s) for all actions.

In Option E: the total future reward from state s = `C(s) - Σ(layer_drop_jumps)`.
The layer_drop_jumps depend on what actual costs the policy incurs in future layers.
So V(s) = C(s) - E_π[future_actual_costs], which is POLICY-DEPENDENT.
A good policy incurs lower costs → higher V. This prevents R + V from cancelling.

### Empirical results (Map 2, 5x5, 50 sims)

| Seed | eval_best | eval_avg (converged) |
|------|-----------|---------------------|
| 12315 | **12** | **12.0** |
| 99999 | **12** | 13.0 |

Both seeds reach eval_best=12. Seed A converges to eval_avg=12.0 (every game finds cost 12).

### Failed alternatives

- **Suppress auto-execute to zero**: Forces reward=0 at boundaries, losing the last
  placement's impact. eval_best=16 (both seeds). Worse because it discards genuine signal.
- **accumulated_cost tracking**: Equivalent rewards but unnecessary complexity.
  The clean version (compute reward before auto-execute) is simpler and produces
  identical results.

### Is "biased" the right word for Option E?

No — strictly speaking, Option E is not biased in the RL sense. The total reward formula is:
```
total = initial_heuristic(s0) - actual_solution_cost
      = do_nothing_remaining(s0) - actual_cost
```

`do_nothing_remaining(s0)` is a **state-dependent but action-independent baseline** — it depends only on the initial positions, not on which actions the agent takes. Subtracting an action-independent baseline from the reward does not bias the policy gradient (∇_θ E[total] is unchanged). This is exactly the advantage function trick: subtract a baseline to reduce variance without introducing bias.

The practical consequence: V_θ learns a **shifted** value function — it learns `V(s) ≈ do_nothing_remaining(s) - remaining_actual_cost(s)` instead of `-remaining_actual_cost(s)`. This shifted target is harder to interpret but not wrong. The policy gradient is identical to using the unbiased formulation.

**Why we still care**: For curriculum learning, the do_nothing baseline varies per map, so the value head must learn a different shift per map. Option G removes this by making V_θ learn the true remaining cost directly.

### Comparison table

| Property | A (Δ all-layers) | B (Δ current-layer) | C (layer completion) | D (raw cost) | E (plan cost) | G (plan_cost_unbiased) |
|---|---|---|---|---|---|---|
| Telescopes? | Yes → constant | Within-layer only | No | No | **No** | **No** |
| Total = −actual_cost? | No (constant) | **Yes** | **Yes** | No (offset) | No (shifted) | **Yes** |
| Cross-layer? | Yes (useless) | No | Via value net | Yes (buried) | **Yes (direct)** | **Yes (direct)** |
| Dense? | Every step | Every step | Every ~8 steps | Every step | **Every step** | **Every step** |
| Q-collapse? | Yes (fatal) | Partial | No | No | **No** | **No** |
| Value function learns | do-nothing → wrong | **True V^π** | **True V^π** | integral of cost | Shifted V^π | **True V^π** |
| Best eval (Map 2) | Never tested | 18 (from scratch, 100ep) | 25 (from scratch, 100ep) | 27 | **11** (sims=200) | ~13 (ongoing) |
| Status | Broken | Requires aug; slow | Requires td_steps≥k | Legacy | **Current default** | **New, testing** |

## Option G: Plan Cost Unbiased

Combines Option E's dense cross-layer signal with the unbiased total of Options B/C.

```python
# reset(): save do-nothing total for this episode
self._do_nothing_total = self._compute_remaining_cost()  # at initial positions

# step(): reward = E's signal minus a constant per-step correction
cost_before = self._compute_remaining_cost()
# apply move
cost_after_move = self._compute_remaining_cost()
reward = (cost_before - cost_after_move - self._do_nothing_total / self.episode_length) * reward_scale
```

### Why the correction works

The correction `-do_nothing_total / episode_length` is a **constant per episode** (does not depend on actions). Distributing it uniformly across all steps means:

```
total = Σ_t (cost_before_t - cost_after_t) - do_nothing_total
      = (do_nothing_remaining(s0) - actual_cost) - do_nothing_total
      = -actual_cost
```

The first term is Option E's total (`do_nothing_remaining(s0) - actual_cost`). The subtracted correction cancels `do_nothing_remaining(s0)`, leaving `-actual_cost`. Distributing the correction uniformly avoids the n-step propagation issues of a terminal correction.

### Value function target

V_θ(s) now converges to the true remaining expected cost under π (same as Options B/C), not the shifted version from Option E. This is especially useful for curriculum learning where the do-nothing baseline varies per map.

### Policy gradient

The per-step correction `-do_nothing_total/episode_length` is constant within each episode (independent of actions), so it does not change ∇_θ E[total] compared to Option E. The policy gradient direction is identical; only the value function target improves.

### Status

Run `925e114e` (sims=200, curriculum from ef54adc7, seed=42): ep262 at last check, entropy=1.796, still converging. Results pending.
