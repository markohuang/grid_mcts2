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

## Option B: Current-Layer-Only Cost Delta (what we have now)

```python
def compute_current_layer_cost(atom_positions, tasks, tasks_done, current_phase_moves):
    if tasks_done >= len(tasks):
        return 0
    total = 0
    # Only count reconfig groups for moves in THIS layer
    if len(current_phase_moves) > 0:
        reconfig_moves = torch.stack(current_phase_moves)
        total += count_groups(reconfig_moves, canonicalize=False)
    # Only count gate groups for THIS layer
    gate_moves = gates_to_moves(tasks[tasks_done], atom_positions)
    if len(gate_moves) > 0:
        total += 2 * count_groups(gate_moves, canonicalize=True)
    return total
```

Reward: `prev_current_layer_cost - curr_current_layer_cost`

### Does it telescope?

Within a layer: yes, the sum telescopes to `initial_layer_cost - final_layer_cost`. But across layers, the layer costs are independent — each layer resets. Total reward = `Σ_layers (initial_layer_cost - final_layer_cost)`, which varies with actions. **Not fully constant.**

### What it misses

The cost function ONLY looks at the current layer. If moving qubit q3 to cell X:
- Makes the current layer worse (costs 1 reconfig group): reward = -1
- But perfectly positions q3 for layers 2 and 3 (saving 4 groups total): NOT SEEN

The agent has no incentive to make cross-layer tradeoffs. It optimizes each layer greedily.

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

### Trade-off

Reward is sparser — only at layer boundaries (every ~8 steps on Map 2). Within a layer, all steps get reward=0. MCTS simulations that don't reach a layer completion only get signal from the value network at the leaf.

This is actually fine — it's how AlphaGo/AlphaZero work (reward only at game end). The value network is what carries the strategic reasoning. The reward just provides the ground truth signal for training it.

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

## Summary

| Property | Option A (all-layers delta) | Option B (current-layer delta) | Option C (layer-completion) |
|---|---|---|---|
| Telescopes? | Yes → constant total | Partially (within-layer) | No |
| Cross-layer signal? | Yes (but useless due to telescoping) | No (fully myopic) | Yes (via value network) |
| Dense signal? | Yes (every step) | Yes (every step) | Sparse (every ~8 steps) |
| MCTS can distinguish? | No (as V improves) | Yes (within a layer) | Yes |
| Value function role | Learn cost(s) → Q-values collapse | Learn within-layer future | Learn remaining solution cost |

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

### Comparison table (updated)

| Property | A (all-layers Δ) | B (layer Δ) | C (layer completion) | D (raw cost) |
|---|---|---|---|---|
| Telescopes? | Yes → constant | Within-layer | No | No |
| Cross-layer? | Yes (useless) | No | Via value net | Yes (direct) |
| Dense signal? | Every step | Every step | Every ~8 steps | Every step |
| MCTS distinguishes? | No (Q collapse) | Yes (within-layer) | Yes | Yes |
| Incentivizes early improvement? | No | No | Weakly | Strongly |
| Needs accurate V? | Fatally (causes collapse) | Somewhat | Yes (carries cross-layer) | Somewhat |

## Option E: Plan Cost (accumulated_actual + remaining_heuristic)

```
plan_cost(state) = accumulated_actual_cost + compute_remaining_cost(state)
reward = plan_cost(before_step) - plan_cost(after_step)
```

### How it works

Track two components:
- `accumulated_cost`: sum of actual costs of completed layers (grows at each auto-execute)
- `_compute_remaining_cost()`: heuristic cost of all remaining layers from current positions

At each step, reward = delta of their sum.

### During atom placements (within a layer)

accumulated_cost doesn't change. Reward = delta of remaining heuristic cost.
This is identical to Option A — cross-layer, dense signal.

### At auto-execute

This is where Option E differs from both A and the simplified version:

1. `_compute_layer_cost_fast()` computes the actual cost of the just-completed layer
2. This actual cost gets added to `accumulated_cost`
3. Meanwhile, `_compute_remaining_cost()` drops the completed layer from its sum

The reward = `plan_before - plan_after` is usually small but NOT forced to zero.

**Why it's nonzero**: The remaining heuristic cost BEFORE auto-execute included the
current layer's estimated cost. The actual layer cost (computed at auto-execute) may
differ because the last atom placement changed the reconfig group count. The reward
captures this discrepancy: "the last placement made this layer X units more/less
expensive than the heuristic predicted."

Empirically: 60% of auto-execute rewards are nonzero, ranging from -5 to +2.
This provides a calibration signal that helps the value network learn the gap
between heuristic estimates and actual costs.

### Why it doesn't telescope

```
total_reward = plan_cost(start) - plan_cost(end)
             = [0 + initial_heuristic] - [actual_solution_cost + 0]
             = initial_heuristic - actual_solution_cost
```

initial_heuristic is constant. actual_solution_cost varies with actions.
Different action sequences → different total rewards. No telescoping.

### Why Q-values don't collapse

In Option A: V*(s) = C(s) (mathematical identity, policy-independent).
In Option E: V(s) = plan_cost(s) - E_π[solution_cost from s].

E_π[solution_cost] depends on how well the policy performs on remaining layers.
A perfect V must predict policy quality, not just state cost.
This makes V(s) policy-dependent → R + V doesn't perfectly cancel → 
Q-values differ between actions even with a well-trained V.

### Empirical results

| Metric (Map 2, 50 sims) | Option B | Option E |
|---|---|---|
| eval_best | 13-16 | **13** |
| eval_avg (converged) | 15-18 | **14.6** |
| noop_frac (epoch 10) | 36% | **66%** |
| MCTS depth (epoch 10) | 3.1 | **3.6** |
| reward_frac | 82% | 73% |

Option E learns "don't move" faster (66% noops vs 36%), goes deeper in MCTS (3.6 vs 3.1),
and achieves lower eval cost (13 vs 16).

### Simplified version (suppress to zero) performs WORSE

Forcing auto-execute reward to exactly 0 loses the calibration signal.
Result: eval_best=16 (both seeds) vs 13 for accumulated_cost version.
The residual nonzero rewards at boundaries help the value network calibrate
its heuristic estimates against actual costs.

### Updated comparison table

| Property | A (Δ all-layers) | B (Δ current-layer) | C (layer completion) | D (raw cost) | E (plan cost) |
|---|---|---|---|---|---|
| Telescopes? | Yes → constant | Within-layer only | No | No | No |
| Cross-layer? | Yes (useless) | No | Via value net | Yes (buried) | **Yes (direct)** |
| Dense? | Every step | Every step | Every ~8 steps | Every step | **Every step** |
| Q-collapse? | Yes (fatal) | Partial | No | No | **No** |
| Auto-exec reward | Large jump | N/A | -layer_cost | N/A | **Small calibration** |
| Signal quality | High then collapses | High (myopic) | Depends on V | Weak (offset) | **High + cross-layer** |
| Best eval (Map 2) | Never tested | 13 | 26 | 27 | **13** |
