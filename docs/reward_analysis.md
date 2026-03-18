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
