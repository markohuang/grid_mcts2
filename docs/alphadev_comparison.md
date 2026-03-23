# AlphaDev vs NeutralAtoms: Systematic Comparison

AlphaDev (sorting assembly programs) is the closest reference implementation to our system. Both use AlphaZero-style MCTS with dual value heads (correctness + latency), categorical value distributions, and environment cloning for search. This doc maps every component and flags divergences.

## Environment Parallel

| | AlphaDev | NeutralAtoms |
|---|---|---|
| **State** | Program (list of instructions) + execution state (memory, registers) | Board (atom positions) + tasks_done + current_atom_idx + current_phase_moves |
| **Action** | Append an assembly instruction | Cell index (0..board_size-1) for the current qubit |
| **Terminal** | All test cases pass OR max program length | All task layers auto-executed (deterministic) |
| **Episode length** | Up to `max_program_size` (100) | `sum(k_t)` — fixed per map (e.g., 22 for Map 0, 24 for Map 1) |
| **Action space** | 271 (fixed) | `board_size`: 12 for Map 0, 16 for Map 1, 25 for Map 2, 64 for Maps 3/4 |

Both are sequential decision problems where the agent builds a solution step by step. Key structural differences: AlphaDev actions are always "append" (program grows monotonically) while NeutralAtoms actions place one qubit at a time in a fixed order, with layers auto-executing after all relevant qubits are placed. NeutralAtoms has no explicit "execute" action and 100% episode completion by construction.

## Reward Structure — THE Critical Comparison

### AlphaDev

```
correctness_reward (per-step):
    correct_items = weighted_sum(output[i] == expected[i])
    reward = correctness_reward_weight * (correct_items - previous_correct_items)
    + correct_reward * all_correct   # bonus for full correctness

latency_reward (terminal-only, ONLY if correct):
    latency = quantile(measure_latency(program), latency_quantile)
    reward = latency * latency_reward_weight
```

**Weights**: `correctness_reward_weight = 2.0`, `latency_reward_weight = 0.5`.

The correctness reward is **dense and incremental** — every instruction that places more items correctly gets positive reward. The latency reward is terminal-only and only kicks in when the program is fully correct.

### NeutralAtoms

```
correctness_reward (per-step, plan_cost mode):
    cost_before = _compute_remaining_cost()   # BEFORE the move
    # apply move
    cost_after  = _compute_remaining_cost()   # AFTER move, BEFORE auto-execute
    reward = reward_scale * (cost_before - cost_after)
    # then: tasks_done++, current_phase_moves cleared — no reward for this

latency_reward (terminal-only, ONLY if all tasks completed):
    reward = -total_move_distance / episode_length
```

**Weights**: `reward_scale = 1.0`, `correctness_weight = 1.0`, `latency_weight = 0.1`.

`_compute_remaining_cost()` sums layer costs for all remaining layers. Computing reward before auto-execute avoids the telescoping-sum problem (see `docs/reward_analysis.md`). The remaining-cost view provides cross-layer signal directly in the reward, unlike older versions.

### Key Differences

1. **Reward weighting**: AlphaDev explicitly weights correctness 4x more than latency (2.0 vs 0.5). We weight them equally. This means AlphaDev's MCTS heavily prioritizes getting the program correct first, then optimizes latency. We don't have this prioritization.

2. **Reward monotonicity**: AlphaDev's correctness reward is nearly monotonic — once items are correctly placed, they tend to stay correct. Our cost_delta can oscillate (move increases cost, execute decreases cost), making credit assignment harder.

3. **Latency gating**: AlphaDev's latency reward is **gated on correctness** — you get 0 latency reward unless the program is fully correct. This creates a clean two-phase learning signal: first learn to be correct, then learn to be fast. Our latency reward (`-final_cost`) is gated on completing all tasks, which is similar but our "correctness" is just "did you execute all layers" (trivially achievable by pressing execute 3 times).

## Value Heads and Training Targets

### AlphaDev

```python
# make_target returns:
correctness_value = sum(rewards[i:i+td_steps] * discount^k)  # TD return
latency_value = env.latency_reward()  # terminal only, same for all steps
bootstrap_discount = discount^td_steps if non-terminal else 0

# In training loss:
target_correctness += bootstrap_discount * V_target(s_{t+n})  # standard TD(n)
target_latency = latency_value  # no bootstrap
```

### NeutralAtoms

```python
# make_target returns: (identical structure)
correctness_value = sum(rewards[i:i+td_steps] * discount^k)
latency_value = -final_cost  # terminal only
bootstrap_discount = discount^td_steps if non-terminal else 0

# In training loss (network.py:289-292):
bootstrap_cv = logits2values(bootstrap_predictions.correctness_value_logits)
target_correctness = (1 - bd) * tc + 0.5 * bd * (tc + bootstrap_cv)
target_latency = latency_value  # no bootstrap (matches AlphaDev)
```

### Bootstrap formula

**Fixed.** Current implementation (`network.py`):

```python
target_correctness = (target_correctness + bootstrap_discount * bootstrap_cv).clip(value_min, value_max)
```

This is `tc + bd * bcv` — the correct AlphaDev standard formula.

## Categorical Value Distribution

| | AlphaDev | NeutralAtoms |
|---|---|---|
| **Range** | [-3, 3] | [-20, 5] |
| **Bins** | 301 | 101 |
| **Resolution** | ~0.02 per bin | ~0.25 per bin |
| **Encoding** | Two-hot (interpolation) | Two-hot (interpolation) |

### Two-hot encoding

**Fixed.** Both now use two-hot encoding. `Network.scalar_to_two_hot` distributes probability between the two adjacent bins proportional to distance, matching AlphaDev's approach.

### Resolution

With `plan_cost` reward mode, typical per-step reward magnitudes are in [-5, +2] and value targets are in [-20, 5]. 101 bins over [-20, 5] gives ~0.25 resolution — fine enough for meaningful gradient signal on single-move improvements.

## MCTS Implementation

The MCTS core (UCB, select, expand, backpropagate, Dirichlet noise) is **identical** between AlphaDev and NeutralAtoms. The implementation is a faithful port. Specific matches:

- `_ucb_score`: same formula
- `_select_child`: same max over UCB
- `_expand_node`: same exp + normalize over legal actions
- `_backpropagate`: same reversed search path with discount
- `_add_exploration_noise`: same Dirichlet + fraction blend
- `MinMaxStats`: same known bounds + normalize

One minor difference: AlphaDev's `run_mcts` calls `sim_env.step(action)` at the end of the while loop (on the leaf node), while ours puts the step inside the while loop. Both are correct — ours just gets the reward from the `result` after the loop exits. Functionally equivalent.

### Scale Difference

| | AlphaDev | NeutralAtoms |
|---|---|---|
| **Simulations/move** | 800 | 50 |
| **Actors** | 128 (TPU) | 1-4 (CPU) |
| **Training batch** | 512 | 128 |
| **Training steps** | 1,000,000 total | 200 per epoch × 50 epochs = 10,000 |
| **Buffer** | 1,000,000 games | 1,000 transitions |

AlphaDev runs ~100x more search and ~100x more training. Our 50 simulations with 109-129 actions means most actions get 0-1 visits — the policy prior dominates. More simulations would let the value estimates actually influence action selection.

## Target Network

| | AlphaDev | NeutralAtoms |
|---|---|---|
| **Method** | Hard copy every 100 steps | EMA with decay=0.995 |
| **Bootstrap usage** | Correctness value only | Correctness value only |

Both are valid. EMA is smoother and doesn't have the stale-copy problem. AlphaDev's hard copy every 100 steps (out of 1M) is effectively similar to a slow-moving average.

## Replay Buffer

| | AlphaDev | NeutralAtoms |
|---|---|---|
| **Storage** | Raw `Game` objects | Pre-computed `TensorDict` |
| **Sampling** | 1 random position per game per batch | Uniform over all materialized transitions |
| **Observation cost** | O(N) per sample (replay env to position) | O(1) per sample (pre-computed) |
| **Save cost** | O(1) per game | O(N²) per game (replay env for each position) |

AlphaDev stores raw games and lazily computes observations at sample time. This means each training batch requires O(batch_size × N) env steps. But with 1M+ games in the buffer and 512 batch size, sampling is well amortized.

Our approach eagerly materializes all transitions when saving a game. This makes sampling O(1) but saving O(N²) — for a 24-step game, that's ~600 env steps per game just for data preparation. With 20 games per epoch, that's 12,000 wasted env steps. **Caching observations during self-play would eliminate this entirely.**

## Optimizer

| | AlphaDev | NeutralAtoms |
|---|---|---|
| **Optimizer** | SGD + momentum (0.9) | AdamW |
| **Learning rate** | 2e-4 | 2e-4 |

Both use the same learning rate. Adam is more forgiving of hyperparameter choices and adapts per-parameter learning rates, which is helpful for small-scale training. SGD + momentum can generalize better at scale but requires more careful tuning.

## Summary of Action Items

| Item | Status |
|------|--------|
| Fix bootstrap formula (`tc + bd * bcv`) | **Done** |
| Implement two-hot encoding | **Done** |
| Add reward weighting config (`correctness_weight`, `latency_weight`) | **Done** (defaults: 1.0, 0.1) |
| Tighten value bins (range + resolution) | **Done** ([-20, 5], 101 bins) |
| Fix telescoping reward (plan_cost mode) | **Done** |
| Cache observations in self-play (eliminate O(N²) replay) | Open |
| Scale up simulations (50 is sparse for larger maps) | Open (experiments use 100–200) |
