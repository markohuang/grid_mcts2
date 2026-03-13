# AlphaDev vs NeutralAtoms: Systematic Comparison

AlphaDev (sorting assembly programs) is the closest reference implementation to our system. Both use AlphaZero-style MCTS with dual value heads (correctness + latency), categorical value distributions, and environment cloning for search. This doc maps every component and flags divergences.

## Environment Parallel

| | AlphaDev | NeutralAtoms |
|---|---|---|
| **State** | Program (list of instructions) + execution state (memory, registers) | Board (atom positions) + tasks_done + current_phase_moves |
| **Action** | Append an assembly instruction | Execute gate layer (action 0) or move qubit q to cell p (action 1+) |
| **Terminal** | All test cases pass OR max program length | All task layers executed OR budget exhausted |
| **Episode length** | Up to `max_program_size` (100) | Up to `budget` (24) |
| **Action space** | 271 (fixed) | 109 for map 0, 129 for map 1 (varies by map) |

Both are sequential decision problems where the agent builds a solution step by step. Key structural difference: AlphaDev actions are always "append" (program grows monotonically), while NeutralAtoms has two action types (execute vs move) with the execute action being irreversible (advances `tasks_done`).

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
correctness_reward (per-step):
    reward = reward_scale * (prev_cost - curr_cost)
    # cost = compute_total_cost(remaining_layers)

latency_reward (terminal-only, ONLY if all tasks completed):
    reward = -final_total_cost
```

**Weights**: `reward_scale = 1.0`, no separate weighting for correctness vs latency heads.

The correctness reward is the cost delta from `compute_total_cost`, which simulates all remaining gate layers and counts parallel groups. This is dense but has a problem: reconfig moves often INCREASE total cost (adding a reconfig group) even when they improve gate parallelism, making the immediate reward negative.

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

### **BUG: Bootstrap formula is wrong**

Expanding our formula with `bd = bootstrap_discount`, `tc = td_return`, `bcv = V_target(s_{t+n})`:

```
result = (1 - bd) * tc + 0.5 * bd * (tc + bcv)
       = tc * (1 - 0.5 * bd) + 0.5 * bd * bcv
```

When `bd = 1` (non-terminal, discount=1.0): `result = 0.5 * tc + 0.5 * bcv`

**AlphaDev standard**: `result = tc + bd * bcv = tc + bcv`

Our formula produces targets that are roughly **half** the correct value. The network learns compressed value estimates. This doesn't break training entirely (the network can still learn relative ordering), but it:
- Wastes representational capacity (all values squeezed into half the bin range)
- Makes the value head less informative for MCTS (UCB scores are diluted)
- Means `known_bounds` of [-6, 6] are miscalibrated

**Fix**: Replace with `target_correctness = tc + bd * bootstrap_cv`.

## Categorical Value Distribution

| | AlphaDev | NeutralAtoms |
|---|---|---|
| **Range** | [-3, 3] | [-25, 25] |
| **Bins** | 301 | 51 |
| **Resolution** | ~0.02 per bin | ~1.0 per bin |
| **Encoding** | Two-hot (interpolation) | One-hot (hard assignment) |

### Two-hot vs One-hot

AlphaDev uses `scalar_to_two_hot`: for a target value between bins i and i+1, it distributes probability proportionally to both bins. This preserves gradient flow — a value of 3.7 provides gradient signal to both the bin at 3.0 and the bin at 4.0.

Our `to_onehot` uses `torch.bucketize` to find the nearest bin and creates a hard one-hot. A target of 3.7 assigns all probability to whichever bin is closest. This loses information and creates a discretization bias.

**Fix**: Implement two-hot encoding (linear interpolation between adjacent bins).

### Resolution

With cost values typically in [6, 24] and cost deltas in [-6, 6], our 51 bins over [-25, 25] give ~1.0 resolution. A reconfig move that reduces future gate cost by 0.3 (below our bin resolution) produces no meaningful gradient. AlphaDev's 301 bins over [-3, 3] give 50x finer resolution.

**Fix**: Either increase bins (e.g., 201) or tighten range to match actual value distribution (e.g., [-10, 10] with 101 bins).

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

## Summary of Action Items (priority order)

1. **Fix bootstrap formula** — this is a bug that halves value targets
2. **Implement two-hot encoding** — improves gradient flow for value learning
3. **Add reward weighting config** — `correctness_weight`, `latency_weight` for the combined value
4. **Cache observations in self-play** — eliminates O(N²) waste in save_game
5. **Tighten value bins** — increase resolution to match actual value range
6. **Experiment with more simulations** — 50 is likely too few for 109-129 action space
