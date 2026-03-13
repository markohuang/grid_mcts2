# Revised Roadmap

## What changed (Phase 0A/0B, completed)

The old MDP had a 129-action space with a GATE_ACTION that let the agent "just execute" layers without reconfiguration. This was the root cause of the execute-immediately attractor (Rounds 01-02).

The new **layer-level per-qubit placement MDP**:
- Action space = `board_size` (12-25 actions, not 129)
- No GATE_ACTION — layers auto-execute after all relevant atoms are placed
- Deterministic episode length = `sum(k_t)` (22-24 steps)
- 100% completion by construction
- Branching factor ~12 vs ~129 → MCTS is ~10x more effective per simulation

## Reflection: what makes sense, what might be over-engineered

### The reward signal

The dense reward (`prev_cost - curr_cost`) works correctly but has a subtle property: **total cumulative reward is constant** (= initial cost) regardless of the agent's moves. This is because `_cached_cost` = remaining cost from current state, and at terminal state remaining cost = 0.

This doesn't break learning:
- The **value network** learns to predict `_cached_cost` at intermediate states, which IS action-dependent and informative
- The **policy network** is trained on MCTS visit counts, which correctly prefer actions leading to lower cost
- TD returns at intermediate states are informative — they differ based on the agent's choices

But it means the value network is essentially learning `compute_total_cost` as a function of board state + task info. This is useful — it needs to understand parallel grouping to predict cost — but it's worth noting that the value target is really just the forward-looking cost heuristic.

### Is the IRM over-engineered for now?

**Yes, probably.** The case for building the IRM first was to bypass the attractor problem. But the attractor is now eliminated by architecture. The new MCTS should be much more effective because:

1. **10x smaller branching factor** — 50 MCTS sims explore ~50/12 ≈ 4 distinct actions per qubit (meaningful coverage) vs 50/129 ≈ 0.4 per original action (essentially random)
2. **No degenerate solution** — the agent must place atoms; there's no "just execute" escape hatch
3. **Episodes are short** — 22-24 steps, well within what MCTS can handle

**Recommendation**: Run Round 04 experiments first. If MCTS + learning finds solutions near or below the old best of 13, the current system may be sufficient. Only build the IRM if MCTS plateaus at a cost significantly above the lower bound.

### What about the qubit placement order?

Currently, qubits are placed in a fixed sorted order within each layer. This is a simplification — the optimal placement for qubit 5 might depend on where qubit 3 was placed, but the agent can't choose the ORDER of placement.

This is fine for now. The action space is already informative (the agent sees the current board state after each placement), and the fixed order avoids the complexity of a "which qubit next?" meta-decision. If we find that order matters, we could add a "choose next qubit" action, but that doubles the decision space.

### What about the cost function being expensive?

`compute_total_cost` is called at every step (both during gameplay and MCTS simulation). It simulates all remaining layers, calling `gates_to_moves` + `count_groups` for each. With 22 steps × 50 sims, that's ~1100 evaluations per decision point.

For the current maps (3 layers, 4-8 gates/layer), this is fast enough. For larger maps, we might need to:
- Cache partial cost computations
- Only recompute the current layer's contribution (since future layers don't change during a layer's placement phase)
- Use the value network to estimate remaining cost instead of computing it exactly

## Development plan

### Phase 1: Validate (Round 04) — immediate next

Run the experiments in `experiments/04_layer_mdp_validation.md`:
1. MCTS-only baseline (FakeNet, varying sims)
2. Learning baseline (20 epochs, Map 0 and Map 1)
3. Simulation count comparison
4. Temperature schedule sweep

**Decision point**: If best_cost < 13 → the system is working, focus on optimization. If best_cost plateaus above 15 → need architectural improvements.

### Phase 2: Optimize (if Phase 1 is promising)

Potential improvements within the current architecture:
- **Cost caching**: only recompute current layer's cost contribution, not all remaining layers
- **Network capacity**: try larger v_hsize/p_hsize, deeper MLPMixer
- **Curriculum**: start with 1-layer problems, scale up
- **Batched MCTS inference**: multiple sims share a single network forward pass
- **Parallel games**: use num_parallel_games > 1 for faster self-play

### Phase 3: IRM (if Phase 1 plateaus)

Only build the IRM if MCTS + learning can't find good solutions on its own. The IRM would:
- Provide a better policy prior for MCTS (currently uniform/random at first)
- Provide leaf evaluation (complete the plan for remaining layers)
- Be trained on MCTS-discovered solutions rather than independently

The IRM architecture would follow the Sudoku solver pattern (see `docs/iterative_refinement_reference.md`):
- Dual-stream IRBackbone (z_H = plan, z_L = scratchpad)
- Differentiable cost via soft logits
- ACT training loop

### Phase 4: Scaling

- Larger maps (Map 2: 5x5, 12 qubits)
- More layers per map
- Generalization across map instances (not just fixed maps)

## Key metrics to track

| Metric | What it tells you |
|--------|-------------------|
| `best_cost` | Solution quality (lower = better, lb = 6) |
| `avg_cost` | Average solution quality (should decrease over training) |
| `completion_rate` | Should always be 100% (by construction) |
| `avg_policy_entropy` | Exploration health (should stay > 0.3, not collapse) |
| `train_total` (loss) | Learning convergence |
| `avg_root_value` | Value estimate quality (should correlate with actual cost) |

## Risk register

| Risk | Impact | Mitigation |
|------|--------|------------|
| Value network can't learn `compute_total_cost` | No search guidance | Check: does loss decrease? Does root_value correlate with actual cost? |
| Policy entropy collapses again | Poor exploration | Monitor entropy; temperature schedule may need tuning |
| MCTS plateaus above 13 | Need IRM or other improvements | Run curriculum, increase sims, try larger networks first |
| Cost computation too slow for large maps | Training bottleneck | Implement cost caching for current-layer-only recomputation |
