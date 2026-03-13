# TODO — Active Work Items

Workspace scratch pad. Checked items are done but kept for context.

## Critical: Align with AlphaDev

See `docs/alphadev_comparison.md` for full analysis.

- [x] **Fix bootstrap formula** — was `0.5*(td_return + bootstrap_value)`, now standard `td_return + bootstrap_discount * bootstrap_value`
- [x] **Switch to two-hot encoding** — `scalar_to_two_hot` with linear interpolation between adjacent bins, verified roundtrip accuracy
- [x] **Add reward weighting** — `config.network.correctness_weight` and `latency_weight` (default 1.0). Applied in both inference (combined MCTS value) and loss
- [x] **Increase value resolution** — 101 bins over [-10, 10] = ~0.2 per bin (was 51 bins over [-25, 25] = ~1.0 per bin)

## Critical: Cache observations during self-play

- [x] **Eliminate O(N²) make_observation in save_game** — `game.cache_observation()` called in `play_game` at each step + terminal. `save_game` hits cache instead of replaying env. Verified N+1 cache entries for N-step games.

## Reward Shaping

- [x] Implement `gate_only`, `conflict_count`, `manhattan` reward modes
- [ ] Test `gate_only` mode in a proper experiment (Round 03)
- [ ] Consider potential-based shaping: `reward = γ*φ(s') - φ(s)` where φ = gate-only cost

## Performance

- [x] GPU auto-detection (`accelerator='auto'`)
- [x] Parallel self-play with `forkserver` + `torch.set_num_threads(1)`
- [x] Sequential self-play also uses `num_threads=1` (1.6x speedup)
- [ ] Batched MCTS inference — currently 1 network call per simulation per move step (50 sequential calls per move). Collecting multiple leaf nodes and batching would reduce overhead.
- [ ] Vectorize `make_features` inner loop (Python for-loop over tasks/gates → scatter_add)
- [ ] Vectorize `_expand_node` (Python dict comprehension with `math.exp` → `torch.softmax`)
