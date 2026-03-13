# TODO — Active Work Items

Workspace scratch pad. Checked items are done but kept for context.

## Critical: Align with AlphaDev

See `docs/alphadev_comparison.md` for full analysis.

- [ ] **Fix bootstrap formula** — our correctness value target is `0.5*(td_return + bootstrap_value)` instead of standard `td_return + bootstrap_discount * bootstrap_value`. This halves the effective value targets. See `network.py:289-292`.
- [ ] **Switch to two-hot encoding** — we use hard one-hot (`to_onehot` via `torch.bucketize`), AlphaDev uses two-hot interpolation (`scalar_to_two_hot`). Two-hot preserves gradient information between adjacent bins.
- [ ] **Add reward weighting** — AlphaDev uses `correctness_reward_weight=2.0`, `latency_reward_weight=0.5`. We weight both heads equally in the combined value (`correctness_mean + latency_mean`). Need config params for these weights.
- [ ] **Increase value resolution** — our 51 bins over [-25, 25] gives ~1.0 resolution per bin. AlphaDev uses 301 bins over [-3, 3] giving ~0.02 resolution. Our rewards are bounded by ±6, so we could tighten the range and/or increase bins.

## Critical: Cache observations during self-play

- [ ] **Eliminate O(N²) make_observation in save_game** — currently, saving a game of N steps replays the env from scratch for every step index (and again for each bootstrap index). For a 24-step game this is ~600 env steps of pure waste. Fix: cache observations during `play_game` (store `obs` and `features` at each step of self-play), then `save_game` just stacks cached tensors. This is the single biggest CPU bottleneck after MCTS itself.
- [ ] **Cache make_features too** — `make_features` is called during MCTS (for every simulation leaf) and again in `save_game`. The root observation features could be cached.

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
