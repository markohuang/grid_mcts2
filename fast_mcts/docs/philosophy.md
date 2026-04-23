# Philosophy

Two sources of truth. Neither overridden.

## AlphaDev (`docs/alphadev.py`) — semantic layer

AlphaZero-style MCTS. Our `neutral_atoms/mcts.py` is a faithful port
(per `docs/alphadev_comparison.md`). `fast_mcts` **reuses** those primitives
via direct import — never reimplements:

```python
from neutral_atoms.mcts import (
    Node, MinMaxStats,
    _select_child, _expand_node, _backpropagate, _add_exploration_noise,
    _select_action, get_temperature,
)
```

Consequence: UCB formula, single-player backup (`value = reward + discount *
value`, no to-play flip), Dirichlet root noise, softmax visit sampling,
min-max normalization — all identical. Any edit to the AlphaDev-derived
formulas automatically propagates to fast; we don't maintain a second copy.

### Invariants inherited

- `root.visit_count == 1 + num_simulations` (1 seed backup + N sim backups)
- `sum(child.visit_count for direct children) == num_simulations`
- `_expand_node` early-returns on empty legal actions but **sets `node.reward`
  first** (load-bearing in `_backpropagate`'s `value = node.reward + discount *
  value`). Missing this was the Phase 1 bug.
- Terminal-leaf bootstrap: `leaf_value = 0.0`, no NN call.

## lc0 (`docs/lc0_gpu_selfplay.md`) — efficiency layer

Bolt-on, orthogonal to semantics:

- **`BackendComputation` analog** (`fast_mcts/nn_backend.py::NNBackend`):
  `add(features, current_qubit) -> slot_id`, `compute_blocking()`, `get(slot)`.
  Mirrors lc0 `AddInput` / `ComputeBlocking` / slot-ptr writeback.
- **Virtual loss** on pending paths. Single-player max: `visit_count += 1;
  value_sum -= vl` on descend; reverted before real backup. Steers parallel
  descents toward different leaves.
- **Collision dedup** by `id(leaf_node)`: if descent reaches an already-pending
  leaf within the same batch, share the slot instead of duplicating the NN call.
- **Terminal-leaf short-circuit**: no NN request, backup `leaf_value=0.0`
  immediately. Analog of lc0 `FETCHED_IMMEDIATELY`.

### Not yet adopted

- `memcache` transposition cache → Phase 2
- Cross-process task workers → deliberately skipped (Python GIL; instead we
  gather across many parallel games in one process in a future Phase 1b)
- `batchsplit` auto-split for `maximum_batch_size` → trivial, deferred

## Parity contract

At the "safe knob" set, fast must produce **byte-identical** output to classic:

```
nn_batch_size = 1
virtual_loss  = 0.0
prior_mix_weight = 0.0
```

Enforced by `test_parity.py::test_cross_classic_vs_fast_defaults` across
6 map / sim / net combinations. Not negotiable: any PR that breaks this
must either be reverted or explicitly re-baseline the snapshots with
justification.

At other knobs, parity is **statistical** — `test_stat_parity.py` asserts
mean-cost tracking on real net with paired seeds.

## Guardrails we actively enforce

- Fail fast on unsupported combos: fast backend asserts `prior_mix_weight == 0`
  (Phase 1 doesn't batch the prior-mix lookahead probes; use classic).
- `nn_batch_size >= 1`, `virtual_loss >= 0` — fast backend checks both.
- `NNBackend._infer_single` path when `cap == 1` or `use_fake` — guarantees
  bit-exact equivalence to classic `network.inference(obs, aslist=True)`.
  Batched path only activates when the user explicitly sets `cap > 1`.

## What we do NOT claim

- Fast at `batch > 1` produces equal-or-better quality. It produces
  equivalent-in-distribution quality on real net (z < 2 across tested
  configurations). With fake net it visibly degrades — uniform priors +
  batch staleness. This is an inherent lc0 tradeoff, documented, bounded.
- No bit-exact batched path. Torch reductions across batch dims are not
  guaranteed identical to per-sample calls (~1e-7 drift). Fine for training,
  would fail snapshot parity — so byte-parity tests force `cap = 1`.
