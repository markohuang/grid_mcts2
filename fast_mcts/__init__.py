"""Efficient / batched MCTS backend.

Non-invasive alternative to `neutral_atoms.mcts`. Consumes the same
`Game`, `NeutralAtomsEnv`, and `Network` interfaces; produces the same
Game output so replay buffer + trainer are unchanged.

Phase 0 ships `bench.py` (profiling harness) and `test_parity.py`
(deterministic classic-vs-future-fast equivalence check). Later phases
will add `backend.py`, `search.py`, `vec_env.py`, `cache.py`, `play.py`.
"""

from .backends import CLASSIC_BACKEND, get_backend  # noqa: F401
