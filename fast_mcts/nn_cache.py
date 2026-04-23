"""NN transposition cache (Phase 2).

LRU dict keyed by `(features.tobytes(), current_qubit)`. Value is the
`NetworkOutput` returned by `Network.inference(aslist=True)`.

Ownership: caller (self-play worker / trainer / test) creates the cache once,
passes into `play_game` (or equivalent). Lifetime decides scope:
  - None                    → no cache (Phase 1 behaviour)
  - fresh per play_game     → per-game cache (catches within-game reuse)
  - shared across games     → per-map cache (catches cross-game reuse, the
                              big win — see cache_estimator.py scouting data)

Thread-unsafe. Matches our single-process search model. If we ever move to
multi-threaded search, wrap with a lock.

Correctness: the NN is deterministic in eval mode, so cache hit ≡ re-eval.
Byte-parity tests (`test_parity.py`) must pass with cache enabled at the
safe-knob set (batch=1, vl=0).
"""

from __future__ import annotations
from collections import OrderedDict
from typing import Any, Optional


class NNCache:
    __slots__ = ('_cap', '_store', 'hits', 'misses', 'insertions', 'evictions')

    def __init__(self, cap: int = 50_000):
        assert cap >= 1, "cache cap must be >= 1"
        self._cap = cap
        self._store: OrderedDict[Any, Any] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.insertions = 0
        self.evictions = 0

    # --- core ops ---

    def get(self, key) -> Optional[Any]:
        v = self._store.get(key)
        if v is None:
            # Miss: could be absent, or stored value literally None. We don't
            # store None, so None = absent.
            self.misses += 1
            return None
        # Promote to MRU
        self._store.move_to_end(key)
        self.hits += 1
        return v

    def put(self, key, value) -> None:
        assert value is not None, "cache does not store None"
        if key in self._store:
            # Overwrite + promote (shouldn't normally happen given callers)
            self._store.move_to_end(key)
            self._store[key] = value
            return
        if len(self._store) >= self._cap:
            self._store.popitem(last=False)
            self.evictions += 1
        self._store[key] = value
        self.insertions += 1

    # --- introspection ---

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, key) -> bool:
        return key in self._store

    @property
    def cap(self) -> int:
        return self._cap

    def clear(self) -> None:
        self._store.clear()
        self.hits = 0
        self.misses = 0
        self.insertions = 0
        self.evictions = 0

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            'size': len(self._store),
            'cap': self._cap,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.hits / total if total else 0.0,
            'insertions': self.insertions,
            'evictions': self.evictions,
        }


def make_feature_key(features, current_qubit) -> tuple:
    """Standard key: tensor bytes + qubit index.

    Features are built by env.get_features() on CPU. If the caller passes a
    GPU tensor (shouldn't happen under current fast_mcts), we force a host
    copy — `.cpu()` is a no-op for already-CPU tensors.
    """
    t = features
    if hasattr(t, 'is_cuda') and t.is_cuda:
        t = t.cpu()
    return (t.numpy().tobytes(), int(current_qubit))
