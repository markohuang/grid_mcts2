"""Batched NN backend (lc0 `BackendComputation` analog) + optional cache hook.

Contract:
  - `add(features, current_qubit)` queues a leaf request, returns `slot_id`.
  - `compute_blocking()` evaluates all pending un-cached requests in one go;
    cache hits were already written into `_slots_out` during `add`.
  - `get(slot_id)` retrieves the `NetworkOutput` for a queued leaf.

Parity rules:
  - When the cap is 1 (or `use_fake`), we call `Network.inference` per slot —
    byte-identical to classic. This is the path parity tests exercise.
  - When cap > 1, we stack features and call `nnet` once. Outputs are split
    back per slot. Not bit-exact vs per-sample in all corners (torch batched
    reductions can differ by ~1e-7), so parity tests MUST keep cap=1.

Cache (Phase 2): optional `cache` kwarg. None = Phase 1 behaviour unchanged.
When provided, `add` does an O(1) lookup; hit → slot pre-filled and
`compute_blocking` skips it (lc0 `FETCHED_IMMEDIATELY` analog). Newly-computed
misses are inserted after compute. Byte-identical to no-cache at safe knobs
(determinism in eval mode).
"""

from __future__ import annotations
import time
from typing import NamedTuple, Optional

import torch

from neutral_atoms.network import Network, NetworkOutput

from .nn_cache import NNCache, make_feature_key


class LeafObs(NamedTuple):
    features: torch.Tensor   # (num_tasks+1, board_size, num_qubits)
    current_qubit: int


class NNBackend:
    """Pluggable NN evaluator. One instance per `run_mcts` call is fine;
    slot_ids are reset every `compute_blocking()`. Cache (if provided)
    persists across resets — ownership is the caller's."""

    def __init__(self, network: Network, *, cap: int = 1,
                 cache: Optional[NNCache] = None):
        assert cap >= 1
        self._net = network
        self._cap = cap
        self._use_fake = getattr(network, 'use_fake', False)
        self._cache = cache
        self._slots_obs: list[LeafObs] = []
        self._slots_out: list[NetworkOutput | None] = []
        # Per-slot cache key (None if cache disabled OR slot was a cache hit —
        # in the hit case we don't need to re-insert after compute).
        self._slot_keys: list[Optional[tuple]] = []
        # Stats
        self.total_requests = 0
        self.total_batches = 0
        self.max_batch_seen = 0
        self.total_h2d_ms = 0.0   # host-to-device transfer time (0 on CPU)
        self.total_nn_ms = 0.0    # total NN compute time (h2d + forward)
        self.cache_hits = 0       # slots short-circuited via cache
        self.nn_calls = 0         # slots that actually required the net forward

    # --- submission ---

    @property
    def cap(self) -> int:
        return self._cap

    def pending_size(self) -> int:
        return len(self._slots_obs)

    def has_pending(self) -> bool:
        # Only slots that still need compute count as "pending" for gather logic.
        return any(out is None for out in self._slots_out)

    def add(self, features: torch.Tensor, current_qubit: int) -> int:
        slot_id = len(self._slots_obs)
        self._slots_obs.append(LeafObs(features, int(current_qubit)))
        if self._cache is not None:
            key = make_feature_key(features, current_qubit)
            hit = self._cache.get(key)
            if hit is not None:
                # Cache hit: pre-fill slot, don't queue for compute.
                self._slots_out.append(hit)
                self._slot_keys.append(None)
                self.cache_hits += 1
                return slot_id
            # Miss: queue for compute, remember key for post-insert.
            self._slots_out.append(None)
            self._slot_keys.append(key)
            return slot_id
        # No cache
        self._slots_out.append(None)
        self._slot_keys.append(None)
        return slot_id

    def get(self, slot_id: int) -> NetworkOutput:
        out = self._slots_out[slot_id]
        assert out is not None, f"slot {slot_id} not yet computed"
        return out

    # --- compute ---

    def compute_blocking(self) -> None:
        # Indices of slots that still need the NN.
        pending_idx = [i for i, out in enumerate(self._slots_out) if out is None]
        n = len(pending_idx)
        if n == 0:
            return
        self.total_requests += n
        self.nn_calls += n
        self.total_batches += 1
        self.max_batch_seen = max(self.max_batch_seen, n)
        pending_obs = [self._slots_obs[i] for i in pending_idx]
        if n == 1 or self._cap == 1 or self._use_fake:
            outs = [self._infer_single(o) for o in pending_obs]
        else:
            outs = self._infer_batch(pending_obs)
        for i, out in zip(pending_idx, outs):
            self._slots_out[i] = out
            # Insert newly-computed result into cache.
            if self._cache is not None:
                k = self._slot_keys[i]
                if k is not None:
                    self._cache.put(k, out)

    def reset(self) -> None:
        self._slots_obs.clear()
        self._slots_out.clear()
        self._slot_keys.clear()

    # --- internals ---

    @torch.no_grad()
    def _infer_single(self, obs: LeafObs) -> NetworkOutput:
        # Identical to classic leaf eval path.
        return self._net.inference(
            {'features': obs.features, 'current_qubit': obs.current_qubit},
            aslist=True,
        )

    @torch.no_grad()
    def _infer_batch(self, obses: list[LeafObs]) -> list[NetworkOutput]:
        # Stack features and qubit indices, call nnet once, split results.
        features = torch.stack([o.features for o in obses])  # (B, T+1, bs, Q)
        qubits = torch.tensor([o.current_qubit for o in obses], dtype=torch.long)
        device = next(self._net.nnet.parameters()).device
        is_cuda = device.type == 'cuda'
        if is_cuda:
            torch.cuda.synchronize()
        t_nn_start = time.perf_counter()
        features = features.to(device)
        qubits = qubits.to(device)
        if is_cuda:
            torch.cuda.synchronize()
        t_h2d_done = time.perf_counter()
        self.total_h2d_ms += (t_h2d_done - t_nn_start) * 1000
        cv_log, lv_log, pi_log = self._net.nnet(features, current_qubit=qubits)
        if is_cuda:
            torch.cuda.synchronize()
        self.total_nn_ms += (time.perf_counter() - t_nn_start) * 1000
        # (B, num_bins), (B, num_bins), (B, board_size)
        cv_mean = self._net.logits2values(cv_log)  # (B,)
        lv_mean = self._net.logits2values(lv_log)
        cw = self._net.cfg.correctness_weight
        lw = self._net.cfg.latency_weight
        values = (cw * cv_mean + lw * lv_mean)
        # Move to CPU/float for slot consumers (mirrors aslist=True path).
        cv_log_cpu = cv_log.detach().cpu()
        lv_log_cpu = lv_log.detach().cpu()
        pi_log_cpu = pi_log.detach().cpu()
        values_cpu = values.detach().cpu()
        outs: list[NetworkOutput] = []
        for b in range(len(obses)):
            outs.append(NetworkOutput(
                value=float(values_cpu[b].item()),
                correctness_value_logits=cv_log_cpu[b],
                latency_value_logits=lv_log_cpu[b],
                policy_logits=pi_log_cpu[b].tolist(),
            ))
        return outs
