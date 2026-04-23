"""Batched NN backend (lc0 `BackendComputation` analog).

Contract:
  - `add(obs)` queues a leaf request, returns `slot_id`.
  - `compute_blocking()` evaluates all pending requests in one go.
  - `get(slot_id)` retrieves the `NetworkOutput` for a queued leaf.

Parity rules:
  - When the cap is 1 (or `use_fake`), we call `Network.inference` per slot —
    byte-identical to classic. This is the path parity tests exercise.
  - When cap > 1, we stack features and call `nnet` once. Outputs are split
    back per slot. Not bit-exact vs per-sample in all corners (torch batched
    reductions can differ by ~1e-7), so parity tests MUST keep cap=1.

We intentionally do not build our own caching layer here — Phase 2 adds a
`memcache`-style short-circuit on top of this interface (`FETCHED_IMMEDIATELY`).
"""

from __future__ import annotations
from typing import NamedTuple
import time

import torch

from neutral_atoms.network import Network, NetworkOutput


class LeafObs(NamedTuple):
    features: torch.Tensor   # (num_tasks+1, board_size, num_qubits)
    current_qubit: int


class NNBackend:
    """Pluggable NN evaluator. One instance per `run_mcts` call is fine;
    slot_ids are reset every `compute_blocking()`."""

    def __init__(self, network: Network, *, cap: int = 1):
        assert cap >= 1
        self._net = network
        self._cap = cap
        self._use_fake = getattr(network, 'use_fake', False)
        self._slots_obs: list[LeafObs] = []
        self._slots_out: list[NetworkOutput | None] = []
        # Stats
        self.total_requests = 0
        self.total_batches = 0
        self.max_batch_seen = 0
        self.total_h2d_ms = 0.0   # host-to-device transfer time (0 on CPU)
        self.total_nn_ms = 0.0    # total NN compute time (h2d + forward)

    # --- submission ---

    @property
    def cap(self) -> int:
        return self._cap

    def pending_size(self) -> int:
        return len(self._slots_obs)

    def has_pending(self) -> bool:
        return bool(self._slots_obs)

    def add(self, features: torch.Tensor, current_qubit: int) -> int:
        slot_id = len(self._slots_obs)
        self._slots_obs.append(LeafObs(features, int(current_qubit)))
        self._slots_out.append(None)
        return slot_id

    def get(self, slot_id: int) -> NetworkOutput:
        out = self._slots_out[slot_id]
        assert out is not None, f"slot {slot_id} not yet computed"
        return out

    # --- compute ---

    def compute_blocking(self) -> None:
        n = len(self._slots_obs)
        if n == 0:
            return
        self.total_requests += n
        self.total_batches += 1
        self.max_batch_seen = max(self.max_batch_seen, n)
        # Parity path: cap==1 OR fake net → per-slot Network.inference (exact).
        # Bulk path: real batch forward, split outputs.
        if n == 1 or self._cap == 1 or self._use_fake:
            for i, obs in enumerate(self._slots_obs):
                out = self._infer_single(obs)
                self._slots_out[i] = out
        else:
            outs = self._infer_batch(self._slots_obs)
            for i, out in enumerate(outs):
                self._slots_out[i] = out

    def reset(self) -> None:
        self._slots_obs.clear()
        self._slots_out.clear()

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
