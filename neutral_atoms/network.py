import math
import torch
from torch import nn
from torch.nn import functional as F
from functools import partial
from contextlib import contextmanager
from typing import NamedTuple

from einops.layers.torch import Rearrange



# ---- Feature Construction ----

def make_features(observation: dict, tasks: list) -> torch.Tensor:
    """Convert raw env observation to network features with cross-positional pair encoding.

    Returns: (num_tasks+1, board_size, num_qubits) tensor.
    Slot 0: board state — board_feat[cell, qubit] = 1.0 if qubit occupies cell.
    Slots 1..num_tasks: board state + cross-positional gate pair markers.
      For each gate pair (q1, q2) in layer t (where t >= tasks_done):
        feat[cell_of_q1, q2] = 1.0  (mark partner identity at partner's position)
        feat[cell_of_q2, q1] = 1.0
      This encodes both the pairing structure and spatial relationship between
      gate partners, which is critical for predicting parallel group cost.
    """
    board_onehot = observation['board_onehot']  # (board_size, num_qubits+1)
    board_feat = board_onehot[:, :-1]  # (board_size, num_qubits) drop empty channel
    tasks_done = observation['tasks_done']
    num_tasks = len(tasks)
    num_qubits = board_feat.shape[1]

    # Build qubit-to-cell mapping from the board one-hot encoding
    qubit_cells = board_feat.argmax(dim=0)  # (num_qubits,) — cell index per qubit
    qubit_present = board_feat.sum(dim=0) > 0  # which qubits are on the board

    features = [board_feat.clone()]  # slot 0: raw board state
    for t in range(num_tasks):
        feat = board_feat.clone()
        if t >= tasks_done:
            for q1, q2 in tasks[t]:
                if qubit_present[q1] and qubit_present[q2]:
                    feat[qubit_cells[q1], q2] = 1.0
                    feat[qubit_cells[q2], q1] = 1.0
        features.append(feat)
    return torch.stack(features)  # (num_tasks+1, board_size, num_qubits)


# ---- Network Output ----

class NetworkOutput(NamedTuple):
    value: float
    correctness_value_logits: torch.Tensor
    latency_value_logits: torch.Tensor
    policy_logits: list


# ---- EMA ----

class EMA:
    def __init__(self, parameters, decay):
        self.decay = decay
        self.params = list(parameters)
        self.shadow = [p.data.clone() for p in self.params]

    def update(self, parameters=None):
        params = list(parameters) if parameters is not None else self.params
        for s, p in zip(self.shadow, params):
            s.mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def to(self, device):
        self.shadow = [s.to(device) for s in self.shadow]
        return self

    @contextmanager
    def average_parameters(self):
        old = [p.data.clone() for p in self.params]
        for p, s in zip(self.params, self.shadow):
            p.data.copy_(s)
        yield
        for p, o in zip(self.params, old):
            p.data.copy_(o)


# ---- Building Blocks ----

pair = lambda x: x if isinstance(x, tuple) else (x, x)


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x


def FeedForward(dim, expansion_factor=4, dropout=0., dense=nn.Linear):
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        dense(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(inner_dim, dim),
        nn.Dropout(dropout)
    )


def MLPMixer(*, image_size, channels, patch_size, dim, depth,
             expansion_factor=4, expansion_factor_token=0.5, dropout=0.):
    image_h, image_w = pair(image_size)
    assert (image_h % patch_size) == 0 and (image_w % patch_size) == 0
    num_patches = (image_h // patch_size) * (image_w // patch_size)
    chan_first, chan_last = partial(nn.Conv1d, kernel_size=1), nn.Linear
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                  p1=patch_size, p2=patch_size),
        nn.Linear((patch_size ** 2) * channels, dim),
        *[nn.Sequential(
            PreNormResidual(dim, FeedForward(num_patches, expansion_factor,
                                            dropout, chan_first)),
            PreNormResidual(dim, FeedForward(dim, expansion_factor_token,
                                            dropout, chan_last))
        ) for _ in range(depth)],
        nn.LayerNorm(dim)
    )


# ---- FakeNet (uniform random for testing) ----

class FakeNet:
    def __init__(self, cfg):
        self.num_bins = cfg.num_bins
        self.num_actions = cfg.num_actions  # board_size

    def __call__(self, *args, **kwargs):
        val = torch.ones(1, self.num_bins)
        pi = torch.ones(1, self.num_actions)
        return (F.log_softmax(val, dim=1),
                F.log_softmax(val, dim=1),
                F.log_softmax(pi, dim=1))

    def parameters(self):
        return iter([])

    def update(self, *args, **kwargs):
        pass

    @contextmanager
    def average_parameters(self):
        yield

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


# ---- Qubit Identity Embedding ----

def make_qubit_embedding(num_qubits: int, dim: int) -> torch.Tensor:
    pe = torch.zeros(num_qubits, dim)
    position = torch.arange(num_qubits, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    if dim > 1:
        pe[:, 1::2] = torch.cos(position * div_term[:dim // 2])
    return pe


# ---- Value & Policy Networks ----

class ValueNetwork(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ntasks = cfg.num_tasks + 1
        self.nqubits = cfg.num_qubits
        self.board_size = cfg.board_size
        dim = cfg.v_hsize
        # Stage 1: project board features to qubit embeddings, then self-attend over qubits.
        # Self-attention lets the network learn to attend to gate partners dynamically,
        # which is critical for computing pairwise costs (parallel group counting).
        self.proj_in = nn.Linear(self.board_size, dim)
        self.qubit_emb = nn.Parameter(torch.randn(1, self.nqubits, dim) * 0.02)
        qubit_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=max(1, dim // 16), dim_feedforward=dim * 4,
            dropout=0.0, batch_first=True)
        self.qubit_attn = nn.TransformerEncoder(qubit_layer, num_layers=cfg.mlp_depth)
        # Stage 2: aggregate qubit representations per task, then attend over tasks.
        # This lets the network reason about cross-layer dependencies.
        self.task_proj = nn.Linear(self.nqubits * dim, dim)
        task_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=max(1, dim // 16), dim_feedforward=dim * 4,
            dropout=0.0, batch_first=True)
        self.task_attn = nn.TransformerEncoder(task_layer, num_layers=max(1, cfg.mlp_depth // 2))
        self.W_correctness = nn.Linear(self.ntasks * dim, cfg.num_bins)
        self.W_latency = nn.Linear(self.ntasks * dim, cfg.num_bins)
        for w in [self.W_correctness, self.W_latency]:
            w.weight.data /= 100
            w.bias.data /= 100

    def forward(self, grids):
        bs, ntasks = grids.shape[:2]
        # grids: (bs, ntasks, board_size, nqubits)
        # Transpose to (bs*ntasks, nqubits, board_size) so each qubit sees all cells
        x = grids.flatten(0, 1).permute(0, 2, 1)  # (bs*ntasks, nqubits, board_size)
        x = self.proj_in(x)  # (bs*ntasks, nqubits, dim)
        x = x + self.qubit_emb
        x = self.qubit_attn(x)  # (bs*ntasks, nqubits, dim) — qubits attend to partners
        # Aggregate over qubits, attend over tasks
        x = x.reshape(bs, ntasks, -1)  # (bs, ntasks, nqubits*dim)
        x = self.task_proj(x)  # (bs, ntasks, dim)
        x = self.task_attn(x)  # (bs, ntasks, dim)
        flat = x.flatten(1)
        return self.W_correctness(flat), self.W_latency(flat)


class PolicyNetwork(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ntasks = cfg.num_tasks + 1
        self.nqubits = cfg.num_qubits
        self.board_size = cfg.board_size
        dim = cfg.p_hsize
        # Stage 1: per-task qubit attention (same structure as ValueNetwork).
        self.proj_in = nn.Linear(self.board_size, dim)
        self.qubit_emb = nn.Parameter(torch.randn(1, self.nqubits, dim) * 0.02)
        qubit_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=max(1, dim // 16), dim_feedforward=dim * 4,
            dropout=0.0, batch_first=True)
        self.qubit_attn = nn.TransformerEncoder(qubit_layer, num_layers=cfg.mlp_depth)
        # Stage 2: cross-task attention per qubit.
        # Reshape so each qubit's representations across tasks can attend to each other,
        # then project to per-qubit cell logits.
        self.task_proj = nn.Linear(self.ntasks * dim, dim)
        task_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=max(1, dim // 16), dim_feedforward=dim * 4,
            dropout=0.0, batch_first=True)
        self.task_attn = nn.TransformerEncoder(task_layer, num_layers=max(1, cfg.mlp_depth // 2))
        self.W_pi = nn.Linear(dim, self.board_size)
        self.W_pi.weight.data /= 100
        self.W_pi.bias.data /= 100
        self.softplus = nn.Softplus()

    def forward(self, grids):
        bs, ntasks = grids.shape[:2]
        # grids: (bs, ntasks, board_size, nqubits)
        x = grids.flatten(0, 1).permute(0, 2, 1)  # (bs*ntasks, nqubits, board_size)
        x = self.proj_in(x)  # (bs*ntasks, nqubits, dim)
        x = x + self.qubit_emb
        x = self.qubit_attn(x)  # (bs*ntasks, nqubits, dim) — qubits attend to partners
        # Reshape to per-qubit cross-task: (bs, nqubits, ntasks*dim)
        _, nqubits, dim = x.shape
        x = x.reshape(bs, ntasks, nqubits, dim)
        x = x.permute(0, 2, 1, 3).reshape(bs, nqubits, ntasks * dim)  # (bs, nqubits, ntasks*dim)
        x = self.task_proj(x)  # (bs, nqubits, dim)
        x = self.task_attn(x)  # (bs, nqubits, dim) — per-qubit cross-task reasoning
        return self.softplus(self.W_pi(x))  # (bs, nqubits, board_size)


class NeutralAtomsMLP2(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.value_net = ValueNetwork(cfg)
        self.pi_net = PolicyNetwork(cfg)

    def forward(self, features, current_qubit=None):
        cval, lval = self.value_net(features)
        pi = self.pi_net(features)  # (bs, nqubits, board_size)
        if current_qubit is not None:
            if isinstance(current_qubit, (int, float)):
                # Single qubit index (inference)
                pi = pi[:, current_qubit, :]  # (bs, board_size)
            else:
                # Batched qubit indices (training): (bs,) tensor
                bs = pi.shape[0]
                pi = pi[torch.arange(bs, device=pi.device), current_qubit]  # (bs, board_size)
        return (F.log_softmax(cval, dim=1),
                F.log_softmax(lval, dim=1),
                F.log_softmax(pi, dim=-1))


# ---- Main Network Wrapper ----

class Network(nn.Module):
    def __init__(self, cfg, use_fake: bool = False):
        super().__init__()
        self.cfg = cfg
        self.use_fake = use_fake
        self.action_space_size = cfg.num_actions
        if use_fake:
            self.nnet = FakeNet(cfg)
            self.t_nnet = FakeNet(cfg)
        else:
            self.nnet = NeutralAtomsMLP2(cfg)
            self.t_nnet = EMA(self.nnet.parameters(), decay=cfg.ema_decay)
        self.register_buffer(
            'categories',
            torch.linspace(cfg.value_min, cfg.value_max, steps=cfg.num_bins)[:, None]
        )
        self._training_steps = 0

    def inference(self, observation: dict, aslist=False) -> NetworkOutput:
        features = observation['features']
        current_qubit = observation.get('current_qubit', None)
        if features.dim() == 3:
            features = features[None, :]  # add batch dim
        if not self.use_fake:
            device = next(self.nnet.parameters()).device
            features = features.to(device)
        if self.use_fake:
            output = self.nnet(features)
        else:
            output = self.nnet(features, current_qubit=current_qubit)
        correctness_logits, latency_logits, pi = output
        correctness_mean = self.logits2values(correctness_logits)
        latency_mean = self.logits2values(latency_logits)
        cw, lw = self.cfg.correctness_weight, self.cfg.latency_weight
        if aslist:
            return NetworkOutput(
                value=(cw * correctness_mean + lw * latency_mean).item(),
                correctness_value_logits=correctness_logits.squeeze(),
                latency_value_logits=latency_logits.squeeze(),
                policy_logits=pi.squeeze().tolist(),
            )
        return NetworkOutput(
            value=cw * correctness_mean + lw * latency_mean,
            correctness_value_logits=correctness_logits,
            latency_value_logits=latency_logits,
            policy_logits=pi,
        )

    def forward(self, batch):
        obs_with_qubit = {
            'features': batch['obs']['features'],
            'current_qubit': batch['obs']['current_qubit'],
        }
        predictions = self.inference(obs_with_qubit)
        with self.t_nnet.average_parameters(), torch.no_grad():
            bootstrap_predictions = self.inference(batch['bootstrap_obs'])

        target_correctness = batch['target']['correctness_values']
        target_latency = batch['target']['latency_values']
        target_policy = batch['target']['policies']
        bootstrap_discount = batch['target']['bootstrap_discounts']

        bootstrap_cv = self.logits2values(
            bootstrap_predictions.correctness_value_logits
        )
        target_correctness = (
            target_correctness + bootstrap_discount * bootstrap_cv
        ).clip(self.cfg.value_min, self.cfg.value_max)

        policy_loss = F.cross_entropy(predictions.policy_logits, target_policy)
        correctness_loss = F.cross_entropy(
            predictions.correctness_value_logits,
            self.scalar_to_two_hot(target_correctness)
        )
        latency_loss = F.cross_entropy(
            predictions.latency_value_logits,
            self.scalar_to_two_hot(target_latency)
        )
        cw, lw = self.cfg.correctness_weight, self.cfg.latency_weight
        total = (policy_loss + cw * correctness_loss + lw * latency_loss).mean()

        return {
            'total': total,
            'policy': policy_loss.mean().item(),
            'correctness': correctness_loss.mean().item(),
            'latency': latency_loss.mean().item(),
        }

    def logits2values(self, logits):
        return (torch.exp(logits) @ self.categories).squeeze(-1)

    def scalar_to_two_hot(self, val):
        bins = self.categories.squeeze()
        val = val.clamp(bins[0], bins[-1])
        idx_below = torch.bucketize(val, bins, right=True).clamp(1, len(bins) - 1) - 1
        idx_above = idx_below + 1
        weight_above = (val - bins[idx_below]) / (bins[idx_above] - bins[idx_below])
        weight_below = 1.0 - weight_above
        two_hot = torch.zeros(*val.shape, self.cfg.num_bins, device=val.device)
        two_hot.scatter_(-1, idx_below.unsqueeze(-1), weight_below.unsqueeze(-1))
        two_hot.scatter_(-1, idx_above.unsqueeze(-1), weight_above.unsqueeze(-1))
        return two_hot

    def training_steps(self) -> int:
        return self._training_steps
