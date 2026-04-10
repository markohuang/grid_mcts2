"""
Architecture comparison for 5x5 cost prediction.

Tests whether the MLPMixer can learn cost at scale, and whether
alternative architectures with better inductive biases do better.

Architectures tested:
  1. MLPMixer (current) — baseline
  2. MLPMixer + classifier features (cross-positional pairs)
  3. Transformer (self-attention over qubits replaces patch mixing)
  4. MLPMixer + pairwise distance features (explicit pair distances as channels)

Usage:
    python -m neutral_atoms.arch_sanity_check [--num_samples=200000] [--epochs=500]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time

from .network import MLPMixer
from .config import get_config, set_derived_config, MAPS, atom_map_to_positions
from .map_generator import generate_random_map
from .rewards import compute_total_cost


# ---- Data generation ----

def generate_data(num_samples, board_dim, num_qubits, num_tasks, gates_per_layer):
    """Generate random board configs with their costs. Returns raw data for flexible featurization."""
    board_h, board_w = board_dim
    board_size = board_h * board_w
    boards, positions_list, tasks_list, costs = [], [], [], []
    for i in range(num_samples):
        m = generate_random_map(board_dim, num_qubits, num_tasks, gates_per_layer=gates_per_layer, seed=i)
        ip = atom_map_to_positions(m['atom_map'], board_w)
        board = torch.full((board_h, board_w), -1, dtype=torch.long)
        atom_pos = torch.tensor(ip, dtype=torch.long)
        for q in range(num_qubits):
            board[ip[q][0], ip[q][1]] = q
        cost = compute_total_cost(board, atom_pos, m['tasks'], 0, [])
        boards.append(board)
        positions_list.append(ip)
        tasks_list.append(m['tasks'])
        costs.append(cost)
    return boards, positions_list, tasks_list, torch.tensor(costs, dtype=torch.long)


def augment_board(board, positions, tasks, board_h, board_w):
    """Apply 8 dihedral symmetries to a board config. Returns list of (board, positions, tasks)."""
    results = []
    num_qubits = len(positions)
    for rot in range(4):
        for flip in [False, True]:
            new_pos = []
            for r, c in positions:
                nr, nc = r, c
                for _ in range(rot):
                    nr, nc = nc, board_h - 1 - nr  # 90° clockwise
                if flip:
                    nc = board_w - 1 - nc
                new_pos.append((nr, nc))
            new_board = torch.full((board_h, board_w), -1, dtype=torch.long)
            for q, (nr, nc) in enumerate(new_pos):
                new_board[nr, nc] = q
            results.append((new_board, new_pos, tasks))  # tasks don't change (qubit indices stay same)
    return results


# ---- Feature encoders ----

def featurize_current(board, positions, tasks, board_size, num_qubits, num_tasks):
    """Current env encoding: flat gate participation flags."""
    board_feat = torch.zeros(board_size, num_qubits)
    flat = board.flatten().long()
    for i in range(board_size):
        q = flat[i].item()
        if q >= 0:
            board_feat[i, q] = 1.0
    features = board_feat.unsqueeze(0).expand(num_tasks + 1, -1, -1).clone()
    for t in range(num_tasks):
        for q1, q2 in tasks[t]:
            features[t + 1, :, q1] += 1.0
            features[t + 1, :, q2] += 1.0
    return features  # (num_tasks+1, board_size, num_qubits)


def featurize_classifier(board, positions, tasks, board_size, num_qubits, num_tasks):
    """Classifier encoding: cross-positional pair markers."""
    board_feat = torch.zeros(board_size, num_qubits)
    flat = board.flatten().long()
    q2c = {}
    for i in range(board_size):
        q = flat[i].item()
        if q >= 0:
            board_feat[i, q] = 1.0
            q2c[q] = i
    features = board_feat.unsqueeze(0).expand(num_tasks + 1, -1, -1).clone()
    for t in range(num_tasks):
        for q1, q2 in tasks[t]:
            c1, c2 = q2c.get(q1), q2c.get(q2)
            if c1 is not None and c2 is not None:
                features[t + 1, c1, q2] = 1.0
                features[t + 1, c2, q1] = 1.0
    return features


def featurize_pairwise(board, positions, tasks, board_size, num_qubits, num_tasks, board_h, board_w):
    """Current encoding + explicit pairwise distance channels.
    Adds per-layer channels encoding Manhattan distance to gate partner(s) for each qubit at each cell."""
    base = featurize_current(board, positions, tasks, board_size, num_qubits, num_tasks)
    # Add distance channels: for each layer, each qubit gets its distance to partner(s)
    dist_channels = torch.zeros(num_tasks, board_size, num_qubits)
    for t in range(num_tasks):
        for q1, q2 in tasks[t]:
            r1, c1 = positions[q1]
            r2, c2 = positions[q2]
            d = abs(r1 - r2) + abs(c1 - c2)
            # Normalize by board diagonal
            max_dist = board_h + board_w - 2
            norm_d = d / max(max_dist, 1)
            # Mark distance at both qubits' positions across all cells
            cell1 = r1 * board_w + c1
            cell2 = r2 * board_w + c2
            dist_channels[t, cell1, q1] = norm_d
            dist_channels[t, cell2, q2] = norm_d
            # Also mark partner identity
            dist_channels[t, cell1, q2] = norm_d
            dist_channels[t, cell2, q1] = norm_d
    # Concatenate along task dimension: (num_tasks+1 + num_tasks, board_size, num_qubits)
    return torch.cat([base, dist_channels], dim=0)


# ---- Architectures ----

class MLPMixerPredictor(nn.Module):
    """Current architecture (baseline)."""
    def __init__(self, ntasks_in, board_size, num_qubits, hsize=64, depth=5, num_classes=30):
        super().__init__()
        self.ntasks = ntasks_in
        self.mixer1 = MLPMixer(channels=board_size, depth=depth, dim=hsize, image_size=(num_qubits, 1), patch_size=1)
        self.mixer2 = MLPMixer(channels=num_qubits * hsize, depth=depth, dim=hsize, image_size=(ntasks_in, 1), patch_size=1)
        self.head = nn.Linear(ntasks_in * hsize, num_classes)

    def forward(self, x):
        bs = x.shape[0]
        x = self.mixer1(x.flatten(0, 1).unsqueeze(-1))
        x = self.mixer2(x.reshape(bs, self.ntasks, -1).permute(0, 2, 1).unsqueeze(-1))
        return self.head(x.flatten(1))


class TransformerPredictor(nn.Module):
    """Self-attention over qubits instead of fixed patch mixing."""
    def __init__(self, ntasks_in, board_size, num_qubits, hsize=64, depth=4, num_classes=30, nhead=4):
        super().__init__()
        self.ntasks = ntasks_in
        self.num_qubits = num_qubits
        self.proj_in = nn.Linear(board_size, hsize)
        self.qubit_emb = nn.Parameter(torch.randn(1, num_qubits, hsize) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hsize, nhead=nhead, dim_feedforward=hsize * 4, dropout=0.0, batch_first=True)
        self.qubit_attn = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.task_proj = nn.Linear(num_qubits * hsize, hsize)
        task_layer = nn.TransformerEncoderLayer(d_model=hsize, nhead=nhead, dim_feedforward=hsize * 4, dropout=0.0, batch_first=True)
        self.task_attn = nn.TransformerEncoder(task_layer, num_layers=2)
        self.head = nn.Linear(ntasks_in * hsize, num_classes)

    def forward(self, x):
        bs, ntasks, board_size, nqubits = x.shape
        # Per-task: project board features, attend over qubits
        x = x.permute(0, 1, 3, 2)  # (bs, ntasks, nqubits, board_size)
        x = x.reshape(bs * ntasks, nqubits, board_size)
        x = self.proj_in(x)  # (bs*ntasks, nqubits, hsize)
        x = x + self.qubit_emb
        x = self.qubit_attn(x)  # (bs*ntasks, nqubits, hsize)
        # Aggregate over qubits, attend over tasks
        x = x.reshape(bs, ntasks, -1)  # (bs, ntasks, nqubits*hsize)
        x = self.task_proj(x)  # (bs, ntasks, hsize)
        x = self.task_attn(x)  # (bs, ntasks, hsize)
        return self.head(x.flatten(1))


# ---- Training loop ----

def train_and_eval(model, train_feat, train_cost, test_feat, test_cost,
                   epochs=500, lr=2e-5, batch_size=256, label='', device='cuda'):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    tr_f, tr_c = train_feat.to(device), train_cost.to(device)
    te_f, te_c = test_feat.to(device), test_cost.to(device)
    n_train = len(tr_f)
    params = sum(p.numel() for p in model.parameters())
    print(f'\n{"="*70}')
    print(f'{label} — {params:,} params, {n_train} train, {len(te_f)} test')
    print(f'{"="*70}')

    best_acc, best_pearson = 0, -1
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        idx = torch.randperm(n_train, device=device)
        for s in range(0, n_train, batch_size):
            i = idx[s:s + batch_size]
            loss = F.cross_entropy(model(tr_f[i]), tr_c[i])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (ep + 1) % 50 == 0 or ep == 0:
            model.eval()
            with torch.no_grad():
                preds = []
                for s in range(0, len(te_f), 500):
                    preds.append(model(te_f[s:s + 500]).argmax(1))
                pred = torch.cat(preds).float()
                actual = te_c.float()
                acc = (pred == actual).float().mean().item()
                mae = (pred - actual).abs().mean().item()
                p_num = ((pred - pred.mean()) * (actual - actual.mean())).mean()
                pearson = (p_num / (pred.std() * actual.std() + 1e-8)).item()
                within2 = ((pred - actual).abs() <= 2).float().mean().item()
                if acc > best_acc: best_acc = acc
                if pearson > best_pearson: best_pearson = pearson
            elapsed = time.time() - t0
            print(f'  ep{ep + 1:>4}: acc={acc:.3f} (best={best_acc:.3f}), MAE={mae:.2f}, '
                  f'r={pearson:.3f} (best={best_pearson:.3f}), ±2={within2:.3f}, [{elapsed:.0f}s]')

    print(f'  FINAL: best_acc={best_acc:.3f}, best_pearson={best_pearson:.3f}')
    return {'best_acc': best_acc, 'best_pearson': best_pearson}


# ---- Main ----

def main():
    from absl import app, flags
    flags.DEFINE_integer('num_samples', 200000, 'Raw samples before augmentation')
    flags.DEFINE_integer('epochs', 500, 'Training epochs')
    flags.DEFINE_integer('map_num', 2, 'Map number')
    flags.DEFINE_bool('augment', True, 'Apply 8-fold dihedral augmentation')

    def run(_):
        FLAGS = flags.FLAGS
        config = get_config()
        config.map_num = FLAGS.map_num
        set_derived_config(config)
        map_data = MAPS[config.map_num]
        board_dim = map_data['board_dim']
        board_h, board_w = board_dim
        board_size = board_h * board_w
        num_qubits = map_data['num_qubits']
        num_tasks = len(map_data['tasks'])
        gates_per_layer = max(len(layer) for layer in map_data['tasks'])

        print(f'Map {config.map_num}: {board_dim}, {num_qubits}q, {num_tasks} tasks')
        print(f'Generating {FLAGS.num_samples} raw samples...')
        t0 = time.time()
        boards, positions_list, tasks_list, costs = generate_data(
            FLAGS.num_samples, board_dim, num_qubits, num_tasks, gates_per_layer)
        print(f'  Done in {time.time() - t0:.0f}s, cost range [{costs.min()},{costs.max()}]')
        num_classes = costs.max().item() + 1

        # Augmentation
        if FLAGS.augment:
            print(f'Augmenting with 8-fold symmetry...')
            aug_boards, aug_pos, aug_tasks, aug_costs = [], [], [], []
            for b, p, t, c in zip(boards, positions_list, tasks_list, costs):
                for ab, ap, at in augment_board(b, p, t, board_h, board_w):
                    aug_boards.append(ab)
                    aug_pos.append(ap)
                    aug_tasks.append(at)
                    aug_costs.append(c.item())
            boards, positions_list, tasks_list = aug_boards, aug_pos, aug_tasks
            costs = torch.tensor(aug_costs, dtype=torch.long)
            print(f'  {len(boards)} total samples after augmentation')

        # Split BEFORE augmentation to prevent data leakage.
        # Original indices 0..num_samples-1, each has 8 augmented copies at i*8..i*8+7
        n_raw = FLAGS.num_samples
        n_test_raw = min(5000, n_raw // 5)
        n_train_raw = n_raw - n_test_raw
        perm_raw = torch.randperm(n_raw)
        train_raw_idx = perm_raw[:n_train_raw]
        test_raw_idx = perm_raw[n_train_raw:]

        if FLAGS.augment:
            # Map raw index to augmented indices: raw_i → [raw_i*8 .. raw_i*8+7]
            train_idx = torch.cat([torch.arange(i * 8, i * 8 + 8) for i in train_raw_idx])
            test_idx = torch.cat([torch.arange(i * 8, i * 8 + 8) for i in test_raw_idx])
        else:
            train_idx = train_raw_idx
            test_idx = test_raw_idx

        print(f'Split: {len(train_idx)} train ({n_train_raw} raw), {len(test_idx)} test ({n_test_raw} raw)')

        # Featurize for each encoding
        print(f'Featurizing...')

        def batch_featurize(fn, indices, **kwargs):
            feats = []
            for i in indices:
                feats.append(fn(boards[i], positions_list[i], tasks_list[i],
                               board_size, num_qubits, num_tasks, **kwargs))
            return torch.stack(feats), costs[indices]

        # 1. Current encoding
        print('  Current encoding...')
        tr_curr, tr_c = batch_featurize(featurize_current, train_idx)
        te_curr, te_c = batch_featurize(featurize_current, test_idx)

        # 2. Classifier encoding
        print('  Classifier encoding...')
        tr_cls, _ = batch_featurize(featurize_classifier, train_idx)
        te_cls, _ = batch_featurize(featurize_classifier, test_idx)

        # 3. Pairwise encoding
        print('  Pairwise encoding...')
        tr_pair, _ = batch_featurize(featurize_pairwise, train_idx, board_h=board_h, board_w=board_w)
        te_pair, _ = batch_featurize(featurize_pairwise, test_idx, board_h=board_h, board_w=board_w)

        ntasks_base = num_tasks + 1
        ntasks_pair = ntasks_base + num_tasks  # extra distance channels

        results = {}

        # Experiment 1: MLPMixer + current features (scaled up)
        m1 = MLPMixerPredictor(ntasks_base, board_size, num_qubits, hsize=64, depth=5, num_classes=num_classes)
        results['mlp_current'] = train_and_eval(m1, tr_curr, tr_c, te_curr, te_c,
                                                 epochs=FLAGS.epochs, label='1. MLPMixer + current features')

        # Experiment 2: MLPMixer + classifier features
        m2 = MLPMixerPredictor(ntasks_base, board_size, num_qubits, hsize=64, depth=5, num_classes=num_classes)
        results['mlp_classifier'] = train_and_eval(m2, tr_cls, tr_c, te_cls, te_c,
                                                    epochs=FLAGS.epochs, label='2. MLPMixer + classifier features')

        # Experiment 3: Transformer + current features
        m3 = TransformerPredictor(ntasks_base, board_size, num_qubits, hsize=64, depth=4, num_classes=num_classes)
        results['transformer_current'] = train_and_eval(m3, tr_curr, tr_c, te_curr, te_c,
                                                         epochs=FLAGS.epochs, label='3. Transformer + current features')

        # Experiment 4: Transformer + classifier features
        m4 = TransformerPredictor(ntasks_base, board_size, num_qubits, hsize=64, depth=4, num_classes=num_classes)
        results['transformer_classifier'] = train_and_eval(m4, tr_cls, tr_c, te_cls, te_c,
                                                            epochs=FLAGS.epochs, label='4. Transformer + classifier features')

        # Experiment 5: MLPMixer + pairwise distance features
        m5 = MLPMixerPredictor(ntasks_pair, board_size, num_qubits, hsize=64, depth=5, num_classes=num_classes)
        results['mlp_pairwise'] = train_and_eval(m5, tr_pair, tr_c, te_pair, te_c,
                                                  epochs=FLAGS.epochs, label='5. MLPMixer + pairwise features')

        # Experiment 6: Transformer + pairwise distance features
        m6 = TransformerPredictor(ntasks_pair, board_size, num_qubits, hsize=64, depth=4, num_classes=num_classes)
        results['transformer_pairwise'] = train_and_eval(m6, tr_pair, tr_c, te_pair, te_c,
                                                          epochs=FLAGS.epochs, label='6. Transformer + pairwise features')

        # Summary
        print(f'\n{"="*70}')
        print(f'SUMMARY ({len(boards)} samples, {FLAGS.epochs} epochs)')
        print(f'{"="*70}')
        for name, r in results.items():
            print(f'  {name:30s}: acc={r["best_acc"]:.3f}, pearson={r["best_pearson"]:.3f}')

    app.run(run)


if __name__ == '__main__':
    main()
