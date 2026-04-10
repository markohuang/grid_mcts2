"""
Sanity checks for architecture and feature representation.

Tests whether the current feature encoding retains enough information
for the network to predict cost. Compares against the richer encoding
used in the original rewards_classifier_mlp.py.

Usage:
    python -m neutral_atoms.sanity_check --num_samples=5000 --epochs=100
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from itertools import chain

from .config import get_config, set_derived_config, MAPS, atom_map_to_positions
from .env import NeutralAtomsEnv
from .experiment import compute_solution_cost
from .network import MLPMixer
from .map_generator import generate_random_map


# ---- Feature encodings ----

def current_features(board, atom_positions, tasks, tasks_done, gate_indicators, num_qubits, board_size):
    """Current env encoding: flat participation flags."""
    board_feat = torch.zeros(board_size, num_qubits)
    flat = board.flatten().long()
    for i in range(board_size):
        q = flat[i].item()
        if q >= 0:
            board_feat[i, q] = 1.0
    num_tasks = len(tasks)
    features = board_feat.unsqueeze(0).expand(num_tasks + 1, -1, -1).clone()
    if tasks_done < num_tasks:
        features[tasks_done + 1:] += gate_indicators[tasks_done:].unsqueeze(1)
    return features


def classifier_features(board, atom_positions, tasks, tasks_done, num_qubits, board_size):
    """Original classifier encoding: cross-positional pair markers."""
    board_feat = torch.zeros(board_size, num_qubits)
    flat = board.flatten().long()
    qubit_to_cell = {}
    for i in range(board_size):
        q = flat[i].item()
        if q >= 0:
            board_feat[i, q] = 1.0
            qubit_to_cell[q] = i
    num_tasks = len(tasks)
    features = board_feat.unsqueeze(0).expand(num_tasks + 1, -1, -1).clone()
    for t in range(num_tasks):
        if t >= tasks_done:
            for q1, q2 in tasks[t]:
                c1 = qubit_to_cell.get(q1)
                c2 = qubit_to_cell.get(q2)
                if c1 is not None and c2 is not None:
                    features[t + 1, c1, q2] = 1.0
                    features[t + 1, c2, q1] = 1.0
    return features


# ---- Data generation ----

def generate_dataset(num_samples, config, encoding='current'):
    """Generate (features, cost) pairs from random board configurations."""
    map_data = MAPS[config.map_num]
    tasks = map_data['tasks']
    board_w = map_data['board_dim'][1]
    board_h = map_data['board_dim'][0]
    board_size = board_h * board_w
    num_qubits = map_data['num_qubits']
    num_tasks = len(tasks)
    gate_indicators = torch.zeros(num_tasks, num_qubits)
    for t, layer in enumerate(tasks):
        for q1, q2 in layer:
            gate_indicators[t, q1] = 1.0
            gate_indicators[t, q2] = 1.0

    features_list = []
    costs_list = []
    gates_per_layer = max(len(layer) for layer in tasks)

    for i in range(num_samples):
        # Generate a random map (random initial positions + random gates)
        m = generate_random_map(
            (board_h, board_w), num_qubits, num_tasks,
            gates_per_layer=gates_per_layer, seed=i,
        )
        sample_tasks = m['tasks']
        ip = atom_map_to_positions(m['atom_map'], board_w)

        # Build board
        board = torch.full((board_h, board_w), -1, dtype=torch.long)
        for q in range(num_qubits):
            r, c = ip[q]
            board[r, c] = q

        # Recompute gate indicators for this map's tasks
        gi = torch.zeros(num_tasks, num_qubits)
        for t, layer in enumerate(sample_tasks):
            for q1, q2 in layer:
                gi[t, q1] = 1.0
                gi[t, q2] = 1.0

        tasks_done = 0
        if encoding == 'current':
            feat = current_features(board, ip, sample_tasks, tasks_done, gi, num_qubits, board_size)
        elif encoding == 'classifier':
            feat = classifier_features(board, ip, sample_tasks, tasks_done, num_qubits, board_size)
        else:
            raise ValueError(f"Unknown encoding: {encoding}")

        # Compute do-nothing cost (no moves, just execute all layers as-is)
        from .rewards import compute_total_cost
        atom_pos = torch.tensor(ip, dtype=torch.long)
        cost = compute_total_cost(board, atom_pos, sample_tasks, 0, [])
        features_list.append(feat)
        costs_list.append(cost)

    features = torch.stack(features_list)
    costs = torch.tensor(costs_list, dtype=torch.long)
    return features, costs


# ---- Simple predictor ----

class CostPredictor(nn.Module):
    def __init__(self, num_tasks, board_size, num_qubits, hsize=64, depth=2, num_classes=30):
        super().__init__()
        self.ntasks = num_tasks + 1
        self.mixer1 = MLPMixer(
            channels=board_size, depth=depth, dim=hsize,
            image_size=(num_qubits, 1), patch_size=1
        )
        self.mixer2 = MLPMixer(
            channels=num_qubits * hsize, depth=depth, dim=hsize,
            image_size=(self.ntasks, 1), patch_size=1
        )
        self.head = nn.Linear(self.ntasks * hsize, num_classes)

    def forward(self, grids):
        bs, ntasks = grids.shape[:2]
        x = self.mixer1(grids.flatten(0, 1).unsqueeze(-1))
        x = self.mixer2(
            x.reshape(bs, ntasks, -1).permute(0, 2, 1).unsqueeze(-1)
        )
        return self.head(x.flatten(1))


# ---- Training and evaluation ----

def train_and_eval(features, costs, num_tasks, board_size, num_qubits,
                   epochs=100, lr=1e-3, batch_size=256, label=''):
    """Train a cost predictor and return final accuracy."""
    num_classes = costs.max().item() + 1
    model = CostPredictor(num_tasks, board_size, num_qubits, num_classes=num_classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    n = len(features)
    n_train = int(0.8 * n)
    perm = torch.randperm(n)
    train_feat, train_cost = features[perm[:n_train]], costs[perm[:n_train]]
    test_feat, test_cost = features[perm[n_train:]], costs[perm[n_train:]]

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}")
    print(f"Training cost predictor: {label}")
    print(f"  Samples: {n_train} train, {n - n_train} test, {num_classes} classes")
    print(f"  Model params: {total_params:,}")
    print(f"{'='*60}")

    for epoch in range(epochs):
        model.train()
        indices = torch.randperm(n_train)
        total_loss, total_correct, total_count = 0, 0, 0
        for start in range(0, n_train, batch_size):
            idx = indices[start:start + batch_size]
            logits = model(train_feat[idx])
            loss = F.cross_entropy(logits, train_cost[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)
            total_correct += (logits.argmax(1) == train_cost[idx]).sum().item()
            total_count += len(idx)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                test_logits = model(test_feat)
                test_acc = (test_logits.argmax(1) == test_cost).float().mean().item()
                train_acc = total_correct / total_count
                # Per-class accuracy for analysis
                pred = test_logits.argmax(1)
                class_acc = {}
                for c in test_cost.unique().tolist():
                    mask = test_cost == c
                    if mask.sum() > 0:
                        class_acc[c] = round((pred[mask] == c).float().mean().item(), 2)
            print(f"  ep{epoch+1:>3}: train_acc={train_acc:.3f}, test_acc={test_acc:.3f}, "
                  f"loss={total_loss/total_count:.4f}")

    # Final evaluation
    model.eval()
    with torch.no_grad():
        test_logits = model(test_feat)
        test_pred = test_logits.argmax(1)
        test_acc = (test_pred == test_cost).float().mean().item()
        # Mean absolute error
        mae = (test_pred - test_cost).float().abs().mean().item()
        # Per-class accuracy
        class_accs = {}
        for c in sorted(test_cost.unique().tolist()):
            mask = test_cost == c
            if mask.sum() >= 5:
                class_accs[c] = round((test_pred[mask] == c).float().mean().item(), 2)

    print(f"\nFinal: test_acc={test_acc:.3f}, MAE={mae:.2f}")
    print(f"Per-class accuracy: {class_accs}")
    return {'test_acc': test_acc, 'mae': mae, 'class_accs': class_accs}


# ---- Gradient flow analysis ----

def check_gradient_flow(model, features, costs, batch_size=128):
    """Measure gradient norms per module."""
    model.train()
    idx = torch.randperm(len(features))[:batch_size]
    logits = model(features[idx])
    loss = F.cross_entropy(logits, costs[idx])
    loss.backward()

    module_grads = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            module = name.split('.')[0]
            g = param.grad.norm().item()
            module_grads.setdefault(module, []).append(g)

    print("\nGradient norms by module:")
    for module, norms in sorted(module_grads.items()):
        mean_norm = sum(norms) / len(norms)
        max_norm = max(norms)
        print(f"  {module:20s}: mean={mean_norm:.4f}, max={max_norm:.4f}, n_params={len(norms)}")
    return module_grads


# ---- Pairwise distance baseline ----

def pairwise_distance_correlation(features, costs, tasks_list, positions_list):
    """Check if sum of Manhattan distances between gate pairs predicts cost."""
    pair_dists = []
    for ip, tasks in zip(positions_list, tasks_list):
        total_dist = 0
        for layer in tasks:
            for q1, q2 in layer:
                r1, c1 = ip[q1]
                r2, c2 = ip[q2]
                total_dist += abs(r1 - r2) + abs(c1 - c2)
        pair_dists.append(total_dist)

    pair_dists = torch.tensor(pair_dists, dtype=torch.float)
    costs_f = costs.float()
    mean_d, mean_c = pair_dists.mean(), costs_f.mean()
    cov = ((pair_dists - mean_d) * (costs_f - mean_c)).mean()
    std_d = pair_dists.std()
    std_c = costs_f.std()
    corr = (cov / (std_d * std_c + 1e-8)).item()
    print(f"\nPairwise distance → cost correlation: r={corr:.3f}")
    print(f"  Distance: mean={mean_d:.1f}, std={std_d:.1f}")
    print(f"  Cost: mean={mean_c:.1f}, std={std_c:.1f}")
    return corr


# ---- Main ----

def main():
    from absl import app, flags
    flags.DEFINE_integer('num_samples', 5000, 'Number of random configurations to generate')
    flags.DEFINE_integer('epochs', 100, 'Training epochs per encoding')
    flags.DEFINE_integer('map_num', 2, 'Map number to use')

    def run(_):
        FLAGS = flags.FLAGS
        config = get_config()
        config.map_num = FLAGS.map_num
        set_derived_config(config)
        map_data = MAPS[config.map_num]
        num_tasks = len(map_data['tasks'])
        board_size = map_data['board_dim'][0] * map_data['board_dim'][1]
        num_qubits = map_data['num_qubits']

        print(f"Map {config.map_num}: {map_data['board_dim']}, {num_qubits} qubits, {num_tasks} tasks")
        print(f"Generating {FLAGS.num_samples} random configurations...")

        # Generate data with both encodings
        print("\n--- Generating with CURRENT encoding ---")
        feat_curr, costs = generate_dataset(FLAGS.num_samples, config, encoding='current')
        print(f"  Features shape: {feat_curr.shape}, Cost range: [{costs.min()}, {costs.max()}]")

        print("\n--- Generating with CLASSIFIER encoding ---")
        feat_cls, costs_cls = generate_dataset(FLAGS.num_samples, config, encoding='classifier')
        assert (costs == costs_cls).all(), "Cost mismatch between encodings"

        # Check feature difference
        diff = (feat_curr - feat_cls).abs()
        print(f"\nFeature difference: mean={diff.mean():.4f}, max={diff.max():.4f}, "
              f"nonzero_frac={(diff > 1e-6).float().mean():.4f}")

        # Train cost predictors with each encoding
        results_curr = train_and_eval(
            feat_curr, costs, num_tasks, board_size, num_qubits,
            epochs=FLAGS.epochs, label='CURRENT encoding (flat participation flags)')

        results_cls = train_and_eval(
            feat_cls, costs, num_tasks, board_size, num_qubits,
            epochs=FLAGS.epochs, label='CLASSIFIER encoding (cross-positional pairs)')

        # Summary
        print(f"\n{'='*60}")
        print(f"COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(f"  Current encoding:    test_acc={results_curr['test_acc']:.3f}, MAE={results_curr['mae']:.2f}")
        print(f"  Classifier encoding: test_acc={results_cls['test_acc']:.3f}, MAE={results_cls['mae']:.2f}")
        gap = results_cls['test_acc'] - results_curr['test_acc']
        print(f"  Gap: {gap:+.3f} accuracy ({'+' if gap > 0 else ''}{gap*100:.1f}%)")
        if gap > 0.05:
            print(f"  → Classifier encoding is significantly better. Feature gap confirmed.")
        elif gap > 0.01:
            print(f"  → Small difference. Feature encoding matters but isn't the only factor.")
        else:
            print(f"  → No meaningful difference. Feature encoding is NOT the bottleneck.")

    app.run(run)


if __name__ == '__main__':
    main()
