"""Constraint-preserving augmentations for neutral atoms reconfiguration.

All augmentations preserve AOD compatibility (coordinate-order-preserving moves).
Applied per-sample to: init_cells (cell indices), task_partner (atom refs), tasks (gate pairs).
"""
import random
import torch


def build_transform(cfg):
    if not cfg.data.augmentation.enabled:
        return None
    H, W = cfg.env.board_height, cfg.env.board_width
    N = cfg.env.num_qubits
    transforms = list(cfg.data.augmentation.transforms)
    return lambda sample: apply_transforms(sample, transforms, H, W, N)


def apply_transforms(sample, transforms, H, W, N):
    init_cells = sample['init_cells'].clone()
    task_partner = sample['task_partner'].clone()
    tasks = [[(a, b) for a, b in layer] for layer in sample['tasks']]

    for t in transforms:
        if t == 'atom_permutation':
            init_cells, task_partner, tasks = _atom_permutation(init_cells, task_partner, tasks, N)
        elif t == 'rotation':
            if H == W:
                init_cells = _rotate_cells(init_cells, H, W)
        elif t == 'reflection':
            init_cells = _reflect_cells(init_cells, H, W)
        elif t == 'transpose':
            if H == W:
                init_cells = _transpose_cells(init_cells, H, W)

    sample = dict(sample)
    sample['init_cells'] = init_cells
    sample['task_partner'] = task_partner
    sample['tasks'] = tasks
    return sample


def _atom_permutation(init_cells, task_partner, tasks, N):
    """Random bijection on atom indices {0..N-1}."""
    perm = torch.randperm(N)
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(N)

    # Permute init_cells: new_init[i] = old_init[perm[i]]
    init_cells = init_cells[perm]

    # Remap tasks: old atom a -> inv_perm[a]
    new_tasks = []
    for layer in tasks:
        new_tasks.append([(inv_perm[a].item(), inv_perm[b].item()) for a, b in layer])

    # Remap task_partner: for each (t, new_atom_i), partner = inv_perm[old_partner_of_perm[i]]
    T = task_partner.shape[0]
    new_tp = torch.full_like(task_partner, N)  # sentinel
    for t_idx in range(T):
        for i in range(N):
            old_i = perm[i].item()
            old_partner = task_partner[t_idx, old_i].item()
            if old_partner < N:  # not sentinel
                new_tp[t_idx, i] = inv_perm[old_partner]

    return init_cells, new_tp, new_tasks


def _rotate_cells(init_cells, H, W):
    """Rotate cell indices by random k*90 degrees (square boards only)."""
    k = random.randint(0, 3)
    if k == 0:
        return init_cells
    r, c = init_cells // W, init_cells % W
    for _ in range(k):
        r, c = c, H - 1 - r
    return r * W + c


def _reflect_cells(init_cells, H, W):
    """Random horizontal or vertical flip."""
    if random.random() < 0.5:
        # Horizontal: c -> W-1-c
        r, c = init_cells // W, init_cells % W
        return r * W + (W - 1 - c)
    else:
        # Vertical: r -> H-1-r
        r, c = init_cells // W, init_cells % W
        return (H - 1 - r) * W + c


def _transpose_cells(init_cells, H, W):
    """Swap rows and columns (square boards only)."""
    if random.random() < 0.5:
        r, c = init_cells // W, init_cells % W
        return c * H + r
    return init_cells
