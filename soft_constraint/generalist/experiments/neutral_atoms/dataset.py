"""Streaming dataset for neutral atoms reconfiguration."""
import random
import torch
from torch.utils.data import IterableDataset, Dataset

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from config import generate_random_map


def _build_task_partner(tasks, N):
    """Build (T, N) tensor of gate partner indices. Sentinel = N (no gate)."""
    T = len(tasks)
    partner = torch.full((T, N), N, dtype=torch.long)
    for t, gates in enumerate(tasks):
        for a, b in gates:
            partner[t, a] = b
            partner[t, b] = a
    return partner


class NeutralAtomsStreamDataset(IterableDataset):
    def __init__(self, board_dim, num_qubits, num_layers, gates_per_layer,
                 seed=42, transform=None):
        self.board_dim = board_dim
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.gates_per_layer = gates_per_layer
        self.seed = seed
        self.transform = transform

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_seed = self.seed + (worker_info.id if worker_info else 0)
        rng = random.Random(worker_seed)
        while True:
            map_seed = rng.randint(0, 2**31)
            m = generate_random_map(
                self.board_dim, self.num_qubits, self.num_layers,
                self.gates_per_layer, seed=map_seed)
            init_cells = torch.tensor(m['atom_map'], dtype=torch.long)
            task_partner = _build_task_partner(m['tasks'], self.num_qubits)
            sample = {'init_cells': init_cells, 'task_partner': task_partner,
                      'tasks': m['tasks']}
            if self.transform is not None:
                sample = self.transform(sample)
            yield sample['init_cells'], sample['task_partner'], sample['tasks']


class NeutralAtomsFixedDataset(Dataset):
    def __init__(self, board_dim, num_qubits, num_layers, gates_per_layer,
                 pool_size=500, base_seed=99999):
        self.samples = []
        for i in range(pool_size):
            m = generate_random_map(board_dim, num_qubits, num_layers,
                                    gates_per_layer, seed=base_seed + i)
            init_cells = torch.tensor(m['atom_map'], dtype=torch.long)
            task_partner = _build_task_partner(m['tasks'], num_qubits)
            self.samples.append((init_cells, task_partner, m['tasks']))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class SingleMapDataset(IterableDataset):
    """Yields the same fixed map forever (for overfitting experiments)."""
    def __init__(self, map_data, transform=None):
        self.init_cells = torch.tensor(map_data['atom_map'], dtype=torch.long)
        self.task_partner = _build_task_partner(map_data['tasks'], map_data['num_qubits'])
        self.tasks = map_data['tasks']
        self.transform = transform

    def __iter__(self):
        while True:
            sample = {'init_cells': self.init_cells.clone(),
                      'task_partner': self.task_partner.clone(),
                      'tasks': self.tasks}
            if self.transform is not None:
                sample = self.transform(sample)
            yield sample['init_cells'], sample['task_partner'], sample['tasks']


def neutral_atoms_collate_fn(batch):
    init_cells_list, task_partner_list, tasks_list = zip(*batch)
    return (torch.stack(init_cells_list),
            torch.stack(task_partner_list),
            list(tasks_list))
