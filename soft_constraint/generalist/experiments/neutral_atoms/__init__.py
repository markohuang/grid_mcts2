"""Data pipeline for neutral atoms experiments."""
from torch.utils.data import DataLoader
from .dataset import (NeutralAtomsStreamDataset, NeutralAtomsFixedDataset,
                      SingleMapDataset, neutral_atoms_collate_fn)
from .augmentations import build_transform


def setup_experiment(cfg):
    transform = build_transform(cfg)
    bd = (cfg.env.board_height, cfg.env.board_width)

    if cfg.experiment == 'single_map':
        from generalist.configs.experiments.single_map import SINGLE_MAP
        ds = SingleMapDataset(SINGLE_MAP, transform=transform)
        tloader = DataLoader(ds, batch_size=cfg.data.batch_size,
                             collate_fn=neutral_atoms_collate_fn,
                             num_workers=cfg.data.num_workers, pin_memory=True)
        vloader = DataLoader(SingleMapDataset(SINGLE_MAP),
                             batch_size=cfg.data.batch_size,
                             collate_fn=neutral_atoms_collate_fn,
                             num_workers=cfg.data.num_workers, pin_memory=True)
        return tloader, vloader

    tloader = DataLoader(
        NeutralAtomsStreamDataset(
            board_dim=bd,
            num_qubits=cfg.env.num_qubits,
            num_layers=cfg.env.num_layers,
            gates_per_layer=cfg.env.gates_per_layer,
            seed=cfg.seed,
            transform=transform,
        ),
        batch_size=cfg.data.batch_size,
        collate_fn=neutral_atoms_collate_fn,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )

    vloader = DataLoader(
        NeutralAtomsFixedDataset(
            board_dim=bd,
            num_qubits=cfg.env.num_qubits,
            num_layers=cfg.env.num_layers,
            gates_per_layer=cfg.env.gates_per_layer,
            pool_size=cfg.data.val_pool_size,
            base_seed=cfg.data.val_seed,
        ),
        batch_size=cfg.data.batch_size,
        collate_fn=neutral_atoms_collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    return tloader, vloader
