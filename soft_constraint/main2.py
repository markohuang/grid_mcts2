#!/usr/bin/env python3
"""
Main entry point for neutral atoms IR training.

Usage:
    python main2.py --cfg=config2.py:trm_dit/iter_T4N4
    python main2.py --cfg=config2.py:trm_dit/tiny_T2N2 --cfg.trainer.max_steps=100
    python main2.py --cfg=config2.py:trm_dit/iter_T4N4 --cfg.optim.lr=1e-4
"""

import os
import sys
import hashlib
import torch
import wandb
import rich.tree
import lightning as L
from loguru import logger
from time import time
from importlib import import_module
from pathlib import Path

# Ensure soft_constraint/ is on sys.path for surrogate imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from lightning.pytorch.strategies import DDPStrategy

from generalist.wrapper import NeutralAtomsWrapper
from generalist.callbacks import get_default_callbacks


@L.pytorch.utilities.rank_zero_only
def print_config(cfg, save_dir: str = None) -> None:
    style = 'dim'
    tree = rich.tree.Tree('CONFIG', style=style, guide_style=style)
    config_dict = cfg.to_dict() if hasattr(cfg, 'to_dict') else dict(cfg)

    def add_branch(node, key, value):
        if isinstance(value, dict):
            branch = node.add(str(key), style=style, guide_style=style)
            for k, v in value.items():
                add_branch(branch, k, v)
        else:
            node.add(f"{key}: {value}", style=style, guide_style=style)

    for key, value in config_dict.items():
        add_branch(tree, key, value)
    rich.print(tree)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        with open(f'{save_dir}/config_tree.txt', 'w') as fp:
            rich.print(tree, file=fp)


def train(cfg):
    seed_everything(cfg.seed, workers=True)
    torch.set_float32_matmul_precision('medium')

    run_id = hashlib.sha256(f"{time()}{os.urandom(4).hex()}".encode()).hexdigest()[:8]
    run_dir = Path(cfg.save_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg.run_id = run_id
    cfg.run_dir = str(run_dir)

    print_config(cfg, save_dir=str(run_dir))
    logger.info(f"Run: {run_id} -> {run_dir}")

    # Setup data
    from generalist.experiments.neutral_atoms import setup_experiment
    tloader, vloader = setup_experiment(cfg)
    logger.info(f"Data ready: streaming neutral atoms ({cfg.env.board_height}x{cfg.env.board_width}, "
                f"{cfg.env.num_qubits} atoms, {cfg.env.num_layers} layers)")

    # Loggers
    loggers = [
        WandbLogger(
            project=cfg.wandb.project,
            name=run_id,
            config=cfg.to_dict(),
            save_code=True,
        ),
        CSVLogger(save_dir=run_dir, name="", version=""),
    ]

    # Callbacks
    checkpoint_cb = ModelCheckpoint(
        dirpath=run_dir / 'checkpoints',
        filename='{step}_{val_loss:.4f}',
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        save_last=True,
    )

    callbacks = [
        checkpoint_cb,
        LearningRateMonitor(logging_interval='step'),
        EarlyStopping(**cfg.early_stopping),
        *get_default_callbacks(cfg),
    ]

    # Trainer
    use_muon = cfg.optim.type == 'muon'
    trainer = Trainer(
        logger=loggers,
        strategy=DDPStrategy() if cfg.trainer.devices > 1 else 'auto',
        callbacks=callbacks,
        accumulate_grad_batches=1 if use_muon else cfg.accumulate_grad_batches,
        **cfg.trainer
    )

    model = NeutralAtomsWrapper(cfg)
    trainer.fit(model, tloader, vloader)
    wandb.finish()


if __name__ == '__main__':
    from absl import app
    from ml_collections import config_flags

    CONFIG = config_flags.DEFINE_config_file('cfg', default='config2.py:trm_dit/iter_T4N4')

    def main(_):
        train(cfg=CONFIG.value)

    app.run(main)
