"""Iterative Refinement training for neutral atom reconfiguration.

Usage:
    ../grid_mcts2/.venv/bin/python train_ir.py
    ../grid_mcts2/.venv/bin/python train_ir.py --config.map_num=3 --config.model.T=4
    ../grid_mcts2/.venv/bin/python train_ir.py --config.random_board=True --config.training.max_steps=10000
"""
import os
from absl import app, flags
from ml_collections import config_flags

from iterative_refinement.config import get_config, set_derived_config
from iterative_refinement.wrapper import IRWrapper
from neutral_atoms.config import MAPS, _resolve_map

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

FLAGS = flags.FLAGS
config_flags.DEFINE_config_dict('config', get_config())
flags.DEFINE_integer('eval_map', -1, 'Map number to run inference on after training (-1 = same as training)')


def main(_):
    cfg = FLAGS.config
    map_data = set_derived_config(cfg)
    L.seed_everything(cfg.training.seed)

    model = IRWrapper(cfg)
    print(f"PlanNet parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Board: {cfg.board_height}x{cfg.board_width}, "
          f"qubits: {cfg.num_qubits}, layers: {cfg.num_tasks}")

    callbacks = [
        LearningRateMonitor(logging_interval='step'),
        ModelCheckpoint(
            dirpath=f"{cfg.experiment.output_dir}/checkpoints",
            every_n_train_steps=cfg.training.save_every,
            save_top_k=3,
            monitor='val_hard_cost',
            mode='min',
        ),
    ]

    trainer = L.Trainer(
        max_steps=cfg.training.max_steps,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        gradient_clip_val=cfg.training.grad_clip,
        log_every_n_steps=cfg.training.log_every,
        val_check_interval=cfg.training.eval_every,
        limit_val_batches=cfg.training.num_eval_samples // cfg.training.batch_size,
        callbacks=callbacks,
        default_root_dir=cfg.experiment.output_dir,
    )
    trainer.fit(model)

    # ---- Post-training inference ----
    eval_map_num = FLAGS.eval_map if FLAGS.eval_map >= 0 else cfg.map_num
    eval_map = _resolve_map(MAPS[eval_map_num])
    # Ensure board dims match (can only infer on same-shape boards)
    eval_h, eval_w = eval_map['board_dim']
    eval_q = eval_map['num_qubits']
    if eval_h != cfg.board_height or eval_w != cfg.board_width or eval_q != cfg.num_qubits:
        print(f"WARNING: eval map {eval_map_num} has shape {eval_h}x{eval_w}/{eval_q}q, "
              f"but model trained on {cfg.board_height}x{cfg.board_width}/{cfg.num_qubits}q. "
              f"Skipping inference.")
        return
    if len(eval_map['tasks']) != cfg.num_tasks:
        print(f"WARNING: eval map has {len(eval_map['tasks'])} layers but model expects {cfg.num_tasks}. "
              f"Skipping inference.")
        return
    sol_dir = os.path.join(cfg.experiment.output_dir, 'solutions')
    model.save_solution(eval_map, os.path.join(sol_dir, f'map{eval_map_num}_solution.json'))


if __name__ == '__main__':
    app.run(main)
