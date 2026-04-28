"""Offline training entry point for the autonomous pipeline.

Loads games from one or more dataset dirs into a replay buffer, loads an
optional parent checkpoint, runs N gradient steps, saves a lineage-stamped
checkpoint. No self-play, no run_dir, no registry writes — the pipeline
orchestrator owns those.

Usage:
  python train_offline.py \
    --preset=hpc \
    --dataset_dirs=/project/.../pipeline_abc/epoch_00,/project/.../pipeline_abc/epoch_01 \
    --load_ckpt=/project/.../pipeline_abc/checkpoints/epoch_00.ckpt \
    --save_ckpt=/project/.../pipeline_abc/checkpoints/epoch_01.ckpt \
    --training_steps=2000 \
    --run_id=abc12345 \
    --epoch=1
"""

import argparse
import math
import os
import sys
import time
import torch

from neutral_atoms.config import (
    set_derived_config, get_map_data, atom_map_to_positions,
)
from neutral_atoms.network import Network
from neutral_atoms.trainer import AlphaAtomsTrainer, load_dataset_into_buffer


def _load_parent_ckpt(network, path):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    if 'training_steps' not in ckpt:
        raise KeyError(
            f"checkpoint {path} has no 'training_steps' field — "
            "re-save with updated trainer before use in pipeline."
        )
    state = ckpt['model'] if 'model' in ckpt else ckpt
    network.load_state_dict(state)
    network._training_steps = int(ckpt['training_steps'])
    return ckpt


def _atomic_save_checkpoint(trainer, path, **kwargs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f'{path}.tmp.{os.getpid()}'
    trainer.save_checkpoint(tmp, **kwargs)
    os.rename(tmp, path)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--dataset_dirs', required=True,
                        help='Comma-separated list of dataset dirs (sliding window).')
    parser.add_argument('--load_ckpt', default='',
                        help='Parent ckpt to warm-start from. Empty = random init (epoch 0).')
    parser.add_argument('--save_ckpt', required=True)
    parser.add_argument('--training_steps', type=int, default=None,
                        help='Absolute number of gradient steps. Mutually exclusive with --train_epochs.')
    parser.add_argument('--train_epochs', type=float, default=None,
                        help='Number of full passes through the loaded replay buffer. '
                             'Computes steps as ceil(epochs * len(buffer) / batch_size). '
                             'Takes precedence over --training_steps.')
    parser.add_argument('--run_id', required=True)
    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--preset', choices=['default', 'hpc'], default='default')
    args, remaining = parser.parse_known_args()

    if args.preset == 'hpc':
        from neutral_atoms.config_hpc import get_config as _get_config
    else:
        from neutral_atoms.config import get_config as _get_config

    sys.argv = [sys.argv[0]] + remaining
    from absl import app
    from ml_collections import config_flags
    _CONFIG = config_flags.DEFINE_config_dict('config', _get_config())

    def _main(_):
        config = _CONFIG.value
        set_derived_config(config)
        map_data = get_map_data(config)
        tasks = map_data['tasks']
        initial_positions = atom_map_to_positions(map_data['atom_map'],
                                                  config.env.board_width)

        if config.use_fake:
            raise ValueError("train_offline.py does not support use_fake=True "
                             "(FakeNet has no trainable parameters).")
        if args.training_steps is None and args.train_epochs is None:
            raise ValueError("must pass either --training_steps or --train_epochs")

        dataset_dirs = [d.strip() for d in args.dataset_dirs.split(',') if d.strip()]
        print(f"train_offline: run_id={args.run_id} epoch={args.epoch} "
              f"preset={args.preset}")
        print(f"  dataset_dirs: {dataset_dirs}")
        print(f"  load_ckpt={args.load_ckpt or '(random init)'}")
        print(f"  save_ckpt={args.save_ckpt}")

        network = Network(config.network, use_fake=False)
        if args.load_ckpt:
            parent = _load_parent_ckpt(network, args.load_ckpt)
            print(f"  loaded parent ckpt: training_steps={network.training_steps()}, "
                  f"run_id={parent.get('run_id','')}, "
                  f"git_sha={(parent.get('git_sha') or '')[:8]}")
        else:
            print(f"  using randomly initialized network (training_steps=0)")

        trainer = AlphaAtomsTrainer(network, config, tasks, initial_positions)

        t0 = time.time()
        loaded = load_dataset_into_buffer(
            dataset_dirs, config.env, config.training.td_steps, trainer.replay_buffer,
            max_transitions=config.training.buffer_size,
        )
        load_time = time.time() - t0
        print(f"  loaded {loaded} games ({len(trainer.replay_buffer)} transitions) "
              f"in {load_time:.1f}s")

        if len(trainer.replay_buffer) < config.training.batch_size:
            raise RuntimeError(
                f"replay buffer has {len(trainer.replay_buffer)} transitions, "
                f"< batch_size={config.training.batch_size}. "
                f"Dataset dirs may be empty or too small."
            )

        if args.train_epochs is not None:
            n_steps = int(math.ceil(
                args.train_epochs * len(trainer.replay_buffer) / config.training.batch_size
            ))
            print(f"  train_epochs={args.train_epochs} x {len(trainer.replay_buffer)} transitions "
                  f"/ batch={config.training.batch_size} -> {n_steps} steps")
        else:
            n_steps = args.training_steps
            print(f"  training_steps={n_steps} (explicit)")

        t0 = time.time()
        losses = trainer.fit(training_steps=n_steps)
        train_time = time.time() - t0
        if losses is None:
            raise RuntimeError("trainer.fit() returned None on a real network — "
                               "buffer check should have caught this.")
        print(f"  trained {n_steps} steps in {train_time:.1f}s "
              f"(final: total={losses['total']:.4f} pi={losses['policy']:.3f} "
              f"cv={losses['correctness']:.3f} lv={losses['latency']:.3f})")

        _atomic_save_checkpoint(
            trainer, args.save_ckpt,
            run_id=args.run_id, epoch=args.epoch,
            parent_ckpt=args.load_ckpt,
        )
        print(f"  saved ckpt: {args.save_ckpt} "
              f"(training_steps={network.training_steps()})")

    app.run(_main)


if __name__ == '__main__':
    main()
