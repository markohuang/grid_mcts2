"""Shared test scaffolding: deterministic build + snapshot."""

from __future__ import annotations
import random
from typing import Any

import numpy as np
import torch

from neutral_atoms.config import (
    get_config, set_derived_config, get_map_data, atom_map_to_positions,
)
from neutral_atoms.network import Network
from neutral_atoms.game import Game

from .backends import get_backend


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_cfg(*, map_num: int = 2, sims: int = 25, use_fake: bool = True,
              prior_mix_weight: float = 0.0,
              temperature_init: float = 1.0,
              temperature_final: float = 1.0,
              temperature_decay_moves: int = 0) -> Any:
    cfg = get_config()
    cfg.map_num = map_num
    cfg.use_fake = use_fake
    cfg.mcts.num_simulations = sims
    cfg.mcts.prior_mix_weight = prior_mix_weight
    cfg.mcts.temperature_init = temperature_init
    cfg.mcts.temperature_final = temperature_final
    cfg.mcts.temperature_decay_moves = temperature_decay_moves
    if prior_mix_weight > 0:
        # prior-mix requires plan_cost-family reward or explicit track_plan_delta
        with cfg.env.unlocked():
            cfg.env.track_plan_delta = True
    set_derived_config(cfg)
    return cfg


def build_env_spec(cfg):
    md = get_map_data(cfg)
    tasks = md['tasks']
    ip = atom_map_to_positions(md['atom_map'], cfg.env.board_width)
    return tasks, ip


def make_network(cfg, seed: int) -> Network:
    seed_all(seed)
    net = Network(cfg.network, use_fake=cfg.use_fake)
    net.eval()
    return net


def play_once(backend_name: str, cfg, net, tasks, ip,
              *, seed: int, deterministic: bool, add_exploration_noise: bool,
              temperature_override: float | None = None) -> Game:
    seed_all(seed)
    game = Game(cfg, tasks, ip)
    play_game = get_backend(backend_name)
    return play_game(
        game, cfg.mcts, net,
        add_exploration_noise=add_exploration_noise,
        deterministic=deterministic,
        temperature_override=temperature_override,
    )


def game_snapshot(game: Game) -> dict:
    """Full trajectory fingerprint. Equality → identical sequences across runs."""
    return {
        'history': list(game.history),
        'rewards': [round(float(r), 8) for r in game.rewards],
        'root_values': [round(float(v), 8) for v in game.root_values],
        'child_visits': [[round(float(x), 8) for x in row]
                         for row in game.child_visits],
        'latency_reward': round(float(game.latency_reward), 8),
        'done': bool(game.done),
        'tasks_done': int(game.last_info.get('tasks_done', 0)),
    }


def assert_snapshot_equal(a: dict, b: dict, *, label: str = '') -> None:
    assert a.keys() == b.keys(), f"snapshot key mismatch ({label})"
    for k in a:
        assert a[k] == b[k], (
            f"snapshot mismatch on '{k}' ({label})\n"
            f"  a={a[k]}\n  b={b[k]}"
        )
