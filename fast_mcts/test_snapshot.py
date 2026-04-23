"""Regression snapshot: hash of the full classic trajectory at fixed seeds.

Catches silent behavior changes in mcts.py / env.py / network.py / rewards.py.
Reference snapshots live in fast_mcts/snapshots/. To refresh after an
intentional change:

    UPDATE_SNAPSHOTS=1 ../grid_mcts2/.venv/bin/python -m fast_mcts.test_snapshot
    git diff fast_mcts/snapshots/    # review
    git add fast_mcts/snapshots/     # commit with the behavior change

Without UPDATE_SNAPSHOTS, tests assert byte-equality vs the stored JSON.
"""

from __future__ import annotations
import json
import os
import pytest

from ._common import (
    build_cfg, build_env_spec, make_network, play_once, game_snapshot,
)


SNAP_DIR = os.path.join(os.path.dirname(__file__), 'snapshots')

SCENARIOS = [
    dict(name='classic_map0_fake_25sims', backend='classic',
         map_num=0, sims=25, use_fake=True, seed=42),
    dict(name='classic_map1_fake_25sims', backend='classic',
         map_num=1, sims=25, use_fake=True, seed=42),
    dict(name='classic_map2_fake_25sims', backend='classic',
         map_num=2, sims=25, use_fake=True, seed=42),
    dict(name='classic_map2_fake_50sims', backend='classic',
         map_num=2, sims=50, use_fake=True, seed=42),
    dict(name='classic_map0_real_10sims', backend='classic',
         map_num=0, sims=10, use_fake=False, seed=42),
    dict(name='classic_map2_priormix_25sims', backend='classic',
         map_num=2, sims=25, use_fake=True, seed=42,
         prior_mix_weight=0.5),
]


def _snapshot_path(name: str) -> str:
    return os.path.join(SNAP_DIR, f'{name}.json')


def _capture(scenario: dict) -> dict:
    backend = scenario['backend']
    seed = scenario['seed']
    cfg_kw = {k: v for k, v in scenario.items()
              if k not in ('name', 'backend', 'seed')}
    cfg = build_cfg(**cfg_kw)
    tasks, ip = build_env_spec(cfg)
    net = make_network(cfg, seed=seed)
    g = play_once(backend, cfg, net, tasks, ip,
                  seed=seed, deterministic=True, add_exploration_noise=False)
    return game_snapshot(g)


def _update_mode() -> bool:
    return os.environ.get('UPDATE_SNAPSHOTS', '') == '1'


def _compare_or_write(name: str, snap: dict) -> None:
    os.makedirs(SNAP_DIR, exist_ok=True)
    path = _snapshot_path(name)
    if _update_mode() or not os.path.exists(path):
        with open(path, 'w') as f:
            json.dump(snap, f, indent=2, sort_keys=True)
        print(f"[WROTE] {path}")
        return
    with open(path) as f:
        ref = json.load(f)
    # json stores tuples as lists → normalize current snap
    cur = json.loads(json.dumps(snap))
    assert cur == ref, (
        f"snapshot drift: {name}\n"
        f"  path: {path}\n"
        f"  re-run with UPDATE_SNAPSHOTS=1 to accept if change is intended"
    )


@pytest.mark.parametrize('scenario', SCENARIOS, ids=[s['name'] for s in SCENARIOS])
def test_snapshot(scenario):
    snap = _capture(scenario)
    _compare_or_write(scenario['name'], snap)


if __name__ == '__main__':
    for s in SCENARIOS:
        snap = _capture(s)
        _compare_or_write(s['name'], snap)
        print(f"[OK] {s['name']}")
