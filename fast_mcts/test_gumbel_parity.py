"""Parity tests for fast_gumbel backend.

Self-parity: same seed → byte-identical trajectory.
Cross-parity: fast_gumbel(batch=1, vl=0) == classic Gumbel.
"""

from __future__ import annotations
import pytest

from ._common import (
    seed_all, build_cfg, build_env_spec, make_network, play_once,
    game_snapshot, assert_snapshot_equal,
)


def _gumbel_cfg(map_num=2, sims=25, use_fake=True, nn_batch_size=1,
                virtual_loss=0.0):
    cfg = build_cfg(map_num=map_num, sims=sims, use_fake=use_fake)
    with cfg.mcts.unlocked():
        cfg.mcts.gumbel.enabled = True
        cfg.mcts.nn_batch_size = nn_batch_size
        cfg.mcts.virtual_loss = virtual_loss
    with cfg.env.unlocked():
        cfg.env.reward_mode = 'plan_cost'
    return cfg


def _self_parity_gumbel(backend, *, map_num=2, sims=25, use_fake=True,
                         nn_batch_size=1, virtual_loss=0.0, seed=42):
    cfg = _gumbel_cfg(map_num=map_num, sims=sims, use_fake=use_fake,
                      nn_batch_size=nn_batch_size, virtual_loss=virtual_loss)
    tasks, ip = build_env_spec(cfg)
    net = make_network(cfg, seed)
    g1 = play_once(backend, cfg, net, tasks, ip,
                   seed=seed, deterministic=True, add_exploration_noise=False)
    g2 = play_once(backend, cfg, net, tasks, ip,
                   seed=seed, deterministic=True, add_exploration_noise=False)
    assert_snapshot_equal(
        game_snapshot(g1), game_snapshot(g2),
        label=f'{backend}/map={map_num}/sims={sims}/fake={use_fake}/batch={nn_batch_size}/vl={virtual_loss}',
    )
    return g1


def _cross_parity(map_num=2, sims=25, use_fake=True, seed=42):
    """classic Gumbel vs fast_gumbel(batch=1, vl=0) must be byte-identical."""
    cfg_classic = _gumbel_cfg(map_num=map_num, sims=sims, use_fake=use_fake)
    cfg_fast = _gumbel_cfg(map_num=map_num, sims=sims, use_fake=use_fake,
                            nn_batch_size=1, virtual_loss=0.0)
    tasks, ip = build_env_spec(cfg_classic)
    net = make_network(cfg_classic, seed)
    gc = play_once('classic', cfg_classic, net, tasks, ip,
                   seed=seed, deterministic=True, add_exploration_noise=False)
    gf = play_once('fast_gumbel', cfg_fast, net, tasks, ip,
                   seed=seed, deterministic=True, add_exploration_noise=False)
    assert_snapshot_equal(
        game_snapshot(gc), game_snapshot(gf),
        label=f'classic-vs-fast_gumbel/map={map_num}/sims={sims}/fake={use_fake}',
    )


# ---- self-parity: classic Gumbel ----

@pytest.mark.parametrize('map_num', [0, 1, 2])
def test_classic_gumbel_self_parity_maps(map_num):
    _self_parity_gumbel('classic', map_num=map_num, sims=25)


def test_classic_gumbel_self_parity_real_net():
    _self_parity_gumbel('classic', map_num=0, sims=10, use_fake=False)


@pytest.mark.parametrize('sims', [5, 25, 75])
def test_classic_gumbel_self_parity_sims_scan(sims):
    _self_parity_gumbel('classic', sims=sims)


# ---- self-parity: fast_gumbel batch=1, vl=0 ----

@pytest.mark.parametrize('map_num', [0, 1, 2])
def test_fast_gumbel_self_parity_batch1_maps(map_num):
    _self_parity_gumbel('fast_gumbel', map_num=map_num, sims=25)


def test_fast_gumbel_self_parity_real_net():
    _self_parity_gumbel('fast_gumbel', map_num=0, sims=10, use_fake=False)


# ---- self-parity: fast_gumbel with batching and VL (production config) ----

@pytest.mark.parametrize('map_num', [0, 1, 2])
def test_fast_gumbel_self_parity_batched_maps(map_num):
    _self_parity_gumbel('fast_gumbel', map_num=map_num, sims=25,
                         nn_batch_size=8, virtual_loss=1.0)


@pytest.mark.parametrize('sims', [25, 75])
def test_fast_gumbel_self_parity_batched_sims_scan(sims):
    _self_parity_gumbel('fast_gumbel', sims=sims, nn_batch_size=8, virtual_loss=1.0)


# ---- cross-parity: classic == fast_gumbel(batch=1, vl=0) ----

@pytest.mark.parametrize('map_num,sims,fake', [
    (0, 10, True), (0, 10, False),
    (1, 25, True),
    (2, 25, True), (2, 50, True), (2, 75, True),
])
def test_cross_classic_vs_fast_gumbel(map_num, sims, fake):
    _cross_parity(map_num=map_num, sims=sims, use_fake=fake)


if __name__ == '__main__':
    for m in (0, 1, 2):
        _self_parity_gumbel('classic', map_num=m, sims=25)
        print(f'[OK] classic Gumbel self-parity map={m} fake 25sims')
    _self_parity_gumbel('classic', map_num=0, sims=10, use_fake=False)
    print('[OK] classic Gumbel self-parity real net')
    for m in (0, 1, 2):
        _self_parity_gumbel('fast_gumbel', map_num=m, sims=25)
        print(f'[OK] fast_gumbel self-parity map={m} fake 25sims batch=1 vl=0')
    for m in (0, 1, 2):
        _self_parity_gumbel('fast_gumbel', map_num=m, sims=25,
                             nn_batch_size=8, virtual_loss=1.0)
        print(f'[OK] fast_gumbel self-parity map={m} fake 25sims batch=8 vl=1')
    _cross_parity(map_num=2, sims=25)
    print('[OK] cross-parity classic == fast_gumbel(batch=1,vl=0) map=2 25sims')
    _cross_parity(map_num=0, sims=10, use_fake=False)
    print('[OK] cross-parity classic == fast_gumbel(batch=1,vl=0) map=0 real')
