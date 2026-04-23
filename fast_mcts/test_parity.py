"""Parity harness.

Scope:
  - self-parity: same backend + same seed + deterministic → byte-identical trajectories
  - config-path coverage: noise on/off, temperature decay, prior-mix, multiple maps
  - cross-backend scaffold: `run_cross_backend_parity` ready for Phase 1 `fast`
"""

from __future__ import annotations
import pytest

from ._common import (
    seed_all, build_cfg, build_env_spec, make_network, play_once,
    game_snapshot, assert_snapshot_equal,
)


# ---- self-parity ----

def _self_parity(backend: str, **cfg_kw):
    seed = cfg_kw.pop('seed', 42)
    deterministic = cfg_kw.pop('deterministic', True)
    add_noise = cfg_kw.pop('add_noise', False)
    cfg = build_cfg(**cfg_kw)
    tasks, ip = build_env_spec(cfg)
    net = make_network(cfg, seed)

    g1 = play_once(backend, cfg, net, tasks, ip,
                   seed=seed, deterministic=deterministic,
                   add_exploration_noise=add_noise)
    g2 = play_once(backend, cfg, net, tasks, ip,
                   seed=seed, deterministic=deterministic,
                   add_exploration_noise=add_noise)
    assert_snapshot_equal(
        game_snapshot(g1), game_snapshot(g2),
        label=f'{backend}/{cfg_kw}/det={deterministic}/noise={add_noise}',
    )
    return g1


# --- parametrized config branches ---

@pytest.mark.parametrize('map_num', [0, 1, 2])
def test_classic_self_parity_maps_fake(map_num):
    _self_parity('classic', map_num=map_num, sims=25, use_fake=True)


def test_classic_self_parity_real_net_small():
    _self_parity('classic', map_num=0, sims=10, use_fake=False)


def test_classic_self_parity_with_noise():
    # Dirichlet path reproducible under numpy seed.
    _self_parity('classic', map_num=2, sims=25, use_fake=True,
                 deterministic=False, add_noise=True)


def test_classic_self_parity_temperature_decay():
    # Exercises get_temperature linear-decay path + softmax sampling.
    _self_parity('classic', map_num=2, sims=25, use_fake=True,
                 deterministic=False, add_noise=False,
                 temperature_init=1.0, temperature_final=0.1,
                 temperature_decay_moves=8)


def test_classic_self_parity_prior_mix():
    # Exercises _expand_node prior-mix clone-per-action branch.
    _self_parity('classic', map_num=2, sims=25, use_fake=True,
                 prior_mix_weight=0.5)


@pytest.mark.parametrize('sims', [5, 25, 75])
def test_classic_self_parity_sims_scan(sims):
    _self_parity('classic', map_num=2, sims=sims, use_fake=True)


# --- fast backend ---

@pytest.mark.parametrize('map_num', [0, 1, 2])
def test_fast_self_parity_maps_fake(map_num):
    _self_parity('fast', map_num=map_num, sims=25, use_fake=True)


def test_fast_self_parity_real_net_small():
    _self_parity('fast', map_num=0, sims=10, use_fake=False)


# --- cross-backend parity: fast (batch=1, vl=0) must equal classic ---

@pytest.mark.parametrize('map_num,sims,fake', [
    (0, 10, True), (0, 10, False),
    (1, 25, True),
    (2, 25, True), (2, 50, True), (2, 75, True),
])
def test_cross_classic_vs_fast_defaults(map_num, sims, fake):
    run_cross_backend_parity('classic', 'fast',
                             map_num=map_num, sims=sims, use_fake=fake)


# ---- cross-backend (scaffold — activated once 'fast' lands) ----

def run_cross_backend_parity(a: str, b: str, **cfg_kw) -> None:
    """Byte-identical parity between backends requires the safe knob set:
    nn_batch_size=1, virtual_loss=0. Force it here — repo defaults now carry
    production values (batch=32, vl=1.0) that break byte-parity by design."""
    seed = cfg_kw.pop('seed', 42)
    cfg = build_cfg(**cfg_kw)
    with cfg.mcts.unlocked():
        cfg.mcts.nn_batch_size = 1
        cfg.mcts.virtual_loss = 0.0
    tasks, ip = build_env_spec(cfg)
    net = make_network(cfg, seed)
    ga = play_once(a, cfg, net, tasks, ip, seed=seed,
                   deterministic=True, add_exploration_noise=False)
    gb = play_once(b, cfg, net, tasks, ip, seed=seed,
                   deterministic=True, add_exploration_noise=False)
    assert_snapshot_equal(game_snapshot(ga), game_snapshot(gb),
                          label=f'{a}-vs-{b}/{cfg_kw}')


if __name__ == '__main__':
    # Run subset without pytest for quick dev loop.
    for m in (0, 1, 2):
        _self_parity('classic', map_num=m, sims=25, use_fake=True)
        print(f"[OK] self-parity classic map={m} fake 25sims")
    _self_parity('classic', map_num=0, sims=10, use_fake=False)
    print("[OK] self-parity classic map=0 real 10sims")
    _self_parity('classic', map_num=2, sims=25, use_fake=True,
                 deterministic=False, add_noise=True)
    print("[OK] self-parity classic noise-on")
    _self_parity('classic', map_num=2, sims=25, use_fake=True,
                 deterministic=False,
                 temperature_init=1.0, temperature_final=0.1,
                 temperature_decay_moves=8)
    print("[OK] self-parity classic temperature-decay")
    _self_parity('classic', map_num=2, sims=25, use_fake=True,
                 prior_mix_weight=0.5)
    print("[OK] self-parity classic prior-mix")
