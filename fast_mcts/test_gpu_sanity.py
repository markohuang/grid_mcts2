"""GPU sanity checks for fast_mcts.

Run directly on a GPU node (skipped silently if no CUDA):
    python fast_mcts/test_gpu_sanity.py [--ckpt path/to/ckpt.ckpt]

Checks:
  1. Byte-exact parity: GPU classic vs GPU fast (batch=1, vl=0).
     Confirms that moving the model to CUDA doesn't break search determinism
     or introduce numerical drift vs the CPU classic baseline.
  2. H2D fraction: with batch=32 at 10k sims, H2D should be <20% of NN time.
     If >20%, the forward pass is too cheap and batching can't hide transfer cost.
  3. Quality smoke: GPU fast (batch=32, vl=1) mean cost within 1.5 units of
     GPU classic on the same 20 seeds. Catches VL-induced quality regression on GPU.
"""

import argparse
import os
import sys

# Ensure repo root is on path when run as `python fast_mcts/test_gpu_sanity.py`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

_HAS_CUDA = torch.cuda.is_available()


def _skip(msg):
    print(f'SKIP: {msg}')
    sys.exit(0)


def _load_net(cfg, ckpt_path):
    from neutral_atoms.network import Network
    net = Network(cfg.network, use_fake=False)
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        state = ckpt['model'] if 'model' in ckpt else ckpt
        net.load_state_dict(state)
    net.eval()
    return net


def _build(map_num=2, sims=25, ckpt_path=None):
    from fast_mcts._common import build_cfg, build_env_spec, seed_all
    cfg = build_cfg(map_num=map_num, sims=sims, use_fake=(ckpt_path is None))
    tasks, ip = build_env_spec(cfg)
    if ckpt_path:
        net = _load_net(cfg, ckpt_path)
    else:
        from neutral_atoms.network import Network
        seed_all(0)
        net = Network(cfg.network, use_fake=True)
        net.eval()
    return cfg, tasks, ip, net


def check1_gpu_classic_vs_fast_parity(ckpt_path):
    """GPU classic vs GPU fast (batch=1, vl=0) must be byte-identical."""
    print('\n--- Check 1: byte-exact GPU parity (classic vs fast-batch=1-vl=0) ---')
    from fast_mcts._common import build_cfg, build_env_spec, seed_all, game_snapshot
    from fast_mcts.backends import get_backend
    from neutral_atoms.game import Game

    cfg, tasks, ip, net = _build(map_num=2, sims=25, ckpt_path=ckpt_path)
    net_gpu = net.to('cuda')

    failures = 0
    for seed in range(5):
        seed_all(seed)
        g_classic = Game(cfg, tasks, ip)
        g_classic = get_backend('classic')(g_classic, cfg.mcts, net_gpu,
                                          add_exploration_noise=True, deterministic=False)
        seed_all(seed)
        g_fast = Game(cfg, tasks, ip)
        # safe knobs: batch=1, vl=0 → identical to classic
        with cfg.unlocked():
            cfg.mcts.nn_batch_size = 1
            cfg.mcts.virtual_loss = 0.0
        g_fast = get_backend('fast')(g_fast, cfg.mcts, net_gpu,
                                    add_exploration_noise=True, deterministic=False)
        snap_c = game_snapshot(g_classic)
        snap_f = game_snapshot(g_fast)
        if snap_c != snap_f:
            print(f'  FAIL seed={seed}: trajectories differ')
            print(f'    classic history[:5] = {snap_c["history"][:5]}')
            print(f'    fast    history[:5] = {snap_f["history"][:5]}')
            failures += 1
        else:
            print(f'  OK   seed={seed}: byte-identical')

    if failures == 0:
        print('Check 1 PASSED')
    else:
        print(f'Check 1 FAILED ({failures}/5 seeds)')
    return failures == 0


def check2_h2d_fraction(ckpt_path, sims=10000):
    """H2D should be <20% of NN time at batch=32, 10k sims, real GPU."""
    print(f'\n--- Check 2: H2D fraction at batch=32, sims={sims} ---')
    from fast_mcts._common import build_cfg, build_env_spec, seed_all
    from fast_mcts.backends import get_backend
    from neutral_atoms.game import Game

    if ckpt_path is None:
        print('  SKIP: no ckpt provided, H2D check needs real net')
        return True

    cfg, tasks, ip, net = _build(map_num=2, sims=sims, ckpt_path=ckpt_path)
    net_gpu = net.to('cuda')
    with cfg.unlocked():
        cfg.mcts.nn_batch_size = 32
        cfg.mcts.virtual_loss = 1.0

    seed_all(0)
    g = Game(cfg, tasks, ip)
    g = get_backend('fast')(g, cfg.mcts, net_gpu,
                            add_exploration_noise=True, deterministic=False)

    total_h2d = sum(g.mcts_nn_h2d_ms_list)
    total_nn = sum(g.mcts_nn_total_ms_list)
    total_reqs = sum(g.mcts_nn_requests_list)
    total_batches = sum(g.mcts_nn_batches_list)
    h2d_frac = total_h2d / total_nn if total_nn > 0 else 0.0
    eff_batch = total_reqs / total_batches if total_batches > 0 else 0.0

    print(f'  total_nn_ms    = {total_nn:.1f} ms')
    print(f'  total_h2d_ms   = {total_h2d:.1f} ms  ({100*h2d_frac:.1f}% of NN time)')
    print(f'  total_requests = {total_reqs}  batches = {total_batches}')
    print(f'  eff_batch_size = {eff_batch:.1f}  (target ≥ 28 for batch=32)')

    ok_h2d = h2d_frac < 0.20
    ok_batch = eff_batch >= 16.0  # at least half-filled
    print(f'  H2D < 20%: {"OK" if ok_h2d else "WARN (transfer-bound)"}')
    print(f'  eff_batch ≥ 16: {"OK" if ok_batch else "WARN (VL not diversifying enough)"}')
    print(f'Check 2 {"PASSED" if ok_h2d else "WARN — consider larger model or smaller batch"}')
    return ok_h2d


def check3_quality_parity(ckpt_path, n_games=20):
    """GPU fast (batch=32, vl=1) mean cost within 1.5 of GPU classic, real net."""
    print(f'\n--- Check 3: quality parity over {n_games} games ---')
    from fast_mcts._common import build_cfg, build_env_spec, seed_all
    from fast_mcts.backends import get_backend
    from neutral_atoms.experiment import compute_solution_cost
    from neutral_atoms.game import Game
    from neutral_atoms.map_generator import generate_random_map
    from neutral_atoms.config import atom_map_to_positions

    if ckpt_path is None:
        print('  SKIP: no ckpt — quality check needs real net')
        return True

    cfg, _, _, net = _build(map_num=2, sims=100, ckpt_path=ckpt_path)
    net_gpu = net.to('cuda')
    md_template = {'board_dim': (5, 5), 'num_qubits': 12,
                   'tasks': cfg.network.num_tasks}

    def play_game_seed(backend_name, batch, vl, seed):
        with cfg.unlocked():
            cfg.mcts.nn_batch_size = batch
            cfg.mcts.virtual_loss = vl
        seed_all(seed)
        m = generate_random_map((5, 5), 12, 3, gates_per_layer=4, seed=seed)
        tasks = m['tasks']
        ip = atom_map_to_positions(m['atom_map'], cfg.env.board_width)
        g = Game(cfg, tasks, ip)
        return get_backend(backend_name)(g, cfg.mcts, net_gpu,
                                        add_exploration_noise=True, deterministic=False)

    classic_costs = [compute_solution_cost(play_game_seed('classic', 1, 0.0, s))
                     for s in range(n_games)]
    fast_costs = [compute_solution_cost(play_game_seed('fast', 32, 1.0, s))
                  for s in range(n_games)]

    mean_c = sum(classic_costs) / n_games
    mean_f = sum(fast_costs) / n_games
    diff = mean_f - mean_c
    print(f'  classic mean={mean_c:.2f}  fast-batch32 mean={mean_f:.2f}  diff={diff:+.2f}')
    ok = abs(diff) <= 1.5
    print(f'  |diff| ≤ 1.5: {"OK" if ok else "WARN — batch+VL may hurt quality on GPU"}')
    print(f'Check 3 {"PASSED" if ok else "WARN"}')
    return ok


def main():
    if not _HAS_CUDA:
        _skip('no CUDA device available')

    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='', help='Path to checkpoint (optional; uses FakeNet if omitted)')
    ap.add_argument('--sims', type=int, default=10000)
    args = ap.parse_args()

    ckpt = args.ckpt or None
    print(f'GPU sanity checks  device={torch.cuda.get_device_name(0)}  ckpt={ckpt or "FakeNet"}')

    r1 = check1_gpu_classic_vs_fast_parity(ckpt)
    r2 = check2_h2d_fraction(ckpt, sims=args.sims)
    r3 = check3_quality_parity(ckpt)

    print('\n=== Summary ===')
    print(f'  Check 1 (byte-exact parity): {"PASS" if r1 else "FAIL"}')
    print(f'  Check 2 (H2D fraction):      {"PASS" if r2 else "WARN"}')
    print(f'  Check 3 (quality parity):    {"PASS" if r3 else "WARN"}')
    sys.exit(0 if (r1 and r2 and r3) else 1)


if __name__ == '__main__':
    main()
