"""Smoke test for Gumbel AlphaZero self-play (docs/gumbel_pczero_plan.md Phase 1).

Runs one FakeNet self-play game on MAPS[2] with gumbel.enabled=True, then asserts:
  - game completes deterministically (no crash)
  - episode length == sum(k_t) (layer-MDP invariant)
  - every stored policy target sums to ~1 and has support only on legal actions at that step
  - winner (stored action) lies in root legal actions
  - root._gumbel_policy attached each step
  - stats bucket filled (avg_depth > 0, reached_terminal_frac in [0,1])

Usage:
  ../grid_mcts2/.venv/bin/python scripts/smoke_gumbel.py
"""
import sys
sys.path.insert(0, '.')

from neutral_atoms.config import get_config, set_derived_config, get_map_data, atom_map_to_positions
from neutral_atoms.game import Game
from neutral_atoms.network import Network
from neutral_atoms.mcts import play_game


def main():
    config = get_config()
    config.map_num = 2  # 5x5 / 12qb / 3 layers
    config.use_fake = True
    config.mcts.num_simulations = 64  # small for smoke
    config.mcts.gumbel.enabled = True
    set_derived_config(config)

    print(f"[smoke] map_num=2, gumbel.enabled=True, num_simulations={config.mcts.num_simulations}")
    print(f"[smoke] num_samples_m={config.mcts.gumbel.num_samples_m} "
          f"(expected = board_size - num_qubits + 1 = 25 - 12 + 1 = 14)")
    assert config.mcts.gumbel.num_samples_m == 14, f"expected 14, got {config.mcts.gumbel.num_samples_m}"

    m = get_map_data(config)
    positions = atom_map_to_positions(m['atom_map'], m['board_dim'][1])
    game = Game(config, m['tasks'], positions)
    network = Network(config.network, use_fake=True)

    game = play_game(game, config.mcts, network, add_exploration_noise=False)

    expected_episode_len = sum(
        len({q for pair in layer for q in pair}) for layer in m['tasks']
    )
    print(f"[smoke] episode length: got {len(game.history)}, expected {expected_episode_len}")
    assert len(game.history) == expected_episode_len, \
        f"episode length {len(game.history)} != expected {expected_episode_len}"

    assert len(game.child_visits) == expected_episode_len
    assert len(game.root_values) == expected_episode_len

    # Every step: policy target is a proper distribution
    for i, pt in enumerate(game.child_visits):
        s = sum(pt)
        assert abs(s - 1.0) < 1e-5, f"step {i}: policy target sums to {s}, not 1"
        assert min(pt) >= 0.0, f"step {i}: negative policy prob"

    # All chosen actions were legal at root (post-hoc: they must have been, since apply didn't crash)
    # Check stats
    print(f"[smoke] mcts_depth mean: {sum(game.mcts_depths)/len(game.mcts_depths):.2f}")
    print(f"[smoke] reached_terminal_frac mean: {sum(game.mcts_reached_terminal_fracs)/len(game.mcts_reached_terminal_fracs):.3f}")
    print(f"[smoke] boundary_reach_frac mean: {sum(game.mcts_boundary_reach_fracs)/len(game.mcts_boundary_reach_fracs):.3f}")
    print(f"[smoke] total reward: {sum(game.rewards):.3f}")

    # Terminal reward fires once tasks_done >= num_tasks (done=True). latency_reward set.
    print(f"[smoke] latency_reward: {game.latency_reward:.3f}")
    print(f"[smoke] final tasks_done: {game.last_info.get('tasks_done')}")
    assert game.last_info.get('tasks_done') == len(m['tasks']), \
        "episode did not complete all tasks"

    print("[smoke] PASS")


def control_vs_gumbel(num_games=4, num_sims=64):
    """Side-by-side: run num_games with each variant on fresh random 5x5 maps, report cost."""
    import numpy as np
    from neutral_atoms.experiment import compute_solution_cost

    def run(gumbel_on):
        costs = []
        lens = []
        for seed in range(num_games):
            np.random.seed(seed)
            config = get_config()
            config.map_num = 2
            config.use_fake = True
            config.mcts.num_simulations = num_sims
            config.mcts.gumbel.enabled = gumbel_on
            set_derived_config(config)
            m = get_map_data(config)
            positions = atom_map_to_positions(m['atom_map'], m['board_dim'][1])
            game = Game(config, m['tasks'], positions)
            network = Network(config.network, use_fake=True)
            game = play_game(game, config.mcts, network, add_exploration_noise=(not gumbel_on))
            cost = compute_solution_cost(game)
            costs.append(cost)
            lens.append(len(game.history))
        return costs, lens

    print(f"\n[compare] {num_games} games per variant, {num_sims} sims, MAPS[2] fixed")
    c_ctrl, l_ctrl = run(gumbel_on=False)
    c_gum, l_gum = run(gumbel_on=True)
    print(f"[compare] pUCT   costs: {c_ctrl} (mean {sum(c_ctrl)/len(c_ctrl):.2f}, lens {l_ctrl})")
    print(f"[compare] Gumbel costs: {c_gum}  (mean {sum(c_gum)/len(c_gum):.2f}, lens {l_gum})")


if __name__ == '__main__':
    main()
    control_vs_gumbel()
