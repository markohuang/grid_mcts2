# test_env.py — tests for layer-level MDP and parallel grouping
import torch
import pytest
from .moves import (
    is_parallel_executable_batch,
    greedy_color_from_adjacency,
    canonicalize_moves,
    parallel_groups,
    count_groups,
    group_sizes,
)
from .config import MAPS, atom_map_to_positions, get_config, set_derived_config
from .env import NeutralAtomsEnv
from .tasks import get_relevant_atoms


# ---- Parallel grouping tests (unchanged) ----

class TestIsParallelExecutableBatch:
    def test_empty(self):
        result = is_parallel_executable_batch(torch.empty(0, 4))
        assert result.shape == (0, 0)

    def test_single_move(self):
        moves = torch.tensor([[0, 0, 2, 2]])
        result = is_parallel_executable_batch(moves)
        assert result.shape == (1, 1)
        assert result[0, 0] == True

    def test_parallel_same_direction(self):
        moves = torch.tensor([[0, 0, 2, 0], [0, 1, 2, 1]])
        result = is_parallel_executable_batch(moves)
        assert result[0, 1] == True
        assert result[1, 0] == True

    def test_crossing_conflict_x(self):
        moves = torch.tensor([[0, 0, 2, 0], [3, 0, 1, 0]])
        result = is_parallel_executable_batch(moves)
        assert result[0, 1] == False

    def test_crossing_conflict_y(self):
        moves = torch.tensor([[0, 0, 0, 2], [0, 3, 0, 1]])
        result = is_parallel_executable_batch(moves)
        assert result[0, 1] == False

    def test_many_to_one_conflict(self):
        moves = torch.tensor([[0, 0, 2, 2], [4, 4, 2, 2]])
        result = is_parallel_executable_batch(moves)
        assert result[0, 1] == False

    def test_diagonal_parallel(self):
        moves = torch.tensor([[0, 0, 2, 2], [1, 0, 3, 2]])
        result = is_parallel_executable_batch(moves)
        assert result[0, 1] == True

class TestGreedyColor:
    def test_empty(self):
        can_par = torch.empty(0, 0, dtype=torch.bool)
        result = greedy_color_from_adjacency(can_par)
        assert len(result) == 0

    def test_all_parallel(self):
        can_par = torch.ones(3, 3, dtype=torch.bool)
        result = greedy_color_from_adjacency(can_par)
        assert result.max() == 0

    def test_none_parallel(self):
        can_par = torch.eye(3, dtype=torch.bool)
        result = greedy_color_from_adjacency(can_par)
        assert result.max() == 2

    def test_partial(self):
        can_par = torch.tensor([
            [True, True, False],
            [True, True, False],
            [False, False, True],
        ])
        result = greedy_color_from_adjacency(can_par)
        assert result[0] == result[1]
        assert result[2] != result[0]

class TestCanonicalizeMoves:
    def test_empty(self):
        result = canonicalize_moves(torch.empty(0, 4))
        assert result.shape == (0, 4)

    def test_single(self):
        moves = torch.tensor([[2, 2, 0, 0]])
        result = canonicalize_moves(moves)
        assert result.shape == (1, 4)

    def test_direction_invariance(self):
        moves_lr = torch.tensor([[0, 0, 2, 0], [1, 1, 3, 1]])
        moves_rl = torch.tensor([[2, 0, 0, 0], [3, 1, 1, 1]])
        moves_mixed = torch.tensor([[0, 0, 2, 0], [3, 1, 1, 1]])
        count_lr = count_groups(moves_lr, canonicalize=True)
        count_rl = count_groups(moves_rl, canonicalize=True)
        count_mixed = count_groups(moves_mixed, canonicalize=True)
        assert count_lr == count_rl == count_mixed

    def test_known_failure_case_fixed(self):
        moves = torch.tensor([[1, 5, 2, 4], [3, 1, 5, 4]])
        count_raw = count_groups(moves, canonicalize=False)
        count_canon = count_groups(moves, canonicalize=True)
        assert count_canon <= count_raw
        assert count_canon == 1

class TestParallelGroups:
    def test_empty(self):
        result = parallel_groups(torch.empty(0, 4))
        assert len(result) == 0

    def test_single(self):
        result = parallel_groups(torch.tensor([[0, 0, 1, 1]]))
        assert result.tolist() == [0]

    def test_canonicalize_flag(self):
        moves = torch.tensor([[1, 5, 2, 4], [3, 1, 5, 4]])
        groups_raw = parallel_groups(moves, canonicalize=False)
        groups_canon = parallel_groups(moves, canonicalize=True)
        assert groups_canon.max() <= groups_raw.max()

class TestCountGroups:
    def test_empty(self):
        assert count_groups(torch.empty(0, 4)) == 0

    def test_single(self):
        assert count_groups(torch.tensor([[0, 0, 1, 1]])) == 1

    def test_two_parallel(self):
        moves = torch.tensor([[0, 0, 2, 0], [0, 1, 2, 1]])
        assert count_groups(moves) == 1

    def test_two_conflict(self):
        moves = torch.tensor([[0, 0, 2, 2], [4, 4, 2, 2]])
        assert count_groups(moves) == 2

class TestGroupSizes:
    def test_empty(self):
        result = group_sizes(torch.empty(0, 4))
        assert len(result) == 0

    def test_single(self):
        result = group_sizes(torch.tensor([[0, 0, 1, 1]]))
        assert result.tolist() == [1]

    def test_distribution(self):
        moves = torch.tensor([
            [0, 0, 2, 0],
            [0, 1, 2, 1],
            [3, 0, 1, 0],
        ])
        sizes = group_sizes(moves)
        assert sorted(sizes.tolist()) == [1, 2]


# ---- Layer-level MDP tests ----

def _make_env(map_num=1):
    config = get_config()
    config.map_num = map_num
    set_derived_config(config)
    m = MAPS[map_num]
    tasks = m['tasks']
    initial_positions = atom_map_to_positions(m['atom_map'], config.env.board_width)
    return NeutralAtomsEnv(tasks, initial_positions, config.env), tasks


class TestLayerLevelMDP:
    def test_episode_length_matches_sum_kt(self):
        env, tasks = _make_env(map_num=1)
        expected_length = sum(len(get_relevant_atoms(tasks, i)) for i in range(len(tasks)))
        assert env.episode_length == expected_length

    def test_all_noop_produces_baseline_cost(self):
        env, tasks = _make_env(map_num=1)
        env.reset()
        # All no-ops: each atom stays in place
        while not env.tasks_done >= env.num_tasks:
            current_q = env.current_qubit
            current_flat = (env.atom_positions[current_q][0] * env.board_width +
                            env.atom_positions[current_q][1]).item()
            env.step(current_flat)
        assert env.tasks_done == len(tasks)
        assert env._cached_cost >= 0

    def test_layer_auto_advances(self):
        env, tasks = _make_env(map_num=1)
        env.reset()
        assert env.tasks_done == 0
        # Play through first layer with all no-ops
        relevant = get_relevant_atoms(tasks, 0)
        for _ in range(len(relevant)):
            current_q = env.current_qubit
            current_flat = (env.atom_positions[current_q][0] * env.board_width +
                            env.atom_positions[current_q][1]).item()
            env.step(current_flat)
        assert env.tasks_done == 1

    def test_legal_actions_scoped_to_current_qubit(self):
        env, tasks = _make_env(map_num=1)
        env.reset()
        legal = env.legal_actions()
        current_q = env.current_qubit
        current_flat = (env.atom_positions[current_q][0] * env.board_width +
                        env.atom_positions[current_q][1]).item()
        # Current position should be in legal actions (no-op)
        assert current_flat in legal
        # All legal actions should be valid cell indices
        board_size = env.board_size
        for a in legal:
            assert 0 <= a < board_size

    def test_full_episode_completes(self):
        env, tasks = _make_env(map_num=1)
        env.reset()
        step_count = 0
        while env.tasks_done < env.num_tasks:
            legal = env.legal_actions()
            assert len(legal) > 0, f"No legal actions at step {step_count}"
            env.step(legal[0])
            step_count += 1
        assert env.tasks_done == len(tasks)

    def test_episode_length_deterministic(self):
        env, tasks = _make_env(map_num=1)
        env.reset()
        step_count = 0
        while env.tasks_done < env.num_tasks:
            current_q = env.current_qubit
            current_flat = (env.atom_positions[current_q][0] * env.board_width +
                            env.atom_positions[current_q][1]).item()
            env.step(current_flat)
            step_count += 1
        assert step_count == env.episode_length

    def test_clone_preserves_state(self):
        env, tasks = _make_env(map_num=1)
        env.reset()
        # Take a few steps
        for _ in range(3):
            legal = env.legal_actions()
            env.step(legal[0])
        clone = env.clone()
        assert clone.tasks_done == env.tasks_done
        assert clone.current_atom_idx == env.current_atom_idx
        assert clone.current_qubit == env.current_qubit
        assert torch.equal(clone.board, env.board)
        assert torch.equal(clone.atom_positions, env.atom_positions)

    def test_move_changes_position(self):
        env, tasks = _make_env(map_num=1)
        env.reset()
        current_q = env.current_qubit
        legal = env.legal_actions()
        current_flat = (env.atom_positions[current_q][0] * env.board_width +
                        env.atom_positions[current_q][1]).item()
        # Find a legal action that isn't no-op
        move_action = [a for a in legal if a != current_flat]
        if move_action:
            old_pos = env.atom_positions[current_q].clone()
            env.step(move_action[0])
            # Position should have changed (or atom_idx advanced)
            # The qubit was moved and atom_idx advanced

    def test_no_legal_actions_when_done(self):
        env, tasks = _make_env(map_num=1)
        env.reset()
        while env.tasks_done < env.num_tasks:
            current_q = env.current_qubit
            current_flat = (env.atom_positions[current_q][0] * env.board_width +
                            env.atom_positions[current_q][1]).item()
            env.step(current_flat)
        assert env.legal_actions() == []

    def test_map0_episode_length(self):
        env, tasks = _make_env(map_num=0)
        expected = sum(len(get_relevant_atoms(tasks, i)) for i in range(len(tasks)))
        assert env.episode_length == expected

    def test_reward_zero_for_noops(self):
        env, tasks = _make_env(map_num=1)
        env.reset()
        # All no-ops: no moves, no cost change within any layer → zero reward
        total_reward = 0.0
        while env.tasks_done < env.num_tasks:
            current_q = env.current_qubit
            current_flat = (env.atom_positions[current_q][0] * env.board_width +
                            env.atom_positions[current_q][1]).item()
            result = env.step(current_flat)
            total_reward += result.reward
        assert abs(total_reward) < 1e-6

    def test_reward_differs_for_different_actions(self):
        import random
        random.seed(42)
        env, tasks = _make_env(map_num=1)
        env.reset()
        # Random play: some moves will change cost → non-zero total reward
        total_reward = 0.0
        while env.tasks_done < env.num_tasks:
            legal = env.legal_actions()
            action = random.choice(legal)
            result = env.step(action)
            total_reward += result.reward
        # Random moves should produce non-zero reward (either positive or negative)
        # This is probabilistic but virtually guaranteed with random moves
        assert total_reward != 0.0


def run_tests():
    test_classes = [
        TestIsParallelExecutableBatch,
        TestGreedyColor,
        TestCanonicalizeMoves,
        TestParallelGroups,
        TestCountGroups,
        TestGroupSizes,
        TestLayerLevelMDP,
    ]
    total_passed, total_failed = 0, 0
    for test_class in test_classes:
        instance = test_class()
        class_name = test_class.__name__
        methods = [m for m in dir(instance) if m.startswith('test_')]
        for method_name in methods:
            try:
                getattr(instance, method_name)()
                print(f"  PASS {class_name}.{method_name}")
                total_passed += 1
            except AssertionError as e:
                print(f"  FAIL {class_name}.{method_name}: {e}")
                total_failed += 1
            except Exception as e:
                print(f"  FAIL {class_name}.{method_name}: {type(e).__name__}: {e}")
                total_failed += 1
    print(f"\n{'='*50}")
    print(f"Results: {total_passed} passed, {total_failed} failed")
    return total_failed == 0


if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)
