# test_parallel.py
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


def run_tests():
    """Run all tests and print results."""
    test_classes = [
        TestIsParallelExecutableBatch,
        TestGreedyColor,
        TestCanonicalizeMoves,
        TestParallelGroups,
        TestCountGroups,
        TestGroupSizes,
    ]
    total_passed, total_failed = 0, 0
    for test_class in test_classes:
        instance = test_class()
        class_name = test_class.__name__
        methods = [m for m in dir(instance) if m.startswith('test_')]
        for method_name in methods:
            try:
                getattr(instance, method_name)()
                print(f"  ✓ {class_name}.{method_name}")
                total_passed += 1
            except AssertionError as e:
                print(f"  ✗ {class_name}.{method_name}: {e}")
                total_failed += 1
            except Exception as e:
                print(f"  ✗ {class_name}.{method_name}: {type(e).__name__}: {e}")
                total_failed += 1
    print(f"\n{'='*50}")
    print(f"Results: {total_passed} passed, {total_failed} failed")
    return total_failed == 0


if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)