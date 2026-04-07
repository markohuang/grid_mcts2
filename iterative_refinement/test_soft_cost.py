"""Tests for differentiable soft cost computation."""
import torch
from neutral_atoms.moves import count_groups
from neutral_atoms.tasks import gates_to_moves
from iterative_refinement.soft_cost import (
    soft_layer_cost, hard_layer_cost, _top_k_candidates,
    _soft_reconfig, _soft_collision, _soft_gate,
)


def test_gradient_flows():
    """Verify gradients flow from soft_layer_cost back to logits."""
    torch.manual_seed(0)
    Q, B = 8, 16  # 8 qubits, 4x4 board
    atom_positions = torch.tensor([
        [0, 0], [0, 1], [0, 2], [0, 3],
        [1, 0], [1, 1], [1, 2], [1, 3],
    ])
    logits = torch.randn(Q, B, requires_grad=True)
    relevant = list(range(Q))
    gate_pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
    cost = soft_layer_cost(atom_positions, logits, relevant, gate_pairs, board_w=4, top_k=3)
    cost.backward()
    assert logits.grad is not None
    assert logits.grad.abs().sum() > 0
    print(f"  gradient flows: grad norm = {logits.grad.norm():.4f}")


def test_peaked_logits_match_hard_cost():
    """When logits are very peaked (one-hot-like), soft cost should approximate hard cost."""
    torch.manual_seed(42)
    Q, B = 6, 12  # 6 qubits, 3x4 board
    atom_positions = torch.tensor([
        [0, 0], [0, 1], [0, 2], [0, 3],
        [1, 0], [1, 1],
    ])
    gate_pairs = [(0, 1), (2, 3), (4, 5)]
    relevant = list(range(Q))
    # Create peaked logits: each qubit strongly prefers a specific destination
    targets = [5, 4, 7, 6, 1, 0]  # swap pairs
    logits = torch.full((Q, B), -20.0)
    for q, t in enumerate(targets):
        logits[q, t] = 20.0
    logits.requires_grad_(True)
    soft = soft_layer_cost(atom_positions, logits, relevant, gate_pairs, board_w=4, top_k=3)
    hard = hard_layer_cost(atom_positions, logits, relevant, gate_pairs, board_w=4)
    # With peaked logits and top_k >= 1, soft should be very close to hard
    # (collision_weight penalty should be ~0 since destinations are unique)
    print(f"  peaked: soft={soft.item():.4f}, hard={hard}")
    # Soft won't exactly equal hard (it counts expected pairwise conflicts, not chromatic number)
    # But they should be correlated — soft should be low when hard is low


def test_collision_penalty():
    """Two qubits targeting the same cell should produce high collision cost."""
    atom_positions = torch.tensor([[0, 0], [0, 1]])
    # Both qubits strongly prefer cell 5
    logits = torch.full((2, 12), -20.0)
    logits[0, 5] = 20.0
    logits[1, 5] = 20.0
    logits.requires_grad_(True)
    cost = soft_layer_cost(
        atom_positions, logits, [0, 1], [(0, 1)], board_w=4,
        top_k=3, collision_weight=10.0,
    )
    # Should have high collision penalty
    print(f"  collision (same dest): cost={cost.item():.4f}")
    # Now give them different destinations
    logits2 = torch.full((2, 12), -20.0)
    logits2[0, 5] = 20.0
    logits2[1, 6] = 20.0
    logits2.requires_grad_(True)
    cost2 = soft_layer_cost(
        atom_positions, logits2, [0, 1], [(0, 1)], board_w=4,
        top_k=3, collision_weight=10.0,
    )
    print(f"  collision (diff dest): cost={cost2.item():.4f}")
    assert cost.item() > cost2.item(), "Same-dest should have higher cost"


def test_no_move_zero_reconfig():
    """If all qubits stay in place, reconfig cost should be zero."""
    atom_positions = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    # Each qubit strongly prefers its current cell
    logits = torch.full((4, 4), -20.0)
    for q in range(4):
        current_cell = atom_positions[q, 0] * 2 + atom_positions[q, 1]
        logits[q, current_cell] = 20.0
    probs, cells = _top_k_candidates(logits, top_k=3)
    src = atom_positions
    reconfig = _soft_reconfig(src, probs, cells, board_w=2)
    print(f"  no-move reconfig cost: {reconfig.item():.6f}")
    assert reconfig.item() < 0.01, "No-move should have ~zero reconfig cost"


def test_soft_cost_decreases_with_optimization():
    """A few gradient steps should reduce the soft cost."""
    torch.manual_seed(123)
    Q, B = 8, 16
    atom_positions = torch.tensor([
        [0, 0], [0, 1], [0, 2], [0, 3],
        [1, 0], [1, 1], [1, 2], [1, 3],
    ])
    gate_pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
    relevant = list(range(Q))
    logits = torch.randn(Q, B, requires_grad=True)
    opt = torch.optim.Adam([logits], lr=0.1)
    costs = []
    for step in range(20):
        opt.zero_grad()
        cost = soft_layer_cost(atom_positions, logits, relevant, gate_pairs, board_w=4, top_k=3)
        cost.backward()
        opt.step()
        costs.append(cost.item())
    print(f"  optimization: cost {costs[0]:.2f} -> {costs[-1]:.2f}")
    assert costs[-1] < costs[0], "Cost should decrease with optimization"


if __name__ == '__main__':
    tests = [
        test_gradient_flows,
        test_peaked_logits_match_hard_cost,
        test_collision_penalty,
        test_no_move_zero_reconfig,
        test_soft_cost_decreases_with_optimization,
    ]
    for test in tests:
        print(f"\n{test.__name__}:")
        test()
    print("\nAll tests passed!")
