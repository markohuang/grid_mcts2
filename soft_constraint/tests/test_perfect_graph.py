"""
tests/test_perfect_graph.py — Check if AOD conflict graphs are perfect.

A graph is perfect iff it has no induced odd hole or odd antihole of length >= 5
(Strong Perfect Graph Theorem, 2006).

We generate random AOD conflict graphs from:
  - Random atom placements on HxW grids
  - Random gate pairs
  - Both reconfig and gate conflict structures

and check each for perfectness.
"""

import torch
import sys, os
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from surrogate.primitives import aod_compatible


def build_aod_conflict_adj(moves):
    """Build binary adjacency matrix for AOD conflict graph."""
    n = moves.shape[0]
    A = torch.zeros(n, n, dtype=torch.bool)
    for i in range(n):
        for j in range(i + 1, n):
            if not aod_compatible(moves[i], moves[j]):
                A[i, j] = A[j, i] = True
    return A


def is_cycle(adj, nodes):
    """Check if induced subgraph on `nodes` is a cycle."""
    k = len(nodes)
    if k < 5: return False
    # Each node must have exactly degree 2 in the induced subgraph
    sub = adj[nodes][:, nodes]
    degs = sub.sum(dim=1)
    if not (degs == 2).all():
        return False
    # Must be connected (a single cycle, not union of cycles)
    visited = set()
    stack = [0]
    while stack:
        v = stack.pop()
        if v in visited: continue
        visited.add(v)
        for u in range(k):
            if sub[v, u] and u not in visited:
                stack.append(u)
    return len(visited) == k


def is_odd_hole(adj, nodes):
    """Check if induced subgraph is an odd hole (odd cycle, length >= 5)."""
    sub = adj[nodes][:, nodes]
    local_nodes = list(range(len(nodes)))
    return len(nodes) % 2 == 1 and is_cycle(sub, local_nodes)


def is_odd_antihole(adj, nodes):
    """Check if complement of induced subgraph is an odd hole."""
    sub = adj[nodes][:, nodes]
    complement = ~sub
    complement.fill_diagonal_(False)
    local_nodes = list(range(len(nodes)))
    return len(nodes) % 2 == 1 and is_cycle(complement, local_nodes)


def check_perfect(adj):
    """Check if graph is perfect. Returns (is_perfect, witness) where
    witness is the forbidden induced subgraph if not perfect."""
    n = adj.shape[0]
    # Check subsets of odd size 5, 7, 9, ...
    for k in range(5, n + 1, 2):
        for subset in combinations(range(n), k):
            nodes = list(subset)
            if is_odd_hole(adj, nodes):
                return False, ('odd_hole', nodes)
            if is_odd_antihole(adj, nodes):
                return False, ('odd_antihole', nodes)
    return True, None


def random_gate_conflict_graph(H, W, num_atoms, num_gates, seed=None):
    """Generate a random gate conflict graph.

    Places atoms randomly on HxW grid, pairs them into gates,
    picks random directions, returns the conflict adjacency matrix.
    """
    if seed is not None: torch.manual_seed(seed)
    C = H * W
    # Random distinct positions
    perm = torch.randperm(C)[:num_atoms]
    rows, cols = perm // W, perm % W
    # Random gate pairs (each gate uses 2 distinct atoms)
    atom_perm = torch.randperm(num_atoms)
    gates = [(atom_perm[2*g].item(), atom_perm[2*g+1].item())
             for g in range(num_gates)]
    # Random directions
    dirs = torch.randint(0, 2, (num_gates,))
    # Build moves
    moves = []
    for g, (a, b) in enumerate(gates):
        if dirs[g]: a, b = b, a
        moves.append(torch.tensor([rows[a], cols[a], rows[b], cols[b]], dtype=torch.long))
    moves = torch.stack(moves)
    return build_aod_conflict_adj(moves), moves


def random_reconfig_conflict_graph(H, W, num_movers, seed=None):
    """Generate a random reconfig conflict graph.

    Picks random src and dst positions for each mover,
    returns the conflict adjacency matrix.
    """
    if seed is not None: torch.manual_seed(seed)
    C = H * W
    # Random distinct source positions
    src_perm = torch.randperm(C)[:num_movers]
    # Random distinct destination positions
    dst_perm = torch.randperm(C)[:num_movers]
    src_rows, src_cols = src_perm // W, src_perm % W
    dst_rows, dst_cols = dst_perm // W, dst_perm % W
    moves = torch.stack([src_rows, src_cols, dst_rows, dst_cols], dim=1)
    # Filter out non-movers (src == dst)
    moved = (src_perm != dst_perm)
    if moved.sum() < 2:
        return torch.zeros(0, 0, dtype=torch.bool), torch.zeros(0, 4, dtype=torch.long)
    moves = moves[moved]
    return build_aod_conflict_adj(moves), moves


# ═════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    N_TRIALS = 500

    for grid_label, H, W in [('5x5', 5, 5), ('8x8', 8, 8)]:
        print(f"\n{'═' * 70}")
        print(f"  GRID: {grid_label}  ({H}×{W} = {H*W} cells)")
        print(f"{'═' * 70}")

        # ── Gate conflict graphs ──
        for num_gates in [3, 4, 5]:
            num_atoms = num_gates * 2
            if num_atoms > H * W:
                continue
            perfect_count = 0
            total = 0
            witnesses = []
            for trial in range(N_TRIALS):
                adj, moves = random_gate_conflict_graph(
                    H, W, num_atoms, num_gates, seed=trial * 1000 + num_gates)
                if adj.shape[0] < 3:
                    continue
                total += 1
                is_perf, witness = check_perfect(adj)
                if is_perf:
                    perfect_count += 1
                elif len(witnesses) < 3:
                    witnesses.append((trial, witness, adj, moves))

            pct = 100 * perfect_count / total if total > 0 else 0
            print(f"\n  Gate conflicts | {num_gates} gates | "
                  f"{perfect_count}/{total} perfect ({pct:.1f}%)")
            for trial, (wtype, wnodes), adj, moves in witnesses:
                n_edges = adj.sum().item() // 2
                print(f"    trial={trial}: {wtype} on nodes {wnodes}, "
                      f"|V|={adj.shape[0]}, |E|={n_edges}")
                print(f"    moves: {moves.tolist()}")

        # ── Reconfig conflict graphs ──
        for num_movers in [4, 6, 8, 10]:
            if num_movers > H * W:
                continue
            perfect_count = 0
            total = 0
            witnesses = []
            for trial in range(N_TRIALS):
                adj, moves = random_reconfig_conflict_graph(
                    H, W, num_movers, seed=trial * 1000 + num_movers + 100)
                if adj.shape[0] < 3:
                    continue
                total += 1
                is_perf, witness = check_perfect(adj)
                if is_perf:
                    perfect_count += 1
                elif len(witnesses) < 3:
                    witnesses.append((trial, witness, adj, moves))

            pct = 100 * perfect_count / total if total > 0 else 0
            print(f"\n  Reconfig conflicts | {num_movers} movers | "
                  f"{perfect_count}/{total} perfect ({pct:.1f}%)")
            for trial, (wtype, wnodes), adj, moves in witnesses:
                n_edges = adj.sum().item() // 2
                print(f"    trial={trial}: {wtype} on nodes {wnodes}, "
                      f"|V|={adj.shape[0]}, |E|={n_edges}")

    print(f"\n{'═' * 70}")
    print(f"  DONE")
    print(f"{'═' * 70}")
