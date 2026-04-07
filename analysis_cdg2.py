"""Deep analysis of CDG action equivalence.

1. Separate reconfig vs gate cost to find where violations come from
2. Test if reconfig-only signature is exact for last atom
3. Quantify actual MCTS branching reduction
4. Test intermediate atoms with random completions
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
from collections import defaultdict
from neutral_atoms.config import get_config, set_derived_config, get_map_data, atom_map_to_positions
from neutral_atoms.env import NeutralAtomsEnv
from neutral_atoms.fast_moves import count_groups_fast


def sign(x):
    if x > 0: return 1
    if x < 0: return -1
    return 0


def reconfig_signature(r, c, sr, sc, committed_moves):
    """Signature based on actual AOD displacement-crossing physics."""
    sig = []
    for (sr_j, sc_j, dr_j, dc_j) in committed_moves:
        v1_x = sr - sr_j
        v1_y = sc - sc_j
        v2_x = r - dr_j
        v2_y = c - dc_j
        if v1_x == 0 and v2_x == 0: h_ok = True
        elif v1_x == 0 or v2_x == 0: h_ok = False
        else: h_ok = (v1_x > 0) == (v2_x > 0)
        if v1_y == 0 and v2_y == 0: v_ok = True
        elif v1_y == 0 or v2_y == 0: v_ok = False
        else: v_ok = (v1_y > 0) == (v2_y > 0)
        collision = (r == dr_j and c == dc_j)
        sig.append((h_ok, v_ok, collision))
    return tuple(sig)


def compute_costs_separately(atom_positions, assignments, gate_pairs, board_w):
    """Return (reconfig_cost, gate_cost) separately for a complete assignment."""
    reconfig_moves = []
    sim_positions = atom_positions.clone()
    for q, (dr, dc) in sorted(assignments.items()):
        sr, sc = atom_positions[q].tolist()
        if sr != dr or sc != dc:
            reconfig_moves.append([sr, sc, dr, dc])
        sim_positions[q, 0] = dr
        sim_positions[q, 1] = dc

    rc = 0
    if reconfig_moves:
        rm = torch.tensor(reconfig_moves, dtype=torch.long)
        rc = count_groups_fast(rm, canonicalize=False)

    gc = 0
    if gate_pairs:
        gate_moves = []
        for q1, q2 in gate_pairs:
            gate_moves.append([sim_positions[q1, 0].item(), sim_positions[q1, 1].item(),
                              sim_positions[q2, 0].item(), sim_positions[q2, 1].item()])
        gm = torch.tensor(gate_moves, dtype=torch.long)
        gc = 2 * count_groups_fast(gm, canonicalize=True)

    return rc, gc


# ======================================================================
# TEST 1: Which cost component causes violations in last-atom signature?
# ======================================================================
def test_reconfig_vs_gate_violations(map_num=2, num_trials=50):
    config = get_config()
    config.map_num = map_num
    set_derived_config(config)
    map_data = get_map_data(config)
    positions = atom_map_to_positions(map_data['atom_map'], map_data['board_dim'][1])
    tasks = map_data['tasks']
    board_h, board_w = map_data['board_dim']
    env = NeutralAtomsEnv(tasks, positions, config.env)

    print(f"\n=== Map {map_num}: {board_h}x{board_w}, {config.env.num_qubits}q ===")

    rc_violations = 0
    gc_violations = 0
    total_violations = 0
    total_classes = 0

    for layer_idx in range(len(tasks)):
        relevant = env.layer_relevant_atoms[layer_idx]
        gate_pairs = tasks[layer_idx]
        n_atoms = len(relevant)

        for trial in range(num_trials):
            env.reset()
            env.tasks_done = layer_idx
            env.current_atom_idx = 0
            env.current_phase_moves = []
            committed = {}

            # Random prefix for all but last atom
            for i in range(n_atoms - 1):
                legal = env.legal_actions()
                action = np.random.choice(legal)
                q = env.current_qubit
                dr, dc = action // board_w, action % board_w
                committed[q] = (dr, dc)
                env.step(action, skip_obs=True)

            last_q = env.current_qubit
            legal = env.legal_actions()
            sr, sc = env.atom_positions[last_q].tolist()
            committed_moves_list = [tuple(m.tolist()) for m in env.current_phase_moves]

            # Group by reconfig-only signature
            sig_to_cells = defaultdict(list)
            cell_costs = {}  # action -> (rc, gc, total)

            for action in legal:
                dr, dc = action // board_w, action % board_w
                rsig = reconfig_signature(dr, dc, sr, sc, committed_moves_list)
                sig_to_cells[rsig].append(action)

                full_assign = dict(committed)
                full_assign[last_q] = (dr, dc)
                rc, gc = compute_costs_separately(env.atom_positions, full_assign, gate_pairs, board_w)
                cell_costs[action] = (rc, gc, rc + gc)

            for sig, cells in sig_to_cells.items():
                total_classes += 1
                rcs = set(cell_costs[c][0] for c in cells)
                gcs = set(cell_costs[c][1] for c in cells)
                tots = set(cell_costs[c][2] for c in cells)
                if len(rcs) > 1:
                    rc_violations += 1
                if len(gcs) > 1:
                    gc_violations += 1
                if len(tots) > 1:
                    total_violations += 1

    print(f"  Reconfig-only signature violations (last atom):")
    print(f"    Reconfig cost varies within class: {rc_violations}/{total_classes}")
    print(f"    Gate cost varies within class:     {gc_violations}/{total_classes}")
    print(f"    Total cost varies within class:    {total_violations}/{total_classes}")


# ======================================================================
# TEST 2: Full conflict signature (reconfig + destination position info)
# ======================================================================
def full_last_atom_signature(r, c, sr, sc, committed_moves, gate_pairs, qubit_idx, atom_positions, committed_dests):
    """Extended signature: reconfig sig + gate-relevant position info."""
    rsig = reconfig_signature(r, c, sr, sc, committed_moves)

    # For gate cost: the gate move for this qubit's couple depends on
    # the relative position to its partner. Include this directly.
    gsig = []
    for q1, q2 in gate_pairs:
        if qubit_idx != q1 and qubit_idx != q2:
            continue
        partner = q2 if qubit_idx == q1 else q1
        if partner in committed_dests:
            pr, pc = committed_dests[partner]
        else:
            pr, pc = atom_positions[partner].tolist()
        # Gate move: (my_pos -> partner_pos) or vice versa
        # Include exact relative position to partner (not just signs)
        # because canonicalization depends on exact displacement magnitude
        gsig.append((r - pr, c - pc))

    # Also: position of our destination relative to ALL other couples' atoms
    # (affects gate-gate conflicts after canonicalization)
    for q1, q2 in gate_pairs:
        if qubit_idx == q1 or qubit_idx == q2:
            continue
        # Other couple's atom positions
        for oq in [q1, q2]:
            if oq in committed_dests:
                opr, opc = committed_dests[oq]
            else:
                opr, opc = atom_positions[oq].tolist()
            gsig.append((sign(r - opr), sign(c - opc)))

    return (rsig, tuple(gsig))


def test_full_signature_violations(map_num=2, num_trials=50):
    config = get_config()
    config.map_num = map_num
    set_derived_config(config)
    map_data = get_map_data(config)
    positions = atom_map_to_positions(map_data['atom_map'], map_data['board_dim'][1])
    tasks = map_data['tasks']
    board_h, board_w = map_data['board_dim']
    env = NeutralAtomsEnv(tasks, positions, config.env)

    print(f"\n=== FULL SIGNATURE: Map {map_num} ===")

    violations = 0
    total_classes = 0
    total_legal = 0
    total_sig_classes = 0

    for layer_idx in range(len(tasks)):
        relevant = env.layer_relevant_atoms[layer_idx]
        gate_pairs = tasks[layer_idx]
        n_atoms = len(relevant)

        layer_legal = 0
        layer_classes = 0

        for trial in range(num_trials):
            env.reset()
            env.tasks_done = layer_idx
            env.current_atom_idx = 0
            env.current_phase_moves = []
            committed = {}

            for i in range(n_atoms - 1):
                legal = env.legal_actions()
                action = np.random.choice(legal)
                q = env.current_qubit
                dr, dc = action // board_w, action % board_w
                committed[q] = (dr, dc)
                env.step(action, skip_obs=True)

            last_q = env.current_qubit
            legal = env.legal_actions()
            sr, sc = env.atom_positions[last_q].tolist()
            committed_moves_list = [tuple(m.tolist()) for m in env.current_phase_moves]

            sig_to_cells = defaultdict(list)
            cell_costs = {}

            for action in legal:
                dr, dc = action // board_w, action % board_w
                fsig = full_last_atom_signature(
                    dr, dc, sr, sc, committed_moves_list, gate_pairs,
                    last_q, env.atom_positions, committed)
                sig_to_cells[fsig].append(action)

                full_assign = dict(committed)
                full_assign[last_q] = (dr, dc)
                rc, gc = compute_costs_separately(env.atom_positions, full_assign, gate_pairs, board_w)
                cell_costs[action] = rc + gc

            for sig, cells in sig_to_cells.items():
                total_classes += 1
                costs = set(cell_costs[c] for c in cells)
                if len(costs) > 1:
                    violations += 1

            layer_legal += len(legal)
            layer_classes += len(sig_to_cells)

        total_legal += layer_legal
        total_sig_classes += layer_classes
        print(f"  Layer {layer_idx}: avg legal={layer_legal/num_trials:.1f}, "
              f"avg classes={layer_classes/num_trials:.1f}, "
              f"reduction={layer_legal/max(layer_classes,1):.2f}x")

    print(f"  Violations: {violations}/{total_classes}")
    print(f"  Overall reduction: {total_legal/max(total_sig_classes,1):.2f}x")


# ======================================================================
# TEST 3: Intermediate atoms — sample-based cost comparison
# ======================================================================
def test_intermediate_atom_divergence(map_num=2, num_trials=20, num_completions=200):
    """For each atom position, pick two cells with same reconfig signature,
    estimate optimal cost via random completions, measure divergence."""
    config = get_config()
    config.map_num = map_num
    set_derived_config(config)
    map_data = get_map_data(config)
    positions = atom_map_to_positions(map_data['atom_map'], map_data['board_dim'][1])
    tasks = map_data['tasks']
    board_h, board_w = map_data['board_dim']
    env = NeutralAtomsEnv(tasks, positions, config.env)

    print(f"\n=== INTERMEDIATE DIVERGENCE: Map {map_num} ===")

    for layer_idx in range(len(tasks)):
        relevant = env.layer_relevant_atoms[layer_idx]
        n_atoms = len(relevant)
        gate_pairs = tasks[layer_idx]

        print(f"\nLayer {layer_idx}: {n_atoms} atoms")

        for atom_pos in range(n_atoms - 1):  # skip last (already tested exactly)
            divergences = []
            tested_pairs = 0

            for trial in range(num_trials):
                env.reset()
                env.tasks_done = layer_idx
                env.current_atom_idx = 0
                env.current_phase_moves = []

                # Random prefix up to atom_pos
                for i in range(atom_pos):
                    legal = env.legal_actions()
                    env.step(np.random.choice(legal), skip_obs=True)

                q = env.current_qubit
                legal = env.legal_actions()
                sr, sc = env.atom_positions[q].tolist()
                committed_moves_list = [tuple(m.tolist()) for m in env.current_phase_moves]

                # Group by reconfig signature
                sig_to_cells = defaultdict(list)
                for action in legal:
                    dr, dc = action // board_w, action % board_w
                    rsig = reconfig_signature(dr, dc, sr, sc, committed_moves_list)
                    sig_to_cells[rsig].append(action)

                # Pick classes with >1 cell
                for sig, cells in sig_to_cells.items():
                    if len(cells) < 2:
                        continue
                    c1, c2 = cells[0], cells[1]

                    # Estimate best achievable cost for each via random completions
                    best1 = _estimate_best_cost(env, c1, num_completions)
                    best2 = _estimate_best_cost(env, c2, num_completions)
                    divergences.append(abs(best1 - best2))
                    tested_pairs += 1

            if divergences:
                divs = np.array(divergences)
                print(f"  Atom {atom_pos}/{n_atoms}: {tested_pairs} pairs tested, "
                      f"cost divergence: mean={divs.mean():.2f}, "
                      f"max={divs.max():.0f}, "
                      f"fraction>0: {(divs>0).mean():.2f}")
            else:
                print(f"  Atom {atom_pos}/{n_atoms}: no multi-cell classes found")


def _estimate_best_cost(env, first_action, num_completions):
    """Complete the layer randomly many times, return best cost found."""
    best = float('inf')
    layer = env.tasks_done
    for _ in range(num_completions):
        test_env = env.clone()
        result = test_env.step(first_action, skip_obs=True)
        while test_env.tasks_done == layer:
            la = test_env.legal_actions()
            if not la:
                break
            result = test_env.step(np.random.choice(la), skip_obs=True)
        # The step that completed the layer has reward = -layer_cost * reward_scale
        if result.reward != 0:
            cost = -result.reward / env.reward_scale
            best = min(best, cost)
    return best


# ======================================================================
# TEST 4: What if we use exact cost grouping? (oracle upper bound on reduction)
# ======================================================================
def test_oracle_cost_grouping(map_num=2, num_trials=50):
    """Group legal actions by the actual cost they produce (for last atom).
    This gives the maximum achievable reduction if we had a perfect signature."""
    config = get_config()
    config.map_num = map_num
    set_derived_config(config)
    map_data = get_map_data(config)
    positions = atom_map_to_positions(map_data['atom_map'], map_data['board_dim'][1])
    tasks = map_data['tasks']
    board_h, board_w = map_data['board_dim']
    env = NeutralAtomsEnv(tasks, positions, config.env)

    print(f"\n=== ORACLE COST GROUPING (last atom): Map {map_num} ===")

    for layer_idx in range(len(tasks)):
        relevant = env.layer_relevant_atoms[layer_idx]
        gate_pairs = tasks[layer_idx]
        n_atoms = len(relevant)

        total_legal = 0
        total_cost_classes = 0
        total_rsig_classes = 0

        for trial in range(num_trials):
            env.reset()
            env.tasks_done = layer_idx
            env.current_atom_idx = 0
            env.current_phase_moves = []
            committed = {}

            for i in range(n_atoms - 1):
                legal = env.legal_actions()
                action = np.random.choice(legal)
                q = env.current_qubit
                dr, dc = action // board_w, action % board_w
                committed[q] = (dr, dc)
                env.step(action, skip_obs=True)

            last_q = env.current_qubit
            legal = env.legal_actions()
            sr, sc = env.atom_positions[last_q].tolist()
            committed_moves_list = [tuple(m.tolist()) for m in env.current_phase_moves]

            # Group by actual cost (oracle)
            cost_to_cells = defaultdict(list)
            rsig_to_cells = defaultdict(list)
            for action in legal:
                dr, dc = action // board_w, action % board_w
                full_assign = dict(committed)
                full_assign[last_q] = (dr, dc)
                rc, gc = compute_costs_separately(env.atom_positions, full_assign, gate_pairs, board_w)
                cost_to_cells[rc + gc].append(action)

                rsig = reconfig_signature(dr, dc, sr, sc, committed_moves_list)
                rsig_to_cells[rsig].append(action)

            total_legal += len(legal)
            total_cost_classes += len(cost_to_cells)
            total_rsig_classes += len(rsig_to_cells)

        avg_legal = total_legal / num_trials
        avg_cost = total_cost_classes / num_trials
        avg_rsig = total_rsig_classes / num_trials
        print(f"  Layer {layer_idx}: legal={avg_legal:.1f}, "
              f"cost_classes={avg_cost:.1f} ({avg_legal/avg_cost:.2f}x oracle), "
              f"rsig_classes={avg_rsig:.1f} ({avg_legal/avg_rsig:.2f}x reconfig-only)")


if __name__ == '__main__':
    print("=" * 70)
    print("TEST 1: Where do violations come from? (reconfig vs gate cost)")
    print("=" * 70)
    for m in [0, 1, 2]:
        test_reconfig_vs_gate_violations(m, num_trials=50)

    print("\n" + "=" * 70)
    print("TEST 2: Full signature with gate position info")
    print("=" * 70)
    for m in [0, 1, 2]:
        test_full_signature_violations(m, num_trials=50)

    print("\n" + "=" * 70)
    print("TEST 3: Oracle cost grouping — maximum achievable reduction")
    print("=" * 70)
    for m in [0, 1, 2]:
        test_oracle_cost_grouping(m, num_trials=50)

    print("\n" + "=" * 70)
    print("TEST 4: Intermediate atom cost divergence within signature classes")
    print("=" * 70)
    test_intermediate_atom_divergence(map_num=2, num_trials=10, num_completions=100)
