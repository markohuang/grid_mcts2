"""Empirical analysis of action equivalence classes for CDG action-space reduction.

Tests the core claim: cells with the same "conflict signature" produce identical
count_groups costs. Measures reduction ratio on real maps.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
from collections import defaultdict
from itertools import product
from neutral_atoms.config import get_config, set_derived_config, get_map_data, atom_map_to_positions, MAPS
from neutral_atoms.env import NeutralAtomsEnv
from neutral_atoms.fast_moves import count_groups_fast

def sign(x):
    if x > 0: return 1
    if x < 0: return -1
    return 0

def reconfig_signature(r, c, sr, sc, committed_moves):
    """Conflict signature of destination (r,c) for atom with source (sr,sc)
    w.r.t. committed moves. Based on actual AOD crossing physics."""
    sig = []
    for (sr_j, sc_j, dr_j, dc_j) in committed_moves:
        # From is_parallel_executable: what determines conflict is
        # the sign relationship between source diffs and dest diffs
        v1_x = sr - sr_j  # source row diff (fixed for this atom)
        v1_y = sc - sc_j  # source col diff (fixed)
        v2_x = r - dr_j   # dest row diff (varies with candidate)
        v2_y = c - dc_j   # dest col diff (varies)

        # h_ok check
        if v1_x == 0 and v2_x == 0:
            h_ok = True
        elif v1_x == 0 or v2_x == 0:
            h_ok = False
        else:
            h_ok = (v1_x > 0) == (v2_x > 0)

        # v_ok check
        if v1_y == 0 and v2_y == 0:
            v_ok = True
        elif v1_y == 0 or v2_y == 0:
            v_ok = False
        else:
            v_ok = (v1_y > 0) == (v2_y > 0)

        # collision check
        collision = (r == dr_j and c == dc_j)

        # is_parallel = h_ok and v_ok and not collision
        sig.append((h_ok, v_ok, collision))
    return tuple(sig)


def gate_signature(r, c, qubit_idx, gate_pairs, atom_positions, committed_destinations):
    """Signature capturing how destination (r,c) for qubit_idx affects gate move conflicts.

    For each gate pair involving qubit_idx, compute the gate move that would result.
    For each OTHER gate pair, compute conflict pattern.
    """
    # Find which couple this qubit belongs to
    my_couples = []
    other_couples = []
    for pair in gate_pairs:
        q1, q2 = pair
        if qubit_idx == q1 or qubit_idx == q2:
            my_couples.append(pair)
        else:
            other_couples.append(pair)

    if not my_couples:
        return ('no_gate',)

    sig = []
    for pair in my_couples:
        q1, q2 = pair
        # Determine positions of both atoms in this couple
        if qubit_idx == q1:
            my_pos = (r, c)
            partner = q2
        else:
            my_pos = (r, c)
            partner = q1

        # Partner position: committed or current
        if partner in committed_destinations:
            partner_pos = committed_destinations[partner]
        else:
            partner_pos = tuple(atom_positions[partner].tolist())

        # Gate move: q1_pos -> q2_pos (before canonicalization)
        if qubit_idx == q1:
            gate_move = (my_pos[0], my_pos[1], partner_pos[0], partner_pos[1])
        else:
            gate_move = (partner_pos[0], partner_pos[1], my_pos[0], my_pos[1])

        # Compute conflict pattern with each other couple's gate move
        for other_pair in other_couples:
            oq1, oq2 = other_pair
            op1 = committed_destinations.get(oq1, tuple(atom_positions[oq1].tolist()))
            op2 = committed_destinations.get(oq2, tuple(atom_positions[oq2].tolist()))
            other_gate = (op1[0], op1[1], op2[0], op2[1])

            # Check parallel executability between gate_move and other_gate
            # (before canonicalization — canonicalization could flip either)
            # For signature purposes, compute all 4 combinations (flip/no-flip)
            for flip_mine in [False, True]:
                for flip_other in [False, True]:
                    gm = gate_move if not flip_mine else (gate_move[2], gate_move[3], gate_move[0], gate_move[1])
                    og = other_gate if not flip_other else (other_gate[2], other_gate[3], other_gate[0], other_gate[1])
                    v1_x = gm[0] - og[0]
                    v1_y = gm[1] - og[1]
                    v2_x = gm[2] - og[2]
                    v2_y = gm[3] - og[3]
                    if v1_x == 0 and v2_x == 0: h = True
                    elif v1_x == 0 or v2_x == 0: h = False
                    else: h = (v1_x > 0) == (v2_x > 0)
                    if v1_y == 0 and v2_y == 0: v = True
                    elif v1_y == 0 or v2_y == 0: v = False
                    else: v = (v1_y > 0) == (v2_y > 0)
                    coll = (v2_x == 0 and v2_y == 0)
                    sig.append((flip_mine, flip_other, h, v, coll))

    return tuple(sig)


def compute_layer_cost_for_assignment(atom_positions, assignments, gate_pairs, board_width):
    """Compute full layer cost given a complete assignment of atom destinations.
    assignments: dict mapping qubit_idx -> (row, col)
    """
    # Build reconfig moves (only for atoms that actually moved)
    reconfig_moves = []
    sim_positions = atom_positions.clone()
    for q, (dr, dc) in sorted(assignments.items()):
        sr, sc = atom_positions[q].tolist()
        if sr != dr or sc != dc:
            reconfig_moves.append([sr, sc, dr, dc])
        sim_positions[q, 0] = dr
        sim_positions[q, 1] = dc

    total = 0
    if reconfig_moves:
        rm = torch.tensor(reconfig_moves, dtype=torch.long)
        total += count_groups_fast(rm, canonicalize=False)

    # Gate moves from final positions
    if gate_pairs:
        gate_moves = []
        for q1, q2 in gate_pairs:
            gate_moves.append([sim_positions[q1, 0].item(), sim_positions[q1, 1].item(),
                              sim_positions[q2, 0].item(), sim_positions[q2, 1].item()])
        gm = torch.tensor(gate_moves, dtype=torch.long)
        total += 2 * count_groups_fast(gm, canonicalize=True)

    return total


def analyze_last_atom_equivalence(map_num=2, num_random_prefixes=50):
    """For the LAST atom in each layer, verify that same-signature cells -> same cost.
    This is where the equivalence should be exact."""
    config = get_config()
    config.map_num = map_num
    set_derived_config(config)
    map_data = get_map_data(config)
    positions = atom_map_to_positions(map_data['atom_map'], map_data['board_dim'][1])
    tasks = map_data['tasks']
    board_h, board_w = map_data['board_dim']

    env = NeutralAtomsEnv(tasks, positions, config.env)
    print(f"\n=== Map {map_num}: {board_h}x{board_w}, {config.env.num_qubits} qubits ===")

    violations = 0
    total_checks = 0
    reduction_ratios = []

    for layer_idx in range(len(tasks)):
        relevant = env.layer_relevant_atoms[layer_idx]
        gate_pairs = tasks[layer_idx]
        n_atoms = len(relevant)
        print(f"\nLayer {layer_idx}: {n_atoms} atoms, {len(gate_pairs)} gate pairs")
        print(f"  Relevant atoms: {relevant}")

        for trial in range(num_random_prefixes):
            # Reset env to initial state, advance to this layer
            env.reset()
            env.tasks_done = layer_idx
            env.current_atom_idx = 0
            env.current_phase_moves = []

            # Random prefix: place all atoms except the last one randomly
            committed = {}
            for i in range(n_atoms - 1):
                legal = env.legal_actions()
                action = np.random.choice(legal)
                q = env.current_qubit
                dr, dc = action // board_w, action % board_w
                committed[q] = (dr, dc)
                env.step(action, skip_obs=True)

            # Now we're at the last atom
            last_q = env.current_qubit
            legal = env.legal_actions()
            sr, sc = env.atom_positions[last_q].tolist()

            # Get committed moves as list
            committed_moves_list = []
            for m in env.current_phase_moves:
                committed_moves_list.append(tuple(m.tolist()))

            # Compute signature + actual cost for each legal action
            sig_to_cells = defaultdict(list)
            cell_to_cost = {}

            for action in legal:
                dr, dc = action // board_w, action % board_w

                # Reconfig signature
                rsig = reconfig_signature(dr, dc, sr, sc, committed_moves_list)
                # Gate signature
                gsig = gate_signature(dr, dc, last_q, gate_pairs,
                                     env.atom_positions, committed)
                full_sig = (rsig, gsig)
                sig_to_cells[full_sig].append(action)

                # Compute actual cost by completing the assignment
                full_assignment = dict(committed)
                full_assignment[last_q] = (dr, dc)
                cost = compute_layer_cost_for_assignment(
                    env.atom_positions, full_assignment, gate_pairs, board_w)
                cell_to_cost[action] = cost

            # Verify: cells in same signature class -> same cost
            for sig, cells in sig_to_cells.items():
                costs = [cell_to_cost[c] for c in cells]
                total_checks += 1
                if len(set(costs)) > 1:
                    violations += 1
                    if violations <= 5:
                        print(f"  VIOLATION: sig={sig}, cells={cells}, costs={costs}")

            ratio = len(legal) / max(len(sig_to_cells), 1)
            reduction_ratios.append((len(legal), len(sig_to_cells)))

    # Summary
    total_legal = sum(r[0] for r in reduction_ratios)
    total_classes = sum(r[1] for r in reduction_ratios)
    print(f"\n--- LAST-ATOM RESULTS ---")
    print(f"Violations: {violations}/{total_checks} signature classes")
    print(f"Avg legal actions: {total_legal/len(reduction_ratios):.1f}")
    print(f"Avg equiv classes: {total_classes/len(reduction_ratios):.1f}")
    print(f"Avg reduction ratio: {total_legal/max(total_classes,1):.2f}x")
    return violations


def analyze_intermediate_atom_equivalence(map_num=2, num_trials=30):
    """For intermediate atoms, check if same-signature cells can produce different
    costs when future atoms are optimally/pessimally assigned. Quantify the gap."""
    config = get_config()
    config.map_num = map_num
    set_derived_config(config)
    map_data = get_map_data(config)
    positions = atom_map_to_positions(map_data['atom_map'], map_data['board_dim'][1])
    tasks = map_data['tasks']
    board_h, board_w = map_data['board_dim']

    env = NeutralAtomsEnv(tasks, positions, config.env)
    print(f"\n=== INTERMEDIATE ATOM ANALYSIS: Map {map_num} ===")

    for layer_idx in range(len(tasks)):
        relevant = env.layer_relevant_atoms[layer_idx]
        gate_pairs = tasks[layer_idx]
        n_atoms = len(relevant)
        if n_atoms <= 2:
            continue

        print(f"\nLayer {layer_idx}: {n_atoms} atoms")

        # Test atom at position 0 (most future uncertainty)
        for trial in range(min(num_trials, 5)):
            env.reset()
            env.tasks_done = layer_idx
            env.current_atom_idx = 0
            env.current_phase_moves = []

            first_q = env.current_qubit
            legal = env.legal_actions()
            sr, sc = env.atom_positions[first_q].tolist()

            # No committed moves yet -> reconfig signature is empty for everyone
            # Gate signature depends only on initial positions of other atoms
            sig_to_cells = defaultdict(list)

            for action in legal:
                dr, dc = action // board_w, action % board_w
                rsig = reconfig_signature(dr, dc, sr, sc, [])
                gsig = gate_signature(dr, dc, first_q, gate_pairs,
                                     env.atom_positions, {})
                sig = (rsig, gsig)
                sig_to_cells[sig].append(action)

            # For each signature class with >1 cell, check if different cells
            # can lead to different optimal costs via brute-force over remaining atoms
            disagree_count = 0
            agree_count = 0

            for sig, cells in sig_to_cells.items():
                if len(cells) <= 1:
                    continue
                # Sample two cells from this class
                c1, c2 = cells[0], cells[1]

                # For each, try a few random completions and compare costs
                costs_c1 = []
                costs_c2 = []
                for _ in range(20):
                    for test_cell, cost_list in [(c1, costs_c1), (c2, costs_c2)]:
                        test_env = env.clone()
                        test_env.step(test_cell, skip_obs=True)
                        # Random completion
                        while test_env.current_atom_idx > 0:  # still in same layer
                            la = test_env.legal_actions()
                            test_env.step(np.random.choice(la), skip_obs=True)
                        # Cost was computed at layer completion via reward
                        # Actually, let's reconstruct...
                        pass

                # Simpler: just measure if the optimal achievable cost differs
                # For small action spaces, try all completions
                if n_atoms <= 6:  # feasible to enumerate
                    best_c1 = _best_cost_bruteforce(env, c1, gate_pairs, board_w)
                    best_c2 = _best_cost_bruteforce(env, c2, gate_pairs, board_w)
                    if best_c1 != best_c2:
                        disagree_count += 1
                        if disagree_count <= 3:
                            print(f"  Disagree: cells {c1},{c2} same sig, best costs {best_c1} vs {best_c2}")
                    else:
                        agree_count += 1

            if agree_count + disagree_count > 0:
                print(f"  Trial {trial}: {agree_count} agree, {disagree_count} disagree "
                      f"(out of {len([s for s,c in sig_to_cells.items() if len(c)>1])} multi-cell classes)")


def _best_cost_bruteforce(env, first_action, gate_pairs, board_w, max_completions=500):
    """Try random completions after first_action, return best cost found."""
    best = float('inf')
    for _ in range(max_completions):
        test_env = env.clone()
        # Apply first action
        result = test_env.step(first_action, skip_obs=True)
        # Complete layer randomly
        while test_env.tasks_done == env.tasks_done:
            la = test_env.legal_actions()
            if not la:
                break
            result = test_env.step(np.random.choice(la), skip_obs=True)
        # Reward is -layer_cost * reward_scale at layer completion
        # The step that completed the layer had the reward
        cost = -result.reward / env.reward_scale if result.reward != 0 else float('inf')
        best = min(best, cost)
    return best


def analyze_reduction_throughout_layer(map_num=2, num_trials=50):
    """Measure reduction ratio at each atom position within a layer."""
    config = get_config()
    config.map_num = map_num
    set_derived_config(config)
    map_data = get_map_data(config)
    positions = atom_map_to_positions(map_data['atom_map'], map_data['board_dim'][1])
    tasks = map_data['tasks']
    board_h, board_w = map_data['board_dim']

    env = NeutralAtomsEnv(tasks, positions, config.env)
    print(f"\n=== REDUCTION BY POSITION: Map {map_num} ===")

    for layer_idx in range(len(tasks)):
        relevant = env.layer_relevant_atoms[layer_idx]
        n_atoms = len(relevant)
        print(f"\nLayer {layer_idx}: {n_atoms} atoms")

        # Collect stats per atom position
        stats = defaultdict(lambda: {'legal': [], 'classes': [], 'reconfig_classes': []})

        for trial in range(num_trials):
            env.reset()
            env.tasks_done = layer_idx
            env.current_atom_idx = 0
            env.current_phase_moves = []

            for atom_pos in range(n_atoms):
                q = env.current_qubit
                legal = env.legal_actions()
                sr, sc = env.atom_positions[q].tolist()

                committed_moves_list = [tuple(m.tolist()) for m in env.current_phase_moves]
                committed_dests = {}
                for m in env.current_phase_moves:
                    # Find which qubit this move belongs to
                    # Actually we need to track this... let me use atom_positions
                    pass

                # Just compute reconfig signature (the simpler, provably-correct part)
                rsig_to_cells = defaultdict(list)
                for action in legal:
                    dr, dc = action // board_w, action % board_w
                    rsig = reconfig_signature(dr, dc, sr, sc, committed_moves_list)
                    rsig_to_cells[rsig].append(action)

                stats[atom_pos]['legal'].append(len(legal))
                stats[atom_pos]['reconfig_classes'].append(len(rsig_to_cells))

                # Take random action and continue
                action = np.random.choice(legal)
                env.step(action, skip_obs=True)

        # Print stats
        for pos in range(n_atoms):
            s = stats[pos]
            avg_legal = np.mean(s['legal'])
            avg_rc = np.mean(s['reconfig_classes'])
            print(f"  Atom {pos}/{n_atoms}: avg legal={avg_legal:.1f}, "
                  f"avg reconfig classes={avg_rc:.1f}, "
                  f"reduction={avg_legal/max(avg_rc,1):.2f}x")


if __name__ == '__main__':
    # Test 1: Verify equivalence is exact for last atom
    print("=" * 60)
    print("TEST 1: Last-atom equivalence (should be exact)")
    print("=" * 60)
    for map_num in [0, 1, 2]:
        analyze_last_atom_equivalence(map_num, num_random_prefixes=30)

    # Test 2: Measure reduction ratios throughout layer
    print("\n" + "=" * 60)
    print("TEST 2: Reduction ratio by atom position in layer")
    print("=" * 60)
    for map_num in [0, 1, 2]:
        analyze_reduction_throughout_layer(map_num, num_trials=30)

    # Test 3: Check intermediate atom equivalence
    print("\n" + "=" * 60)
    print("TEST 3: Intermediate atom — do same-sig cells differ in optimal cost?")
    print("=" * 60)
    analyze_intermediate_atom_equivalence(map_num=2, num_trials=5)
