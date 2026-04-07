"""Final CDG analysis: fix no-op issue, measure oracle for all positions."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
from collections import defaultdict
from neutral_atoms.config import get_config, set_derived_config, get_map_data, atom_map_to_positions
from neutral_atoms.env import NeutralAtomsEnv
from neutral_atoms.fast_moves import count_groups_fast


def reconfig_signature_v2(r, c, sr, sc, committed_moves, is_noop):
    """Fixed signature: separate no-op from non-no-op, since no-op doesn't
    add a reconfig move to the conflict graph."""
    if is_noop:
        return ('NOOP',)
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
# TEST 1: Fixed reconfig signature (no-op separated)
# ======================================================================
def test_fixed_reconfig_sig(map_num=2, num_trials=100):
    config = get_config()
    config.map_num = map_num
    set_derived_config(config)
    map_data = get_map_data(config)
    positions = atom_map_to_positions(map_data['atom_map'], map_data['board_dim'][1])
    tasks = map_data['tasks']
    board_h, board_w = map_data['board_dim']
    env = NeutralAtomsEnv(tasks, positions, config.env)

    print(f"\n=== FIXED RECONFIG SIG: Map {map_num} ({board_h}x{board_w}) ===")

    rc_violations = 0
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
            current_flat = sr * board_w + sc
            committed_moves_list = [tuple(m.tolist()) for m in env.current_phase_moves]

            sig_to_cells = defaultdict(list)
            cell_rc = {}

            for action in legal:
                dr, dc = action // board_w, action % board_w
                is_noop = (action == current_flat)
                rsig = reconfig_signature_v2(dr, dc, sr, sc, committed_moves_list, is_noop)
                sig_to_cells[rsig].append(action)

                full_assign = dict(committed)
                full_assign[last_q] = (dr, dc)
                rc, gc = compute_costs_separately(env.atom_positions, full_assign, gate_pairs, board_w)
                cell_rc[action] = rc

            for sig, cells in sig_to_cells.items():
                total_classes += 1
                rcs = set(cell_rc[c] for c in cells)
                if len(rcs) > 1:
                    rc_violations += 1
                    if rc_violations <= 3:
                        print(f"  RC violation: sig={sig}, cells={cells}, costs={[cell_rc[c] for c in cells]}")
                        # Print the committed moves and atom positions for debugging
                        print(f"    committed_moves: {committed_moves_list}")
                        print(f"    atom src: ({sr},{sc})")

    print(f"  Reconfig cost violations: {rc_violations}/{total_classes}")


# ======================================================================
# TEST 2: Oracle reduction at EVERY atom position (not just last)
# Uses random completions to estimate cost equivalence
# ======================================================================
def test_oracle_all_positions(map_num=2, num_trials=30, num_completions=300):
    """For each atom position, estimate cost for each legal action via
    random completions. Group by estimated-best-cost. Compare to signature."""
    config = get_config()
    config.map_num = map_num
    set_derived_config(config)
    map_data = get_map_data(config)
    positions = atom_map_to_positions(map_data['atom_map'], map_data['board_dim'][1])
    tasks = map_data['tasks']
    board_h, board_w = map_data['board_dim']
    env = NeutralAtomsEnv(tasks, positions, config.env)

    print(f"\n=== ORACLE AT ALL POSITIONS: Map {map_num} ===")

    for layer_idx in range(len(tasks)):
        relevant = env.layer_relevant_atoms[layer_idx]
        n_atoms = len(relevant)
        print(f"\nLayer {layer_idx}: {n_atoms} atoms")

        for atom_pos in range(n_atoms):
            legal_counts = []
            oracle_counts = []
            rsig_counts = []

            for trial in range(num_trials):
                env.reset()
                env.tasks_done = layer_idx
                env.current_atom_idx = 0
                env.current_phase_moves = []

                for i in range(atom_pos):
                    la = env.legal_actions()
                    env.step(np.random.choice(la), skip_obs=True)

                q = env.current_qubit
                legal = env.legal_actions()
                sr, sc = env.atom_positions[q].tolist()
                current_flat = sr * board_w + sc
                committed_moves_list = [tuple(m.tolist()) for m in env.current_phase_moves]

                # For last atom: exact cost. For others: sampled best cost.
                is_last = (atom_pos == n_atoms - 1)
                gate_pairs = tasks[layer_idx]
                committed = {}
                for m in env.current_phase_moves:
                    s_r, s_c, d_r, d_c = m.tolist()
                    # Find which qubit moved from (s_r, s_c)
                    # Since board is updated, we need to find by destination
                    qid = env.board[d_r, d_c].item()
                    committed[qid] = (d_r, d_c)

                cost_to_cells = defaultdict(list)
                rsig_to_cells = defaultdict(list)

                if is_last:
                    # Exact cost computation
                    for action in legal:
                        dr, dc = action // board_w, action % board_w
                        full_assign = dict(committed)
                        full_assign[q] = (dr, dc)
                        rc, gc = compute_costs_separately(env.atom_positions, full_assign, gate_pairs, board_w)
                        cost_to_cells[rc + gc].append(action)

                        is_noop = (action == current_flat)
                        rsig = reconfig_signature_v2(dr, dc, sr, sc, committed_moves_list, is_noop)
                        rsig_to_cells[rsig].append(action)
                else:
                    # Sampled best cost
                    layer = env.tasks_done
                    for action in legal:
                        best_cost = float('inf')
                        for _ in range(num_completions):
                            test_env = env.clone()
                            result = test_env.step(action, skip_obs=True)
                            while test_env.tasks_done == layer:
                                la2 = test_env.legal_actions()
                                if not la2: break
                                result = test_env.step(np.random.choice(la2), skip_obs=True)
                            if result.reward != 0:
                                cost = int(round(-result.reward / env.reward_scale))
                                best_cost = min(best_cost, cost)
                        cost_to_cells[best_cost].append(action)

                        dr, dc = action // board_w, action % board_w
                        is_noop = (action == current_flat)
                        rsig = reconfig_signature_v2(dr, dc, sr, sc, committed_moves_list, is_noop)
                        rsig_to_cells[rsig].append(action)

                legal_counts.append(len(legal))
                oracle_counts.append(len(cost_to_cells))
                rsig_counts.append(len(rsig_to_cells))

            avg_l = np.mean(legal_counts)
            avg_o = np.mean(oracle_counts)
            avg_r = np.mean(rsig_counts)
            label = "(LAST - exact)" if atom_pos == n_atoms - 1 else "(sampled)"
            print(f"  Atom {atom_pos}/{n_atoms} {label}: "
                  f"legal={avg_l:.1f}, "
                  f"oracle_classes={avg_o:.1f} ({avg_l/avg_o:.1f}x), "
                  f"rsig_classes={avg_r:.1f} ({avg_l/avg_r:.1f}x)")


# ======================================================================
# TEST 3: Can we compute a gate-aware signature that actually works?
# Key idea: for the last atom, compute actual reconfig+gate cost directly
# and group by cost. For intermediate atoms, compute partial cost.
# ======================================================================
def test_direct_cost_grouping(map_num=2, num_trials=100):
    """For last atom: direct cost computation gives perfect grouping.
    Measure how many count_groups calls this requires per expansion."""
    config = get_config()
    config.map_num = map_num
    set_derived_config(config)
    map_data = get_map_data(config)
    positions = atom_map_to_positions(map_data['atom_map'], map_data['board_dim'][1])
    tasks = map_data['tasks']
    board_h, board_w = map_data['board_dim']
    env = NeutralAtomsEnv(tasks, positions, config.env)

    print(f"\n=== DIRECT COST GROUPING: Map {map_num} ===")

    import time
    total_time = 0
    total_calls = 0

    for layer_idx in range(len(tasks)):
        relevant = env.layer_relevant_atoms[layer_idx]
        gate_pairs = tasks[layer_idx]
        n_atoms = len(relevant)
        legal_sum = 0
        class_sum = 0

        for trial in range(num_trials):
            env.reset()
            env.tasks_done = layer_idx
            env.current_atom_idx = 0
            env.current_phase_moves = []
            committed = {}

            for i in range(n_atoms - 1):
                la = env.legal_actions()
                action = np.random.choice(la)
                q = env.current_qubit
                dr, dc = action // board_w, action % board_w
                committed[q] = (dr, dc)
                env.step(action, skip_obs=True)

            last_q = env.current_qubit
            legal = env.legal_actions()

            t0 = time.perf_counter()
            cost_to_cells = defaultdict(list)
            for action in legal:
                dr, dc = action // board_w, action % board_w
                full_assign = dict(committed)
                full_assign[last_q] = (dr, dc)
                rc, gc = compute_costs_separately(env.atom_positions, full_assign, gate_pairs, board_w)
                cost_to_cells[rc + gc].append(action)
            t1 = time.perf_counter()

            total_time += (t1 - t0)
            total_calls += len(legal)
            legal_sum += len(legal)
            class_sum += len(cost_to_cells)

        avg_l = legal_sum / num_trials
        avg_c = class_sum / num_trials
        print(f"  Layer {layer_idx}: legal={avg_l:.1f}, cost_classes={avg_c:.1f}, reduction={avg_l/avg_c:.1f}x")

    avg_time_per_expansion = total_time / (num_trials * len(tasks)) * 1000
    avg_time_per_call = total_time / total_calls * 1e6
    print(f"\n  Timing: {avg_time_per_expansion:.2f}ms per last-atom expansion "
          f"({avg_time_per_call:.1f}µs per count_groups call)")


# ======================================================================
# TEST 4: The real question — how much does action reduction help MCTS?
# Compare: full action space vs oracle-reduced for last atom only
# ======================================================================
def simulate_mcts_branching(map_num=2, num_trials=50):
    """Simulate MCTS tree size with and without action reduction."""
    config = get_config()
    config.map_num = map_num
    set_derived_config(config)
    map_data = get_map_data(config)
    positions = atom_map_to_positions(map_data['atom_map'], map_data['board_dim'][1])
    tasks = map_data['tasks']
    board_h, board_w = map_data['board_dim']
    env = NeutralAtomsEnv(tasks, positions, config.env)

    print(f"\n=== MCTS BRANCHING IMPACT: Map {map_num} ===")

    for layer_idx in range(len(tasks)):
        relevant = env.layer_relevant_atoms[layer_idx]
        n_atoms = len(relevant)

        # Estimate tree size: product of branching factors along a path
        raw_product = []
        reduced_product = []

        for trial in range(num_trials):
            env.reset()
            env.tasks_done = layer_idx
            env.current_atom_idx = 0
            env.current_phase_moves = []

            raw_bf = 1.0
            red_bf = 1.0

            for atom_pos in range(n_atoms):
                legal = env.legal_actions()
                raw_bf *= len(legal)

                # For now, reduction only at last atom (exact cost grouping)
                if atom_pos == n_atoms - 1:
                    # Count cost classes
                    committed = {}
                    for m in env.current_phase_moves:
                        s_r, s_c, d_r, d_c = m.tolist()
                        qid = env.board[d_r, d_c].item()
                        committed[qid] = (d_r, d_c)
                    q = env.current_qubit
                    gate_pairs = tasks[layer_idx]
                    cost_classes = defaultdict(list)
                    for action in legal:
                        dr, dc = action // board_w, action % board_w
                        full_assign = dict(committed)
                        full_assign[q] = (dr, dc)
                        rc, gc = compute_costs_separately(env.atom_positions, full_assign, gate_pairs, board_w)
                        cost_classes[rc + gc].append(action)
                    red_bf *= len(cost_classes)
                else:
                    red_bf *= len(legal)

                action = np.random.choice(legal)
                env.step(action, skip_obs=True)

            raw_product.append(raw_bf)
            reduced_product.append(red_bf)

        avg_raw = np.mean(raw_product)
        avg_red = np.mean(reduced_product)
        print(f"  Layer {layer_idx} ({n_atoms} atoms): "
              f"raw tree={avg_raw:.0f}, "
              f"last-atom-reduced={avg_red:.0f}, "
              f"saved={avg_raw/avg_red:.1f}x")


if __name__ == '__main__':
    print("=" * 70)
    print("TEST 1: Fixed reconfig sig (no-op separated) — last atom")
    print("=" * 70)
    for m in [0, 1, 2]:
        test_fixed_reconfig_sig(m, num_trials=100)

    print("\n" + "=" * 70)
    print("TEST 2: Direct cost grouping — last atom, exact")
    print("=" * 70)
    for m in [0, 1, 2]:
        test_direct_cost_grouping(m, num_trials=100)

    print("\n" + "=" * 70)
    print("TEST 3: Oracle at ALL atom positions (sampled for intermediate)")
    print("=" * 70)
    test_oracle_all_positions(map_num=2, num_trials=10, num_completions=200)

    print("\n" + "=" * 70)
    print("TEST 4: MCTS branching impact")
    print("=" * 70)
    for m in [0, 1, 2]:
        simulate_mcts_branching(m, num_trials=50)
