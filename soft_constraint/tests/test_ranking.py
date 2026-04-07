"""
tests/test_ranking.py — Surrogate ranking validation.

Does the partition surrogate correctly rank discrete solutions by true cost?

Tests:
  1. Calibration: sigmoid χ̂ vs true cost at one-hot (with direction enumeration)
  2. Ranking correlation: Spearman ρ across diverse instances
  3. Gate 2× weighting verification
  4. Noisy distribution ranking
"""

import torch
import torch.nn.functional as F
import sys, os, random, time
import numpy as np
from collections import defaultdict
from itertools import combinations
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import generate_random_map, atom_map_to_positions, MAPS, _resolve_map
from surrogate.primitives import (
    true_total_cost, true_gate_cost, get_relevant_atoms, greedy_chromatic
)
from surrogate.partition import chromatic_estimate, chromatic_cost
from surrogate.erdos import build_reconfig_conflict_matrix, build_gate_conflict_matrix

dtype = torch.float64


def header(name):
    print(f"\n{'═' * 70}")
    print(f"  {name}")
    print(f"{'═' * 70}")


def check(name, cond, detail=""):
    status = "✓ PASS" if cond else "✗ FAIL"
    print(f"  {status}: {name}" + (f"  ({detail})" if detail else ""))


# ─────────────────────────────────────────────────────────────────────────────
# Solution generators
# ─────────────────────────────────────────────────────────────────────────────

def do_nothing(init_cells, tasks):
    return [init_cells.clone() for _ in tasks]


def random_perturbation(init_cells, tasks, relevant, H, W, n_moves, rng):
    C = H * W
    layer_cells = [init_cells.clone() for _ in tasks]
    for t in range(len(tasks)):
        occupied = set(layer_cells[t].tolist())
        movers = list(relevant[t])
        rng.shuffle(movers)
        for q in movers[:n_moves]:
            empty = [c for c in range(C) if c not in occupied]
            if not empty: break
            new_cell = rng.choice(empty)
            occupied.discard(layer_cells[t][q].item())
            layer_cells[t][q] = new_cell
            occupied.add(new_cell)
    return layer_cells


def generate_solutions(init_cells, tasks, relevant, H, W, seed=42):
    rng = random.Random(seed)
    sols = [('do-nothing', do_nothing(init_cells, tasks))]
    for n_moves in [1, 2, 3, 4, 6]:
        for trial in range(4):
            lc = random_perturbation(init_cells, tasks, relevant, H, W, n_moves, rng)
            sols.append((f'rand_m{n_moves}_t{trial}', lc))
    return sols


# ─────────────────────────────────────────────────────────────────────────────
# Surrogate evaluation (fixed directions — fast)
# ─────────────────────────────────────────────────────────────────────────────

def eval_at_onehot(layer_cells, init_cells, tasks, relevant, H, W,
                   q_max=5, alpha=20.0):
    N, C = len(init_cells), H * W
    init_dists = torch.zeros(N, C, dtype=dtype)
    init_dists.scatter_(1, init_cells.unsqueeze(1), 1.0)

    sigmoid_total = 0.0
    cost_total = 0.0
    prev = init_dists

    for t in range(len(tasks)):
        dists_t = torch.zeros(N, C, dtype=dtype)
        dists_t.scatter_(1, layer_cells[t].unsqueeze(1), 1.0)
        M_G = len(tasks[t])
        dirs = torch.zeros(M_G, dtype=dtype)

        A_R, _, p_any = build_reconfig_conflict_matrix(prev, dists_t, H, W, relevant[t])
        A_G = build_gate_conflict_matrix(tasks[t], dists_t, H, W, dirs)

        chi_R = chromatic_estimate(A_R, q_max, alpha).item()
        chi_G = chromatic_estimate(A_G, q_max, alpha).item()
        cR = chromatic_cost(A_R, q_max).item()
        cG = chromatic_cost(A_G, q_max).item()
        p = p_any.item()

        sigmoid_total += p * chi_R + 2.0 * chi_G
        cost_total += p * cR + 2.0 * cG
        prev = dists_t

    return sigmoid_total, cost_total


def eval_noisy(layer_cells, init_cells, tasks, relevant, H, W,
               noise_scale=0.3, q_max=5, alpha=20.0, seed=0):
    torch.manual_seed(seed)
    N, C = len(init_cells), H * W
    init_dists = torch.zeros(N, C, dtype=dtype)
    init_dists.scatter_(1, init_cells.unsqueeze(1), 1.0)

    sigmoid_total = 0.0
    cost_total = 0.0
    prev = init_dists

    for t in range(len(tasks)):
        logits = torch.zeros(N, C, dtype=dtype)
        logits.scatter_(1, layer_cells[t].unsqueeze(1), 5.0)
        logits += torch.randn(N, C, dtype=dtype) * noise_scale
        dists_t = F.softmax(logits, dim=-1)

        M_G = len(tasks[t])
        dirs = torch.full((M_G,), 0.5, dtype=dtype)

        A_R, _, p_any = build_reconfig_conflict_matrix(prev, dists_t, H, W, relevant[t])
        A_G = build_gate_conflict_matrix(tasks[t], dists_t, H, W, dirs)

        chi_R = chromatic_estimate(A_R, q_max, alpha).item()
        chi_G = chromatic_estimate(A_G, q_max, alpha).item()
        cR = chromatic_cost(A_R, q_max).item()
        cG = chromatic_cost(A_G, q_max).item()
        p = p_any.item()

        sigmoid_total += p * chi_R + 2.0 * chi_G
        cost_total += p * cR + 2.0 * cG
        prev = dists_t

    return sigmoid_total, cost_total


# ═════════════════════════════════════════════════════════════════════════════
# TEST 1: Calibration — does sigmoid match true cost at one-hot?
# ═════════════════════════════════════════════════════════════════════════════

header("TEST 1: Calibration at one-hot (Map 2, with direction enumeration)")

md = _resolve_map(MAPS[2])
H, W = md['board_dim']
C, N = H * W, md['num_qubits']
tasks = md['tasks']
relevant = [get_relevant_atoms(t) for t in tasks]
pos_t = torch.tensor(atom_map_to_positions(md['atom_map'], W), dtype=torch.long)
init_cells = pos_t[:, 0] * W + pos_t[:, 1]

solutions = generate_solutions(init_cells, tasks, relevant, H, W)

# Full calibration check: enumerate directions per layer
print(f"  {'Solution':25s} {'True':>5s} {'χ̂(d=0)':>8s} {'χ̂(best)':>8s}")
print(f"  {'-'*25} {'-'*5} {'-'*8} {'-'*8}")

init_dists = torch.zeros(N, C, dtype=dtype)
init_dists.scatter_(1, init_cells.unsqueeze(1), 1.0)

exact_matches = 0
for name, lc in solutions[:10]:
    tc, bd = true_total_cost(init_cells, lc, tasks, H, W)
    sig_fixd, _ = eval_at_onehot(lc, init_cells, tasks, relevant, H, W)

    # Best-direction evaluation (only for 10 solutions to stay fast)
    sig_bestd = 0.0
    prev = init_dists
    for t in range(len(tasks)):
        dists_t = torch.zeros(N, C, dtype=dtype)
        dists_t.scatter_(1, lc[t].unsqueeze(1), 1.0)
        A_R, _, p_any = build_reconfig_conflict_matrix(prev, dists_t, H, W, relevant[t])
        chi_R = chromatic_estimate(A_R, 5, 20.0).item()

        M_G = len(tasks[t])
        best_cg = float('inf')
        for d in range(1 << M_G):
            dirs = torch.tensor([(d >> g) & 1 for g in range(M_G)], dtype=dtype)
            A_G = build_gate_conflict_matrix(tasks[t], dists_t, H, W, dirs)
            best_cg = min(best_cg, chromatic_estimate(A_G, 5, 20.0).item())
        sig_bestd += p_any.item() * chi_R + 2.0 * best_cg
        prev = dists_t

    match = abs(sig_bestd - tc) < 1.0
    if match: exact_matches += 1
    print(f"  {name:25s} {tc:5d} {sig_fixd:8.1f} {sig_bestd:8.1f} {'✓' if match else '✗'}")

check("χ̂(best dir) ≈ true cost at one-hot", exact_matches >= 8,
      f"{exact_matches}/10 match")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 2: Ranking correlation across instances
# ═════════════════════════════════════════════════════════════════════════════

header("TEST 2: Ranking correlation (Spearman ρ)")

CONFIGS = [
    ((5, 5), 12, [101, 102]),
    ((8, 8), 20, [201, 202]),
]
GATES = [3, 4]
LAYERS = [3, 4]

corrs = defaultdict(list)
all_rows = []
t_start = time.time()

for (board_dim, nq, seeds) in CONFIGS:
    for seed in seeds:
        for gpl in GATES:
            for nl in LAYERS:
                if gpl > nq // 2: continue
                H, W = board_dim
                C = H * W
                md = generate_random_map(board_dim, nq, nl,
                                         gates_per_layer=gpl, seed=seed)
                tasks = md['tasks']
                relevant = [get_relevant_atoms(t) for t in tasks]
                pos_t = torch.tensor(atom_map_to_positions(md['atom_map'], W),
                                     dtype=torch.long)
                init_cells = pos_t[:, 0] * W + pos_t[:, 1]
                label = f"{H}x{W}_s{seed}_g{gpl}_l{nl}"

                sols = generate_solutions(init_cells, tasks, relevant, H, W, seed=seed)

                tc_list, sig_list, cost_list = [], [], []
                sig_n_list, cost_n_list = [], []

                for sname, lc in sols:
                    tc, _ = true_total_cost(init_cells, lc, tasks, H, W)
                    sig, cst = eval_at_onehot(lc, init_cells, tasks, relevant, H, W)
                    sig_n, cst_n = eval_noisy(lc, init_cells, tasks, relevant,
                                              H, W, noise_scale=0.3, seed=seed)

                    tc_list.append(tc)
                    sig_list.append(sig)
                    cost_list.append(cst)
                    sig_n_list.append(sig_n)
                    cost_n_list.append(cst_n)

                    all_rows.append({
                        'instance': label, 'board': f"{H}x{W}",
                        'method': sname, 'true_cost': tc,
                        'sigmoid': sig, 'cost': cst,
                        'sigmoid_noisy': sig_n, 'cost_noisy': cst_n,
                    })

                if len(set(tc_list)) >= 3:
                    for key, vals in [('sigmoid', sig_list), ('cost', cost_list),
                                      ('sigmoid_noisy', sig_n_list),
                                      ('cost_noisy', cost_n_list)]:
                        rho, _ = spearmanr(tc_list, vals)
                        corrs[key].append(rho)

elapsed = time.time() - t_start
print(f"  Evaluated {len(all_rows)} solutions across "
      f"{len(set(r['instance'] for r in all_rows))} instances ({elapsed:.1f}s)\n")

print(f"  {'Variant':20s} {'Mean ρ':>7s} {'Min ρ':>7s} {'Max ρ':>7s} {'N':>4s}")
print(f"  {'-'*20} {'-'*7} {'-'*7} {'-'*7} {'-'*4}")

for key in ['sigmoid', 'cost', 'sigmoid_noisy', 'cost_noisy']:
    vals = [v for v in corrs[key] if not np.isnan(v)]
    if not vals: continue
    print(f"  {key:20s} {np.mean(vals):7.3f} {np.min(vals):7.3f} "
          f"{np.max(vals):7.3f} {len(vals):4d}")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 3: Per board-size breakdown
# ═════════════════════════════════════════════════════════════════════════════

header("TEST 3: Per board-size breakdown")

for board in ['5x5', '8x8']:
    br = [r for r in all_rows if r['board'] == board]
    if not br: continue
    tc = [r['true_cost'] for r in br]
    print(f"\n  {board} ({len(br)} pairs):")
    for key in ['sigmoid', 'cost', 'sigmoid_noisy', 'cost_noisy']:
        vals = [r[key] for r in br]
        rho, _ = spearmanr(tc, vals)
        print(f"    {key:20s}  ρ = {rho:.3f}")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 4: Pairwise violations
# ═════════════════════════════════════════════════════════════════════════════

header("TEST 4: Pairwise violations")

instances = sorted(set(r['instance'] for r in all_rows))
for surr_key in ['sigmoid', 'cost']:
    total_v, total_p = 0, 0
    for inst in instances:
        ir = [r for r in all_rows if r['instance'] == inst]
        for i, j in combinations(range(len(ir)), 2):
            ti, tj = ir[i]['true_cost'], ir[j]['true_cost']
            si, sj = ir[i][surr_key], ir[j][surr_key]
            if ti != tj:
                total_p += 1
                if (ti < tj) != (si < sj):
                    total_v += 1
    pct = total_v / max(total_p, 1) * 100
    print(f"  {surr_key:20s}: {total_v}/{total_p} violations ({pct:.1f}%)")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 5: Gate 2× weighting verification
# ═════════════════════════════════════════════════════════════════════════════

header("TEST 5: Gate 2× weighting")

# At one-hot with best dirs, verify: surrogate = true_reconfig + 2 * true_gate_groups
md = _resolve_map(MAPS[2])
H, W = md['board_dim']
C, N = H * W, md['num_qubits']
tasks = md['tasks']
relevant = [get_relevant_atoms(t) for t in tasks]
pos_t = torch.tensor(atom_map_to_positions(md['atom_map'], W), dtype=torch.long)
init_cells = pos_t[:, 0] * W + pos_t[:, 1]
init_dists = torch.zeros(N, C, dtype=dtype)
init_dists.scatter_(1, init_cells.unsqueeze(1), 1.0)

solutions = generate_solutions(init_cells, tasks, relevant, H, W)

print(f"  Decomposition: true = Σ(χ_R + 2·χ_G)")
print(f"  {'Solution':20s} {'True':>5s} {'ΣR':>3s} {'ΣG':>3s}  "
      f"{'χ̂ΣR':>5s} {'χ̂ΣG':>5s} {'Surr':>6s} {'Δ':>3s}")
print(f"  {'-'*20} {'-'*5} {'-'*3} {'-'*3}  {'-'*5} {'-'*5} {'-'*6} {'-'*3}")

for name, lc in solutions[:8]:
    tc, bd = true_total_cost(init_cells, lc, tasks, H, W)
    true_R_total = sum(r for r, g in bd)
    true_G_total = sum(g for r, g in bd)  # already 2×χ_G

    prev = init_dists
    surr_R, surr_G = 0.0, 0.0
    for t in range(len(tasks)):
        dists_t = torch.zeros(N, C, dtype=dtype)
        dists_t.scatter_(1, lc[t].unsqueeze(1), 1.0)
        A_R, _, p_any = build_reconfig_conflict_matrix(prev, dists_t, H, W, relevant[t])
        surr_R += p_any.item() * chromatic_estimate(A_R, 5, 20.0).item()

        M_G = len(tasks[t])
        best_cg = float('inf')
        for d in range(1 << M_G):
            dirs = torch.tensor([(d >> g) & 1 for g in range(M_G)], dtype=dtype)
            A_G = build_gate_conflict_matrix(tasks[t], dists_t, H, W, dirs)
            best_cg = min(best_cg, chromatic_estimate(A_G, 5, 20.0).item())
        surr_G += 2.0 * best_cg
        prev = dists_t

    surr_total = surr_R + surr_G
    delta = abs(surr_total - tc)
    print(f"  {name:20s} {tc:5d} {true_R_total:3d} {true_G_total:3d}  "
          f"{surr_R:5.1f} {surr_G:5.1f} {surr_total:6.1f} {delta:3.1f}")


# ═════════════════════════════════════════════════════════════════════════════

header("SUMMARY")
print("  See results above. Key metrics:")
for key in ['sigmoid', 'cost']:
    vals = [v for v in corrs[key] if not np.isnan(v)]
    if vals:
        print(f"  - {key}: mean Spearman ρ = {np.mean(vals):.3f} "
              f"(min={np.min(vals):.3f}, max={np.max(vals):.3f})")
