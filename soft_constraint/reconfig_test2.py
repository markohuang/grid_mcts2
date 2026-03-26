"""
Combined multi-layer cost surrogate for neutral atom reconfiguration.

Gate cost: exact enumeration (GPU-accelerated, top-K pruning)
Reconfig cost: inversion-based surrogate with collision penalty

Full pipeline: model outputs placement distributions for ALL layers at once.
Total cost = Σ_t [reconfig_cost(p^(t-1), p^(t)) + 2 * E[χ_gate(p^(t))]]

Tested on Map 2: 5×5 board, 12 qubits, 3 layers.
"""

import torch
import torch.nn.functional as F
import time
import math
import random
from itertools import product as iterproduct
from collections import Counter


# ─────────────────────────────────────────────────────────────────────────────
# AOD primitives
# ─────────────────────────────────────────────────────────────────────────────

def aod_compatible(mi, mj):
    sr_i,sc_i,dr_i,dc_i = mi[...,0],mi[...,1],mi[...,2],mi[...,3]
    sr_j,sc_j,dr_j,dc_j = mj[...,0],mj[...,1],mj[...,2],mj[...,3]
    dcs,dcd = sc_i-sc_j, dc_i-dc_j
    h_ok = ((dcs==0)&(dcd==0)) | (~((dcs==0)|(dcd==0)) & (torch.sign(dcs)==torch.sign(dcd)))
    drs,drd = sr_i-sr_j, dr_i-dr_j
    v_ok = ((drs==0)&(drd==0)) | (~((drs==0)|(drd==0)) & (torch.sign(drs)==torch.sign(drd)))
    no_col = ~((drd==0)&(dcd==0))
    return h_ok & v_ok & no_col


def greedy_chromatic(moves):
    n = moves.shape[0]
    if n <= 1: return n
    conflict = torch.zeros(n,n,dtype=torch.bool)
    for i in range(n):
        for j in range(i+1,n):
            if not aod_compatible(moves[i],moves[j]):
                conflict[i,j]=conflict[j,i]=True
    colors=[-1]*n
    for idx in sorted(range(n), key=lambda x: -conflict[x].sum().item()):
        used={colors[k] for k in range(n) if conflict[idx,k] and colors[k]>=0}
        c=0
        while c in used: c+=1
        colors[idx]=c
    return max(colors)+1


# ─────────────────────────────────────────────────────────────────────────────
# χ table for gate cost
# ─────────────────────────────────────────────────────────────────────────────

_CHI = {}
def chi_table(m):
    if m not in _CHI:
        ne = m*(m-1)//2
        elist = [(i,j) for j in range(1,m) for i in range(j)]
        tbl = torch.zeros(1 << ne)
        for cfg in range(1 << ne):
            adj = torch.zeros(m,m,dtype=torch.bool)
            for ei,(ii,jj) in enumerate(elist):
                if cfg & (1 << ei): adj[ii,jj]=adj[jj,ii]=True
            edges = [(i,j) for i in range(m) for j in range(i+1,m) if adj[i,j]]
            if not edges: tbl[cfg]=1; continue
            for k in range(1,m+1):
                found=False
                for col in iterproduct(range(k), repeat=m):
                    if all(col[u]!=col[v] for u,v in edges):
                        tbl[cfg]=k; found=True; break
                if found: break
        _CHI[m] = tbl
    return _CHI[m]


# ─────────────────────────────────────────────────────────────────────────────
# Gate cost (exact, GPU-accelerated) — from previous work
# ─────────────────────────────────────────────────────────────────────────────

def get_topk_gate_states(placement_dists, gate_atoms, H, W, top_k=10):
    C = H * W
    device = placement_dists.device
    dtype = placement_dists.dtype
    cells = torch.arange(C, device=device)
    rows = (cells // W).to(dtype)
    cols = (cells % W).to(dtype)
    result = []
    for a, b in gate_atoms:
        pa, pb = placement_dists[a], placement_dists[b]
        ka, kb = min(top_k, C), min(top_k, C)
        top_a, top_b = pa.topk(ka), pb.topk(kb)
        active_a, pa_k = top_a.indices, top_a.values
        active_b, pb_k = top_b.indices, top_b.values
        probs_2d = pa_k[:,None] * pb_k[None,:]
        kept_mass = probs_2d.sum()
        probs_renorm = probs_2d / kept_mass.clamp(min=1e-30)
        ca = active_a[:,None].expand(ka,kb)
        cb = active_b[None,:].expand(ka,kb)
        moves_fwd = torch.stack([rows[ca],cols[ca],rows[cb],cols[cb]], dim=-1)
        moves_rev = torch.stack([rows[cb],cols[cb],rows[ca],cols[ca]], dim=-1)
        result.append({'probs': probs_renorm, 'moves_fwd': moves_fwd,
                       'moves_rev': moves_rev, 'ka': ka, 'kb': kb,
                       'kept_mass': kept_mass})
    return result


def build_compat_2d(gs_i, gs_j, dir_i, dir_j):
    mi = gs_i['moves_rev' if dir_i else 'moves_fwd']
    mj = gs_j['moves_rev' if dir_j else 'moves_fwd']
    Ki, Kj = mi.shape[0]*mi.shape[1], mj.shape[0]*mj.shape[1]
    return aod_compatible(mi.reshape(Ki,4)[:,None,:], mj.reshape(Kj,4)[None,:,:])


MAX_ELEMENTS = 500_000_000

def exact_gate_cost(gate_atoms, placement_dists, H, W, canonicalize=True, top_k=10):
    gs = get_topk_gate_states(placement_dists, gate_atoms, H, W, top_k)
    M = len(gate_atoms)
    chi_tbl = chi_table(M).to(device=placement_dists.device, dtype=placement_dists.dtype)
    p = [g['probs'].reshape(-1) for g in gs]
    K = [pp.shape[0] for pp in p]
    joint = p[0][:,None,None,None] * p[1][None,:,None,None] * p[2][None,None,:,None] * p[3][None,None,None,:]

    if canonicalize:
        min_chi = None
        for dir_idx in range(1 << M):
            dirs = [bool((dir_idx>>g)&1) for g in range(M)]
            c01 = (~build_compat_2d(gs[0],gs[1],dirs[0],dirs[1])).to(torch.int8)
            c02 = (~build_compat_2d(gs[0],gs[2],dirs[0],dirs[2])).to(torch.int8)
            c12 = (~build_compat_2d(gs[1],gs[2],dirs[1],dirs[2])).to(torch.int8)
            c03 = (~build_compat_2d(gs[0],gs[3],dirs[0],dirs[3])).to(torch.int8)
            c13 = (~build_compat_2d(gs[1],gs[3],dirs[1],dirs[3])).to(torch.int8)
            c23 = (~build_compat_2d(gs[2],gs[3],dirs[2],dirs[3])).to(torch.int8)
            edge_cfg = (c01[:,:,None,None].to(torch.int16) +
                        c02[:,None,:,None].to(torch.int16)*2 +
                        c12[None,:,:,None].to(torch.int16)*4 +
                        c03[:,None,None,:].to(torch.int16)*8 +
                        c13[None,:,None,:].to(torch.int16)*16 +
                        c23[None,None,:,:].to(torch.int16)*32)
            chi_vals = chi_tbl[edge_cfg.long()]
            min_chi = chi_vals if min_chi is None else torch.min(min_chi, chi_vals)
        return 2.0 * (min_chi * joint).sum()
    else:
        dirs = [False]*M
        c01 = (~build_compat_2d(gs[0],gs[1],dirs[0],dirs[1])).to(torch.int8)
        c02 = (~build_compat_2d(gs[0],gs[2],dirs[0],dirs[2])).to(torch.int8)
        c12 = (~build_compat_2d(gs[1],gs[2],dirs[1],dirs[2])).to(torch.int8)
        c03 = (~build_compat_2d(gs[0],gs[3],dirs[0],dirs[3])).to(torch.int8)
        c13 = (~build_compat_2d(gs[1],gs[3],dirs[1],dirs[3])).to(torch.int8)
        c23 = (~build_compat_2d(gs[2],gs[3],dirs[2],dirs[3])).to(torch.int8)
        edge_cfg = (c01[:,:,None,None].to(torch.int16) +
                    c02[:,None,:,None].to(torch.int16)*2 +
                    c12[None,:,:,None].to(torch.int16)*4 +
                    c03[:,None,None,:].to(torch.int16)*8 +
                    c13[None,:,None,:].to(torch.int16)*16 +
                    c23[None,None,:,:].to(torch.int16)*32)
        chi_vals = chi_tbl[edge_cfg.long()]
        return 2.0 * (chi_vals * joint).sum()


# ─────────────────────────────────────────────────────────────────────────────
# Reconfig cost surrogate (inversion-based + collision)
# ─────────────────────────────────────────────────────────────────────────────

def soft_pairwise_conflict_axis(src_dists_i, dst_dists_i, src_dists_j, dst_dists_j,
                                 H, W, axis='col'):
    """P(axis conflict between atoms i and j).

    Computes from coordinate marginals. O(V^4) where V = W or H.
    """
    C = H * W
    device = src_dists_i.device
    dtype = src_dists_i.dtype
    cells = torch.arange(C, device=device)

    if axis == 'col':
        coords = (cells % W).to(dtype)
        V = W
    else:
        coords = (cells // W).to(dtype)
        V = H

    # Marginals over coordinate values
    def marginal(dist):
        m = torch.zeros(V, device=device, dtype=dtype)
        for v in range(V):
            m[v] = (dist * (coords == v).to(dtype)).sum()
        return m

    si = marginal(src_dists_i)
    sj = marginal(src_dists_j)
    di = marginal(dst_dists_i)
    dj = marginal(dst_dists_j)

    # P(conflict) over all (sv_i, sv_j, dv_i, dv_j)
    v = torch.arange(V, device=device, dtype=dtype)
    ds = v[:,None,None,None] - v[None,:,None,None]  # si - sj
    dd = v[None,None,:,None] - v[None,None,None,:]  # di - dj

    both_same_s = (ds == 0)
    both_same_d = (dd == 0)
    same_dir = torch.sign(ds) == torch.sign(dd)
    ok = (both_same_s & both_same_d) | (~(ds==0) & ~(dd==0) & same_dir)
    conflict = ~ok  # (V, V, V, V)

    prob = si[:,None,None,None] * sj[None,:,None,None] * di[None,None,:,None] * dj[None,None,None,:]

    return (prob * conflict.to(dtype)).sum()


def soft_collision_prob(dst_dists_i, dst_dists_j):
    """P(atoms i and j end up at same destination cell)."""
    return (dst_dists_i * dst_dists_j).sum()


def reconfig_surrogate(src_dists, dst_dists, H, W, mover_indices=None,
                        lambda_inv=0.244, lambda_col=1.0):
    """Differentiable reconfig cost surrogate.

    Args:
        src_dists: (N, C) source distributions (previous layer or initial)
        dst_dists: (N, C) destination distributions (current layer)
        H, W: grid dims
        mover_indices: list of atom indices that might move (default: all)
        lambda_inv: weight per inversion (from calibration: χ ≈ 1.96 + 0.244*I)
        lambda_col: weight for collision penalty

    Returns: scalar surrogate cost, info dict
    """
    N, C = src_dists.shape
    device = src_dists.device
    dtype = src_dists.dtype

    if mover_indices is None:
        mover_indices = list(range(N))

    M = len(mover_indices)
    if M < 2:
        # 0 or 1 movers: cost is 0 or P(moves)
        if M == 0:
            return torch.tensor(0.0, device=device, dtype=dtype), {}
        q = mover_indices[0]
        p_move = 1.0 - (src_dists[q] * dst_dists[q]).sum()
        return p_move, {'n_movers': p_move}

    # Expected number of movers
    p_stay = torch.stack([(src_dists[q] * dst_dists[q]).sum() for q in mover_indices])
    p_move = 1.0 - p_stay  # (M,)
    n_movers = p_move.sum()

    # Pairwise conflict probabilities (axis inversions + collisions)
    total_axis_conflict = torch.tensor(0.0, device=device, dtype=dtype)
    total_collision = torch.tensor(0.0, device=device, dtype=dtype)

    for ii in range(M):
        for jj in range(ii+1, M):
            qi, qj = mover_indices[ii], mover_indices[jj]

            # Weight by P(both move): only count conflicts between actual movers
            p_both_move = p_move[ii] * p_move[jj]

            # Axis conflicts
            col_conf = soft_pairwise_conflict_axis(
                src_dists[qi], dst_dists[qi], src_dists[qj], dst_dists[qj], H, W, 'col')
            row_conf = soft_pairwise_conflict_axis(
                src_dists[qi], dst_dists[qi], src_dists[qj], dst_dists[qj], H, W, 'row')

            # Weight by probability both actually move
            total_axis_conflict = total_axis_conflict + p_both_move * torch.max(col_conf, row_conf)

            # Collision: P(same destination AND both move)
            col_prob = soft_collision_prob(dst_dists[qi], dst_dists[qj])
            total_collision = total_collision + p_both_move * col_prob

    # Soft indicator that any atom moves
    any_moves = torch.sigmoid(n_movers * 10 - 0.5)

    # Surrogate: base cost of 1 (if anything moves) + λ * conflicts
    cost = any_moves * (1.0 + lambda_inv * total_axis_conflict + lambda_col * total_collision)

    return cost, {
        'n_movers': n_movers,
        'axis_conflict': total_axis_conflict,
        'collision': total_collision,
        'any_moves': any_moves,
    }


# ─────────────────────────────────────────────────────────────────────────────
# True costs for validation
# ─────────────────────────────────────────────────────────────────────────────

def true_reconfig_cost(src_cells, dst_cells, H, W):
    moves = []
    for q in range(len(src_cells)):
        if src_cells[q] != dst_cells[q]:
            sr,sc = src_cells[q]//W, src_cells[q]%W
            dr,dc = dst_cells[q]//W, dst_cells[q]%W
            moves.append(torch.tensor([sr,sc,dr,dc], dtype=torch.long))
    if not moves: return 0
    return greedy_chromatic(torch.stack(moves))


def true_gate_cost(cells, gates, H, W, canon=True):
    M = len(gates)
    if M == 0: return 0
    rows, cols = cells // W, cells % W
    def _groups(dirs):
        mvs = [torch.tensor([rows[a if not dirs[g] else b], cols[a if not dirs[g] else b],
                              rows[b if not dirs[g] else a], cols[b if not dirs[g] else a]],
                             dtype=torch.long) for g,(a,b) in enumerate(gates)]
        mvs = torch.stack(mvs); n=mvs.shape[0]
        if n==1: return 1
        conflict = torch.zeros(n,n,dtype=torch.bool)
        for i in range(n):
            for j in range(i+1,n):
                if not aod_compatible(mvs[i],mvs[j]): conflict[i,j]=conflict[j,i]=True
        colors=[-1]*n
        for idx in sorted(range(n), key=lambda x: -conflict[x].sum().item()):
            used={colors[k] for k in range(n) if conflict[idx,k] and colors[k]>=0}
            c=0
            while c in used: c+=1
            colors[idx]=c
        return max(colors)+1
    if not canon: return 2*_groups([0]*M)
    return 2*min(_groups([(d>>g)&1 for g in range(M)]) for d in range(1<<M))


def true_total_cost(initial_pos, layer_positions, tasks, H, W):
    """True total cost for concrete positions across all layers."""
    total = 0
    prev_cells = initial_pos[:, 0] * W + initial_pos[:, 1]
    for t, (gates, pos_t) in enumerate(zip(tasks, layer_positions)):
        curr_cells = pos_t[:, 0] * W + pos_t[:, 1]
        # Reconfig cost
        rc = true_reconfig_cost(prev_cells, curr_cells, H, W)
        # Gate cost
        gc = true_gate_cost(curr_cells, gates, H, W, canon=True)
        total += rc + gc
        prev_cells = curr_cells
    return total


# ═════════════════════════════════════════════════════════════════════════════
# Map 2 setup
# ═════════════════════════════════════════════════════════════════════════════

H, W, C = 5, 5, 25
pos = torch.tensor([
    [0,4],[2,2],[1,2],[2,1],[3,0],[1,0],
    [0,2],[4,1],[4,2],[4,3],[4,0],[1,3],
])
N = pos.shape[0]

tasks = [
    [(4,3), (6,7), (9,8), (10,11)],   # layer 0
    [(2,1), (4,5), (7,6), (8,9)],     # layer 1
    [(0,1), (2,3), (6,5), (9,8)],     # layer 2
]

# Relevant atoms per layer
def get_relevant_atoms(gates):
    atoms = set()
    for a, b in gates:
        atoms.add(a)
        atoms.add(b)
    return sorted(atoms)

relevant = [get_relevant_atoms(t) for t in tasks]


# ═════════════════════════════════════════════════════════════════════════════
# TEST 1: Do-nothing baseline
# ═════════════════════════════════════════════════════════════════════════════

print("═" * 65)
print("TEST 1: Do-nothing baseline (no reconfiguration)")
print("═" * 65)

init_cells = pos[:, 0] * W + pos[:, 1]
total_donothing = 0
for t, gates in enumerate(tasks):
    gc = true_gate_cost(init_cells, gates, H, W, canon=True)
    total_donothing += gc
    print(f"  Layer {t}: gate_cost={gc}  (relevant atoms: {relevant[t]})")
print(f"  Total do-nothing cost: {total_donothing} (reconfig=0, gates={total_donothing})")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 2: Multi-layer optimization
# ═════════════════════════════════════════════════════════════════════════════

print(f"\n{'═' * 65}")
print("TEST 2: Multi-layer optimization (gate + reconfig)")
print("═" * 65)

torch.manual_seed(42)

# Model: logits for each atom in each layer
# Initialize near initial positions
layer_logits = []
for t in range(len(tasks)):
    logits_t = torch.randn(N, C) * 0.1
    for q in range(N):
        logits_t[q, pos[q,0]*W + pos[q,1]] += 5.0  # bias toward initial
    logits_t.requires_grad_(True)
    layer_logits.append(logits_t)

optimizer = torch.optim.Adam(layer_logits, lr=0.03)

print(f"  Layers: {len(tasks)}, atoms: {N}, board: {H}×{W}")
print(f"  Do-nothing baseline: {total_donothing}")
print(f"  Lower bound: {len(tasks) * 2} (all χ_gate=1, zero reconfig)")
print()

TOP_K = 8

for step in range(400):
    optimizer.zero_grad()

    total_loss = torch.tensor(0.0)
    layer_dists = [F.softmax(ll, dim=-1) for ll in layer_logits]

    for t in range(len(tasks)):
        dists_t = layer_dists[t]

        # Gate cost
        gc = exact_gate_cost(tasks[t], dists_t, H, W, canonicalize=True, top_k=TOP_K)

        # Reconfig cost
        if t == 0:
            # Source is initial positions (one-hot)
            src_d = torch.zeros(N, C)
            for q in range(N):
                src_d[q, pos[q,0]*W + pos[q,1]] = 1.0
        else:
            src_d = layer_dists[t-1]

        rc, rc_info = reconfig_surrogate(
            src_d, dists_t, H, W,
            mover_indices=relevant[t],
            lambda_inv=0.244, lambda_col=1.0)

        total_loss = total_loss + gc + rc

    total_loss.backward()
    optimizer.step()

    if step % 50 == 0 or step == 399:
        with torch.no_grad():
            # Evaluate true cost
            hard_positions = []
            for t in range(len(tasks)):
                d = F.softmax(layer_logits[t], dim=-1)
                hard = d.argmax(dim=-1)
                hard_pos = torch.stack([hard // W, hard % W], dim=-1)
                hard_positions.append(hard_pos)

            true_total = true_total_cost(pos, hard_positions, tasks, H, W)

            # Per-layer breakdown
            details = []
            prev = init_cells
            for t in range(len(tasks)):
                curr = hard_positions[t][:,0]*W + hard_positions[t][:,1]
                rc = true_reconfig_cost(prev, curr, H, W)
                gc = true_gate_cost(curr, tasks[t], H, W, canon=True)
                details.append(f"L{t}:r{rc}+g{gc}")
                prev = curr

            ent = sum(-(F.softmax(ll, dim=-1) * torch.log(F.softmax(ll, dim=-1).clamp(min=1e-30))).sum(-1).mean().item()
                      for ll in layer_logits) / len(tasks)

            print(f"  Step {step:3d}: surr={total_loss.item():.3f}  "
                  f"true={true_total}  [{', '.join(details)}]  "
                  f"avg_ent={ent:.2f}")

# Final result
print(f"\n  Final true cost: {true_total}  (baseline: {total_donothing})")

# Check collisions per layer
for t in range(len(tasks)):
    d = F.softmax(layer_logits[t], dim=-1)
    hard = d.argmax(dim=-1).tolist()
    dupes = {c:n for c,n in Counter(hard).items() if n>1}
    if dupes:
        print(f"  Layer {t} ⚠ COLLISIONS: {dupes}")

# Show movements
print(f"\n  Atom movements:")
for t in range(len(tasks)):
    d = F.softmax(layer_logits[t], dim=-1)
    hard = d.argmax(dim=-1)
    moved = []
    for q in relevant[t]:
        r, c = hard[q]//W, hard[q]%W
        o_r, o_c = pos[q].tolist()
        if r != o_r or c != o_c:
            moved.append(f"q{q}:({o_r},{o_c})→({r.item()},{c.item()})")
    if moved:
        print(f"  Layer {t}: {', '.join(moved)}")
    else:
        print(f"  Layer {t}: no moves")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 3: Sweep lambda_inv to find best setting
# ═════════════════════════════════════════════════════════════════════════════

print(f"\n{'═' * 65}")
print("TEST 3: Sensitivity to lambda_inv")
print("═" * 65)

for lam in [0.0, 0.1, 0.244, 0.5, 1.0, 2.0]:
    torch.manual_seed(42)

    ll = []
    for t in range(len(tasks)):
        logits_t = torch.randn(N, C) * 0.1
        for q in range(N):
            logits_t[q, pos[q,0]*W + pos[q,1]] += 5.0
        logits_t.requires_grad_(True)
        ll.append(logits_t)

    opt = torch.optim.Adam(ll, lr=0.03)

    for step in range(300):
        opt.zero_grad()
        loss = torch.tensor(0.0)
        dists = [F.softmax(l, dim=-1) for l in ll]
        for t in range(len(tasks)):
            gc = exact_gate_cost(tasks[t], dists[t], H, W, canonicalize=True, top_k=TOP_K)
            src_d = torch.zeros(N,C) if t==0 else dists[t-1]
            if t == 0:
                for q in range(N):
                    src_d[q, pos[q,0]*W + pos[q,1]] = 1.0
            rc, _ = reconfig_surrogate(src_d, dists[t], H, W,
                                        mover_indices=relevant[t],
                                        lambda_inv=lam, lambda_col=1.0)
            loss = loss + gc + rc
        loss.backward()
        opt.step()

    with torch.no_grad():
        hard_pos = []
        for t in range(len(tasks)):
            d = F.softmax(ll[t], dim=-1)
            hard = d.argmax(dim=-1)
            hard_pos.append(torch.stack([hard//W, hard%W], dim=-1))
        tc = true_total_cost(pos, hard_pos, tasks, H, W)

        # Count total moves
        n_moves = 0
        prev = init_cells
        for t in range(len(tasks)):
            curr = hard_pos[t][:,0]*W + hard_pos[t][:,1]
            n_moves += (prev != curr).sum().item()
            prev = curr

    print(f"  λ_inv={lam:.3f}: true_cost={tc}  n_moves={n_moves}")