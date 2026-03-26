"""
Clean single-instance optimizer for neutral atom reconfiguration.

Key design decisions:
1. Only relevant atoms per layer get free logits; others stay at previous positions
2. Collision avoidance via iterative masking (greedy assignment by confidence)
3. Multiple random restarts to explore the loss landscape
4. Proper tracking of atom positions across layers
"""

import torch
import torch.nn.functional as F
import time
import math
from itertools import product as iterproduct
from collections import Counter


# ─────────────────────────────────────────────────────────────────────────────
# AOD + coloring primitives
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
# Gate cost (exact GPU-accelerated)
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


def get_topk_gate_states(placement_dists, gate_atoms, H, W, top_k=10):
    C = H*W; device = placement_dists.device; dtype = placement_dists.dtype
    cells = torch.arange(C, device=device)
    rows = (cells//W).to(dtype); cols = (cells%W).to(dtype)
    result = []
    for a,b in gate_atoms:
        pa,pb = placement_dists[a],placement_dists[b]
        ka,kb = min(top_k,C),min(top_k,C)
        ta,tb = pa.topk(ka),pb.topk(kb)
        p2d = ta.values[:,None]*tb.values[None,:]
        km = p2d.sum(); pr = p2d/km.clamp(min=1e-30)
        ca = ta.indices[:,None].expand(ka,kb)
        cb = tb.indices[None,:].expand(ka,kb)
        mf = torch.stack([rows[ca],cols[ca],rows[cb],cols[cb]], dim=-1)
        mr = torch.stack([rows[cb],cols[cb],rows[ca],cols[ca]], dim=-1)
        result.append({'probs':pr,'moves_fwd':mf,'moves_rev':mr,'ka':ka,'kb':kb,'kept_mass':km})
    return result


def build_compat_2d(gs_i,gs_j,di,dj):
    mi = gs_i['moves_rev' if di else 'moves_fwd']
    mj = gs_j['moves_rev' if dj else 'moves_fwd']
    Ki,Kj = mi.shape[0]*mi.shape[1], mj.shape[0]*mj.shape[1]
    return aod_compatible(mi.reshape(Ki,4)[:,None,:], mj.reshape(Kj,4)[None,:,:])


def exact_gate_cost(gate_atoms, dists, H, W, canon=True, top_k=8):
    M = len(gate_atoms)
    gs = get_topk_gate_states(dists, gate_atoms, H, W, top_k)
    ct = chi_table(M).to(device=dists.device, dtype=dists.dtype)
    p = [g['probs'].reshape(-1) for g in gs]
    joint = p[0][:,None,None,None]*p[1][None,:,None,None]*p[2][None,None,:,None]*p[3][None,None,None,:]
    if canon:
        mc = None
        for di in range(1<<M):
            ds = [bool((di>>g)&1) for g in range(M)]
            c01 = (~build_compat_2d(gs[0],gs[1],ds[0],ds[1])).to(torch.int8)
            c02 = (~build_compat_2d(gs[0],gs[2],ds[0],ds[2])).to(torch.int8)
            c12 = (~build_compat_2d(gs[1],gs[2],ds[1],ds[2])).to(torch.int8)
            c03 = (~build_compat_2d(gs[0],gs[3],ds[0],ds[3])).to(torch.int8)
            c13 = (~build_compat_2d(gs[1],gs[3],ds[1],ds[3])).to(torch.int8)
            c23 = (~build_compat_2d(gs[2],gs[3],ds[2],ds[3])).to(torch.int8)
            ec = (c01[:,:,None,None].to(torch.int16)+c02[:,None,:,None].to(torch.int16)*2+
                  c12[None,:,:,None].to(torch.int16)*4+c03[:,None,None,:].to(torch.int16)*8+
                  c13[None,:,None,:].to(torch.int16)*16+c23[None,None,:,:].to(torch.int16)*32)
            cv = ct[ec.long()]
            mc = cv if mc is None else torch.min(mc,cv)
        return 2.0*(mc*joint).sum()
    else:
        ds = [False]*M
        c01 = (~build_compat_2d(gs[0],gs[1],ds[0],ds[1])).to(torch.int8)
        c02 = (~build_compat_2d(gs[0],gs[2],ds[0],ds[2])).to(torch.int8)
        c12 = (~build_compat_2d(gs[1],gs[2],ds[1],ds[2])).to(torch.int8)
        c03 = (~build_compat_2d(gs[0],gs[3],ds[0],ds[3])).to(torch.int8)
        c13 = (~build_compat_2d(gs[1],gs[3],ds[1],ds[3])).to(torch.int8)
        c23 = (~build_compat_2d(gs[2],gs[3],ds[2],ds[3])).to(torch.int8)
        ec = (c01[:,:,None,None].to(torch.int16)+c02[:,None,:,None].to(torch.int16)*2+
              c12[None,:,:,None].to(torch.int16)*4+c03[:,None,None,:].to(torch.int16)*8+
              c13[None,:,None,:].to(torch.int16)*16+c23[None,None,:,:].to(torch.int16)*32)
        return 2.0*(ct[ec.long()]*joint).sum()


# ─────────────────────────────────────────────────────────────────────────────
# Reconfig cost surrogate
# ─────────────────────────────────────────────────────────────────────────────

def soft_pairwise_conflict_axis(si, di, sj, dj, H, W, axis='col'):
    C = H*W; device = si.device; dtype = si.dtype
    cells = torch.arange(C, device=device)
    V = W if axis=='col' else H
    coords = ((cells%W) if axis=='col' else (cells//W)).to(dtype)
    def marg(d):
        m = torch.zeros(V, device=device, dtype=dtype)
        for v in range(V): m[v] = (d * (coords==v).to(dtype)).sum()
        return m
    mi,mj,ni,nj = marg(si),marg(sj),marg(di),marg(dj)
    v = torch.arange(V, device=device, dtype=dtype)
    ds = v[:,None,None,None]-v[None,:,None,None]
    dd = v[None,None,:,None]-v[None,None,None,:]
    ok = ((ds==0)&(dd==0)) | (~(ds==0)&~(dd==0)&(torch.sign(ds)==torch.sign(dd)))
    prob = mi[:,None,None,None]*mj[None,:,None,None]*ni[None,None,:,None]*nj[None,None,None,:]
    return (prob*(~ok).to(dtype)).sum()


def reconfig_surrogate(src_dists, dst_dists, H, W, mover_indices,
                        lambda_inv=0.244, lambda_col=2.0):
    N,C = src_dists.shape; device = src_dists.device; dtype = src_dists.dtype
    M = len(mover_indices)
    if M < 2:
        if M == 0: return torch.tensor(0.0, device=device, dtype=dtype)
        q = mover_indices[0]
        return 1.0 - (src_dists[q]*dst_dists[q]).sum()

    p_stay = torch.stack([(src_dists[q]*dst_dists[q]).sum() for q in mover_indices])
    p_move = 1.0 - p_stay
    n_movers = p_move.sum()

    total_conflict = torch.tensor(0.0, device=device, dtype=dtype)
    total_collision = torch.tensor(0.0, device=device, dtype=dtype)
    for ii in range(M):
        for jj in range(ii+1, M):
            qi,qj = mover_indices[ii],mover_indices[jj]
            pbm = p_move[ii]*p_move[jj]
            cc = soft_pairwise_conflict_axis(src_dists[qi],dst_dists[qi],src_dists[qj],dst_dists[qj],H,W,'col')
            rc = soft_pairwise_conflict_axis(src_dists[qi],dst_dists[qi],src_dists[qj],dst_dists[qj],H,W,'row')
            total_conflict = total_conflict + pbm*torch.max(cc,rc)
            total_collision = total_collision + pbm*(dst_dists[qi]*dst_dists[qj]).sum()

    any_moves = torch.sigmoid(n_movers*10-0.5)
    return any_moves*(1.0 + lambda_inv*total_conflict + lambda_col*total_collision)


# ─────────────────────────────────────────────────────────────────────────────
# True cost evaluation
# ─────────────────────────────────────────────────────────────────────────────

def true_reconfig_cost(src_cells, dst_cells, H, W):
    moves = []
    for q in range(len(src_cells)):
        if src_cells[q] != dst_cells[q]:
            moves.append(torch.tensor([src_cells[q]//W,src_cells[q]%W,
                                       dst_cells[q]//W,dst_cells[q]%W], dtype=torch.long))
    if not moves: return 0
    return greedy_chromatic(torch.stack(moves))


def true_gate_cost(cells, gates, H, W):
    M = len(gates)
    if M == 0: return 0
    rows,cols = cells//W, cells%W
    best = M+1
    for di in range(1<<M):
        ds = [(di>>g)&1 for g in range(M)]
        mvs = []
        for g,(a,b) in enumerate(gates):
            if ds[g]: a,b=b,a
            mvs.append(torch.tensor([rows[a],cols[a],rows[b],cols[b]], dtype=torch.long))
        mvs = torch.stack(mvs)
        best = min(best, greedy_chromatic(mvs))
    return 2*best


# ─────────────────────────────────────────────────────────────────────────────
# Hard cell assignment (greedy by confidence, no collisions)
# ─────────────────────────────────────────────────────────────────────────────

def hard_assign_no_collisions(dists, occupied_cells=None):
    """Greedily assign atoms to cells by confidence, avoiding collisions.

    Args:
        dists: (N, C) placement distributions
        occupied_cells: set of cells already taken (by non-relevant atoms)

    Returns: (N,) cell assignments
    """
    N, C = dists.shape
    taken = set(occupied_cells) if occupied_cells else set()
    assignments = torch.full((N,), -1, dtype=torch.long)

    # Order atoms by confidence (highest max-prob first)
    confidences = dists.max(dim=1).values
    order = torch.argsort(confidences, descending=True)

    for idx in order:
        probs = dists[idx].clone()
        for c in taken:
            probs[c] = 0.0
        if probs.sum() < 1e-30:
            # Fallback: assign to any free cell
            for c in range(C):
                if c not in taken:
                    assignments[idx] = c
                    taken.add(c)
                    break
        else:
            cell = probs.argmax().item()
            assignments[idx] = cell
            taken.add(cell)

    return assignments


# ─────────────────────────────────────────────────────────────────────────────
# Full position tracking across layers
# ─────────────────────────────────────────────────────────────────────────────

def get_layer_positions(layer_logits, tasks, initial_pos, H, W, top_k_softmax=None):
    """Convert per-layer logits to full atom position distributions.

    For each layer, only relevant atoms get distributions from logits.
    Non-relevant atoms inherit their position from the previous layer.
    For soft computation, non-relevant atoms get one-hot at previous hard position.

    Returns: list of (N, C) distribution tensors per layer
    """
    N = initial_pos.shape[0]
    C = H * W

    # Initial one-hot
    init_dist = torch.zeros(N, C, dtype=layer_logits[0].dtype)
    for q in range(N):
        init_dist[q, initial_pos[q, 0] * W + initial_pos[q, 1]] = 1.0

    layer_dists = []
    prev_dist = init_dist

    for t, (logits_t, gates_t) in enumerate(zip(layer_logits, tasks)):
        relevant = sorted(set(a for pair in gates_t for a in pair))

        # Start from previous layer's distribution
        dist_t = prev_dist.clone().detach()

        # Override relevant atoms with learned logits
        for local_idx, global_idx in enumerate(relevant):
            dist_t[global_idx] = F.softmax(logits_t[local_idx], dim=-1)

        layer_dists.append(dist_t)
        prev_dist = dist_t

    return layer_dists


def extract_hard_positions(layer_dists, tasks, initial_pos, H, W):
    """Extract collision-free hard positions for each layer."""
    N = initial_pos.shape[0]
    C = H * W

    # Track positions
    current_cells = initial_pos[:, 0] * W + initial_pos[:, 1]
    layer_cells = []

    for t, (dist_t, gates_t) in enumerate(zip(layer_dists, tasks)):
        relevant = sorted(set(a for pair in gates_t for a in pair))
        non_relevant = [q for q in range(N) if q not in relevant]

        # Non-relevant atoms keep their current positions
        occupied = set(current_cells[q].item() for q in non_relevant)

        # Relevant atoms get assigned greedily
        rel_dists = dist_t[relevant]
        rel_assignments = hard_assign_no_collisions(rel_dists, occupied)

        # Build full assignment
        cells_t = current_cells.clone()
        for i, q in enumerate(relevant):
            cells_t[q] = rel_assignments[i]

        layer_cells.append(cells_t.clone())
        current_cells = cells_t

    return layer_cells


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
    [(4,3), (6,7), (9,8), (10,11)],
    [(2,1), (4,5), (7,6), (8,9)],
    [(0,1), (2,3), (6,5), (9,8)],
]

relevant_per_layer = [sorted(set(a for p in t for a in p)) for t in tasks]


# ═════════════════════════════════════════════════════════════════════════════
# Do-nothing baseline
# ═════════════════════════════════════════════════════════════════════════════

print("═" * 65)
print("Map 2: 5×5, 12 qubits, 3 layers")
print("═" * 65)

init_cells = pos[:, 0] * W + pos[:, 1]
donothing = sum(true_gate_cost(init_cells, t, H, W) for t in tasks)
print(f"Do-nothing baseline: {donothing}")
print(f"Lower bound: {len(tasks) * 2}")
print()


# ═════════════════════════════════════════════════════════════════════════════
# Multi-restart optimization
# ═════════════════════════════════════════════════════════════════════════════

print("═" * 65)
print("Multi-restart optimization")
print("═" * 65)

best_cost = donothing
best_solution = None
TOP_K = 8

results = []

for restart in range(10):
    torch.manual_seed(restart * 100 + 42)

    # Only allocate logits for RELEVANT atoms per layer
    layer_logits = []
    for t in range(len(tasks)):
        n_rel = len(relevant_per_layer[t])
        logits_t = torch.randn(n_rel, C) * 0.3
        # Initialize: bias toward current positions
        for local_idx, global_idx in enumerate(relevant_per_layer[t]):
            cell = pos[global_idx, 0] * W + pos[global_idx, 1]
            logits_t[local_idx, cell] += 4.0
        logits_t.requires_grad_(True)
        layer_logits.append(logits_t)

    optimizer = torch.optim.Adam(layer_logits, lr=0.03)

    for step in range(400):
        optimizer.zero_grad()

        # Build full distributions
        layer_dists = get_layer_positions(layer_logits, tasks, pos, H, W)

        total_loss = torch.tensor(0.0)
        for t in range(len(tasks)):
            # Gate cost
            gc = exact_gate_cost(tasks[t], layer_dists[t], H, W, canon=True, top_k=TOP_K)

            # Reconfig cost
            src_d = layer_dists[t-1] if t > 0 else layer_dists[0]  # for t=0, use initial
            if t == 0:
                src_d = torch.zeros(N, C)
                for q in range(N):
                    src_d[q, pos[q,0]*W + pos[q,1]] = 1.0

            rc = reconfig_surrogate(src_d, layer_dists[t], H, W,
                                     mover_indices=relevant_per_layer[t],
                                     lambda_inv=0.2, lambda_col=3.0)

            total_loss = total_loss + gc + rc

        total_loss.backward()
        optimizer.step()

    # Extract collision-free hard solution
    with torch.no_grad():
        final_dists = get_layer_positions(layer_logits, tasks, pos, H, W)
        hard_cells = extract_hard_positions(final_dists, tasks, pos, H, W)

        # Evaluate true cost
        true_total = 0
        prev = init_cells
        details = []
        for t in range(len(tasks)):
            rc = true_reconfig_cost(prev, hard_cells[t], H, W)
            gc = true_gate_cost(hard_cells[t], tasks[t], H, W)
            true_total += rc + gc
            details.append(f"r{rc}+g{gc}")
            prev = hard_cells[t]

        # Check collisions
        has_collision = False
        for t in range(len(tasks)):
            cells_list = hard_cells[t].tolist()
            if len(set(cells_list)) < len(cells_list):
                has_collision = True

    results.append({
        'seed': restart,
        'true_cost': true_total,
        'details': details,
        'collision': has_collision,
        'surrogate': total_loss.item(),
    })

    if true_total < best_cost and not has_collision:
        best_cost = true_total
        best_solution = (hard_cells, details)

    print(f"  Restart {restart:2d}: true={true_total}  [{', '.join(details)}]  "
          f"surr={total_loss.item():.2f}  {'⚠COL' if has_collision else '✓'}")

print(f"\nBest collision-free cost: {best_cost}  (baseline: {donothing})")

# Show best solution details
if best_solution:
    hard_cells, details = best_solution
    print(f"Best breakdown: [{', '.join(details)}]")
    print(f"\nBest solution movements:")
    prev = init_cells
    for t in range(len(tasks)):
        moved = []
        for q in relevant_per_layer[t]:
            if prev[q] != hard_cells[t][q]:
                sr,sc = prev[q]//W, prev[q]%W
                dr,dc = hard_cells[t][q]//W, hard_cells[t][q]%W
                moved.append(f"q{q}:({sr.item()},{sc.item()})→({dr.item()},{dc.item()})")
        if moved:
            print(f"  Layer {t}: {', '.join(moved)}")
        else:
            print(f"  Layer {t}: no moves")
        prev = hard_cells[t]

    # Show board state per layer
    print(f"\nBoard states:")
    for t in range(len(tasks)):
        board = [['.' for _ in range(W)] for _ in range(H)]
        for q in range(N):
            r, c = hard_cells[t][q]//W, hard_cells[t][q]%W
            r, c = r.item(), c.item()
            if board[r][c] == '.':
                board[r][c] = str(q) if q < 10 else chr(55+q)
            else:
                board[r][c] = '!'  # collision
        print(f"  Layer {t}: {[''.join(row) for row in board]}")


# ═════════════════════════════════════════════════════════════════════════════
# Summary statistics
# ═════════════════════════════════════════════════════════════════════════════

print(f"\n{'═' * 65}")
print("Summary across restarts")
print("═" * 65)

costs = [r['true_cost'] for r in results if not r['collision']]
if costs:
    print(f"  Valid solutions: {len(costs)}/{len(results)}")
    print(f"  Cost range: {min(costs)} - {max(costs)}")
    print(f"  Mean: {sum(costs)/len(costs):.1f}")
    print(f"  Best: {min(costs)}")
else:
    print(f"  No collision-free solutions found!")

costs_all = [r['true_cost'] for r in results]
print(f"  Including collisions: {min(costs_all)} - {max(costs_all)}")