"""
Fast exact gate cost surrogate v2.

Key fix: prune per-ATOM marginals (top-K cells per atom) instead of
per-gate-state thresholding. This is much more effective because
|active gate states| = K_a * K_b, and controlling K_a, K_b directly
controls the complexity.

For top-K=8 per atom: 64 gate states per gate, 64^2 = 4096 inner loop
size, 64^2 outer iterations = 4096 total → 4096 * 4096 = 16.7M ops.
Should run in <1s.

We also renormalize after pruning to maintain a valid distribution.
"""

import torch
import torch.nn.functional as F
import time
import math
from itertools import product as iterproduct

# ─────────────────────────────────────────────────────────────────────────────
# Primitives
# ─────────────────────────────────────────────────────────────────────────────

def aod_compatible_batch(mi, mj):
    sr_i, sc_i, dr_i, dc_i = mi[...,0], mi[...,1], mi[...,2], mi[...,3]
    sr_j, sc_j, dr_j, dc_j = mj[...,0], mj[...,1], mj[...,2], mj[...,3]
    dcs, dcd = sc_i - sc_j, dc_i - dc_j
    h_ok = ((dcs==0)&(dcd==0)) | (~((dcs==0)|(dcd==0)) & (torch.sign(dcs)==torch.sign(dcd)))
    drs, drd = sr_i - sr_j, dr_i - dr_j
    v_ok = ((drs==0)&(drd==0)) | (~((drs==0)|(drd==0)) & (torch.sign(drs)==torch.sign(drd)))
    no_col = ~((drd==0)&(dcd==0))
    return h_ok & v_ok & no_col


_CHI = {}
def chi_table(m):
    if m not in _CHI:
        ne = m*(m-1)//2
        elist = [(i,j) for j in range(1,m) for i in range(j)]
        tbl = torch.zeros(1 << ne)
        for cfg in range(1 << ne):
            adj = torch.zeros(m, m, dtype=torch.bool)
            for ei, (ii,jj) in enumerate(elist):
                if cfg & (1 << ei):
                    adj[ii,jj] = adj[jj,ii] = True
            edges = [(i,j) for i in range(m) for j in range(i+1,m) if adj[i,j]]
            if not edges:
                tbl[cfg] = 1; continue
            for k in range(1, m+1):
                found = False
                for col in iterproduct(range(k), repeat=m):
                    if all(col[u] != col[v] for u,v in edges):
                        tbl[cfg] = k; found = True; break
                if found: break
        _CHI[m] = tbl
    return _CHI[m]


def true_gate_cost(cells, gates, H, W, canon=True):
    M = len(gates)
    if M == 0: return 0
    rows, cols = cells // W, cells % W
    def _groups(dirs):
        mvs = []
        for g, (a, b) in enumerate(gates):
            if dirs[g]: a, b = b, a
            mvs.append(torch.tensor([rows[a], cols[a], rows[b], cols[b]], dtype=torch.long))
        mvs = torch.stack(mvs)
        n = mvs.shape[0]
        if n == 1: return 1
        conflict = torch.zeros(n, n, dtype=torch.bool)
        for i in range(n):
            for j in range(i+1,n):
                if not aod_compatible_batch(mvs[i], mvs[j]):
                    conflict[i,j] = conflict[j,i] = True
        colors = [-1]*n
        for idx in sorted(range(n), key=lambda x: -conflict[x].sum().item()):
            used = {colors[k] for k in range(n) if conflict[idx,k] and colors[k]>=0}
            c = 0
            while c in used: c += 1
            colors[idx] = c
        return max(colors)+1
    if not canon:
        return 2 * _groups([0]*M)
    return 2 * min(_groups([(d>>g)&1 for g in range(M)]) for d in range(1<<M))


def mc_estimate(gates, dists, H, W, K, canon=True):
    N, C = dists.shape
    samps = torch.stack([torch.multinomial(dists[q], K, replacement=True) for q in range(N)])
    return torch.tensor([true_gate_cost(samps[:,s], gates, H, W, canon) for s in range(K)],
                        dtype=dists.dtype)


# ─────────────────────────────────────────────────────────────────────────────
# Top-K pruned gate states
# ─────────────────────────────────────────────────────────────────────────────

def get_pruned_gate_states(placement_dists, gate_atoms, H, W, top_k=10):
    """Get active gate states using top-K pruning per atom.

    For each atom, keep only the top_k highest-probability cells.
    Gate states are all pairs of active cells for the two atoms.
    Renormalize joint probabilities to sum to 1 (introducing small bias
    from discarded tail probability).

    Returns list of dicts per gate with:
        probs: (Kg,) joint probabilities (renormalized)
        moves_fwd: (Kg, 4) forward moves
        moves_rev: (Kg, 4) reverse moves
        n: number of active states
        tail_mass: total probability mass discarded
    """
    C = H * W
    cells = torch.arange(C, device=placement_dists.device)
    rows = (cells // W).to(placement_dists.dtype)
    cols = (cells % W).to(placement_dists.dtype)

    result = []
    for a, b in gate_atoms:
        pa, pb = placement_dists[a], placement_dists[b]

        # Top-K cells per atom
        ka = min(top_k, (pa > 0).sum().item())
        kb = min(top_k, (pb > 0).sum().item())
        top_a = pa.topk(ka)
        top_b = pb.topk(kb)

        active_a = top_a.indices  # (ka,)
        active_b = top_b.indices  # (kb,)
        pa_active = top_a.values  # (ka,)
        pb_active = top_b.values  # (kb,)

        # Track tail mass for bias estimation
        tail_a = 1.0 - pa_active.sum()
        tail_b = 1.0 - pb_active.sum()
        tail_mass = 1.0 - pa_active.sum() * pb_active.sum()

        # All pairs
        ca = active_a[:, None].expand(ka, kb).reshape(-1)
        cb = active_b[None, :].expand(ka, kb).reshape(-1)
        probs = pa_active[:, None] * pb_active[None, :]  # (ka, kb)
        probs_flat = probs.reshape(-1)

        # Renormalize
        total = probs_flat.sum()
        probs_renorm = probs_flat / total.clamp(min=1e-30)

        # Moves
        r_a, c_a = rows[ca], cols[ca]
        r_b, c_b = rows[cb], cols[cb]
        moves_fwd = torch.stack([r_a, c_a, r_b, c_b], dim=-1)
        moves_rev = torch.stack([r_b, c_b, r_a, c_a], dim=-1)

        result.append({
            'probs': probs_renorm,
            'probs_unnorm': probs_flat,  # keep for exact (non-renormalized) mode
            'total_mass': total,
            'moves_fwd': moves_fwd,
            'moves_rev': moves_rev,
            'n': len(probs_flat),
            'tail_mass': tail_mass.item() if isinstance(tail_mass, torch.Tensor) else tail_mass,
        })

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Sparse compat tables
# ─────────────────────────────────────────────────────────────────────────────

def build_sparse_compat(gs_i, gs_j, dir_i, dir_j):
    mi = gs_i['moves_rev' if dir_i else 'moves_fwd']
    mj = gs_j['moves_rev' if dir_j else 'moves_fwd']
    return aod_compatible_batch(mi[:, None, :], mj[None, :, :])


# ─────────────────────────────────────────────────────────────────────────────
# Exact E[χ] — sparse, M=4
# ─────────────────────────────────────────────────────────────────────────────

def exact_echi_sparse(gate_states, directions, chi_tbl=None, use_renorm=True):
    """Exact E[χ] for M=4 via sparse enumeration."""
    if chi_tbl is None:
        chi_tbl = chi_table(4)

    gs = gate_states
    K = [g['n'] for g in gs]
    device = gs[0]['probs'].device
    dtype = gs[0]['probs'].dtype
    chi_tbl = chi_tbl.to(device=device, dtype=dtype)

    prob_key = 'probs' if use_renorm else 'probs_unnorm'
    p0, p1, p2, p3 = [g[prob_key] for g in gs]

    # Build compat tables for this direction
    dirs = directions
    c01 = (~build_sparse_compat(gs[0], gs[1], dirs[0], dirs[1])).to(dtype)
    c02 = (~build_sparse_compat(gs[0], gs[2], dirs[0], dirs[2])).to(dtype)
    c12 = (~build_sparse_compat(gs[1], gs[2], dirs[1], dirs[2])).to(dtype)
    c03 = (~build_sparse_compat(gs[0], gs[3], dirs[0], dirs[3])).to(dtype)
    c13 = (~build_sparse_compat(gs[1], gs[3], dirs[1], dirs[3])).to(dtype)
    c23 = (~build_sparse_compat(gs[2], gs[3], dirs[2], dirs[3])).to(dtype)

    result = torch.tensor(0.0, device=device, dtype=dtype)

    for i0 in range(K[0]):
        if p0[i0] < 1e-30:
            continue
        for i1 in range(K[1]):
            if p1[i1] < 1e-30:
                continue

            bit0 = c01[i0, i1]
            bit1 = c02[i0]          # (K2,)
            bit2 = c12[i1]          # (K2,)
            bit3 = c03[i0]          # (K3,)
            bit4 = c13[i1]          # (K3,)
            bit5 = c23              # (K2, K3)

            edge_cfg = (bit0 +
                        bit1[:, None] * 2 +
                        bit2[:, None] * 4 +
                        bit3[None, :] * 8 +
                        bit4[None, :] * 16 +
                        bit5 * 32)

            chi_vals = chi_tbl[edge_cfg.long()]
            joint_p23 = p2[:, None] * p3[None, :]
            inner = (chi_vals * joint_p23).sum()
            result = result + p0[i0] * p1[i1] * inner

    return result


def exact_echi_sparse_canon(gate_states, chi_tbl=None, use_renorm=True):
    """Exact E[min_d χ] with canonicalization — takes min per configuration."""
    if chi_tbl is None:
        chi_tbl = chi_table(4)

    gs = gate_states
    K = [g['n'] for g in gs]
    M = 4
    device = gs[0]['probs'].device
    dtype = gs[0]['probs'].dtype
    chi_tbl_dev = chi_tbl.to(device=device, dtype=dtype)

    prob_key = 'probs' if use_renorm else 'probs_unnorm'
    p0, p1, p2, p3 = [g[prob_key] for g in gs]

    # Precompute all compat tables
    compat = {}
    for i in range(M):
        for j in range(i+1, M):
            for di in [False, True]:
                for dj in [False, True]:
                    compat[(i,j,di,dj)] = (~build_sparse_compat(
                        gs[i], gs[j], di, dj)).to(dtype)

    result = torch.tensor(0.0, device=device, dtype=dtype)

    for i0 in range(K[0]):
        if p0[i0] < 1e-30:
            continue
        for i1 in range(K[1]):
            if p1[i1] < 1e-30:
                continue

            min_chi = None
            for dir_idx in range(1 << M):
                dirs = [bool((dir_idx >> g) & 1) for g in range(M)]
                d0, d1, d2, d3 = dirs

                bit0 = compat[(0,1,d0,d1)][i0, i1]
                bit1 = compat[(0,2,d0,d2)][i0]
                bit2 = compat[(1,2,d1,d2)][i1]
                bit3 = compat[(0,3,d0,d3)][i0]
                bit4 = compat[(1,3,d1,d3)][i1]
                bit5 = compat[(2,3,d2,d3)]

                edge_cfg = (bit0 + bit1[:,None]*2 + bit2[:,None]*4 +
                           bit3[None,:]*8 + bit4[None,:]*16 + bit5*32)
                chi_vals = chi_tbl_dev[edge_cfg.long()]

                if min_chi is None:
                    min_chi = chi_vals.clone()
                else:
                    min_chi = torch.min(min_chi, chi_vals)

            joint_p23 = p2[:, None] * p3[None, :]
            inner = (min_chi * joint_p23).sum()
            result = result + p0[i0] * p1[i1] * inner

    return result


def exact_gate_cost(gate_atoms, placement_dists, H, W,
                    canonicalize=True, top_k=10):
    """Full gate cost: 2 * E[χ*], differentiable."""
    gs = get_pruned_gate_states(placement_dists, gate_atoms, H, W, top_k)
    if canonicalize:
        chi = exact_echi_sparse_canon(gs)
    else:
        chi = exact_echi_sparse(gs, [False]*4)
    return 2.0 * chi


# ═════════════════════════════════════════════════════════════════════════════
# Setup
# ═════════════════════════════════════════════════════════════════════════════

H, W, C = 5, 5, 25
gates_L0 = [(4, 3), (6, 7), (9, 8), (10, 11)]
pos = torch.tensor([
    [0, 4], [2, 2], [1, 2], [2, 1], [3, 0], [1, 0],
    [0, 2], [4, 1], [4, 2], [4, 3], [4, 0], [1, 3],
])
N = pos.shape[0]


# ═════════════════════════════════════════════════════════════════════════════
# TEST 1: One-hot sanity
# ═════════════════════════════════════════════════════════════════════════════

print("═" * 65)
print("TEST 1: One-hot sanity")
print("═" * 65)

onehot = torch.zeros(N, C, dtype=torch.float64)
for q in range(N):
    onehot[q, pos[q, 0] * W + pos[q, 1]] = 1.0

for canon in [False, True]:
    t0 = time.time()
    v = exact_gate_cost(gates_L0, onehot, H, W, canonicalize=canon, top_k=5)
    t = time.time() - t0
    true = true_gate_cost(pos[:,0]*W+pos[:,1], gates_L0, H, W, canon=canon)
    print(f"  canon={canon}: exact={v.item():.4f}  true={true}  time={t:.4f}s  "
          f"{'✓' if abs(v.item()-true)<0.01 else '✗'}")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 2: Speed + bias + tail mass at varying sharpness and top_k
# ═════════════════════════════════════════════════════════════════════════════

print(f"\n{'═' * 65}")
print("TEST 2: Speed, bias, and pruning quality")
print("═" * 65)

torch.manual_seed(42)

for sigma in [0.3, 1.5, 4.0, 10.0]:
    logits = torch.randn(N, C, dtype=torch.float64) * 0.1
    for q in range(N):
        logits[q, pos[q, 0] * W + pos[q, 1]] += sigma
    d = F.softmax(logits, dim=-1)
    ent = -(d * torch.log(d.clamp(min=1e-30))).sum(-1).mean().item()

    mc = mc_estimate(gates_L0, d, H, W, 5000, canon=False)
    mc_mean, mc_se = mc.mean().item(), (mc.std() / math.sqrt(5000)).item()

    print(f"\n  σ={sigma}  (entropy={ent:.2f}, MC={mc_mean:.4f}±{mc_se:.4f})")

    for top_k in [5, 8, 12, 25]:
        gs = get_pruned_gate_states(d, gates_L0, H, W, top_k)
        sizes = [g['n'] for g in gs]
        tails = [g['tail_mass'] for g in gs]
        max_tail = max(tails)

        t0 = time.time()
        v = exact_gate_cost(gates_L0, d, H, W, canonicalize=False, top_k=top_k)
        t = time.time() - t0

        bias = v.item() - mc_mean
        print(f"    top_k={top_k:2d}: states={sizes}  time={t:.3f}s  "
              f"exact={v.item():.4f}  bias={bias:+.4f}  "
              f"max_tail={max_tail:.4f}  "
              f"{'✓' if abs(bias) < 2*mc_se else '✗'}")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 3: Gradient check (analytical vs numerical, both exact, no MC)
# ═════════════════════════════════════════════════════════════════════════════

print(f"\n{'═' * 65}")
print("TEST 3: Gradient check — analytical vs numerical (exact, no MC)")
print("═" * 65)

torch.manual_seed(99)
base_logits = torch.randn(N, C, dtype=torch.float64) * 0.15
for q in range(N):
    base_logits[q, pos[q, 0] * W + pos[q, 1]] += 4.0

TOP_K_GRAD = 10  # use this for gradient tests

# Analytical
gl = base_logits.clone().requires_grad_(True)
d = F.softmax(gl, dim=-1)
cost = exact_gate_cost(gates_L0, d, H, W, canonicalize=False, top_k=TOP_K_GRAD)
cost.backward()
ag = gl.grad.clone()

print(f"  Cost: {cost.item():.6f}")
print(f"  Grad norm: {ag.norm().item():.8f}")

# Numerical FD on top-gradient components
involved = sorted(set(a for pair in gates_L0 for a in pair))
eps = 1e-5

a_list, n_list = [], []
print(f"  FD check: {len(involved)} atoms × 3 cells, ε={eps}")
t0 = time.time()

for qi in involved:
    top3 = ag[qi].abs().topk(3).indices
    for ci in top3:
        c = ci.item()
        lp = base_logits.clone(); lp[qi, c] += eps
        cp = exact_gate_cost(gates_L0, F.softmax(lp, dim=-1), H, W,
                              canonicalize=False, top_k=TOP_K_GRAD)
        lm = base_logits.clone(); lm[qi, c] -= eps
        cm = exact_gate_cost(gates_L0, F.softmax(lm, dim=-1), H, W,
                              canonicalize=False, top_k=TOP_K_GRAD)
        nd = (cp - cm) / (2 * eps)
        a_list.append(ag[qi, c].item())
        n_list.append(nd.item())

t_fd = time.time() - t0
av = torch.tensor(a_list)
nv = torch.tensor(n_list)
cos = F.cosine_similarity(av.unsqueeze(0), nv.unsqueeze(0)).item()
corr = torch.corrcoef(torch.stack([av, nv]))[0, 1].item()

print(f"  Done in {t_fd:.1f}s")
print(f"  Cosine similarity:  {cos:.6f}")
print(f"  Pearson correlation: {corr:.6f}")
print(f"  Analytical: {av[:5].tolist()}")
print(f"  Numerical:  {nv[:5].tolist()}")
print(f"  Scale ratio: {av.norm().item() / nv.norm().item():.4f}")
print(f"  Verdict: {'✓ PASS' if cos > 0.99 else '⚠ CHECK' if cos > 0.9 else '✗ FAIL'}")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 4: Optimization with exact surrogate
# ═════════════════════════════════════════════════════════════════════════════

print(f"\n{'═' * 65}")
print("TEST 4: Optimization")
print("═" * 65)

torch.manual_seed(7)
ol = torch.randn(N, C, dtype=torch.float64) * 0.1
with torch.no_grad():
    for q in range(N):
        ol[q, pos[q, 0] * W + pos[q, 1]] += 5.0
ol.requires_grad_(True)

opt = torch.optim.Adam([ol], lr=0.05)
init_true = true_gate_cost(pos[:,0]*W+pos[:,1], gates_L0, H, W, canon=True)
print(f"  Initial true cost: {init_true}")

for step in range(200):
    opt.zero_grad()
    d = F.softmax(ol, dim=-1)
    loss = exact_gate_cost(gates_L0, d, H, W, canonicalize=True, top_k=10)
    loss.backward()
    opt.step()

    if step % 40 == 0 or step == 199:
        with torch.no_grad():
            d2 = F.softmax(ol, dim=-1)
            hard = d2.argmax(dim=-1)
            hard_cost = true_gate_cost(hard, gates_L0, H, W, canon=True)
            ent = -(d2 * torch.log(d2.clamp(min=1e-30))).sum(-1).mean().item()
            gs = get_pruned_gate_states(d2, gates_L0, H, W, top_k=10)
            sizes = [g['n'] for g in gs]
        print(f"    Step {step:3d}: loss={loss.item():.4f}  hard={hard_cost}  "
              f"ent={ent:.2f}  sizes={sizes}  time/step≈{'-'}")

d_final = F.softmax(ol, dim=-1)
hard_final = d_final.argmax(dim=-1)
final_cost = true_gate_cost(hard_final, gates_L0, H, W, canon=True)
print(f"\n  Result: {init_true} → {final_cost}")

cells_used = hard_final.tolist()
if len(set(cells_used)) < len(cells_used):
    from collections import Counter
    dupes = {c: cnt for c, cnt in Counter(cells_used).items() if cnt > 1}
    print(f"  ⚠ COLLISIONS: {dupes}")

print("\n  Final placements:")
for q in range(N):
    bc = hard_final[q].item()
    r, c = bc // W, bc % W
    o_r, o_c = pos[q].tolist()
    conf = d_final[q].max().item()
    tag = "" if (r==o_r and c==o_c) else f" ← from ({o_r},{o_c})"
    print(f"    q{q:2d}: ({r},{c})  {conf:.1%}{tag}")