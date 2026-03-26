"""
GPU-accelerated exact gate cost surrogate.

Two key improvements:
1. FULLY VECTORIZED — no Python loops. The entire E[χ] computation is a
   single (K0, K1, K2, K3) tensor operation on GPU.
2. TAIL CORRECTION — instead of renormalizing, we compute the exact
   contribution of top-K states and bound/correct the tail contribution.

For top_k=8 on GPU: (64,64,64,64) = 16.7M elements → <100ms.
For top_k=12: (144,144,144,144) = 430M elements → ~1s on GPU.
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
            if not edges: tbl[cfg] = 1; continue
            for k in range(1, m+1):
                found = False
                for col in iterproduct(range(k), repeat=m):
                    if all(col[u]!=col[v] for u,v in edges):
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
        for g,(a,b) in enumerate(gates):
            if dirs[g]: a,b = b,a
            mvs.append(torch.tensor([rows[a],cols[a],rows[b],cols[b]], dtype=torch.long))
        mvs = torch.stack(mvs); n = mvs.shape[0]
        if n == 1: return 1
        conflict = torch.zeros(n,n,dtype=torch.bool)
        for i in range(n):
            for j in range(i+1,n):
                if not aod_compatible_batch(mvs[i],mvs[j]):
                    conflict[i,j]=conflict[j,i]=True
        colors = [-1]*n
        for idx in sorted(range(n), key=lambda x: -conflict[x].sum().item()):
            used = {colors[k] for k in range(n) if conflict[idx,k] and colors[k]>=0}
            c=0
            while c in used: c+=1
            colors[idx]=c
        return max(colors)+1
    if not canon: return 2*_groups([0]*M)
    return 2*min(_groups([(d>>g)&1 for g in range(M)]) for d in range(1<<M))


def mc_estimate(gates, dists, H, W, K, canon=True):
    N,C = dists.shape
    samps = torch.stack([torch.multinomial(dists[q],K,replacement=True) for q in range(N)])
    return torch.tensor([true_gate_cost(samps[:,s],gates,H,W,canon) for s in range(K)],
                        dtype=dists.dtype)


# ─────────────────────────────────────────────────────────────────────────────
# Top-K gate state extraction (no renormalization)
# ─────────────────────────────────────────────────────────────────────────────

def get_topk_gate_states(placement_dists, gate_atoms, H, W, top_k=10):
    """Extract top-K gate states per gate. NO renormalization.

    Returns unnormalized probabilities so the total sums to <1.
    The deficit is the "tail mass" that gets a separate correction.
    """
    C = H * W
    device = placement_dists.device
    dtype = placement_dists.dtype
    cells = torch.arange(C, device=device)
    rows = (cells // W).to(dtype)
    cols = (cells % W).to(dtype)

    result = []
    for a, b in gate_atoms:
        pa, pb = placement_dists[a], placement_dists[b]

        ka = min(top_k, C)
        kb = min(top_k, C)
        top_a = pa.topk(ka)
        top_b = pb.topk(kb)

        active_a, pa_k = top_a.indices, top_a.values
        active_b, pb_k = top_b.indices, top_b.values

        # Gate state probs (unnormalized — these sum to < 1)
        probs_2d = pa_k[:, None] * pb_k[None, :]  # (ka, kb)

        # Moves
        ca = active_a[:, None].expand(ka, kb)
        cb = active_b[None, :].expand(ka, kb)
        r_a, c_a = rows[ca], cols[ca]  # (ka, kb)
        r_b, c_b = rows[cb], cols[cb]

        moves_fwd = torch.stack([r_a, c_a, r_b, c_b], dim=-1)  # (ka, kb, 4)
        moves_rev = torch.stack([r_b, c_b, r_a, c_a], dim=-1)

        kept_mass = probs_2d.sum()
        tail_mass = 1.0 - kept_mass

        result.append({
            'probs_2d': probs_2d,       # (ka, kb) — unnormalized
            'moves_fwd': moves_fwd,      # (ka, kb, 4)
            'moves_rev': moves_rev,
            'ka': ka, 'kb': kb,
            'kept_mass': kept_mass,
            'tail_mass': tail_mass,
        })

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Fully vectorized compat tables
# ─────────────────────────────────────────────────────────────────────────────

def build_compat_2d(gs_i, gs_j, dir_i, dir_j):
    """Build (ka_i*kb_i, ka_j*kb_j) compat table between flattened gate states."""
    mi = gs_i['moves_rev' if dir_i else 'moves_fwd']  # (ka, kb, 4)
    mj = gs_j['moves_rev' if dir_j else 'moves_fwd']  # (ka, kb, 4)

    Ki = mi.shape[0] * mi.shape[1]
    Kj = mj.shape[0] * mj.shape[1]

    mi_flat = mi.reshape(Ki, 4)
    mj_flat = mj.reshape(Kj, 4)

    return aod_compatible_batch(mi_flat[:, None, :], mj_flat[None, :, :])  # (Ki, Kj)


# ─────────────────────────────────────────────────────────────────────────────
# FULLY VECTORIZED E[χ] — no Python loops
# ─────────────────────────────────────────────────────────────────────────────

def exact_echi_vectorized(gate_states, directions, chi_tbl=None):
    """Fully vectorized E[χ] for M=4 gates.

    Computes a (K0, K1, K2, K3) tensor of edge configs, looks up χ,
    and weights by the outer product of gate-state probabilities.

    Args:
        gate_states: list of 4 dicts from get_topk_gate_states
        directions: list of 4 bools

    Returns: scalar tensor (differentiable, unnormalized — sums over kept states only)
    """
    if chi_tbl is None:
        chi_tbl = chi_table(4)

    gs = gate_states
    dirs = directions
    device = gs[0]['probs_2d'].device
    dtype = gs[0]['probs_2d'].dtype
    chi_tbl = chi_tbl.to(device=device, dtype=dtype)

    # Flatten gate state probs: (ka*kb,) per gate
    p = [g['probs_2d'].reshape(-1) for g in gs]
    K = [pp.shape[0] for pp in p]

    # Build conflict bit tensors for all 6 pairs
    # Each is (Ki, Kj) bool → cast to int8 for bit manipulation
    c01 = (~build_compat_2d(gs[0], gs[1], dirs[0], dirs[1])).to(torch.int8)  # (K0, K1)
    c02 = (~build_compat_2d(gs[0], gs[2], dirs[0], dirs[2])).to(torch.int8)  # (K0, K2)
    c12 = (~build_compat_2d(gs[1], gs[2], dirs[1], dirs[2])).to(torch.int8)  # (K1, K2)
    c03 = (~build_compat_2d(gs[0], gs[3], dirs[0], dirs[3])).to(torch.int8)  # (K0, K3)
    c13 = (~build_compat_2d(gs[1], gs[3], dirs[1], dirs[3])).to(torch.int8)  # (K1, K3)
    c23 = (~build_compat_2d(gs[2], gs[3], dirs[2], dirs[3])).to(torch.int8)  # (K2, K3)

    # Build (K0, K1, K2, K3) edge config tensor via broadcasting
    # edge_cfg = bit0 + bit1*2 + bit2*4 + bit3*8 + bit4*16 + bit5*32
    # Shapes for broadcasting:
    # c01: (K0, K1, 1, 1)  * 1
    # c02: (K0, 1, K2, 1)  * 2
    # c12: (1, K1, K2, 1)  * 4
    # c03: (K0, 1, 1, K3)  * 8
    # c13: (1, K1, 1, K3)  * 16
    # c23: (1, 1, K2, K3)  * 32

    edge_cfg = (c01[:, :, None, None].to(torch.int16) +
                c02[:, None, :, None].to(torch.int16) * 2 +
                c12[None, :, :, None].to(torch.int16) * 4 +
                c03[:, None, None, :].to(torch.int16) * 8 +
                c13[None, :, None, :].to(torch.int16) * 16 +
                c23[None, None, :, :].to(torch.int16) * 32)  # (K0, K1, K2, K3) int16

    # Look up χ values
    chi_vals = chi_tbl[edge_cfg.long()]  # (K0, K1, K2, K3) float

    # Weight by probability outer product
    # p[0]: (K0,), p[1]: (K1,), p[2]: (K2,), p[3]: (K3,)
    # joint = p0 * p1 * p2 * p3 via broadcasting
    joint = (p[0][:, None, None, None] *
             p[1][None, :, None, None] *
             p[2][None, None, :, None] *
             p[3][None, None, None, :])  # (K0, K1, K2, K3)

    return (chi_vals * joint).sum()


def exact_echi_vectorized_canon(gate_states, chi_tbl=None):
    """Exact E[min_d χ] with canonicalization, fully vectorized.

    For each direction assignment, compute the (K0,K1,K2,K3) chi tensor,
    then take element-wise min across directions.
    """
    if chi_tbl is None:
        chi_tbl = chi_table(4)

    gs = gate_states
    M = 4
    device = gs[0]['probs_2d'].device
    dtype = gs[0]['probs_2d'].dtype
    chi_tbl = chi_tbl.to(device=device, dtype=dtype)

    p = [g['probs_2d'].reshape(-1) for g in gs]
    K = [pp.shape[0] for pp in p]

    # Joint probability (shared across all directions)
    joint = (p[0][:, None, None, None] *
             p[1][None, :, None, None] *
             p[2][None, None, :, None] *
             p[3][None, None, None, :])

    min_chi = None

    for dir_idx in range(1 << M):
        dirs = [bool((dir_idx >> g) & 1) for g in range(M)]

        c01 = (~build_compat_2d(gs[0], gs[1], dirs[0], dirs[1])).to(torch.int16)
        c02 = (~build_compat_2d(gs[0], gs[2], dirs[0], dirs[2])).to(torch.int16)
        c12 = (~build_compat_2d(gs[1], gs[2], dirs[1], dirs[2])).to(torch.int16)
        c03 = (~build_compat_2d(gs[0], gs[3], dirs[0], dirs[3])).to(torch.int16)
        c13 = (~build_compat_2d(gs[1], gs[3], dirs[1], dirs[3])).to(torch.int16)
        c23 = (~build_compat_2d(gs[2], gs[3], dirs[2], dirs[3])).to(torch.int16)

        edge_cfg = (c01[:, :, None, None] +
                    c02[:, None, :, None] * 2 +
                    c12[None, :, :, None] * 4 +
                    c03[:, None, None, :] * 8 +
                    c13[None, :, None, :] * 16 +
                    c23[None, None, :, :] * 32)

        chi_vals = chi_tbl[edge_cfg.long()]

        if min_chi is None:
            min_chi = chi_vals
        else:
            min_chi = torch.min(min_chi, chi_vals)

    return (min_chi * joint).sum()


# ─────────────────────────────────────────────────────────────────────────────
# Tail correction
# ─────────────────────────────────────────────────────────────────────────────

def tail_correction(gate_states, M=4):
    """Compute bounds on the contribution of pruned tail states.

    The exact E[χ] = E_kept[χ] + E_tail[χ]
    where E_kept is what we compute and E_tail is the contribution from
    configurations involving at least one gate state in the tail.

    Bounds:
      E_tail ∈ [1 * tail_total_mass, χ_max * tail_total_mass]
    where tail_total_mass = 1 - Π_g kept_mass_g
    (probability that at least one gate has a tail state)

    Better estimate: use χ_max for a pessimistic correction,
    or use the empirical mean χ from the kept computation.
    """
    kept_masses = [g['kept_mass'] for g in gate_states]
    all_kept_prob = torch.stack(kept_masses).prod()
    tail_total_mass = 1.0 - all_kept_prob

    chi_max = M  # max chromatic number for M nodes
    chi_min = 1

    return {
        'tail_mass': tail_total_mass,
        'lower_bound': chi_min * tail_total_mass,  # optimistic: all tail configs have χ=1
        'upper_bound': chi_max * tail_total_mass,   # pessimistic: all tail configs have χ=M
        'midpoint': (chi_min + chi_max) / 2 * tail_total_mass,  # rough middle estimate
    }


def exact_gate_cost(gate_atoms, placement_dists, H, W,
                    canonicalize=True, top_k=10,
                    tail_correction_mode='midpoint'):
    """Full gate cost: 2 * E[χ*] with optional tail correction.

    tail_correction_mode:
        None: no correction (slightly negative bias)
        'lower': add χ_min * tail_mass (still negative bias but less)
        'upper': add χ_max * tail_mass (positive bias — conservative)
        'midpoint': add (χ_min + χ_max)/2 * tail_mass
        'empirical': use mean χ from kept computation as estimate for tail
    """
    gs = get_topk_gate_states(placement_dists, gate_atoms, H, W, top_k)

    if canonicalize:
        chi_kept = exact_echi_vectorized_canon(gs)
    else:
        chi_kept = exact_echi_vectorized(gs, [False]*4)

    if tail_correction_mode is None:
        return 2.0 * chi_kept

    tc = tail_correction(gs)

    if tail_correction_mode == 'empirical':
        # Use the mean χ from kept states as estimate for tail states
        kept_mass = 1.0 - tc['tail_mass']
        mean_chi_kept = chi_kept / kept_mass.clamp(min=1e-30)
        correction = mean_chi_kept * tc['tail_mass']
    elif tail_correction_mode == 'lower':
        correction = tc['lower_bound']
    elif tail_correction_mode == 'upper':
        correction = tc['upper_bound']
    elif tail_correction_mode == 'midpoint':
        correction = tc['midpoint']
    else:
        correction = 0.0

    return 2.0 * (chi_kept + correction)


# ═════════════════════════════════════════════════════════════════════════════
# Setup
# ═════════════════════════════════════════════════════════════════════════════

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE = torch.float32  # float32 for GPU speed; float64 for gradient checks
print(f"Device: {DEVICE}")

H, W, C = 5, 5, 25
gates_L0 = [(4, 3), (6, 7), (9, 8), (10, 11)]
pos = torch.tensor([
    [0,4],[2,2],[1,2],[2,1],[3,0],[1,0],
    [0,2],[4,1],[4,2],[4,3],[4,0],[1,3],
], device=DEVICE)
N = pos.shape[0]


# ═════════════════════════════════════════════════════════════════════════════
# TEST 1: One-hot sanity
# ═════════════════════════════════════════════════════════════════════════════

print(f"\n{'═'*65}")
print("TEST 1: One-hot sanity")
print("═"*65)

onehot = torch.zeros(N, C, dtype=DTYPE, device=DEVICE)
for q in range(N):
    onehot[q, pos[q,0]*W + pos[q,1]] = 1.0

for canon in [False, True]:
    t0 = time.time()
    v = exact_gate_cost(gates_L0, onehot, H, W, canonicalize=canon,
                         top_k=5, tail_correction_mode=None)
    t = time.time() - t0
    true = true_gate_cost(pos[:,0]*W+pos[:,1], gates_L0, H, W, canon=canon)
    print(f"  canon={canon}: exact={v.item():.4f}  true={true}  "
          f"time={t:.4f}s  {'✓' if abs(v.item()-true)<0.05 else '✗'}")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 2: Speed benchmark across top_k values
# ═════════════════════════════════════════════════════════════════════════════

print(f"\n{'═'*65}")
print("TEST 2: Speed benchmark (no canon, no correction)")
print("═"*65)

torch.manual_seed(42)

for sigma in [1.5, 4.0, 10.0]:
    logits = torch.randn(N, C, dtype=DTYPE, device=DEVICE) * 0.1
    for q in range(N):
        logits[q, pos[q,0]*W + pos[q,1]] += sigma
    d = F.softmax(logits, dim=-1)
    ent = -(d * torch.log(d.clamp(min=1e-30))).sum(-1).mean().item()

    print(f"\n  σ={sigma}  (entropy={ent:.2f})")

    for top_k in [5, 8, 10, 12, 15, 25]:
        gs = get_topk_gate_states(d, gates_L0, H, W, top_k)
        K_total = math.prod([g['probs_2d'].numel() for g in gs])
        tail = max(g['tail_mass'].item() if isinstance(g['tail_mass'], torch.Tensor)
                    else g['tail_mass'] for g in gs)

        # Warmup
        _ = exact_gate_cost(gates_L0, d, H, W, canonicalize=False,
                             top_k=top_k, tail_correction_mode=None)
        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()

        # Timed run (average of 3)
        times = []
        for _ in range(3):
            if DEVICE.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.time()
            v = exact_gate_cost(gates_L0, d, H, W, canonicalize=False,
                                 top_k=top_k, tail_correction_mode=None)
            if DEVICE.type == 'cuda':
                torch.cuda.synchronize()
            times.append(time.time() - t0)

        avg_t = sum(times) / len(times)
        print(f"    top_k={top_k:2d}: K_total={K_total:>12,}  time={avg_t:.4f}s  "
              f"val={v.item():.4f}  tail={tail:.4f}")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 3: Tail correction quality
# ═════════════════════════════════════════════════════════════════════════════

print(f"\n{'═'*65}")
print("TEST 3: Tail correction quality")
print("═"*65)

torch.manual_seed(42)

for sigma in [1.5, 4.0]:
    logits = torch.randn(N, C, dtype=DTYPE, device=DEVICE) * 0.1
    for q in range(N):
        logits[q, pos[q,0]*W + pos[q,1]] += sigma
    d = F.softmax(logits, dim=-1)

    # Ground truth: top_k=25 (full, no pruning)
    gt = exact_gate_cost(gates_L0, d, H, W, canonicalize=False,
                          top_k=25, tail_correction_mode=None)

    # Also MC for sanity
    d_cpu = d.cpu().to(torch.float64)
    mc = mc_estimate(gates_L0, d_cpu, H, W, 5000, canon=False)
    mc_mean = mc.mean().item()

    print(f"\n  σ={sigma}  (GT={gt.item():.4f}, MC={mc_mean:.4f})")

    for top_k in [5, 8, 10, 12]:
        for mode in [None, 'lower', 'midpoint', 'upper', 'empirical']:
            v = exact_gate_cost(gates_L0, d, H, W, canonicalize=False,
                                 top_k=top_k, tail_correction_mode=mode)
            bias = v.item() - gt.item()
            label = f"{mode or 'none':>10s}"
            print(f"    top_k={top_k:2d} {label}: {v.item():.4f}  "
                  f"bias={bias:+.4f}")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 4: Gradient check (float64 on CPU for precision)
# ═════════════════════════════════════════════════════════════════════════════

print(f"\n{'═'*65}")
print("TEST 4: Gradient check (float64, CPU)")
print("═"*65)

torch.manual_seed(99)
base_logits = torch.randn(N, C, dtype=torch.float64) * 0.15
for q in range(N):
    base_logits[q, pos[q,0]*W + pos[q,1]] += 4.0

TOP_K = 10

# Analytical
gl = base_logits.clone().requires_grad_(True)
d = F.softmax(gl, dim=-1)
cost = exact_gate_cost(gates_L0, d, H, W, canonicalize=False,
                        top_k=TOP_K, tail_correction_mode=None)
cost.backward()
ag = gl.grad.clone()

print(f"  Cost: {cost.item():.6f}")
print(f"  Grad norm: {ag.norm().item():.8f}")

# Numerical FD
involved = sorted(set(a for pair in gates_L0 for a in pair))
eps = 1e-5
a_list, n_list = [], []

t0 = time.time()
for qi in involved:
    top3 = ag[qi].abs().topk(3).indices
    for ci in top3:
        c = ci.item()
        lp = base_logits.clone(); lp[qi,c] += eps
        cp = exact_gate_cost(gates_L0, F.softmax(lp, dim=-1), H, W,
                              canonicalize=False, top_k=TOP_K, tail_correction_mode=None)
        lm = base_logits.clone(); lm[qi,c] -= eps
        cm = exact_gate_cost(gates_L0, F.softmax(lm, dim=-1), H, W,
                              canonicalize=False, top_k=TOP_K, tail_correction_mode=None)
        a_list.append(ag[qi,c].item())
        n_list.append(((cp-cm)/(2*eps)).item())

t_fd = time.time()-t0
av = torch.tensor(a_list)
nv = torch.tensor(n_list)
cos = F.cosine_similarity(av.unsqueeze(0), nv.unsqueeze(0)).item()

print(f"  Done in {t_fd:.1f}s")
print(f"  Cosine similarity: {cos:.6f}")
print(f"  Verdict: {'✓ PASS' if cos > 0.99 else '⚠' if cos > 0.9 else '✗ FAIL'}")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 5: Speed benchmark WITH canonicalization
# ═════════════════════════════════════════════════════════════════════════════

print(f"\n{'═'*65}")
print("TEST 5: Speed with canonicalization (16 direction passes)")
print("═"*65)

torch.manual_seed(42)
for sigma in [4.0, 10.0]:
    logits = torch.randn(N, C, dtype=DTYPE, device=DEVICE) * 0.1
    for q in range(N):
        logits[q, pos[q,0]*W + pos[q,1]] += sigma
    d = F.softmax(logits, dim=-1)

    for top_k in [8, 10, 12]:
        # Warmup
        _ = exact_gate_cost(gates_L0, d, H, W, canonicalize=True,
                             top_k=top_k, tail_correction_mode='empirical')
        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()

        times = []
        for _ in range(3):
            if DEVICE.type == 'cuda': torch.cuda.synchronize()
            t0 = time.time()
            v = exact_gate_cost(gates_L0, d, H, W, canonicalize=True,
                                 top_k=top_k, tail_correction_mode='empirical')
            if DEVICE.type == 'cuda': torch.cuda.synchronize()
            times.append(time.time() - t0)

        avg_t = sum(times)/len(times)
        print(f"  σ={sigma} top_k={top_k:2d}: {avg_t:.4f}s  cost={v.item():.4f}")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 6: Full optimization
# ═════════════════════════════════════════════════════════════════════════════

print(f"\n{'═'*65}")
print("TEST 6: Optimization")
print("═"*65)

torch.manual_seed(7)
ol = torch.randn(N, C, dtype=DTYPE, device=DEVICE) * 0.1
with torch.no_grad():
    for q in range(N):
        ol[q, pos[q,0]*W + pos[q,1]] += 5.0
ol.requires_grad_(True)

opt = torch.optim.Adam([ol], lr=0.05)
init_true = true_gate_cost(pos[:,0]*W+pos[:,1], gates_L0, H, W, canon=True)
print(f"  Initial true cost: {init_true}")

for step in range(200):
    opt.zero_grad()
    d = F.softmax(ol, dim=-1)
    loss = exact_gate_cost(gates_L0, d, H, W, canonicalize=True,
                            top_k=10, tail_correction_mode='empirical')
    loss.backward()
    opt.step()

    if step % 40 == 0 or step == 199:
        with torch.no_grad():
            d2 = F.softmax(ol, dim=-1)
            hard = d2.argmax(dim=-1)
            hard_cost = true_gate_cost(hard, gates_L0, H, W, canon=True)
            ent = -(d2 * torch.log(d2.clamp(min=1e-30))).sum(-1).mean().item()
        print(f"    Step {step:3d}: loss={loss.item():.4f}  hard={hard_cost}  ent={ent:.2f}")

d_final = F.softmax(ol, dim=-1)
hard_final = d_final.argmax(dim=-1)
final_cost = true_gate_cost(hard_final, gates_L0, H, W, canon=True)
print(f"\n  Result: {init_true} → {final_cost}")

cells_used = hard_final.tolist()
if len(set(cells_used)) < len(cells_used):
    from collections import Counter
    dupes = {c: cnt for c,cnt in Counter(cells_used).items() if cnt > 1}
    print(f"  ⚠ COLLISIONS: {dupes}")