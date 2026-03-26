"""
Fast exact gate cost surrogate via sparse gate-state enumeration.

Key speedup: for distributions with entropy < ~3, most gate states have
near-zero probability. Threshold at P(s_g) > ε and only enumerate active
states. For σ=4.0 (entropy ~1.5), each atom concentrates on ~5 cells,
giving ~25 active gate states instead of 625 → 25^4 = 390K vs 625^4 = 150B.

Also: GPU-accelerated, batched over direction assignments.
"""

import torch
import torch.nn.functional as F
import time
import math
from itertools import product as iterproduct

# ─────────────────────────────────────────────────────────────────────────────
# AOD compatibility (vectorized)
# ─────────────────────────────────────────────────────────────────────────────

def aod_compatible_batch(mi, mj):
    """Batched AOD check. Inputs: (..., 4) tensors [sr, sc, dr, dc]."""
    sr_i, sc_i, dr_i, dc_i = mi[...,0], mi[...,1], mi[...,2], mi[...,3]
    sr_j, sc_j, dr_j, dc_j = mj[...,0], mj[...,1], mj[...,2], mj[...,3]
    dcs, dcd = sc_i - sc_j, dc_i - dc_j
    h_ok = ((dcs==0)&(dcd==0)) | (~((dcs==0)|(dcd==0)) & (torch.sign(dcs)==torch.sign(dcd)))
    drs, drd = sr_i - sr_j, dr_i - dr_j
    v_ok = ((drs==0)&(drd==0)) | (~((drs==0)|(drd==0)) & (torch.sign(drs)==torch.sign(drd)))
    no_col = ~((drd==0)&(dcd==0))
    return h_ok & v_ok & no_col


# ─────────────────────────────────────────────────────────────────────────────
# χ table
# ─────────────────────────────────────────────────────────────────────────────

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
            # brute force chi
            n = m
            edges = [(i,j) for i in range(n) for j in range(i+1,n) if adj[i,j]]
            if not edges:
                tbl[cfg] = 1
                continue
            found = False
            for k in range(1, n+1):
                if found: break
                for col in iterproduct(range(k), repeat=n):
                    if all(col[u] != col[v] for u,v in edges):
                        tbl[cfg] = k
                        found = True
                        break
            if not found:
                tbl[cfg] = n
        _CHI[m] = tbl
    return _CHI[m]


# ─────────────────────────────────────────────────────────────────────────────
# Sparse gate-state representation
# ─────────────────────────────────────────────────────────────────────────────

def get_active_gate_states(placement_dists, gate_atoms, H, W, threshold=1e-6):
    """Find active (high-probability) gate states for each gate.

    Returns for each gate:
        indices: (K_g,) tensor of active gate-state indices (s = ca * C + cb)
        probs:   (K_g,) tensor of corresponding probabilities
        moves:   (K_g, 4) tensor of [src_row, src_col, dst_row, dst_col]
    """
    C = H * W
    cells = torch.arange(C, device=placement_dists.device)
    rows = cells // W
    cols = cells % W

    result = []
    for a, b in gate_atoms:
        pa, pb = placement_dists[a], placement_dists[b]

        # Find active cells for each atom
        active_a = (pa > threshold).nonzero(as_tuple=True)[0]
        active_b = (pb > threshold).nonzero(as_tuple=True)[0]

        # All active pairs
        ka, kb = len(active_a), len(active_b)
        ca = active_a[:, None].expand(ka, kb).reshape(-1)  # (ka*kb,)
        cb = active_b[None, :].expand(ka, kb).reshape(-1)  # (ka*kb,)

        # Probabilities
        probs = pa[ca] * pb[cb]  # (ka*kb,)

        # Filter by threshold on joint probability
        mask = probs > threshold
        ca, cb, probs = ca[mask], cb[mask], probs[mask]

        # Gate state indices
        indices = ca * C + cb

        # Move vectors (forward direction: a → b)
        moves_fwd = torch.stack([rows[ca], cols[ca], rows[cb], cols[cb]], dim=-1)
        # Reverse direction: b → a
        moves_rev = torch.stack([rows[cb], cols[cb], rows[ca], cols[ca]], dim=-1)

        result.append({
            'indices': indices,
            'probs': probs,
            'moves_fwd': moves_fwd,
            'moves_rev': moves_rev,
            'n': len(indices),
        })

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Sparse compatibility tables
# ─────────────────────────────────────────────────────────────────────────────

def build_sparse_compat(gate_states_i, gate_states_j, dir_i, dir_j):
    """Build (Ki, Kj) compatibility table between sparse gate states.

    Args:
        gate_states_i, _j: dicts from get_active_gate_states
        dir_i, dir_j: bool, True = reverse direction

    Returns: (Ki, Kj) bool tensor
    """
    mi = gate_states_i['moves_rev' if dir_i else 'moves_fwd']  # (Ki, 4)
    mj = gate_states_j['moves_rev' if dir_j else 'moves_fwd']  # (Kj, 4)

    # Broadcast to (Ki, Kj, 4)
    mi_exp = mi[:, None, :].expand(-1, mj.shape[0], -1)
    mj_exp = mj[None, :, :].expand(mi.shape[0], -1, -1)

    return aod_compatible_batch(mi_exp, mj_exp)  # (Ki, Kj) bool


# ─────────────────────────────────────────────────────────────────────────────
# Exact E[χ] with sparse enumeration — M=4 specific
# ─────────────────────────────────────────────────────────────────────────────

def exact_expected_chi_sparse_m4(gate_states, directions, chi_tbl=None):
    """Exact E[χ] for M=4 gates using sparse gate-state enumeration.

    Loops over (s0, s1) [sparse], vectorizes over (s2, s3) [sparse].

    Args:
        gate_states: list of 4 dicts from get_active_gate_states
        directions: list of 4 bools

    Returns: scalar tensor (differentiable through gate_states[g]['probs'])
    """
    if chi_tbl is None:
        chi_tbl = chi_table(4)

    gs = gate_states
    dirs = directions
    K = [g['n'] for g in gs]

    device = gs[0]['probs'].device
    dtype = gs[0]['probs'].dtype
    chi_tbl = chi_tbl.to(device=device, dtype=dtype)

    # Build sparse compatibility tables for all 6 pairs
    # Edge ordering: (0,1)=bit0, (0,2)=bit1, (1,2)=bit2, (0,3)=bit3, (1,3)=bit4, (2,3)=bit5
    F01 = build_sparse_compat(gs[0], gs[1], dirs[0], dirs[1])  # (K0, K1)
    F02 = build_sparse_compat(gs[0], gs[2], dirs[0], dirs[2])  # (K0, K2)
    F12 = build_sparse_compat(gs[1], gs[2], dirs[1], dirs[2])  # (K1, K2)
    F03 = build_sparse_compat(gs[0], gs[3], dirs[0], dirs[3])  # (K0, K3)
    F13 = build_sparse_compat(gs[1], gs[3], dirs[1], dirs[3])  # (K1, K3)
    F23 = build_sparse_compat(gs[2], gs[3], dirs[2], dirs[3])  # (K2, K3)

    p0, p1, p2, p3 = gs[0]['probs'], gs[1]['probs'], gs[2]['probs'], gs[3]['probs']

    # Convert bool compat to conflict bits (1 = conflict = incompatible)
    # conflict bit = 1 - compat
    c01 = (~F01).to(dtype)  # (K0, K1), 1.0 where conflict
    c02 = (~F02).to(dtype)  # (K0, K2)
    c12 = (~F12).to(dtype)  # (K1, K2)
    c03 = (~F03).to(dtype)  # (K0, K3)
    c13 = (~F13).to(dtype)  # (K1, K3)
    c23 = (~F23).to(dtype)  # (K2, K3)

    result = torch.tensor(0.0, device=device, dtype=dtype)

    # Outer loop: s0, s1 (sparse — typically K0*K1 ~ 25*25 = 625 iterations)
    for i0 in range(K[0]):
        if p0[i0] < 1e-30:
            continue
        for i1 in range(K[1]):
            if p1[i1] < 1e-30:
                continue

            # Fixed bits from (s0, s1): e01
            bit0 = c01[i0, i1]  # scalar, 0 or 1

            # Vectorized over (s2, s3):
            # bit1 = c02[i0, :K2]        — (K2,)  for each s2
            # bit2 = c12[i1, :K2]        — (K2,)  for each s2
            # bit3 = c03[i0, :K3]        — (K3,)  for each s3
            # bit4 = c13[i1, :K3]        — (K3,)  for each s3
            # bit5 = c23[:K2, :K3]       — (K2, K3) for each (s2, s3)

            bit1 = c02[i0]       # (K2,)
            bit2 = c12[i1]       # (K2,)
            bit3 = c03[i0]       # (K3,)
            bit4 = c13[i1]       # (K3,)
            bit5 = c23           # (K2, K3)

            # Edge config = bit0 + bit1*2 + bit2*4 + bit3*8 + bit4*16 + bit5*32
            # Shape: (K2, K3)
            edge_cfg = (bit0 +
                        bit1[:, None] * 2 +
                        bit2[:, None] * 4 +
                        bit3[None, :] * 8 +
                        bit4[None, :] * 16 +
                        bit5 * 32)  # (K2, K3)

            # Look up χ
            chi_vals = chi_tbl[edge_cfg.long()]  # (K2, K3)

            # Weight by p2[s2] * p3[s3]
            joint_p23 = p2[:, None] * p3[None, :]  # (K2, K3)

            # Accumulate
            inner = (chi_vals * joint_p23).sum()
            result = result + p0[i0] * p1[i1] * inner

    return result


def exact_expected_chi_sparse_m4_canonicalized(gate_states, chi_tbl=None):
    """Exact E[min_d χ(G_d)] with canonicalization.

    For each (s0, s1, s2, s3) configuration, find the direction assignment
    that minimizes χ, then weight by probability.

    This is EXACT — no soft-min approximation.
    """
    if chi_tbl is None:
        chi_tbl = chi_table(4)

    gs = gate_states
    K = [g['n'] for g in gs]
    M = 4
    device = gs[0]['probs'].device
    dtype = gs[0]['probs'].dtype
    chi_tbl_dev = chi_tbl.to(device=device, dtype=dtype)

    p0, p1, p2, p3 = gs[0]['probs'], gs[1]['probs'], gs[2]['probs'], gs[3]['probs']

    # Precompute all 24 sparse compat tables (6 pairs × 4 direction combos)
    compat = {}
    for i in range(M):
        for j in range(i+1, M):
            for di in [False, True]:
                for dj in [False, True]:
                    compat[(i,j,di,dj)] = (~build_sparse_compat(
                        gs[i], gs[j], di, dj)).to(dtype)  # conflict bits

    # For each of 16 direction assignments, precompute the edge config
    # for all (s2, s3) given (s0, s1).
    # Then take element-wise min over directions.

    result = torch.tensor(0.0, device=device, dtype=dtype)

    for i0 in range(K[0]):
        if p0[i0] < 1e-30:
            continue
        for i1 in range(K[1]):
            if p1[i1] < 1e-30:
                continue

            # For each direction, compute chi(s2, s3) → (K2, K3) tensor
            # Then take min over directions
            min_chi = None

            for dir_idx in range(1 << M):
                dirs = [bool((dir_idx >> g) & 1) for g in range(M)]
                d0, d1, d2, d3 = dirs

                bit0 = compat[(0,1,d0,d1)][i0, i1]
                bit1 = compat[(0,2,d0,d2)][i0]       # (K2,)
                bit2 = compat[(1,2,d1,d2)][i1]       # (K2,)
                bit3 = compat[(0,3,d0,d3)][i0]       # (K3,)
                bit4 = compat[(1,3,d1,d3)][i1]       # (K3,)
                bit5 = compat[(2,3,d2,d3)]            # (K2, K3)

                edge_cfg = (bit0 +
                            bit1[:, None] * 2 +
                            bit2[:, None] * 4 +
                            bit3[None, :] * 8 +
                            bit4[None, :] * 16 +
                            bit5 * 32)

                chi_vals = chi_tbl_dev[edge_cfg.long()]  # (K2, K3)

                if min_chi is None:
                    min_chi = chi_vals.clone()
                else:
                    min_chi = torch.min(min_chi, chi_vals)

            # Weight by p2 * p3
            joint_p23 = p2[:, None] * p3[None, :]
            inner = (min_chi * joint_p23).sum()
            result = result + p0[i0] * p1[i1] * inner

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Full gate cost function
# ─────────────────────────────────────────────────────────────────────────────

def exact_gate_cost(gate_atoms, placement_dists, H, W,
                    canonicalize=True, threshold=1e-6):
    """Exact differentiable gate cost: 2 * E[χ*].

    Args:
        gate_atoms: list of (atom_a, atom_b) for each gate
        placement_dists: (N_atoms, C) soft placement distributions
        H, W: grid dims
        canonicalize: if True, minimize over directions (exact, not soft-min)
        threshold: probability threshold for sparse enumeration

    Returns: scalar tensor (differentiable)
    """
    gate_states = get_active_gate_states(placement_dists, gate_atoms, H, W, threshold)

    if canonicalize:
        chi = exact_expected_chi_sparse_m4_canonicalized(gate_states)
    else:
        chi = exact_expected_chi_sparse_m4(gate_states, [False]*4)

    return 2.0 * chi


# ─────────────────────────────────────────────────────────────────────────────
# True cost for validation
# ─────────────────────────────────────────────────────────────────────────────

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
            for j in range(i + 1, n):
                if not aod_compatible_batch(mvs[i], mvs[j]):
                    conflict[i, j] = conflict[j, i] = True
        colors = [-1] * n
        for idx in sorted(range(n), key=lambda x: -conflict[x].sum().item()):
            used = {colors[k] for k in range(n) if conflict[idx, k] and colors[k] >= 0}
            c = 0
            while c in used: c += 1
            colors[idx] = c
        return max(colors) + 1
    if not canon:
        return 2 * _groups([0] * M)
    return 2 * min(_groups([(d >> g) & 1 for g in range(M)]) for d in range(1 << M))


def mc_estimate(gates, dists, H, W, K, canon=True):
    N, C = dists.shape
    samps = torch.stack([torch.multinomial(dists[q], K, replacement=True) for q in range(N)])
    return torch.tensor([true_gate_cost(samps[:, s], gates, H, W, canon) for s in range(K)],
                        dtype=dists.dtype)


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
print("TEST 1: One-hot sanity check")
print("═" * 65)

onehot = torch.zeros(N, C, dtype=torch.float64)
for q in range(N):
    onehot[q, pos[q, 0] * W + pos[q, 1]] = 1.0

t0 = time.time()
cost_nocanon = exact_gate_cost(gates_L0, onehot, H, W, canonicalize=False)
t1 = time.time()
cost_canon = exact_gate_cost(gates_L0, onehot, H, W, canonicalize=True)
t2 = time.time()

true_nocanon = true_gate_cost(pos[:,0]*W+pos[:,1], gates_L0, H, W, canon=False)
true_canon = true_gate_cost(pos[:,0]*W+pos[:,1], gates_L0, H, W, canon=True)

print(f"  No canon:  exact={cost_nocanon.item():.4f}  true={true_nocanon}  "
      f"time={t1-t0:.4f}s  {'✓' if abs(cost_nocanon.item()-true_nocanon)<0.01 else '✗'}")
print(f"  Canon:     exact={cost_canon.item():.4f}  true={true_canon}  "
      f"time={t2-t1:.4f}s  {'✓' if abs(cost_canon.item()-true_canon)<0.01 else '✗'}")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 2: Speed + bias at varying sharpness
# ═════════════════════════════════════════════════════════════════════════════

print(f"\n{'═' * 65}")
print("TEST 2: Speed and bias at varying sharpness")
print("═" * 65)

torch.manual_seed(42)

for label, sigma in [("Very soft σ=0.3", 0.3),
                      ("Moderate  σ=1.5", 1.5),
                      ("Sharp     σ=4.0", 4.0),
                      ("Near-1hot σ=10",  10.0)]:
    logits = torch.randn(N, C, dtype=torch.float64) * 0.1
    for q in range(N):
        logits[q, pos[q, 0] * W + pos[q, 1]] += sigma
    d = F.softmax(logits, dim=-1)
    ent = -(d * torch.log(d.clamp(min=1e-30))).sum(-1).mean().item()

    gs = get_active_gate_states(d, gates_L0, H, W, threshold=1e-6)
    active_sizes = [g['n'] for g in gs]

    t0 = time.time()
    exact_val = exact_gate_cost(gates_L0, d, H, W, canonicalize=False, threshold=1e-6)
    t_exact = time.time() - t0

    mc = mc_estimate(gates_L0, d, H, W, 5000, canon=False)
    mm, se = mc.mean(), mc.std() / math.sqrt(5000)
    bias = exact_val.item() - mm.item()

    print(f"\n  {label}  (entropy={ent:.2f})")
    print(f"    Active gate states: {active_sizes}  (product: {math.prod(active_sizes)})")
    print(f"    Exact: {exact_val.item():.4f}  ({t_exact:.3f}s)")
    print(f"    MC:    {mm.item():.4f} ± {se.item():.4f}")
    print(f"    Bias:  {bias:+.4f}  {'(within 2σ ✓)' if abs(bias) < 2*se.item() else '(SIGNIFICANT ✗)'}")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 3: Gradient check — exact analytical vs exact numerical (no MC noise!)
# ═════════════════════════════════════════════════════════════════════════════

print(f"\n{'═' * 65}")
print("TEST 3: Gradient check — analytical vs numerical (no MC noise)")
print("═" * 65)

torch.manual_seed(99)
base_logits = torch.randn(N, C, dtype=torch.float64) * 0.15
for q in range(N):
    base_logits[q, pos[q, 0] * W + pos[q, 1]] += 4.0  # sharper for speed

# Analytical gradient
gl = base_logits.clone().requires_grad_(True)
d = F.softmax(gl, dim=-1)
cost = exact_gate_cost(gates_L0, d, H, W, canonicalize=False, threshold=1e-8)
cost.backward()
analytical_grad = gl.grad.clone()

print(f"  Cost: {cost.item():.6f}")
print(f"  Gradient norm: {analytical_grad.norm().item():.6f}")

# Numerical gradient (using the EXACT surrogate — no MC noise!)
involved = sorted(set(a for pair in gates_L0 for a in pair))
eps = 1e-4  # can use small eps since there's no sampling noise

num_components_a = []
num_components_n = []

print(f"  Computing numerical gradients for {len(involved)} atoms × 3 cells...")
t0 = time.time()

for qi in involved:
    top3 = analytical_grad[qi].abs().topk(3).indices
    for ci in top3:
        c = ci.item()

        lp = base_logits.clone()
        lp[qi, c] += eps
        dp = F.softmax(lp, dim=-1)
        cp = exact_gate_cost(gates_L0, dp, H, W, canonicalize=False, threshold=1e-8)

        lm = base_logits.clone()
        lm[qi, c] -= eps
        dm = F.softmax(lm, dim=-1)
        cm = exact_gate_cost(gates_L0, dm, H, W, canonicalize=False, threshold=1e-8)

        nd = (cp - cm) / (2 * eps)
        num_components_a.append(analytical_grad[qi, c].item())
        num_components_n.append(nd.item())

t_grad = time.time() - t0
print(f"  Done in {t_grad:.1f}s")

av = torch.tensor(num_components_a)
nv = torch.tensor(num_components_n)
cos = F.cosine_similarity(av.unsqueeze(0), nv.unsqueeze(0)).item()
corr = torch.corrcoef(torch.stack([av, nv]))[0, 1].item()

print(f"\n  Analytical vs Numerical ({len(num_components_a)} components):")
print(f"    Cosine similarity:  {cos:.6f}")
print(f"    Pearson correlation: {corr:.6f}")
print(f"    Analytical samples: {av[:6].tolist()}")
print(f"    Numerical samples:  {nv[:6].tolist()}")
print(f"    Scale ratio (analytical/numerical): {av.norm().item() / nv.norm().item():.4f}")
print(f"    Verdict: {'✓ PASS' if cos > 0.99 else '⚠ CHECK' if cos > 0.9 else '✗ FAIL'}")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 4: Optimization
# ═════════════════════════════════════════════════════════════════════════════

print(f"\n{'═' * 65}")
print("TEST 4: Optimization with exact surrogate")
print("═" * 65)

torch.manual_seed(7)
ol = torch.randn(N, C, dtype=torch.float64) * 0.1
with torch.no_grad():
    for q in range(N):
        ol[q, pos[q, 0] * W + pos[q, 1]] += 5.0  # start sharper for speed
ol.requires_grad_(True)

opt = torch.optim.Adam([ol], lr=0.05)
init_true = true_gate_cost(pos[:, 0] * W + pos[:, 1], gates_L0, H, W, canon=True)
print(f"  Initial true cost: {init_true}")

for step in range(200):
    opt.zero_grad()
    d = F.softmax(ol, dim=-1)
    loss = exact_gate_cost(gates_L0, d, H, W, canonicalize=True, threshold=1e-6)
    loss.backward()
    opt.step()

    if step % 40 == 0 or step == 199:
        with torch.no_grad():
            d2 = F.softmax(ol, dim=-1)
            hard = d2.argmax(dim=-1)
            hard_cost = true_gate_cost(hard, gates_L0, H, W, canon=True)
            ent = -(d2 * torch.log(d2.clamp(min=1e-30))).sum(-1).mean().item()
            gs = get_active_gate_states(d2, gates_L0, H, W, threshold=1e-6)
            active = [g['n'] for g in gs]
        print(f"    Step {step:3d}: exact={loss.item():.4f}  hard={hard_cost}  "
              f"ent={ent:.2f}  active={active}")

d_final = F.softmax(ol, dim=-1)
hard_final = d_final.argmax(dim=-1)
final_cost = true_gate_cost(hard_final, gates_L0, H, W, canon=True)
print(f"\n  Result: {init_true} → {final_cost}  "
      f"({'✓ improved!' if final_cost < init_true else '= same' if final_cost == init_true else '✗ worse'})")

# Collision check
cells_used = hard_final.tolist()
if len(set(cells_used)) < len(cells_used):
    from collections import Counter
    dupes = {c: cnt for c, cnt in Counter(cells_used).items() if cnt > 1}
    print(f"  ⚠ COLLISIONS: {dupes}")

print("\n  Final atom placements:")
for q in range(N):
    bc = hard_final[q].item()
    r, c = bc // W, bc % W
    o_r, o_c = pos[q].tolist()
    conf = d_final[q].max().item()
    tag = "" if (r == o_r and c == o_c) else f" ← MOVED from ({o_r},{o_c})"
    print(f"    q{q:2d}: ({r},{c})  conf={conf:.1%}{tag}")