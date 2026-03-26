"""
GPU-accelerated exact gate cost surrogate v2.

Fixes:
1. Tail correction is now INTEGRAL, not optional — the computation always
   returns a meaningful estimate by renormalizing kept probabilities and
   tracking the bias this introduces.
2. Memory-safe: caps 4D tensor size and falls back to chunked computation.
3. Proper benchmarking with ground truth from CPU float64 (no OOM).
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
                if not aod_compatible_batch(mvs[i],mvs[j]): conflict[i,j]=conflict[j,i]=True
        colors=[-1]*n
        for idx in sorted(range(n), key=lambda x: -conflict[x].sum().item()):
            used={colors[k] for k in range(n) if conflict[idx,k] and colors[k]>=0}
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
# Top-K gate state extraction with RENORMALIZED probabilities
# ─────────────────────────────────────────────────────────────────────────────

def get_topk_gate_states(placement_dists, gate_atoms, H, W, top_k=10):
    """Top-K gate states per gate, renormalized to sum to 1.

    Returns the renormalized distribution over kept states, plus metadata
    about how much mass was discarded.
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

        # Joint probs over kept cells (unnormalized)
        probs_2d = pa_k[:, None] * pb_k[None, :]  # (ka, kb)
        kept_mass = probs_2d.sum()

        # Renormalize to get a valid distribution over kept states
        probs_renorm = probs_2d / kept_mass.clamp(min=1e-30)

        # Moves
        ca = active_a[:, None].expand(ka, kb)
        cb = active_b[None, :].expand(ka, kb)
        moves_fwd = torch.stack([rows[ca], cols[ca], rows[cb], cols[cb]], dim=-1)
        moves_rev = torch.stack([rows[cb], cols[cb], rows[ca], cols[ca]], dim=-1)

        result.append({
            'probs': probs_renorm,     # (ka, kb) sums to 1
            'moves_fwd': moves_fwd,    # (ka, kb, 4)
            'moves_rev': moves_rev,
            'ka': ka, 'kb': kb,
            'kept_mass': kept_mass,
        })

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Compat tables
# ─────────────────────────────────────────────────────────────────────────────

def build_compat_2d(gs_i, gs_j, dir_i, dir_j):
    mi = gs_i['moves_rev' if dir_i else 'moves_fwd']
    mj = gs_j['moves_rev' if dir_j else 'moves_fwd']
    Ki = mi.shape[0] * mi.shape[1]
    Kj = mj.shape[0] * mj.shape[1]
    return aod_compatible_batch(mi.reshape(Ki,4)[:,None,:], mj.reshape(Kj,4)[None,:,:])


# ─────────────────────────────────────────────────────────────────────────────
# Fully vectorized E[χ] on renormalized distributions
# ─────────────────────────────────────────────────────────────────────────────

MAX_ELEMENTS = 500_000_000  # ~2GB at float32, safety limit

def exact_echi_vectorized(gate_states, directions, chi_tbl=None):
    """Fully vectorized E[χ] for M=4. Works on renormalized probs."""
    if chi_tbl is None:
        chi_tbl = chi_table(4)

    gs = gate_states
    dirs = directions
    device = gs[0]['probs'].device
    dtype = gs[0]['probs'].dtype
    chi_tbl = chi_tbl.to(device=device, dtype=dtype)

    p = [g['probs'].reshape(-1) for g in gs]
    K = [pp.shape[0] for pp in p]
    total_elements = math.prod(K)

    if total_elements > MAX_ELEMENTS:
        # Fall back to chunked computation
        return _echi_chunked(gs, dirs, chi_tbl, p, K)

    c01 = (~build_compat_2d(gs[0],gs[1],dirs[0],dirs[1])).to(torch.int8)
    c02 = (~build_compat_2d(gs[0],gs[2],dirs[0],dirs[2])).to(torch.int8)
    c12 = (~build_compat_2d(gs[1],gs[2],dirs[1],dirs[2])).to(torch.int8)
    c03 = (~build_compat_2d(gs[0],gs[3],dirs[0],dirs[3])).to(torch.int8)
    c13 = (~build_compat_2d(gs[1],gs[3],dirs[1],dirs[3])).to(torch.int8)
    c23 = (~build_compat_2d(gs[2],gs[3],dirs[2],dirs[3])).to(torch.int8)

    edge_cfg = (c01[:,:,None,None].to(torch.int16) +
                c02[:,None,:,None].to(torch.int16) * 2 +
                c12[None,:,:,None].to(torch.int16) * 4 +
                c03[:,None,None,:].to(torch.int16) * 8 +
                c13[None,:,None,:].to(torch.int16) * 16 +
                c23[None,None,:,:].to(torch.int16) * 32)

    chi_vals = chi_tbl[edge_cfg.long()]

    joint = (p[0][:,None,None,None] * p[1][None,:,None,None] *
             p[2][None,None,:,None] * p[3][None,None,None,:])

    return (chi_vals * joint).sum()


def _echi_chunked(gs, dirs, chi_tbl, p, K):
    """Chunked fallback: loop over s0, vectorize rest."""
    device = p[0].device
    dtype = p[0].dtype

    c01 = (~build_compat_2d(gs[0],gs[1],dirs[0],dirs[1])).to(torch.int8)
    c02 = (~build_compat_2d(gs[0],gs[2],dirs[0],dirs[2])).to(torch.int8)
    c12 = (~build_compat_2d(gs[1],gs[2],dirs[1],dirs[2])).to(torch.int8)
    c03 = (~build_compat_2d(gs[0],gs[3],dirs[0],dirs[3])).to(torch.int8)
    c13 = (~build_compat_2d(gs[1],gs[3],dirs[1],dirs[3])).to(torch.int8)
    c23 = (~build_compat_2d(gs[2],gs[3],dirs[2],dirs[3])).to(torch.int8)

    result = torch.tensor(0.0, device=device, dtype=dtype)

    for i0 in range(K[0]):
        if p[0][i0] < 1e-30:
            continue

        edge_cfg = (c01[i0,:,None,None].to(torch.int16) +
                    c02[i0,None,:,None].to(torch.int16) * 2 +
                    c12[:,:,None].to(torch.int16) * 4 +
                    c03[i0,None,None,:].to(torch.int16) * 8 +
                    c13[:,None,:].to(torch.int16) * 16 +
                    c23[None,:,:].to(torch.int16) * 32)

        chi_vals = chi_tbl[edge_cfg.long()]  # (K1, K2, K3)

        joint_123 = (p[1][:,None,None] * p[2][None,:,None] * p[3][None,None,:])
        result = result + p[0][i0] * (chi_vals * joint_123).sum()

    return result


def exact_echi_canon(gate_states, chi_tbl=None):
    """E[min_d χ] with canonicalization. Element-wise min across directions."""
    if chi_tbl is None:
        chi_tbl = chi_table(4)

    gs = gate_states
    M = 4
    device = gs[0]['probs'].device
    dtype = gs[0]['probs'].dtype
    chi_tbl_d = chi_tbl.to(device=device, dtype=dtype)

    p = [g['probs'].reshape(-1) for g in gs]
    K = [pp.shape[0] for pp in p]
    total = math.prod(K)

    joint = (p[0][:,None,None,None] * p[1][None,:,None,None] *
             p[2][None,None,:,None] * p[3][None,None,None,:])

    use_full = total <= MAX_ELEMENTS

    if use_full:
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
            chi_vals = chi_tbl_d[edge_cfg.long()]

            if min_chi is None:
                min_chi = chi_vals
            else:
                min_chi = torch.min(min_chi, chi_vals)

        return (min_chi * joint).sum()
    else:
        # Chunked: loop over s0, take min per direction for each chunk
        result = torch.tensor(0.0, device=device, dtype=dtype)

        # Precompute all compat tables
        compat = {}
        for i in range(M):
            for j in range(i+1,M):
                for di in [False,True]:
                    for dj in [False,True]:
                        compat[(i,j,di,dj)] = (~build_compat_2d(
                            gs[i],gs[j],di,dj)).to(torch.int8)

        for i0 in range(K[0]):
            if p[0][i0] < 1e-30:
                continue

            min_chi_chunk = None
            for dir_idx in range(1 << M):
                dirs = [bool((dir_idx>>g)&1) for g in range(M)]
                d0,d1,d2,d3 = dirs

                edge_cfg = (compat[(0,1,d0,d1)][i0,:,None,None].to(torch.int16) +
                            compat[(0,2,d0,d2)][i0,None,:,None].to(torch.int16)*2 +
                            compat[(1,2,d1,d2)][:,:,None].to(torch.int16)*4 +
                            compat[(0,3,d0,d3)][i0,None,None,:].to(torch.int16)*8 +
                            compat[(1,3,d1,d3)][:,None,:].to(torch.int16)*16 +
                            compat[(2,3,d2,d3)][None,:,:].to(torch.int16)*32)
                chi_vals = chi_tbl_d[edge_cfg.long()]

                if min_chi_chunk is None:
                    min_chi_chunk = chi_vals
                else:
                    min_chi_chunk = torch.min(min_chi_chunk, chi_vals)

            joint_123 = p[1][:,None,None] * p[2][None,:,None] * p[3][None,None,:]
            result = result + p[0][i0] * (min_chi_chunk * joint_123).sum()

        return result


# ─────────────────────────────────────────────────────────────────────────────
# Full gate cost with renormalization
# ─────────────────────────────────────────────────────────────────────────────

def exact_gate_cost(gate_atoms, placement_dists, H, W,
                    canonicalize=True, top_k=10):
    """Gate cost: 2 * E[χ] using renormalized top-K distributions.

    The computation uses renormalized distributions (sum to 1 over kept
    states), which gives E_renorm[χ]. This is the expected χ conditioned
    on all gates being in their top-K states.

    For well-chosen top_k, the kept mass is high and E_renorm[χ] ≈ E_true[χ].
    """
    gs = get_topk_gate_states(placement_dists, gate_atoms, H, W, top_k)

    if canonicalize:
        chi = exact_echi_canon(gs)
    else:
        chi = exact_echi_vectorized(gs, [False]*4)

    return 2.0 * chi


# ═════════════════════════════════════════════════════════════════════════════
# Setup
# ═════════════════════════════════════════════════════════════════════════════

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE = torch.float32
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
    v = exact_gate_cost(gates_L0, onehot, H, W, canonicalize=canon, top_k=5)
    true = true_gate_cost(pos[:,0]*W+pos[:,1], gates_L0, H, W, canon=canon)
    print(f"  canon={canon}: exact={v.item():.4f}  true={true}  "
          f"{'✓' if abs(v.item()-true)<0.05 else '✗'}")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 2: Speed + bias (using CPU float64 as ground truth)
# ═════════════════════════════════════════════════════════════════════════════

print(f"\n{'═'*65}")
print("TEST 2: Speed and bias at varying sharpness")
print("═"*65)

# Compute CPU ground truth at top_k=25 (full, no pruning)
def cpu_ground_truth(gates, dists_cpu, H, W, canon=False):
    """Exact E[χ] on CPU with full enumeration (no pruning)."""
    gs = get_topk_gate_states(dists_cpu, gates, H, W, top_k=25)
    if canon:
        return 2.0 * exact_echi_canon(gs)
    else:
        return 2.0 * exact_echi_vectorized(gs, [False]*4)


torch.manual_seed(42)

for sigma in [0.3, 1.5, 4.0, 10.0]:
    logits_cpu = torch.randn(N, C, dtype=torch.float64) * 0.1
    for q in range(N):
        logits_cpu[q, pos[q,0].cpu()*W + pos[q,1].cpu()] += sigma
    d_cpu = F.softmax(logits_cpu, dim=-1)
    ent = -(d_cpu * torch.log(d_cpu.clamp(min=1e-30))).sum(-1).mean().item()

    # CPU ground truth
    t0 = time.time()
    gt = cpu_ground_truth(gates_L0, d_cpu, H, W, canon=False)
    gt_time = time.time() - t0

    # GPU at various top_k
    d_gpu = d_cpu.to(device=DEVICE, dtype=DTYPE)

    print(f"\n  σ={sigma}  (entropy={ent:.2f}, GT={gt.item():.4f}, GT_time={gt_time:.1f}s)")

    for top_k in [5, 8, 10, 12, 15]:
        gs = get_topk_gate_states(d_gpu, gates_L0, H, W, top_k)
        K_total = math.prod([g['probs'].numel() for g in gs])
        kept = [g['kept_mass'].item() for g in gs]
        min_kept = min(kept)

        if K_total > MAX_ELEMENTS * 2:
            print(f"    top_k={top_k:2d}: SKIP (too large: {K_total:,})")
            continue

        # Warmup
        try:
            _ = exact_gate_cost(gates_L0, d_gpu, H, W, canonicalize=False, top_k=top_k)
            if DEVICE.type == 'cuda': torch.cuda.synchronize()
        except RuntimeError:
            print(f"    top_k={top_k:2d}: OOM"); continue

        # Timed (3 runs)
        times = []
        for _ in range(3):
            if DEVICE.type == 'cuda': torch.cuda.synchronize()
            t0 = time.time()
            v = exact_gate_cost(gates_L0, d_gpu, H, W, canonicalize=False, top_k=top_k)
            if DEVICE.type == 'cuda': torch.cuda.synchronize()
            times.append(time.time()-t0)

        avg_t = sum(times)/len(times)
        bias = v.item() - gt.item()
        print(f"    top_k={top_k:2d}: K={K_total:>12,}  time={avg_t:.4f}s  "
              f"val={v.item():.4f}  bias={bias:+.4f}  min_kept={min_kept:.4f}")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 3: Gradient check (float64 CPU)
# ═════════════════════════════════════════════════════════════════════════════

print(f"\n{'═'*65}")
print("TEST 3: Gradient check (float64 CPU)")
print("═"*65)

torch.manual_seed(99)
base_logits = torch.randn(N, C, dtype=torch.float64) * 0.15
for q in range(N):
    base_logits[q, pos[q,0].cpu()*W + pos[q,1].cpu()] += 4.0

TOP_K = 10

gl = base_logits.clone().requires_grad_(True)
d = F.softmax(gl, dim=-1)
cost = exact_gate_cost(gates_L0, d, H, W, canonicalize=False, top_k=TOP_K)
cost.backward()
ag = gl.grad.clone()

print(f"  Cost: {cost.item():.6f}")
print(f"  Grad norm: {ag.norm().item():.8f}")

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
                              canonicalize=False, top_k=TOP_K)
        lm = base_logits.clone(); lm[qi,c] -= eps
        cm = exact_gate_cost(gates_L0, F.softmax(lm, dim=-1), H, W,
                              canonicalize=False, top_k=TOP_K)
        a_list.append(ag[qi,c].item())
        n_list.append(((cp-cm)/(2*eps)).item())

t_fd = time.time()-t0
av = torch.tensor(a_list)
nv = torch.tensor(n_list)
cos = F.cosine_similarity(av.unsqueeze(0), nv.unsqueeze(0)).item()
print(f"  Done in {t_fd:.1f}s")
print(f"  Cosine sim: {cos:.6f}  {'✓' if cos>0.99 else '⚠' if cos>0.9 else '✗'}")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 4: Speed with canonicalization
# ═════════════════════════════════════════════════════════════════════════════

print(f"\n{'═'*65}")
print("TEST 4: Speed with canonicalization")
print("═"*65)

torch.manual_seed(42)
for sigma in [4.0, 10.0]:
    logits = torch.randn(N, C, dtype=DTYPE, device=DEVICE) * 0.1
    for q in range(N):
        logits[q, pos[q,0]*W + pos[q,1]] += sigma
    d = F.softmax(logits, dim=-1)

    for top_k in [8, 10, 12]:
        try:
            _ = exact_gate_cost(gates_L0, d, H, W, canonicalize=True, top_k=top_k)
            if DEVICE.type == 'cuda': torch.cuda.synchronize()
        except RuntimeError:
            print(f"  σ={sigma} top_k={top_k}: OOM"); continue

        times = []
        for _ in range(3):
            if DEVICE.type == 'cuda': torch.cuda.synchronize()
            t0 = time.time()
            v = exact_gate_cost(gates_L0, d, H, W, canonicalize=True, top_k=top_k)
            if DEVICE.type == 'cuda': torch.cuda.synchronize()
            times.append(time.time()-t0)
        print(f"  σ={sigma} top_k={top_k:2d}: {sum(times)/len(times):.4f}s  cost={v.item():.4f}")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 5: Optimization
# ═════════════════════════════════════════════════════════════════════════════

print(f"\n{'═'*65}")
print("TEST 5: Optimization")
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

t_start = time.time()
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
            elapsed = time.time() - t_start
        print(f"    Step {step:3d}: loss={loss.item():.4f}  hard={hard_cost}  "
              f"ent={ent:.2f}  wall={elapsed:.1f}s")

d_final = F.softmax(ol, dim=-1)
hard_final = d_final.argmax(dim=-1)
final_cost = true_gate_cost(hard_final, gates_L0, H, W, canon=True)
total_time = time.time() - t_start
print(f"\n  Result: {init_true} → {final_cost}  ({total_time:.1f}s total, "
      f"{total_time/200:.3f}s/step)")

cells_used = hard_final.tolist()
if len(set(cells_used)) < len(cells_used):
    from collections import Counter
    dupes = {c:cnt for c,cnt in Counter(cells_used).items() if cnt>1}
    print(f"  ⚠ COLLISIONS: {dupes}")
else:
    print(f"  ✓ No collisions")

print("\n  Final placements:")
for q in range(N):
    bc = hard_final[q].item()
    r,c = bc//W, bc%W
    o_r,o_c = pos[q].cpu().tolist()
    conf = d_final[q].max().item()
    tag = "" if (r==o_r and c==o_c) else f" ← from ({o_r},{o_c})"
    print(f"    q{q:2d}: ({r},{c})  {conf:.1%}{tag}")