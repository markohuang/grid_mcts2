"""Quick test of DPQA on small synthetic problems to understand output format and timing."""
import sys, json, time, tempfile, os
sys.path.insert(0, '/home/marko/DPQA')

from solve import DPQA

def run_dpqa(name, program, n_qubits, arch, tmpdir, commute=False):
    t0 = time.time()
    solver = DPQA(name, dir=tmpdir + '/', print_detail=False)
    solver.setArchitecture(arch)
    solver.setProgram(program, nqubit=n_qubits)
    if commute:
        solver.setCommutation()
    solver.hybrid_strategy()
    result = solver.solve(save_file=True)
    elapsed = time.time() - t0
    return result, elapsed

def show_result(result, elapsed):
    print(f"  n_t (stages): {result['n_t']}, n_q: {result['n_q']}, duration: {elapsed:.2f}s")
    layers = result['layers']
    print(f"  num layers in output: {len(layers)}")
    if layers:
        l0 = layers[0]
        print(f"  layer[0] keys: {list(l0.keys())}")
        print(f"  layer[0] gates: {l0.get('gates', [])}")
        # Show first 3 qubits positions
        for q in l0['qubits'][:3]:
            print(f"    qubit {q['id']}: pos=({q['x']},{q['y']}), aod={q['a']}, c={q['c']}, r={q['r']}")
    # Show gate assignments across stages
    print("  Gate-to-stage mapping:")
    for i, layer in enumerate(layers):
        gates = layer.get('gates', [])
        if gates:
            print(f"    stage {i}: gates {gates}")

# ─── Test 1: 3x3, 3 layers, 2 gates/layer ───────────────────────────────────
# Qubits: 0,1,2,3,4,5
# Layer 0: (0,1), (2,3)
# Layer 1: (1,2), (4,5)
# Layer 2: (0,3), (2,5)
program_3x3 = [(0,1),(2,3),(1,2),(4,5),(0,3),(2,5)]
# layer_of[g] = which task layer gate g belongs to
layer_of_3x3 = [0,0, 1,1, 2,2]

print("\n=== Test 1: 6 qubits, 3x3 board, 3 layers x 2 gates ===")
with tempfile.TemporaryDirectory() as tmpdir:
    result, elapsed = run_dpqa("test_3x3", program_3x3, 6, [3,3,3,3], tmpdir)
    show_result(result, elapsed)
    # Save full JSON for inspection
    with open('/tmp/dpqa_test_3x3.json', 'w') as f:
        json.dump(result, f, indent=2)

# ─── Test 2: 5x5, 3 layers, 4 gates/layer ───────────────────────────────────
# Qubits: 0..11
# Layer 0: (0,1),(2,3),(4,5),(6,7)
# Layer 1: (1,2),(3,4),(5,6),(8,9)
# Layer 2: (0,3),(2,5),(7,10),(9,11)
program_5x5 = [(0,1),(2,3),(4,5),(6,7), (1,2),(3,4),(5,6),(8,9), (0,3),(2,5),(7,10),(9,11)]
layer_of_5x5 = [0,0,0,0, 1,1,1,1, 2,2,2,2]

print("\n=== Test 2: 12 qubits, 5x5 board, 3 layers x 4 gates ===")
with tempfile.TemporaryDirectory() as tmpdir:
    result, elapsed = run_dpqa("test_5x5", program_5x5, 12, [5,5,5,5], tmpdir)
    show_result(result, elapsed)
    with open('/tmp/dpqa_test_5x5.json', 'w') as f:
        json.dump(result, f, indent=2)

print("\nDone. Full JSONs at /tmp/dpqa_test_3x3.json and /tmp/dpqa_test_5x5.json")
