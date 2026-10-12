#!/usr/bin/env python3
"""
Analyze candidate coherence spectra for PRL majorization experiment. v2 — optimized.
"""
import numpy as np
from itertools import permutations
from scipy.spatial import ConvexHull
import json, os

# ============================================================
# 1. Majorization utilities
# ============================================================
def partial_sums(x):
    xs = np.sort(x)[::-1]
    return np.cumsum(xs)

def majorizes(x, y, tol=1e-12):
    """Check if x majorizes y (y ≺ x)."""
    n = len(x)
    px, py = partial_sums(x), partial_sums(y)
    if abs(px[-1] - py[-1]) > tol:
        return False
    return all(py[k] <= px[k] + tol for k in range(n-1))

def majorization_relation(lam1, lam2):
    if np.allclose(np.sort(lam1)[::-1], np.sort(lam2)[::-1]):
        return 'equal'
    if majorizes(lam1, lam2):
        return 'lam1_maj_lam2'
    if majorizes(lam2, lam1):
        return 'lam2_maj_lam1'
    return 'incomparable'

def partial_sum_crossings(lam1, lam2):
    n = len(lam1)
    ps1, ps2 = partial_sums(lam1), partial_sums(lam2)
    diffs = ps1 - ps2
    crossings = []
    for k in range(n-1):
        crossings.append({
            'k': k+1, 'ps_lam1': round(float(ps1[k]),6),
            'ps_lam2': round(float(ps2[k]),6),
            'diff': round(float(diffs[k]),6),
            'lam1_leads': bool(diffs[k] > 0)
        })
    return crossings

# ============================================================
# 2. Polytope geometry (optimized — use unique permutations)
# ============================================================
def permutohedron_vertices(lam):
    lam_sorted = tuple(sorted(lam, reverse=True))
    verts = set()
    for p in permutations(lam_sorted):
        verts.add(p)
    return np.array(list(verts))

def protrusion_margin(verts_A, verts_B):
    """Max distance any vertex of A protrudes outside B's hull, and vice versa."""
    proj_A = verts_A[:, :-1]
    proj_B = verts_B[:, :-1]
    
    def max_prot(test_pts, hull_pts):
        try:
            hull = ConvexHull(hull_pts)
        except Exception:
            return -1.0
        eqs = hull.equations
        max_v = 0.0
        for pt in test_pts:
            v = np.max(eqs[:, :-1] @ pt + eqs[:, -1])
            if v > max_v:
                max_v = v
        return float(max_v)
    
    return max_prot(proj_A, proj_B), max_prot(proj_B, proj_A)

def containment_check(verts_inner, verts_outer):
    """Check if all inner vertices are inside outer hull."""
    from scipy.spatial import Delaunay
    proj_in = verts_inner[:, :-1]
    proj_out = verts_outer[:, :-1]
    try:
        hull = Delaunay(proj_out)
        inside = hull.find_simplex(proj_in) >= 0
        return bool(np.all(inside)), float(np.mean(inside))
    except:
        return False, 0.0

# ============================================================
# 3. Fast search for strong incomparable pairs (n=4 only)
# ============================================================
def fast_incomp_check(lam1, lam2):
    """Quick check: incomparable iff partial sums cross."""
    ps1, ps2 = partial_sums(lam1), partial_sums(lam2)
    d = ps1[:-1] - ps2[:-1]
    return bool(np.any(d > 1e-12) and np.any(d < -1e-12))

def fast_protrusion_4d(lam1, lam2):
    """Fast protrusion for n=4: compute directly."""
    v1 = permutohedron_vertices(lam1)
    v2 = permutohedron_vertices(lam2)
    return protrusion_margin(v1, v2)

def search_strong_incomparable(n=4, n_trials=10000, seed=42):
    rng = np.random.default_rng(seed)
    best_margin = 0.0
    best_pair = None
    
    for _ in range(n_trials):
        lam1 = np.sort(rng.dirichlet(np.ones(n)))[::-1]
        lam2 = np.sort(rng.dirichlet(np.ones(n)))[::-1]
        
        if not fast_incomp_check(lam1, lam2):
            continue
        
        try:
            p12, p21 = fast_protrusion_4d(lam1, lam2)
        except:
            continue
        
        mm = min(p12, p21)
        if mm > best_margin:
            best_margin = mm
            best_pair = (lam1.copy(), lam2.copy(), p12, p21)
    
    return best_pair, best_margin

# ============================================================
# 4. Analysis driver
# ============================================================
def entropy(lam):
    lam_pos = lam[lam > 1e-15]
    return float(-np.sum(lam_pos * np.log(lam_pos)))

def analyze_pair(name, lam1, lam2):
    print(f"\n{'='*60}")
    print(f"Pair: {name}")
    print(f"  lam1 = {np.round(lam1, 6)}")
    print(f"  lam2 = {np.round(lam2, 6)}")
    
    rel = majorization_relation(lam1, lam2)
    print(f"  Relation: {rel}")
    
    crossings = partial_sum_crossings(lam1, lam2)
    print(f"  Partial-sum details:")
    for c in crossings:
        leader = "lam1" if c['lam1_leads'] else "lam2"
        print(f"    k={c['k']}: ps1={c['ps_lam1']:.6f}, ps2={c['ps_lam2']:.6f}, "
              f"diff={c['diff']:+.6f} ({leader} leads)")
    
    v1 = permutohedron_vertices(lam1)
    v2 = permutohedron_vertices(lam2)
    print(f"  Polytope vertices: {len(v1)} / {len(v2)}")
    
    c12, f12 = containment_check(v1, v2)
    c21, f21 = containment_check(v2, v1)
    print(f"  Containment: v1⊂hull2={c12} ({f12:.0%}), v2⊂hull1={c21} ({f21:.0%})")
    
    try:
        p12, p21 = protrusion_margin(v1, v2)
        print(f"  Protrusion: lam1→outside2={p12:.6f}, lam2→outside1={p21:.6f}")
        print(f"  Min bidirectional protrusion: {min(p12,p21):.6f}")
    except Exception as e:
        p12, p21 = -1, -1
        print(f"  Protrusion failed: {e}")
    
    H1, H2 = entropy(lam1), entropy(lam2)
    print(f"  Entropy: H1={H1:.6f}, H2={H2:.6f}")
    
    # Crossing magnitude: max |diff| at crossing points
    diffs = [c['diff'] for c in crossings]
    max_pos_diff = max(d for d in diffs) if diffs else 0
    max_neg_diff = min(d for d in diffs) if diffs else 0
    crossing_strength = min(abs(max_pos_diff), abs(max_neg_diff))
    print(f"  Crossing strength (min of |max_pos|,|max_neg|): {crossing_strength:.6f}")
    
    return {
        'name': name, 'lam1': lam1.tolist(), 'lam2': lam2.tolist(),
        'relation': rel, 'crossings': crossings,
        'n_verts1': len(v1), 'n_verts2': len(v2),
        'containment_1in2': c12, 'containment_2in1': c21,
        'protrusion_1_outside_2': float(p12),
        'protrusion_2_outside_1': float(p21),
        'min_protrusion': float(min(p12, p21)),
        'entropy_1': H1, 'entropy_2': H2,
        'crossing_strength': crossing_strength,
    }

def main():
    results = {}
    
    # ---- PRL n=3 sanity check ----
    print("#"*60)
    print("# PRL n=3 example: rho_a through rho_e")
    print("#"*60)
    
    prl = {
        'a': np.array([1/3, 1/3, 1/3]),  # fully incoherent
        'b': np.array([0.60, 0.30, 0.10]),
        'c': np.array([0.80, 0.15, 0.05]),
        'd': np.array([1.0, 0.0, 0.0]),     # fully coherent
        'e': np.array([0.55, 0.45, 0.00]),
    }
    names = list(prl.keys())
    
    print("\nAll pairwise relations:")
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            n1, n2 = names[i], names[j]
            rel = majorization_relation(prl[n1], prl[n2])
            print(f"  rho_{n1} vs rho_{n2}: {rel}")
    
    results['prl_b_vs_e'] = analyze_pair("PRL b vs e (incomp)", prl['b'], prl['e'])
    results['prl_c_vs_e'] = analyze_pair("PRL c vs e (incomp)", prl['c'], prl['e'])
    results['prl_a_vs_b'] = analyze_pair("PRL a vs b (comp: a≺b)", prl['a'], prl['b'])
    results['prl_b_vs_c'] = analyze_pair("PRL b vs c (comp: b≺c)", prl['b'], prl['c'])
    
    # ---- Proposed 4-port pairs ----
    print("\n" + "#"*60)
    print("# Proposed 4-port candidate spectra")
    print("#"*60)
    
    lam_C1 = np.array([0.40, 0.30, 0.20, 0.10])
    lam_C2 = np.array([0.70, 0.20, 0.07, 0.03])
    results['4port_comparable'] = analyze_pair("4-port comparable", lam_C1, lam_C2)
    
    lam_I1 = np.array([0.52, 0.23, 0.13, 0.12])
    lam_I2 = np.array([0.45, 0.33, 0.20, 0.02])
    results['4port_incomp_A'] = analyze_pair("4-port incomp A", lam_I1, lam_I2)
    
    # ---- Hand-designed strong incomparable pairs ----
    print("\n" + "#"*60)
    print("# Hand-designed 4-port incomparable pairs")
    print("#"*60)
    
    hand = [
        ("Hand-A", np.array([0.60, 0.15, 0.15, 0.10]),
                   np.array([0.40, 0.35, 0.15, 0.10])),
        ("Hand-B", np.array([0.55, 0.15, 0.15, 0.15]),
                   np.array([0.35, 0.35, 0.25, 0.05])),
        ("Hand-C", np.array([0.50, 0.20, 0.20, 0.10]),
                   np.array([0.40, 0.40, 0.10, 0.10])),
        ("Hand-D", np.array([0.55, 0.20, 0.15, 0.10]),
                   np.array([0.40, 0.35, 0.20, 0.05])),
        ("Hand-E", np.array([0.60, 0.20, 0.10, 0.10]),
                   np.array([0.40, 0.30, 0.25, 0.05])),
    ]
    
    for hname, h1, h2 in hand:
        results[f'4port_{hname}'] = analyze_pair(f"4-port {hname}", h1, h2)
    
    # ---- Random search (reduced) ----
    print("\n" + "#"*60)
    print("# Random search for strong 4-port incomparable pairs (5000 trials)...")
    print("#"*60)
    
    best_pair, best_margin = search_strong_incomparable(n=4, n_trials=5000, seed=42)
    if best_pair is not None:
        lam_opt1, lam_opt2, p12, p21 = best_pair
        results['4port_search_best'] = analyze_pair(
            "4-port SEARCH BEST", lam_opt1, lam_opt2)
    
    # ---- SUMMARY ----
    print("\n" + "#"*60)
    print("# SUMMARY TABLE")
    print("#"*60)
    print(f"\n{'Name':<35} {'Relation':<18} {'MinProt':>9} {'CrossStr':>9} {'H1':>8} {'H2':>8}")
    print("-"*95)
    for key, val in results.items():
        mp = val.get('min_protrusion', -1)
        cs = val.get('crossing_strength', -1)
        print(f"{val.get('name',key):<35} {val['relation']:<18} {mp:>9.6f} {cs:>9.6f} "
              f"{val.get('entropy_1',0):>8.4f} {val.get('entropy_2',0):>8.4f}")
    
    # Save
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, 'majorization_spectra_analysis.json')
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {outpath}")

if __name__ == '__main__':
    main()
