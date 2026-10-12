#!/usr/bin/env python3
"""
Analyze candidate coherence spectra for the PRL majorization experiment.
Tasks:
1. Verify majorization relations for proposed comparable and incomparable pairs.
2. Compute partial-sum crossings for incomparable pairs.
3. Compute power-distribution polytope geometry: vertices, protrusion margins.
4. Search for stronger incomparable pairs with larger protrusion.
5. Also analyze n=3 PRL example spectra (rho_a through rho_e) as sanity check.

Output: quantitative results to stdout + JSON data file.
"""

import numpy as np
from itertools import permutations
from scipy.spatial import ConvexHull
import json, os

# ============================================================
# 1. Majorization utilities
# ============================================================

def partial_sums(x):
    """Cumulative partial sums of sorted-descending vector."""
    xs = np.sort(x)[::-1]
    return np.cumsum(xs)

def majorizes(x, y, tol=1e-12):
    """Check if x majorizes y (y ≺ x), i.e., y is less coherent than x."""
    n = len(x)
    assert len(y) == n
    px = partial_sums(x)
    py = partial_sums(y)
    # y ≺ x iff sum_1^k y_i^↓ <= sum_1^k x_i^↓ for all k, and total sums equal
    if abs(px[-1] - py[-1]) > tol:
        return False
    return all(py[k] <= px[k] + tol for k in range(n-1))

def majorization_relation(lam1, lam2):
    """
    Returns one of: 'equal', 'lam1_maj_lam2' (lam2 ≺ lam1),
    'lam2_maj_lam1' (lam1 ≺ lam2), 'incomparable'.
    """
    if np.allclose(np.sort(lam1)[::-1], np.sort(lam2)[::-1]):
        return 'equal'
    if majorizes(lam1, lam2):
        return 'lam1_maj_lam2'  # lam2 ≺ lam1, lam1 more coherent
    if majorizes(lam2, lam1):
        return 'lam2_maj_lam1'  # lam1 ≺ lam2, lam2 more coherent
    return 'incomparable'

def partial_sum_crossings(lam1, lam2):
    """
    For incomparable pair, find at which k the partial sums cross.
    Returns list of dicts with crossing info.
    """
    n = len(lam1)
    ps1 = partial_sums(lam1)
    ps2 = partial_sums(lam2)
    diffs = ps1 - ps2  # positive means lam1 partial sum > lam2
    crossings = []
    for k in range(n-1):
        crossings.append({
            'k': k+1,
            'ps_lam1': float(ps1[k]),
            'ps_lam2': float(ps2[k]),
            'diff': float(diffs[k]),
            'lam1_leads': bool(diffs[k] > 0)
        })
    return crossings

# ============================================================
# 2. Power-distribution polytope geometry
# ============================================================

def permutohedron_vertices(lam):
    """
    Vertices of the power-distribution polytope {d} = convex hull of all
    permutations of lambda^↓.
    """
    lam_sorted = tuple(np.sort(lam)[::-1])
    verts = set()
    for p in permutations(lam_sorted):
        verts.add(p)
    return np.array(list(verts))

def project_to_simplex_plane(verts):
    """
    Project n-dim vertices (on the hyperplane sum=1) to (n-1)-dim coordinates.
    Uses first (n-1) components as coordinates (since last is determined).
    """
    return verts[:, :-1]

def polytope_containment_test(verts_inner, verts_outer):
    """
    Test if all vertices of inner polytope lie inside the convex hull of outer polytope.
    Works in projected (n-1)-dim space.
    Returns: (all_inside, fraction_inside, max_violation_distance)
    """
    from scipy.spatial import Delaunay
    proj_inner = project_to_simplex_plane(verts_inner)
    proj_outer = project_to_simplex_plane(verts_outer)
    
    try:
        hull = Delaunay(proj_outer)
        inside = hull.find_simplex(proj_inner) >= 0
        return bool(np.all(inside)), float(np.mean(inside)), 0.0
    except Exception:
        # Fallback for degenerate cases
        return False, 0.0, -1.0

def protrusion_margin(verts_A, verts_B):
    """
    Compute the maximum distance that any vertex of A protrudes outside B's convex hull,
    and vice versa. Works in projected space.
    For incomparable pairs, both directions should have positive protrusion.
    """
    from scipy.spatial import ConvexHull
    
    proj_A = project_to_simplex_plane(verts_A)
    proj_B = project_to_simplex_plane(verts_B)
    
    def max_protrusion_outside(test_pts, hull_pts):
        """Max distance any test point lies outside the convex hull of hull_pts."""
        try:
            hull = ConvexHull(hull_pts)
        except Exception:
            return -1.0
        
        # For each facet of the hull, compute signed distance of test points
        # A point is outside if it violates any inequality
        equations = hull.equations  # (A, b) where Ax + b <= 0 for points inside
        max_violation = 0.0
        for pt in test_pts:
            violations = equations[:, :-1] @ pt + equations[:, -1]
            max_v = np.max(violations)
            if max_v > max_violation:
                max_violation = max_v
        return float(max_violation)
    
    prot_A_outside_B = max_protrusion_outside(proj_A, proj_B)
    prot_B_outside_A = max_protrusion_outside(proj_B, proj_A)
    
    return prot_A_outside_B, prot_B_outside_A

# ============================================================
# 3. Search for strong incomparable pairs
# ============================================================

def random_spectrum(n, rng):
    """Generate a random probability vector sorted descending."""
    x = rng.dirichlet(np.ones(n))
    return np.sort(x)[::-1]

def search_strong_incomparable_pairs(n=4, n_trials=50000, seed=42):
    """
    Search for incomparable pairs with large minimum protrusion margin.
    """
    rng = np.random.default_rng(seed)
    best_margin = 0.0
    best_pair = None
    
    for trial in range(n_trials):
        lam1 = random_spectrum(n, rng)
        lam2 = random_spectrum(n, rng)
        
        rel = majorization_relation(lam1, lam2)
        if rel != 'incomparable':
            continue
        
        verts1 = permutohedron_vertices(lam1)
        verts2 = permutohedron_vertices(lam2)
        
        try:
            p12, p21 = protrusion_margin(verts1, verts2)
        except Exception:
            continue
        
        min_prot = min(p12, p21)
        if min_prot > best_margin:
            best_margin = min_prot
            best_pair = (lam1.copy(), lam2.copy(), p12, p21)
    
    return best_pair, best_margin

# ============================================================
# 4. Main analysis
# ============================================================

def analyze_pair(name, lam1, lam2):
    """Full analysis of a spectrum pair."""
    print(f"\n{'='*60}")
    print(f"Pair: {name}")
    print(f"  lam1 = {np.round(lam1, 6)}")
    print(f"  lam2 = {np.round(lam2, 6)}")
    
    rel = majorization_relation(lam1, lam2)
    print(f"  Relation: {rel}")
    
    crossings = partial_sum_crossings(lam1, lam2)
    print(f"  Partial-sum crossings:")
    for c in crossings:
        leader = "lam1" if c['lam1_leads'] else "lam2"
        print(f"    k={c['k']}: ps1={c['ps_lam1']:.6f}, ps2={c['ps_lam2']:.6f}, "
              f"diff={c['diff']:+.6f} ({leader} leads)")
    
    verts1 = permutohedron_vertices(lam1)
    verts2 = permutohedron_vertices(lam2)
    print(f"  Polytope vertices: {len(verts1)} (lam1), {len(verts2)} (lam2)")
    
    # Containment
    c12_all, c12_frac, _ = polytope_containment_test(verts1, verts2)
    c21_all, c21_frac, _ = polytope_containment_test(verts2, verts1)
    print(f"  Containment: verts1 inside hull2: {c12_all} ({c12_frac:.2%})")
    print(f"  Containment: verts2 inside hull1: {c21_all} ({c21_frac:.2%})")
    
    # Protrusion
    try:
        p12, p21 = protrusion_margin(verts1, verts2)
        print(f"  Protrusion: lam1 outside lam2 hull: {p12:.6f}")
        print(f"  Protrusion: lam2 outside lam1 hull: {p21:.6f}")
        print(f"  Min protrusion (bidirectional): {min(p12, p21):.6f}")
    except Exception as e:
        p12, p21 = -1, -1
        print(f"  Protrusion computation failed: {e}")
    
    # Entropy
    def entropy(lam):
        lam_pos = lam[lam > 0]
        return -np.sum(lam_pos * np.log(lam_pos))
    
    H1, H2 = entropy(lam1), entropy(lam2)
    print(f"  Entropy: H(lam1)={H1:.6f}, H(lam2)={H2:.6f}")
    
    return {
        'name': name,
        'lam1': lam1.tolist(),
        'lam2': lam2.tolist(),
        'relation': rel,
        'crossings': crossings,
        'n_verts1': len(verts1),
        'n_verts2': len(verts2),
        'containment_1in2': c12_all,
        'containment_2in1': c21_all,
        'protrusion_1_outside_2': float(p12),
        'protrusion_2_outside_1': float(p21),
        'min_protrusion': float(min(p12, p21)),
        'entropy_1': float(H1),
        'entropy_2': float(H2),
    }

def main():
    results = {}
    
    # ---- PRL n=3 sanity check ----
    print("\n" + "#"*60)
    print("# PRL n=3 example: rho_a through rho_e")
    print("#"*60)
    
    rho_a = np.array([1/3, 1/3, 1/3])
    rho_b = np.array([0.50, 0.30, 0.20])
    rho_c = np.array([0.55, 0.35, 0.10])
    rho_d = np.array([1.0, 0.0, 0.0])
    rho_e = np.array([0.55, 0.45, 0.00])
    
    prl_spectra = {'a': rho_a, 'b': rho_b, 'c': rho_c, 'd': rho_d, 'e': rho_e}
    prl_names = list(prl_spectra.keys())
    
    print("\nAll pairwise relations:")
    for i in range(len(prl_names)):
        for j in range(i+1, len(prl_names)):
            n1, n2 = prl_names[i], prl_names[j]
            rel = majorization_relation(prl_spectra[n1], prl_spectra[n2])
            print(f"  rho_{n1} vs rho_{n2}: {rel}")
    
    # Detailed analysis of key pairs
    results['prl_b_vs_e'] = analyze_pair("PRL rho_b vs rho_e (incomparable)",
                                          rho_b, rho_e)
    results['prl_c_vs_e'] = analyze_pair("PRL rho_c vs rho_e (incomparable)",
                                          rho_c, rho_e)
    results['prl_a_vs_b'] = analyze_pair("PRL rho_a vs rho_b (comparable: a≺b)",
                                          rho_a, rho_b)
    
    # ---- Proposed 4-port pairs ----
    print("\n" + "#"*60)
    print("# Proposed 4-port candidate spectra")
    print("#"*60)
    
    # Comparable pair
    lam_C1 = np.array([0.40, 0.30, 0.20, 0.10])
    lam_C2 = np.array([0.70, 0.20, 0.07, 0.03])
    results['4port_comparable'] = analyze_pair("4-port comparable", lam_C1, lam_C2)
    
    # Incomparable pair A
    lam_I1 = np.array([0.52, 0.23, 0.13, 0.12])
    lam_I2 = np.array([0.45, 0.33, 0.20, 0.02])
    results['4port_incomp_A'] = analyze_pair("4-port incomparable A", lam_I1, lam_I2)
    
    # ---- Search for stronger incomparable pairs ----
    print("\n" + "#"*60)
    print("# Searching for strong incomparable 4-port pairs...")
    print("#"*60)
    
    best_pair, best_margin = search_strong_incomparable_pairs(n=4, n_trials=100000, seed=42)
    if best_pair is not None:
        lam_opt1, lam_opt2, p12, p21 = best_pair
        print(f"\nBest found:")
        print(f"  lam1 = {np.round(lam_opt1, 6)}")
        print(f"  lam2 = {np.round(lam_opt2, 6)}")
        print(f"  Protrusion: {p12:.6f} / {p21:.6f}, min={best_margin:.6f}")
        
        results['4port_incomp_optimal'] = analyze_pair(
            "4-port incomparable OPTIMAL", lam_opt1, lam_opt2)
    
    # ---- Also search for n=10 incomparable pairs ----
    print("\n" + "#"*60)
    print("# Searching for strong incomparable 10-port pairs...")
    print("#"*60)
    
    best10, margin10 = search_strong_incomparable_pairs(n=10, n_trials=50000, seed=123)
    if best10 is not None:
        lam10_1, lam10_2, p12_10, p21_10 = best10
        print(f"\nBest found (n=10):")
        print(f"  lam1 = {np.round(lam10_1, 6)}")
        print(f"  lam2 = {np.round(lam10_2, 6)}")
        print(f"  Protrusion: {p12_10:.6f} / {p21_10:.6f}, min={margin10:.6f}")
        results['10port_incomp_optimal'] = {
            'lam1': lam10_1.tolist(),
            'lam2': lam10_2.tolist(),
            'protrusion_1_outside_2': float(p12_10),
            'protrusion_2_outside_1': float(p21_10),
            'min_protrusion': float(margin10),
            'relation': 'incomparable'
        }
    
    # ---- Additional hand-designed strong incomparable pairs for n=4 ----
    print("\n" + "#"*60)
    print("# Hand-designed strong incomparable 4-port pairs")
    print("#"*60)
    
    # Strategy: maximize partial-sum crossing magnitude
    # lam1 has large first component but small second; lam2 has moderate first but large second
    hand_pairs = [
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
    
    for hname, h1, h2 in hand_pairs:
        results[f'4port_{hname}'] = analyze_pair(f"4-port {hname}", h1, h2)
    
    # ---- Summary ----
    print("\n" + "#"*60)
    print("# SUMMARY OF ALL INCOMPARABLE PAIRS")
    print("#"*60)
    
    print(f"\n{'Name':<40} {'Min Prot':>10} {'Prot 1→2':>10} {'Prot 2→1':>10} {'Relation':<15}")
    print("-"*90)
    for key, val in results.items():
        if 'min_protrusion' in val:
            print(f"{val.get('name', key):<40} {val['min_protrusion']:>10.6f} "
                  f"{val['protrusion_1_outside_2']:>10.6f} {val['protrusion_2_outside_1']:>10.6f} "
                  f"{val['relation']:<15}")
    
    # Save results
    outdir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, 'majorization_spectra_analysis.json')
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {outpath}")

if __name__ == '__main__':
    main()
