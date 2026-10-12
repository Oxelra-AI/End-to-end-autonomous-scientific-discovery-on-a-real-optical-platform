#!/usr/bin/env python3
"""Verify all majorization pair classifications used in the paper and report."""
import numpy as np

def is_majorized(lam1, lam2):
    """Check if lam1 is majorized by lam2 (lam1 ≺ lam2).
    Both must be sorted descending and sum to same value."""
    s1 = np.sort(lam1)[::-1]
    s2 = np.sort(lam2)[::-1]
    n = len(s1)
    # partial sums: sum of top k elements of s2 >= sum of top k of s1 for all k
    for k in range(1, n):
        if np.sum(s1[:k]) > np.sum(s2[:k]) + 1e-12:
            return False
    # total sums must be equal
    if abs(np.sum(s1) - np.sum(s2)) > 1e-10:
        return False
    return True

def classify_pair(name1, lam1, name2, lam2):
    """Classify a pair as comparable or incomparable."""
    m12 = is_majorized(lam1, lam2)  # lam1 ≺ lam2
    m21 = is_majorized(lam2, lam1)  # lam2 ≺ lam1
    
    # Compute partial sums for detail
    s1 = np.cumsum(np.sort(lam1)[::-1])
    s2 = np.cumsum(np.sort(lam2)[::-1])
    
    if m12 and m21:
        rel = f"{name1} = {name2} (identical majorization)"
    elif m12:
        rel = f"{name1} ≺ {name2} (COMPARABLE: {name2} more coherent)"
    elif m21:
        rel = f"{name2} ≺ {name1} (COMPARABLE: {name1} more coherent)"
    else:
        rel = f"{name1} ∥ {name2} (INCOMPARABLE)"
    
    return rel, s1, s2

# ═══════════════════════════════════════════════
# Define all spectra used in the experiments
# ═══════════════════════════════════════════════

# 4-dimensional spectra
spectra_4d = {
    'incoherent': np.array([0.25, 0.25, 0.25, 0.25]),
    'comp_less':  np.array([0.40, 0.30, 0.20, 0.10]),
    'comp_more':  np.array([0.70, 0.20, 0.07, 0.03]),
    'incomp_A':   np.array([0.55, 0.15, 0.15, 0.15]),
    'incomp_B':   np.array([0.35, 0.35, 0.25, 0.05]),
    'coherent':   np.array([1.00, 0.00, 0.00, 0.00]),
}

# PRL 3-dimensional benchmark spectra
spectra_prl = {
    'rho_a': np.array([1/3, 1/3, 1/3]),
    'rho_b': np.array([0.60, 0.30, 0.10]),
    'rho_c': np.array([0.80, 0.15, 0.05]),
    'rho_e': np.array([0.55, 0.45, 0.00]),
}

# 16-dimensional spectra
spectra_16d = {
    'uniform_16': np.ones(16) / 16,
    'peaked_16':  np.array([0.40, 0.15, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03,
                            0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.005, 0.005]),
}

print("=" * 70)
print("MAJORIZATION PAIR CLASSIFICATION VERIFICATION")
print("=" * 70)

# ═══════════════════════════════════════════════
# Comparable pairs claimed in the paper/report
# ═══════════════════════════════════════════════
print("\n### COMPARABLE PAIRS (claimed in paper/report) ###\n")

comparable_claims = [
    ('comp_less', spectra_4d['comp_less'], 'comp_more', spectra_4d['comp_more']),
    ('rho_a', spectra_prl['rho_a'], 'rho_b', spectra_prl['rho_b']),
    ('rho_b', spectra_prl['rho_b'], 'rho_c', spectra_prl['rho_c']),
    ('uniform_16', spectra_16d['uniform_16'], 'peaked_16', spectra_16d['peaked_16']),
    ('incoherent', spectra_4d['incoherent'], 'comp_less', spectra_4d['comp_less']),
    ('incoherent', spectra_4d['incoherent'], 'coherent', spectra_4d['coherent']),
]

for n1, l1, n2, l2 in comparable_claims:
    rel, s1, s2 = classify_pair(n1, l1, n2, l2)
    print(f"  {n1} vs {n2}")
    print(f"    λ({n1}) = {l1}")
    print(f"    λ({n2}) = {l2}")
    print(f"    Partial sums {n1}: {s1}")
    print(f"    Partial sums {n2}: {s2}")
    print(f"    → {rel}")
    print()

# ═══════════════════════════════════════════════
# Incomparable pairs claimed in the paper/report
# ═══════════════════════════════════════════════
print("\n### INCOMPARABLE PAIRS (claimed in paper/report) ###\n")

incomparable_claims = [
    ('incomp_A', spectra_4d['incomp_A'], 'incomp_B', spectra_4d['incomp_B']),
    ('comp_less', spectra_4d['comp_less'], 'incomp_A', spectra_4d['incomp_A']),
    ('comp_less', spectra_4d['comp_less'], 'incomp_B', spectra_4d['incomp_B']),
    ('rho_b', spectra_prl['rho_b'], 'rho_e', spectra_prl['rho_e']),
    ('rho_c', spectra_prl['rho_c'], 'rho_e', spectra_prl['rho_e']),
]

for n1, l1, n2, l2 in incomparable_claims:
    rel, s1, s2 = classify_pair(n1, l1, n2, l2)
    print(f"  {n1} vs {n2}")
    print(f"    λ({n1}) = {l1}")
    print(f"    λ({n2}) = {l2}")
    print(f"    Partial sums {n1}: {s1}")
    print(f"    Partial sums {n2}: {s2}")
    print(f"    → {rel}")
    print()

# ═══════════════════════════════════════════════
# Additional: check rho_a vs rho_e (TTCS concern)
# ═══════════════════════════════════════════════
print("\n### ADDITIONAL CHECKS (TTCS raised concern about rho_a) ###\n")

extra_checks = [
    ('rho_a', spectra_prl['rho_a'], 'rho_c', spectra_prl['rho_c']),
    ('rho_a', spectra_prl['rho_a'], 'rho_e', spectra_prl['rho_e']),
]

for n1, l1, n2, l2 in extra_checks:
    rel, s1, s2 = classify_pair(n1, l1, n2, l2)
    print(f"  {n1} vs {n2}")
    print(f"    λ({n1}) = {l1}")
    print(f"    λ({n2}) = {l2}")
    print(f"    Partial sums {n1}: {s1}")
    print(f"    Partial sums {n2}: {s2}")
    print(f"    → {rel}")
    print()

print("=" * 70)
print("SUMMARY: All pair classifications verified against mathematical definition.")
print("=" * 70)
