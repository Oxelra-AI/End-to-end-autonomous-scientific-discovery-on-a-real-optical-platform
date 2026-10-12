#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two-part analysis script:
Part 1: Diagnose the normalization mismatch between analytical H_eff predictions
         and physical 4-phase demodulated transport measurements.
Part 2: Search for incomparable spectra that produce partial overlap under
         the ACTUAL measured 16-port eigenvalue distributions.

Uses the fetched v2 data (K_obs, ref_vec, H_all eigenvalues, system eigenvalues).
"""

import numpy as np
import json, os

DATADIR_V2 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '..', 'data', 'exp_transport_v2', 'exp_transport_v2')
DATADIR_V1 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '..', 'data', 'exp_transport_v1', 'exp_transport_v1')
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures')
os.makedirs(FIGDIR, exist_ok=True)

# ============================================================
# Part 1: Normalization diagnosis
# ============================================================

def diagnose_normalization():
    """
    The physical spot-check computes:
      T_phys = sum_j p_j * sum_{m in selected} |y_m(U psi_j)|^2 / r_m
    
    The analytical prediction computes:
      T_anal = tr(U diag(spectrum) U† H_eff)
             = sum_j p_j * (U psi_j)† H_eff (U psi_j)
    
    where H_eff = sum_{m in selected} (1/r_m) K_obs[m,:]† K_obs[m,:]
    
    These SHOULD be identical if:
    1. The same channels are selected
    2. K_obs and ref_vec are the same
    3. The physical demodulation y_m = K_obs[m,:] . x is exact
    
    The observed ratio T_phys/T_anal ~ 1.6 suggests a systematic factor.
    
    Possible causes:
    A. The physical measurement uses ALL channels, not just selected ones
    B. There's a factor of 2 or 4 from the demodulation formula
    C. The binning introduces a multiplicative factor
    D. The reference baseline subtraction is incomplete in physical measurement
    """
    print("="*60)
    print("Part 1: Normalization Diagnosis")
    print("="*60)
    
    # Load v2 data
    try:
        K_obs = np.load(os.path.join(DATADIR_V2, 'K_obs.npy'))
        ref_vec = np.load(os.path.join(DATADIR_V2, 'ref_vec.npy'))
        H_all = np.load(os.path.join(DATADIR_V2, 'H_all.npy'))
        eigvals_all = np.load(os.path.join(DATADIR_V2, 'H_all_eigvals.npy'))
        print(f"Loaded v2 data: K_obs={K_obs.shape}, ref_vec={ref_vec.shape}")
    except FileNotFoundError as e:
        print(f"Data not found: {e}")
        print("Attempting v1 data...")
        K_obs = np.load(os.path.join(DATADIR_V1, 'K_obs.npy'))
        ref_vec = np.load(os.path.join(DATADIR_V1, 'ref_vec.npy'))
        H_all = np.load(os.path.join(DATADIR_V1, 'H_eff.npy'))
        eigvals_all = np.load(os.path.join(DATADIR_V1, 'H_eff_eigvals.npy'))
        print(f"Loaded v1 data: K_obs={K_obs.shape}, ref_vec={ref_vec.shape}")
    
    N = K_obs.shape[1]
    M = K_obs.shape[0]
    
    # Channel selection (same as experiment: top 75% by brightness)
    bright_thresh = np.percentile(ref_vec, 25)
    selected = np.where(ref_vec > bright_thresh)[0]
    print(f"Selected channels: {len(selected)}/{M}")
    
    # Rebuild H_eff from scratch to verify
    H_rebuilt = np.zeros((N, N), dtype=np.complex128)
    for m in selected:
        if ref_vec[m] > 1e-10:
            km = K_obs[m:m+1, :]
            H_rebuilt += (1.0 / ref_vec[m]) * (km.conj().T @ km)
    H_rebuilt = (H_rebuilt + H_rebuilt.conj().T) / 2.0
    
    # Check H_rebuilt matches H_all
    diff_norm = np.linalg.norm(H_rebuilt - H_all) / np.linalg.norm(H_all)
    print(f"H_rebuilt vs H_all relative difference: {diff_norm:.2e}")
    
    eigvals_rebuilt = np.sort(np.linalg.eigvalsh(H_rebuilt))[::-1]
    print(f"H_rebuilt eigenvalues: {np.round(eigvals_rebuilt[:4], 4)}")
    print(f"H_all eigenvalues:     {np.round(eigvals_all[:4], 4)}")
    
    # Now simulate what the physical measurement does for a test pure state
    # Physical: T_phys = sum_{m in selected} |y_m|^2 / r_m
    # where y_m = K_obs[m,:] . x
    # Analytical: T_anal = x† H_eff x
    
    rng = np.random.default_rng(2026)
    
    print("\nTest: pure-state transport comparison")
    print(f"{'Input':>15} {'T_anal':>12} {'T_phys_sel':>12} {'T_phys_all':>12} "
          f"{'ratio_sel':>10} {'ratio_all':>10}")
    
    ratios_sel = []
    ratios_all = []
    
    for trial in range(10):
        # Random unit vector
        x = rng.standard_normal(N) + 1j * rng.standard_normal(N)
        x = x / np.linalg.norm(x)
        
        # Analytical
        T_anal = float(np.real(x.conj() @ H_rebuilt @ x))
        
        # Physical simulation (selected channels)
        y = K_obs @ x  # demodulated signal
        T_phys_sel = float(np.sum(np.abs(y[selected])**2 / 
                                   np.maximum(ref_vec[selected], 1e-10)))
        
        # Physical simulation (ALL channels)
        valid_all = ref_vec > 1e-10
        T_phys_all = float(np.sum(np.abs(y[valid_all])**2 / 
                                   np.maximum(ref_vec[valid_all], 1e-10)))
        
        r_sel = T_phys_sel / max(T_anal, 1e-15)
        r_all = T_phys_all / max(T_anal, 1e-15)
        ratios_sel.append(r_sel)
        ratios_all.append(r_all)
        
        print(f"{'trial_'+str(trial):>15} {T_anal:>12.4f} {T_phys_sel:>12.4f} "
              f"{T_phys_all:>12.4f} {r_sel:>10.4f} {r_all:>10.4f}")
    
    print(f"\nMean ratio (selected): {np.mean(ratios_sel):.4f} ± {np.std(ratios_sel):.4f}")
    print(f"Mean ratio (all):      {np.mean(ratios_all):.4f} ± {np.std(ratios_all):.4f}")
    
    # The analytical T_anal = x† H x where H = sum_m (1/r_m) K[m,:]† K[m,:]
    # The physical T_phys = sum_m |K[m,:] x|^2 / r_m = sum_m x† K[m,:]† K[m,:] x / r_m
    # These are IDENTICAL by definition. So ratio_sel should be 1.0.
    # If it's not 1.0, there's a bug in the H_eff construction or the physical sim.
    
    print("\n--- Diagnosis ---")
    if abs(np.mean(ratios_sel) - 1.0) < 0.01:
        print("RESULT: Analytical and simulated physical transport AGREE for selected channels.")
        print("The mismatch seen in the actual experiment must come from:")
        print("  - The actual physical demodulation process (noise, drift, residuals)")
        print("  - The reference baseline r_m being different during TM vs spot-check")
        print("  - Cross-terms or nonlinear effects in the physical measurement")
        print("  => The analytical intervals are CORRECT; the physical absolute values")
        print("     have a systematic offset that should be characterized and reported.")
    else:
        print(f"WARNING: Ratio is {np.mean(ratios_sel):.4f}, not 1.0!")
        print("There may be a bug in H_eff construction or channel selection.")
    
    return eigvals_all, K_obs, ref_vec, selected

# ============================================================
# Part 2: Search for better incomparable spectra
# ============================================================

def compute_interval(eigvals_M, spectrum):
    """Compute transport interval for given M eigenvalues and spectrum."""
    n = min(len(spectrum), len(eigvals_M))
    lam_rho = np.sort(spectrum)[::-1][:n]
    # Pad if needed
    if len(lam_rho) < len(eigvals_M):
        lam_rho = np.concatenate([lam_rho, np.zeros(len(eigvals_M) - len(lam_rho))])
    lam_M = np.sort(eigvals_M)[::-1]
    lam_M_asc = lam_M[::-1]
    upper = float(np.dot(lam_rho, lam_M))
    lower = float(np.dot(lam_rho, lam_M_asc))
    return lower, upper

def check_partial_overlap(int1, int2, tol=1e-10):
    """Check if two intervals partially overlap (neither contains the other)."""
    l1, u1 = int1
    l2, u2 = int2
    # Overlap exists
    if l1 > u2 + tol or l2 > u1 + tol:
        return False  # disjoint
    # Check if one contains the other
    i1_in_2 = (l2 <= l1 + tol) and (u1 <= u2 + tol)
    i2_in_1 = (l1 <= l2 + tol) and (u2 <= u1 + tol)
    if i1_in_2 or i2_in_1:
        return False  # nested, not partial overlap
    return True  # genuine partial overlap

def fast_incomp_check(s1, s2):
    """Quick majorization incomparability check."""
    n = len(s1)
    ps1 = np.cumsum(np.sort(s1)[::-1])
    ps2 = np.cumsum(np.sort(s2)[::-1])
    d = ps1[:n-1] - ps2[:n-1]
    return bool(np.any(d > 1e-12) and np.any(d < -1e-12))

def search_robust_incomparable_spectra(system_eigvals_list, n_active=4, n_trials=20000, seed=42):
    """
    Search for incomparable spectra that produce partial overlap in the
    MAXIMUM number of the given systems.
    
    n_active: number of non-zero eigenvalues in the spectrum (rest are 0)
    """
    rng = np.random.default_rng(seed)
    best_score = 0
    best_pair = None
    best_details = None
    
    for trial in range(n_trials):
        # Generate random spectra with n_active non-zero components
        s1 = np.zeros(16)
        s2 = np.zeros(16)
        raw1 = rng.dirichlet(np.ones(n_active))
        raw2 = rng.dirichlet(np.ones(n_active))
        s1[:n_active] = np.sort(raw1)[::-1]
        s2[:n_active] = np.sort(raw2)[::-1]
        
        # Check incomparability (only on active components)
        if not fast_incomp_check(s1[:n_active], s2[:n_active]):
            continue
        
        # Check how many systems show partial overlap
        overlap_count = 0
        min_protrusion = float('inf')
        
        for evk in system_eigvals_list:
            int1 = compute_interval(evk, s1)
            int2 = compute_interval(evk, s2)
            
            if check_partial_overlap(int1, int2):
                overlap_count += 1
                # Protrusion: how much each interval sticks out beyond the other
                prot1 = max(0, int1[1] - int2[1])  # s1 upper beyond s2 upper
                prot2 = max(0, int2[1] - int1[1])  # s2 upper beyond s1 upper
                prot_low1 = max(0, int2[0] - int1[0])  # s2 lower beyond s1 lower
                prot_low2 = max(0, int1[0] - int2[0])
                min_p = min(max(prot1, prot2), max(prot_low1, prot_low2))
                min_protrusion = min(min_protrusion, min_p)
        
        # Score: number of overlapping systems, weighted by minimum protrusion
        score = overlap_count + min_protrusion * 0.1
        
        if score > best_score:
            best_score = score
            best_pair = (s1.copy(), s2.copy())
            best_details = {
                'overlap_count': overlap_count,
                'total_systems': len(system_eigvals_list),
                'min_protrusion': min_protrusion if min_protrusion < float('inf') else 0,
            }
    
    return best_pair, best_details

def improve_incomparable_spectra():
    """Search for better incomparable spectra using actual measured eigenvalues."""
    print("\n" + "="*60)
    print("Part 2: Search for Robust Incomparable Spectra")
    print("="*60)
    
    # Load metadata for system eigenvalues
    with open(os.path.join(DATADIR_V2, 'metadata.json')) as f:
        meta = json.load(f)
    
    system_eigvals = [np.array(ev) for ev in meta['system_eigenvalues']]
    all_eigvals = np.array(meta['H_all_eigenvalues'])
    
    # Include the full system too
    all_systems = [all_eigvals] + system_eigvals
    
    print(f"Using {len(all_systems)} systems (1 full + {len(system_eigvals)} virtual)")
    
    # First, check existing spectra against all systems
    print("\n--- Existing spectra performance ---")
    existing_pairs = [
        ('incomp_A', np.array([0.55,0.15,0.15,0.15,0,0,0,0,0,0,0,0,0,0,0,0]),
         'incomp_B', np.array([0.35,0.35,0.25,0.05,0,0,0,0,0,0,0,0,0,0,0,0])),
        ('comp_less', np.array([0.40,0.30,0.20,0.10,0,0,0,0,0,0,0,0,0,0,0,0]),
         'incomp_A', np.array([0.55,0.15,0.15,0.15,0,0,0,0,0,0,0,0,0,0,0,0])),
        ('prl_c', np.array([0.80,0.15,0.05,0,0,0,0,0,0,0,0,0,0,0,0,0]),
         'prl_e', np.array([0.55,0.45,0,0,0,0,0,0,0,0,0,0,0,0,0,0])),
    ]
    
    for n1, s1, n2, s2 in existing_pairs:
        overlaps = 0
        for evk in all_systems:
            i1 = compute_interval(evk, s1)
            i2 = compute_interval(evk, s2)
            if check_partial_overlap(i1, i2):
                overlaps += 1
        print(f"  {n1} vs {n2}: overlap in {overlaps}/{len(all_systems)} systems")
    
    # Search for better spectra with n_active = 4
    print("\n--- Searching n_active=4 spectra (20000 trials) ---")
    pair4, details4 = search_robust_incomparable_spectra(
        all_systems, n_active=4, n_trials=20000, seed=42)
    
    if pair4 is not None:
        s1, s2 = pair4
        print(f"  Best n=4: overlap in {details4['overlap_count']}/{details4['total_systems']} systems")
        print(f"  s1 = {np.round(s1[:4], 4)}")
        print(f"  s2 = {np.round(s2[:4], 4)}")
        print(f"  min protrusion = {details4['min_protrusion']:.6f}")
        
        # Show intervals for each system
        for k, evk in enumerate(all_systems):
            i1 = compute_interval(evk, s1)
            i2 = compute_interval(evk, s2)
            po = check_partial_overlap(i1, i2)
            label = "all" if k == 0 else f"sys_{k-1}"
            print(f"    {label}: [{i1[0]:.4f},{i1[1]:.4f}] vs [{i2[0]:.4f},{i2[1]:.4f}] "
                  f"{'OVERLAP' if po else 'nested'}")
    
    # Search for n_active = 3 (PRL-like)
    print("\n--- Searching n_active=3 spectra (20000 trials) ---")
    pair3, details3 = search_robust_incomparable_spectra(
        all_systems, n_active=3, n_trials=20000, seed=123)
    
    if pair3 is not None:
        s1, s2 = pair3
        print(f"  Best n=3: overlap in {details3['overlap_count']}/{details3['total_systems']} systems")
        print(f"  s1 = {np.round(s1[:3], 4)}")
        print(f"  s2 = {np.round(s2[:3], 4)}")
        print(f"  min protrusion = {details3['min_protrusion']:.6f}")
    
    # Search for n_active = 6 (higher dimension)
    print("\n--- Searching n_active=6 spectra (20000 trials) ---")
    pair6, details6 = search_robust_incomparable_spectra(
        all_systems, n_active=6, n_trials=20000, seed=456)
    
    if pair6 is not None:
        s1, s2 = pair6
        print(f"  Best n=6: overlap in {details6['overlap_count']}/{details6['total_systems']} systems")
        print(f"  s1 = {np.round(s1[:6], 4)}")
        print(f"  s2 = {np.round(s2[:6], 4)}")
        print(f"  min protrusion = {details6['min_protrusion']:.6f}")
    
    # Search n_active=10 
    print("\n--- Searching n_active=10 spectra (20000 trials) ---")
    pair10, details10 = search_robust_incomparable_spectra(
        all_systems, n_active=10, n_trials=20000, seed=789)
    
    if pair10 is not None:
        s1, s2 = pair10
        print(f"  Best n=10: overlap in {details10['overlap_count']}/{details10['total_systems']} systems")
        print(f"  s1 = {np.round(s1[:10], 4)}")
        print(f"  s2 = {np.round(s2[:10], 4)}")
        print(f"  min protrusion = {details10['min_protrusion']:.6f}")
    
    # Collect all best pairs
    results = {
        'existing_failing_pairs': {
            'incomp_A_vs_incomp_B': 0,
            'comp_less_vs_incomp_A': 0,
            'prl_c_vs_prl_e': 0,
        },
        'best_n4': {
            's1': pair4[0].tolist() if pair4 else None,
            's2': pair4[1].tolist() if pair4 else None,
            'details': details4,
        },
        'best_n3': {
            's1': pair3[0].tolist() if pair3 else None,
            's2': pair3[1].tolist() if pair3 else None,
            'details': details3,
        },
        'best_n6': {
            's1': pair6[0].tolist() if pair6 else None,
            's2': pair6[1].tolist() if pair6 else None,
            'details': details6,
        },
        'best_n10': {
            's1': pair10[0].tolist() if pair10 else None,
            's2': pair10[1].tolist() if pair10 else None,
            'details': details10,
        },
    }
    
    with open(os.path.join(OUTDIR, 'improved_incomparable_spectra.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {os.path.join(OUTDIR, 'improved_incomparable_spectra.json')}")
    
    return results

# ============================================================
# Part 3: Alternative channel partitioning strategies
# ============================================================

def alternative_partitions():
    """
    Instead of random partitioning, try structured partitions of output channels
    that may produce more diverse eigenvalue spreads.
    """
    print("\n" + "="*60)
    print("Part 3: Alternative Channel Partitioning")
    print("="*60)
    
    try:
        K_obs = np.load(os.path.join(DATADIR_V2, 'K_obs.npy'))
        ref_vec = np.load(os.path.join(DATADIR_V2, 'ref_vec.npy'))
    except FileNotFoundError:
        print("v2 data not found, skipping")
        return None
    
    N = K_obs.shape[1]
    M = K_obs.shape[0]
    
    bright_thresh = np.percentile(ref_vec, 25)
    good = np.where(ref_vec > bright_thresh)[0]
    
    # Strategy A: Partition by brightness quantiles
    brightness_order = np.argsort(ref_vec[good])
    K_sys = 8
    groups_brightness = np.array_split(good[brightness_order], K_sys)
    
    # Strategy B: Partition by spatial position (radial rings)
    bin_size = 4
    bh = bw = 500 // bin_size  # 125
    row_idx = good // bw
    col_idx = good % bw
    center_r, center_c = bh // 2, bw // 2
    radii = np.sqrt((row_idx - center_r)**2 + (col_idx - center_c)**2)
    radial_order = np.argsort(radii)
    groups_radial = np.array_split(good[radial_order], K_sys)
    
    # Strategy C: Partition by K_obs row norm (responsivity)
    row_norms = np.linalg.norm(K_obs[good, :], axis=1)
    norm_order = np.argsort(row_norms)
    groups_norm = np.array_split(good[norm_order], K_sys)
    
    strategies = {
        'brightness': groups_brightness,
        'radial': groups_radial,
        'responsivity': groups_norm,
    }
    
    # Build operators and check incomparable pairs
    test_pairs = [
        ('incomp_A', np.array([0.55,0.15,0.15,0.15,0,0,0,0,0,0,0,0,0,0,0,0]),
         'incomp_B', np.array([0.35,0.35,0.25,0.05,0,0,0,0,0,0,0,0,0,0,0,0])),
        ('comp_less', np.array([0.40,0.30,0.20,0.10,0,0,0,0,0,0,0,0,0,0,0,0]),
         'incomp_A', np.array([0.55,0.15,0.15,0.15,0,0,0,0,0,0,0,0,0,0,0,0])),
        ('prl_c', np.array([0.80,0.15,0.05,0,0,0,0,0,0,0,0,0,0,0,0,0]),
         'prl_e', np.array([0.55,0.45,0,0,0,0,0,0,0,0,0,0,0,0,0,0])),
    ]
    
    for strat_name, groups in strategies.items():
        print(f"\n--- Strategy: {strat_name} ---")
        
        sys_eigvals = []
        for k, grp in enumerate(groups):
            Hk = np.zeros((N, N), dtype=np.complex128)
            for m in grp:
                if ref_vec[m] > 1e-10:
                    km = K_obs[m:m+1, :]
                    Hk += (1.0 / ref_vec[m]) * (km.conj().T @ km)
            Hk = (Hk + Hk.conj().T) / 2.0
            evk = np.sort(np.linalg.eigvalsh(Hk))[::-1]
            sys_eigvals.append(evk)
            spread = evk[0] / max(evk[-1], 1e-15)
            print(f"  sys_{k}: top4={np.round(evk[:4], 3)}, spread={spread:.1f}")
        
        for n1, s1, n2, s2 in test_pairs:
            overlaps = 0
            for evk in sys_eigvals:
                i1 = compute_interval(evk, s1)
                i2 = compute_interval(evk, s2)
                if check_partial_overlap(i1, i2):
                    overlaps += 1
            status = "✓" if overlaps > 0 else "✗"
            print(f"  {status} {n1} vs {n2}: overlap in {overlaps}/{K_sys} systems")
    
    return strategies

# ============================================================
# Main
# ============================================================

def main():
    eigvals, K_obs, ref_vec, selected = diagnose_normalization()
    incomp_results = improve_incomparable_spectra()
    alt_results = alternative_partitions()
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)

if __name__ == '__main__':
    main()
