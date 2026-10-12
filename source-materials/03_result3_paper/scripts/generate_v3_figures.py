#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate report figures from the v3 transport interval experiment.
Figures:
1. Interval diagram for comparable pair (nested) across all 17 systems
2. Interval diagram for incomparable pair (partial overlap) across all 17 systems
3. Summary heatmap: all pairs × all systems
4. Eigenvalue spectrum of full system + representative subsystems
5. Physical spot-check: analytical vs measured transport
6. Combined panel figure
"""

import numpy as np
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

DATADIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', 'data', 'exp_transport_v3', 'exp_transport_v3')
FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures')
os.makedirs(FIGDIR, exist_ok=True)

# Load metadata
with open(os.path.join(DATADIR, 'metadata.json')) as f:
    meta = json.load(f)

def get_intervals(sys_name, spec_name):
    d = meta['all_intervals'][sys_name][spec_name]
    return d['lower'], d['upper']

SYSTEM_NAMES = list(meta['system_info'].keys())
# Reorder: full first, then radial_0..7, then resp_0..7
ordered_systems = ['full'] + [f'radial_{i}' for i in range(8)] + [f'resp_{i}' for i in range(8)]
ordered_systems = [s for s in ordered_systems if s in SYSTEM_NAMES]

SHORT_LABELS = ['Full'] + [f'R{i}' for i in range(8)] + [f'P{i}' for i in range(8)]

# ============================================================
# Figure 1: Comparable pair interval diagram
# ============================================================
def fig_comparable_intervals():
    fig, ax = plt.subplots(figsize=(10, 5))
    
    s1, s2 = 'comp_less', 'comp_more'
    y_positions = np.arange(len(ordered_systems))
    
    for idx, sys_name in enumerate(ordered_systems):
        lo1, hi1 = get_intervals(sys_name, s1)
        lo2, hi2 = get_intervals(sys_name, s2)
        
        # Plot s2 interval (wider, more coherent) — red
        ax.barh(idx + 0.15, hi2 - lo2, left=lo2, height=0.3,
                color='#E91E63', alpha=0.4, edgecolor='#E91E63', linewidth=0.8)
        # Plot s1 interval (narrower, less coherent) — blue
        ax.barh(idx - 0.15, hi1 - lo1, left=lo1, height=0.3,
                color='#2196F3', alpha=0.6, edgecolor='#2196F3', linewidth=0.8)
    
    ax.set_yticks(y_positions)
    ax.set_yticklabels(SHORT_LABELS, fontsize=8)
    ax.set_xlabel('Transport value $\\mathrm{tr}(U\\rho U^\\dagger M)$', fontsize=11)
    ax.set_title('Comparable pair: $\\lambda_1 \\prec \\lambda_2$ → Nested intervals in ALL systems',
                 fontsize=12)
    ax.invert_yaxis()
    
    legend_elements = [
        mpatches.Patch(facecolor='#2196F3', alpha=0.6, edgecolor='#2196F3',
                       label='$\\lambda_1=(0.40,0.30,0.20,0.10)$'),
        mpatches.Patch(facecolor='#E91E63', alpha=0.4, edgecolor='#E91E63',
                       label='$\\lambda_2=(0.70,0.20,0.07,0.03)$'),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc='lower right')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(FIGDIR, f'v3_comparable_intervals.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Saved: v3_comparable_intervals")

# ============================================================
# Figure 2: Incomparable pair interval diagram (best case)
# ============================================================
def fig_incomparable_intervals():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    pairs = [
        ('comp_less', 'incomp_B',
         '$\\lambda_1=(0.40,0.30,0.20,0.10)$',
         '$\\lambda_2=(0.35,0.35,0.25,0.05)$',
         'Incomparable: $\\lambda_1 \\| \\lambda_2$ (strong overlap)'),
        ('prl_b', 'prl_e',
         '$\\rho_b=(0.60,0.30,0.10)$',
         '$\\rho_e=(0.55,0.45,0.00)$',
         'PRL benchmark: $\\rho_b \\| \\rho_e$'),
    ]
    
    for ax, (s1, s2, lab1, lab2, title) in zip(axes, pairs):
        for idx, sys_name in enumerate(ordered_systems):
            lo1, hi1 = get_intervals(sys_name, s1)
            lo2, hi2 = get_intervals(sys_name, s2)
            
            # Check relation
            tol = 1e-10
            i1_in_2 = (lo2 <= lo1 + tol) and (hi1 <= hi2 + tol)
            i2_in_1 = (lo1 <= lo2 + tol) and (hi2 <= hi1 + tol)
            is_overlap = not i1_in_2 and not i2_in_1 and lo1 <= hi2 and lo2 <= hi1
            
            alpha1 = 0.7 if is_overlap else 0.4
            alpha2 = 0.7 if is_overlap else 0.4
            lw = 1.5 if is_overlap else 0.5
            
            # Background highlight for overlap systems
            if is_overlap:
                ax.axhspan(idx - 0.4, idx + 0.4, color='#FFEB3B', alpha=0.2, zorder=0)
            
            ax.barh(idx + 0.15, hi2 - lo2, left=lo2, height=0.3,
                    color='#E91E63', alpha=alpha2, edgecolor='#E91E63', linewidth=lw)
            ax.barh(idx - 0.15, hi1 - lo1, left=lo1, height=0.3,
                    color='#2196F3', alpha=alpha1, edgecolor='#2196F3', linewidth=lw)
        
        ax.set_yticks(range(len(ordered_systems)))
        ax.set_yticklabels(SHORT_LABELS, fontsize=7)
        ax.set_xlabel('Transport value', fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        
        legend_elements = [
            mpatches.Patch(facecolor='#2196F3', alpha=0.6, edgecolor='#2196F3', label=lab1),
            mpatches.Patch(facecolor='#E91E63', alpha=0.6, edgecolor='#E91E63', label=lab2),
            mpatches.Patch(facecolor='#FFEB3B', alpha=0.3, label='Partial overlap'),
        ]
        ax.legend(handles=legend_elements, fontsize=7, loc='lower right')
    
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(FIGDIR, f'v3_incomparable_intervals.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Saved: v3_incomparable_intervals")

# ============================================================
# Figure 3: Summary heatmap
# ============================================================
def fig_summary_heatmap():
    pair_keys = list(meta['pair_results'].keys())
    pair_labels = [k.replace('_vs_', ' vs ').replace('_', ' ') for k in pair_keys]
    
    # Build matrix: 0=nested, 1=partial_overlap, 2=equal
    matrix = np.zeros((len(pair_keys), len(ordered_systems)))
    
    for i, pk in enumerate(pair_keys):
        details = meta['pair_details'][pk]
        for j, sys_name in enumerate(ordered_systems):
            rel = details[sys_name]
            if 'overlap' in rel:
                matrix[i, j] = 1
            elif 'equal' in rel:
                matrix[i, j] = 0.5
            else:
                matrix[i, j] = 0
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    cmap = ListedColormap(['#E3F2FD', '#FF5722'])  # light blue = nested, orange = overlap
    im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1,
                   interpolation='nearest')
    
    ax.set_xticks(range(len(ordered_systems)))
    ax.set_xticklabels(SHORT_LABELS, fontsize=7, rotation=45, ha='right')
    ax.set_yticks(range(len(pair_keys)))
    ax.set_yticklabels(pair_labels, fontsize=8)
    
    # Add majorization labels on the right
    for i, pk in enumerate(pair_keys):
        maj = meta['pair_results'][pk]['majorization']
        counts = meta['pair_results'][pk]['counts']
        n_overlap = counts.get('partial_overlap', 0)
        color = '#4CAF50' if meta['pair_results'][pk]['success'] else '#F44336'
        
        short_maj = 'comp' if 'majorizes' in maj else ('incomp' if maj == 'incomparable' else maj)
        ax.text(len(ordered_systems) + 0.3, i, f'{short_maj} ({n_overlap}/{len(ordered_systems)})',
                fontsize=7, va='center', color=color, fontweight='bold')
    
    # Grid
    for i in range(len(pair_keys) + 1):
        ax.axhline(i - 0.5, color='white', linewidth=0.5)
    for j in range(len(ordered_systems) + 1):
        ax.axvline(j - 0.5, color='white', linewidth=0.5)
    
    # Separator between radial and responsivity
    ax.axvline(8.5, color='black', linewidth=1.5, linestyle='--')
    ax.text(4.5, -1.2, 'Radial partition', ha='center', fontsize=8, style='italic')
    ax.text(13, -1.2, 'Responsivity partition', ha='center', fontsize=8, style='italic')
    
    ax.set_title('Transport interval relations across 17 physical systems\n'
                 '(blue = nested, orange = partial overlap)', fontsize=11)
    
    legend_elements = [
        mpatches.Patch(facecolor='#E3F2FD', edgecolor='gray', label='Nested (theory-consistent for comparable)'),
        mpatches.Patch(facecolor='#FF5722', edgecolor='gray', label='Partial overlap (theory-consistent for incomparable)'),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc='upper center',
              bbox_to_anchor=(0.5, -0.12), ncol=2)
    
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(FIGDIR, f'v3_summary_heatmap.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Saved: v3_summary_heatmap")

# ============================================================
# Figure 4: Eigenvalue spectra
# ============================================================
def fig_eigenvalue_spectra():
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Full system
    ev_full = meta['system_info']['full']['eigenvalues']
    ax.plot(range(1, len(ev_full)+1), ev_full, 'ko-', linewidth=2, markersize=6,
            label='Full system', zorder=5)
    
    # Representative subsystems
    colors = ['#E91E63', '#2196F3', '#4CAF50', '#FF9800']
    for k, (name, color) in enumerate(zip(['radial_0', 'radial_3', 'resp_0', 'resp_7'], colors)):
        ev = meta['system_info'][name]['eigenvalues']
        spread = meta['system_info'][name]['spread']
        ax.plot(range(1, len(ev)+1), ev, 'o--', color=color, linewidth=1.2,
                markersize=4, alpha=0.7,
                label=f'{name} (spread={spread:.1f})')
    
    ax.set_xlabel('Eigenvalue index $k$', fontsize=11)
    ax.set_ylabel('$\\mu_k(M)$', fontsize=11)
    ax.set_title('Transport operator eigenvalue spectra', fontsize=12)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, 17))
    
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(FIGDIR, f'v3_eigenvalue_spectra.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Saved: v3_eigenvalue_spectra")

# ============================================================
# Figure 5: Physical spot-check comparison
# ============================================================
def fig_physical_spotcheck():
    checks = meta['physical_checks']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    
    # Left: analytical vs physical scatter
    for check in checks:
        color = {'comp_less': '#2196F3', 'comp_more': '#E91E63',
                 'incomp_A': '#4CAF50', 'incomp_B': '#FF9800'}[check['spectrum']]
        ax1.scatter(check['T_analytical'], check['T_physical'],
                    c=color, s=60, edgecolors='black', linewidth=0.5, zorder=5)
    
    # Perfect agreement line
    all_vals = [c['T_analytical'] for c in checks] + [c['T_physical'] for c in checks]
    lo, hi = min(all_vals) * 0.9, max(all_vals) * 1.1
    ax1.plot([lo, hi], [lo, hi], 'k--', alpha=0.3, label='Perfect agreement')
    
    # Best-fit line
    T_anal = np.array([c['T_analytical'] for c in checks])
    T_phys = np.array([c['T_physical'] for c in checks])
    slope = np.mean(T_phys / T_anal)
    ax1.plot([lo, hi], [lo * slope, hi * slope], 'r-', alpha=0.5,
             label=f'Systematic offset ×{slope:.2f}')
    
    ax1.set_xlabel('Analytical prediction $\\mathrm{tr}(U\\rho U^\\dagger M)$', fontsize=10)
    ax1.set_ylabel('Physical measurement', fontsize=10)
    ax1.set_title('Physical vs analytical transport', fontsize=11)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    ax1.set_xlim(lo, hi)
    ax1.set_ylim(lo, hi)
    
    # Legend for colors
    for name, color in [('comp_less', '#2196F3'), ('comp_more', '#E91E63'),
                        ('incomp_A', '#4CAF50'), ('incomp_B', '#FF9800')]:
        ax1.scatter([], [], c=color, s=40, edgecolors='black', linewidth=0.5, label=name)
    ax1.legend(fontsize=7, loc='upper left')
    
    # Right: ratio histogram
    ratios = [c['ratio'] for c in checks]
    ax2.hist(ratios, bins=8, color='#607D8B', edgecolor='white', alpha=0.8)
    ax2.axvline(np.mean(ratios), color='red', linestyle='--', linewidth=2,
                label=f'Mean = {np.mean(ratios):.3f}')
    ax2.axvline(1.0, color='black', linestyle='-', linewidth=1, alpha=0.5,
                label='Perfect agreement')
    ax2.set_xlabel('$T_{\\mathrm{physical}} / T_{\\mathrm{analytical}}$', fontsize=10)
    ax2.set_ylabel('Count', fontsize=10)
    ax2.set_title('Physical/analytical ratio distribution', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(FIGDIR, f'v3_physical_spotcheck.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Saved: v3_physical_spotcheck")

# ============================================================
# Figure 6: All incomparable pairs — interval overlap detail
# ============================================================
def fig_all_incomparable_detail():
    incomp_pairs = [
        ('incomp_A', 'incomp_B', '$\\lambda_A \\| \\lambda_B$'),
        ('comp_less', 'incomp_A', 'comp\\_less $\\|$ incomp\\_A'),
        ('comp_less', 'incomp_B', 'comp\\_less $\\|$ incomp\\_B'),
        ('prl_b', 'prl_e', '$\\rho_b \\| \\rho_e$'),
        ('prl_c', 'prl_e', '$\\rho_c \\| \\rho_e$'),
    ]
    
    fig, axes = plt.subplots(len(incomp_pairs), 1, figsize=(12, 3*len(incomp_pairs)),
                             sharex=False)
    
    for ax, (s1, s2, title) in zip(axes, incomp_pairs):
        counts = meta['pair_results'][f'{s1}_vs_{s2}']['counts']
        n_overlap = counts.get('partial_overlap', 0)
        
        for idx, sys_name in enumerate(ordered_systems):
            lo1, hi1 = get_intervals(sys_name, s1)
            lo2, hi2 = get_intervals(sys_name, s2)
            
            tol = 1e-10
            i1_in_2 = (lo2 <= lo1 + tol) and (hi1 <= hi2 + tol)
            i2_in_1 = (lo1 <= lo2 + tol) and (hi2 <= hi1 + tol)
            is_overlap = not i1_in_2 and not i2_in_1 and lo1 <= hi2 and lo2 <= hi1
            
            if is_overlap:
                ax.axvspan(max(lo1, lo2), min(hi1, hi2), alpha=0.15, color='#FFEB3B', zorder=0)
            
            alpha = 0.8 if is_overlap else 0.35
            lw = 1.5 if is_overlap else 0.5
            
            ax.barh(idx + 0.15, hi2 - lo2, left=lo2, height=0.3,
                    color='#E91E63', alpha=alpha, edgecolor='#E91E63', linewidth=lw)
            ax.barh(idx - 0.15, hi1 - lo1, left=lo1, height=0.3,
                    color='#2196F3', alpha=alpha, edgecolor='#2196F3', linewidth=lw)
        
        ax.set_yticks(range(len(ordered_systems)))
        ax.set_yticklabels(SHORT_LABELS, fontsize=6)
        ax.invert_yaxis()
        ax.set_title(f'{title} — overlap in {n_overlap}/{len(ordered_systems)} systems',
                     fontsize=10, fontweight='bold' if n_overlap > 0 else 'normal')
        ax.grid(axis='x', alpha=0.3)
    
    axes[-1].set_xlabel('Transport value', fontsize=10)
    
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(FIGDIR, f'v3_all_incomparable_detail.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Saved: v3_all_incomparable_detail")

# ============================================================
# Figure 7: Drift monitoring
# ============================================================
def fig_drift():
    fig, ax = plt.subplots(figsize=(6, 3.5))
    
    experiments = ['v1 (N=4)', 'v2 (N=16)', 'v3 (N=16)']
    drift_post = [0.9937, 0.9872, 0.9893]
    drift_final = [0.9911, 0.9712, 0.9794]
    
    x = np.arange(len(experiments))
    w = 0.3
    ax.bar(x - w/2, drift_post, w, color='#2196F3', label='Post-TM drift corr.')
    ax.bar(x + w/2, drift_final, w, color='#FF9800', label='Final drift corr.')
    
    ax.axhline(0.95, color='red', linestyle='--', alpha=0.5, label='Stability threshold')
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, fontsize=10)
    ax.set_ylabel('Correlation coefficient', fontsize=10)
    ax.set_title('Platform drift across experiments', fontsize=11)
    ax.set_ylim(0.95, 1.005)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(FIGDIR, f'v3_drift_monitoring.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Saved: v3_drift_monitoring")

# ============================================================
# Main
# ============================================================
def main():
    print("Generating v3 report figures...\n")
    
    fig_comparable_intervals()
    fig_incomparable_intervals()
    fig_summary_heatmap()
    fig_eigenvalue_spectra()
    fig_physical_spotcheck()
    fig_all_incomparable_detail()
    fig_drift()
    
    print("\nAll figures generated. File listing:")
    for f in sorted(os.listdir(FIGDIR)):
        fpath = os.path.join(FIGDIR, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"  {f} ({size_kb:.1f} KB)")

if __name__ == '__main__':
    main()
