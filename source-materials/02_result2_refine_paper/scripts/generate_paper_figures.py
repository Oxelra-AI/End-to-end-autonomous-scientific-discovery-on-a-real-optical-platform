#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate report figures for the TM analysis.
All data is already fetched locally. Runs on Linux.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle
import matplotlib.colors as mcolors

# Paths are relative to this report section.
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_TM = f"{BASE}/source_data/tm_256"
DATA_FOCUS = f"{BASE}/source_data/focusing"
DATA_RAW = f"{BASE}/source_data/focusing/raw_rois"
DATA_FWD = f"{BASE}/source_data/forward_model"
DATA_RING = f"{BASE}/source_data/ring_screening"
FIG_DIR = f"{BASE}/figures"
os.makedirs(FIG_DIR, exist_ok=True)

# Style
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'sans-serif',
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'figure.dpi': 300,
})

N_OUT = 16  # 16x16 output modes

# ══════════════════════════════════════════════════════════════════════════
# Figure 2: TM amplitude, phase, filtered, and operators
# ══════════════════════════════════════════════════════════════════════════
def fig_tm_matrix():
    K = np.load(f"{DATA_TM}/K_obs_canonical.npy")
    K_filt = np.load(f"{DATA_TM}/K_filtered.npy")
    KKd = np.load(f"{DATA_TM}/KKdagger.npy")
    KKd_norm = np.load(f"{DATA_TM}/KKdagger_normalized.npy")
    
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)
    
    # (a) Amplitude
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(np.abs(K), cmap='hot', aspect='equal', interpolation='nearest')
    ax.set_title(r'(a) $|K_{\mathrm{obs}}|$', fontsize=11)
    ax.set_xlabel('Input mode'); ax.set_ylabel('Output mode')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # (b) Phase
    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(np.angle(K), cmap='hsv', aspect='equal', vmin=-np.pi, vmax=np.pi,
                   interpolation='nearest')
    ax.set_title(r'(b) Phase of $K_{\mathrm{obs}}$', fontsize=11)
    ax.set_xlabel('Input mode')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='rad')
    
    # (c) Filtered amplitude
    ax = fig.add_subplot(gs[0, 2])
    im = ax.imshow(np.abs(K_filt), cmap='hot', aspect='equal', interpolation='nearest')
    ax.set_title(r'(c) $|K_{\mathrm{filtered}}|$', fontsize=11)
    ax.set_xlabel('Input mode')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # (d) |KK†|
    ax = fig.add_subplot(gs[1, 0])
    im = ax.imshow(np.abs(KKd), cmap='hot', aspect='equal', interpolation='nearest')
    ax.set_title(r'(d) $|K K^\dagger|$', fontsize=11)
    ax.set_xlabel('Output mode'); ax.set_ylabel('Output mode')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # (e) Normalized |K_norm K_norm†|/N
    ax = fig.add_subplot(gs[1, 1])
    im = ax.imshow(np.abs(KKd_norm)/256, cmap='hot', aspect='equal', interpolation='nearest')
    ax.set_title(r'(e) $|K_{\phi}K_{\phi}^\dagger|/N$', fontsize=11)
    ax.set_xlabel('Output mode')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # (f) Diagonal / off-diagonal line cuts
    ax = fig.add_subplot(gs[1, 2])
    diag_KKd = np.abs(np.diag(KKd))
    diag_norm = np.abs(np.diag(KKd_norm)) / 256
    # Off-diagonal: average of row 128
    row128 = np.abs(KKd[128, :])
    row128_norm = np.abs(KKd_norm[128, :]) / 256
    
    ax.plot(diag_KKd / np.max(diag_KKd), 'r-', lw=1.2, label=r'Diag $|KK^\dagger|$')
    ax.plot(diag_norm / np.max(diag_norm), 'b-', lw=1.2, label=r'Diag $|K_\phi K_\phi^\dagger|/N$')
    ax.plot(row128 / np.max(diag_KKd), 'r--', lw=0.8, alpha=0.5, label='Row 128 (off-diag)')
    ax.plot(row128_norm / np.max(diag_norm), 'b--', lw=0.8, alpha=0.5, label='Row 128 norm')
    ax.set_xlabel('Mode index')
    ax.set_ylabel('Normalized value')
    ax.set_title('(f) Diagonal vs off-diagonal', fontsize=11)
    ax.legend(fontsize=7, loc='upper right')
    
    plt.savefig(f"{FIG_DIR}/fig_tm_matrix.pdf", bbox_inches='tight', dpi=300)
    plt.savefig(f"{FIG_DIR}/fig_tm_matrix.png", bbox_inches='tight', dpi=300)
    plt.close()
    print("  fig_tm_matrix done")


# ══════════════════════════════════════════════════════════════════════════
# Figure 3: Singular value distribution
# ══════════════════════════════════════════════════════════════════════════
def fig_singular_values():
    sv_norm = np.load(f"{DATA_TM}/sv_normalized.npy")
    sv_filt = np.load(f"{DATA_TM}/sv_filtered_normalized.npy")
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # (a) Histogram
    ax = axes[0]
    ax.hist(sv_norm, bins=25, density=True, alpha=0.6, color='steelblue', label='Raw $K_{obs}$', edgecolor='navy')
    ax.hist(sv_filt, bins=25, density=True, alpha=0.6, color='coral', label='Filtered $K$', edgecolor='darkred')
    # Quarter-circle (Marchenko-Pastur for square matrix)
    lam = np.linspace(0, 2.2, 300)
    qc = np.sqrt(np.maximum(4 - lam**2, 0)) / np.pi
    ax.plot(lam, qc, 'k-', lw=2, label='Quarter-circle law')
    ax.set_xlabel('Normalized singular value')
    ax.set_ylabel('Probability density')
    ax.set_title('(a) Singular value distribution')
    ax.legend(fontsize=8)
    ax.set_xlim(0, 2.5)
    
    # (b) Sorted
    ax = axes[1]
    ax.plot(np.arange(1, len(sv_norm)+1), sv_norm, 'b.-', markersize=2, lw=0.8, label='Raw')
    ax.plot(np.arange(1, len(sv_filt)+1), sv_filt, 'r.-', markersize=2, lw=0.8, label='Filtered')
    ax.set_xlabel('Singular value index')
    ax.set_ylabel('Normalized singular value')
    ax.set_title('(b) Sorted singular values')
    ax.legend(fontsize=8)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig_singular_values.pdf", bbox_inches='tight', dpi=300)
    plt.savefig(f"{FIG_DIR}/fig_singular_values.png", bbox_inches='tight', dpi=300)
    plt.close()
    print("  fig_singular_values done")


# ══════════════════════════════════════════════════════════════════════════
# Figure 4: Reference drift
# ══════════════════════════════════════════════════════════════════════════
def fig_reference_drift():
    ref_pre = np.load(f"{DATA_TM}/ref_only_binned.npy")
    ref_post = np.load(f"{DATA_TM}/ref_post_binned.npy")
    
    vmax = max(np.max(ref_pre), np.max(ref_post))
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    
    im0 = axes[0].imshow(ref_pre, cmap='hot', vmin=0, vmax=vmax, interpolation='nearest')
    axes[0].set_title('(a) Reference (pre-acquisition)')
    plt.colorbar(im0, ax=axes[0], fraction=0.046)
    
    im1 = axes[1].imshow(ref_post, cmap='hot', vmin=0, vmax=vmax, interpolation='nearest')
    axes[1].set_title('(b) Reference (post-acquisition)')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    
    diff = ref_post - ref_pre
    vlim = max(abs(np.min(diff)), abs(np.max(diff)))
    im2 = axes[2].imshow(diff, cmap='RdBu_r', vmin=-vlim, vmax=vlim, interpolation='nearest')
    corr = np.corrcoef(ref_pre.flatten(), ref_post.flatten())[0, 1]
    axes[2].set_title(f'(c) Drift ($\\rho$={corr:.3f})')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)
    
    for ax in axes:
        ax.set_xlabel('Output mode $j$')
        ax.set_ylabel('Output mode $i$')
    
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig_reference_drift.pdf", bbox_inches='tight', dpi=300)
    plt.savefig(f"{FIG_DIR}/fig_reference_drift.png", bbox_inches='tight', dpi=300)
    plt.close()
    print("  fig_reference_drift done")


# ══════════════════════════════════════════════════════════════════════════
# Figure 5: Single-spot focusing with raw speckle ROIs
# ══════════════════════════════════════════════════════════════════════════
def fig_single_spot_focusing():
    with open(f"{DATA_FOCUS}/metadata.json") as f:
        meta = json.load(f)
    
    baseline = np.load(f"{DATA_RAW}/baseline_random_roi.npy")
    
    # Select 4 representative targets
    targets = [
        (136, (8, 8)),   # center
        (0, (0, 0)),     # corner
        (240, (15, 0)),  # corner - best enhancement
        (68, (4, 4)),    # interior
    ]
    
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(3, 5, figure=fig, hspace=0.4, wspace=0.3)
    
    # Row 0: baseline + 4 raw speckle crops
    # Baseline crop (central 80x80 from 500x500)
    h, w = baseline.shape
    cy, cx = h//2, w//2
    bl_crop = baseline[cy-40:cy+40, cx-40:cx+40]
    
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(bl_crop, cmap='hot', interpolation='nearest')
    ax.set_title('Baseline\n(random input)', fontsize=9)
    ax.axis('off')
    
    for idx, (mode, (r2d, c2d)) in enumerate(targets):
        crop = np.load(f"{DATA_RAW}/focus_single_m{mode}_po_crop80.npy")
        ax = fig.add_subplot(gs[0, idx+1])
        ax.imshow(crop, cmap='hot', interpolation='nearest')
        # Find the focus result
        res = [r for r in meta["single_focus_results"] if r["target_mode"] == mode][0]
        ax.set_title(f'Target ({r2d},{c2d})\n$\\eta$={res["enhance_phase_only"]:.1f}', fontsize=9)
        ax.axis('off')
    
    # Row 1: binned 16x16 maps with target markers
    bl_binned = np.load(f"{DATA_FOCUS}/baseline_avg_binned.npy")
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(bl_binned, cmap='hot', interpolation='nearest')
    ax.set_title('Baseline (binned)', fontsize=9)
    ax.axis('off')
    
    for idx, (mode, (r2d, c2d)) in enumerate(targets):
        binned = np.load(f"{DATA_RAW}/focus_single_m{mode}_po_binned.npy")
        ax = fig.add_subplot(gs[1, idx+1])
        ax.imshow(binned, cmap='hot', interpolation='nearest')
        ax.plot(c2d, r2d, 'c+', markersize=12, markeredgewidth=2)
        ax.set_title(f'Mode {mode} (binned)', fontsize=9)
        ax.axis('off')
    
    # Row 2: Enhancement summary bar chart
    ax = fig.add_subplot(gs[2, :])
    results = meta["single_focus_results"]
    modes = [r["target_mode"] for r in results]
    labels = [f'({r["target_2d"][0]},{r["target_2d"][1]})' for r in results]
    enh_po = [r["enhance_phase_only"] for r in results]
    pbr_po = [r["pbr_phase_only"] for r in results]
    
    x = np.arange(len(modes))
    bars1 = ax.bar(x - 0.2, enh_po, 0.35, label='Enhancement $\\eta$', color='steelblue', edgecolor='navy')
    bars2 = ax.bar(x + 0.2, pbr_po, 0.35, label='PBR', color='coral', edgecolor='darkred')
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlabel('Target output mode $(i, j)$')
    ax.set_ylabel('Ratio')
    ax.set_title('(c) Single-spot focusing: enhancement and peak-to-background ratio')
    ax.legend(fontsize=9)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5, lw=0.8)
    
    # Add value labels
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.5, f'{h:.0f}', ha='center', va='bottom', fontsize=7)
    
    plt.savefig(f"{FIG_DIR}/fig_single_spot_focusing.pdf", bbox_inches='tight', dpi=300)
    plt.savefig(f"{FIG_DIR}/fig_single_spot_focusing.png", bbox_inches='tight', dpi=300)
    plt.close()
    print("  fig_single_spot_focusing done")


# ══════════════════════════════════════════════════════════════════════════
# Figure 6: Multi-spot focusing
# ══════════════════════════════════════════════════════════════════════════
def fig_multi_spot_focusing():
    with open(f"{DATA_FOCUS}/metadata.json") as f:
        meta = json.load(f)
    
    configs = meta["multi_focus_results"]
    n = len(configs)
    
    fig, axes = plt.subplots(2, n, figsize=(3.5*n, 7))
    
    for idx, cfg in enumerate(configs):
        name = cfg["name"]
        
        # Row 0: raw crop
        crop = np.load(f"{DATA_RAW}/focus_multi_{name}_crop80.npy")
        axes[0, idx].imshow(crop, cmap='hot', interpolation='nearest')
        axes[0, idx].set_title(f'{name}\n$\\eta$={cfg["enhancement"]:.1f}', fontsize=9)
        axes[0, idx].axis('off')
        
        # Row 1: binned with markers
        binned = np.load(f"{DATA_RAW}/focus_multi_{name}_binned.npy")
        axes[1, idx].imshow(binned, cmap='hot', interpolation='nearest')
        for m in cfg["target_modes"]:
            r2d = m // N_OUT
            c2d = m % N_OUT
            axes[1, idx].plot(c2d, r2d, 'c+', markersize=10, markeredgewidth=2)
        axes[1, idx].set_title(f'PBR={cfg["pbr"]:.1f}', fontsize=9)
        axes[1, idx].axis('off')
    
    fig.suptitle('Multi-Spot Focusing (Phase-Only)', fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig_multi_spot_focusing.pdf", bbox_inches='tight', dpi=300)
    plt.savefig(f"{FIG_DIR}/fig_multi_spot_focusing.png", bbox_inches='tight', dpi=300)
    plt.close()
    print("  fig_multi_spot_focusing done")


# ══════════════════════════════════════════════════════════════════════════
# Figure 7: Forward model comparison
# ══════════════════════════════════════════════════════════════════════════
def fig_forward_model():
    with open(f"{DATA_FWD}/full_results.json") as f:
        data = json.load(f)
    
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)
    
    # Row 0: 8->4 system, 4 representative test vectors
    sys1 = data["sys_8x4"]["test_results"]
    for t in range(min(4, len(sys1))):
        ax = fig.add_subplot(gs[0, t])
        r = sys1[t]
        x = np.arange(4)
        w = 0.18
        ax.bar(x-1.5*w, r["I_sr"], w, label='Measured', color='black', alpha=0.9)
        ax.bar(x-0.5*w, r["pred_A"], w, label='Model A', color='#e74c3c', alpha=0.8)
        ax.bar(x+0.5*w, r["pred_C"], w, label='Model C', color='#3498db', alpha=0.8)
        ax.bar(x+1.5*w, r["pred_D"], w, label='Model D', color='#2ecc71', alpha=0.8)
        ax.set_title(f'{r["label"]}\nA:{r["corr_A"]:.2f} C:{r["corr_C"]:.2f}', fontsize=8)
        ax.set_xlabel('Output mode', fontsize=8)
        if t == 0:
            ax.set_ylabel('Intensity', fontsize=8)
            ax.legend(fontsize=5, loc='upper right')
    
    # Row 1: 4->4 system
    sys2 = data["sys_4x4"]["test_results"]
    for t in range(min(4, len(sys2))):
        ax = fig.add_subplot(gs[1, t])
        r = sys2[t]
        x = np.arange(4)
        w = 0.18
        ax.bar(x-1.5*w, r["I_sr"], w, color='black', alpha=0.9)
        ax.bar(x-0.5*w, r["pred_A"], w, color='#e74c3c', alpha=0.8)
        ax.bar(x+0.5*w, r["pred_C"], w, color='#3498db', alpha=0.8)
        ax.bar(x+1.5*w, r["pred_D"], w, color='#2ecc71', alpha=0.8)
        ax.set_title(f'{r["label"]}\nA:{r["corr_A"]:.2f} C:{r["corr_C"]:.2f}', fontsize=8)
        ax.set_xlabel('Output mode', fontsize=8)
        if t == 0:
            ax.set_ylabel('Intensity', fontsize=8)
    
    # Row 2: Summary correlation bar chart
    ax = fig.add_subplot(gs[2, :2])
    models = ['A (naive)', 'C (physics)', 'D (empirical)', 'E (oracle)']
    corr_8x4 = [data["sys_8x4"]["avg_corr_A"], data["sys_8x4"]["avg_corr_C"],
                data["sys_8x4"]["avg_corr_D"], data["sys_8x4"]["avg_corr_E"]]
    corr_4x4 = [data["sys_4x4"]["avg_corr_A"], data["sys_4x4"]["avg_corr_C"],
                data["sys_4x4"]["avg_corr_D"], data["sys_4x4"]["avg_corr_E"]]
    
    x = np.arange(len(models))
    ax.bar(x - 0.2, corr_8x4, 0.35, label='8→4', color='steelblue', edgecolor='navy')
    ax.bar(x + 0.2, corr_4x4, 0.35, label='4→4', color='coral', edgecolor='darkred')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel('Average correlation')
    ax.set_title('(e) Forward model comparison: average correlation')
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.3)
    
    # Add value labels
    for i, (v1, v2) in enumerate(zip(corr_8x4, corr_4x4)):
        ax.text(i - 0.2, v1 + 0.02, f'{v1:.3f}', ha='center', va='bottom', fontsize=7)
        ax.text(i + 0.2, v2 + 0.02, f'{v2:.3f}', ha='center', va='bottom', fontsize=7)
    
    # RMSE summary
    ax2 = fig.add_subplot(gs[2, 2:])
    rmse_8x4 = [data["sys_8x4"]["avg_rmse_A"], data["sys_8x4"]["avg_rmse_C"],
                data["sys_8x4"]["avg_rmse_D"], data["sys_8x4"]["avg_rmse_E"]]
    rmse_4x4 = [data["sys_4x4"]["avg_rmse_A"], data["sys_4x4"]["avg_rmse_C"],
                data["sys_4x4"]["avg_rmse_D"], data["sys_4x4"]["avg_rmse_E"]]
    
    ax2.bar(x - 0.2, rmse_8x4, 0.35, label='8→4', color='steelblue', edgecolor='navy')
    ax2.bar(x + 0.2, rmse_4x4, 0.35, label='4→4', color='coral', edgecolor='darkred')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=9)
    ax2.set_ylabel('Average RMSE')
    ax2.set_title('(f) Forward model comparison: average RMSE')
    ax2.legend(fontsize=9)
    
    plt.savefig(f"{FIG_DIR}/fig_forward_model.pdf", bbox_inches='tight', dpi=300)
    plt.savefig(f"{FIG_DIR}/fig_forward_model.png", bbox_inches='tight', dpi=300)
    plt.close()
    print("  fig_forward_model done")


# ══════════════════════════════════════════════════════════════════════════
# Figure 8: Enhancement vs number of targets
# ══════════════════════════════════════════════════════════════════════════
def fig_enhancement_vs_targets():
    with open(f"{DATA_FOCUS}/metadata.json") as f:
        meta = json.load(f)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Single-spot average
    single_avg = np.mean([r["enhance_phase_only"] for r in meta["single_focus_results"]])
    
    # Multi-spot
    multi = meta["multi_focus_results"]
    n_targets = [1, 2, 2, 3, 4, 5]
    enhancements = [single_avg,
                    multi[0]["enhancement"],  # 2spot diagonal
                    multi[1]["enhancement"],  # 2spot center
                    multi[2]["enhancement"],  # 3spot
                    multi[3]["enhancement"],  # 4spot
                    multi[4]["enhancement"]]  # 5spot
    labels = ['1 (avg)', '2 (diag)', '2 (center)', '3 (line)', '4 (square)', '5 (cross)']
    
    colors = ['steelblue'] + ['coral']*5
    ax.bar(range(len(n_targets)), enhancements, color=colors, edgecolor='black', alpha=0.8)
    ax.set_xticks(range(len(n_targets)))
    ax.set_xticklabels(labels, fontsize=9, rotation=15)
    ax.set_xlabel('Number of target spots (configuration)')
    ax.set_ylabel('Enhancement $\\eta$')
    ax.set_title('Focusing enhancement vs. number of target spots')
    
    # Theoretical: eta ~ N/M for M targets
    # N = 256 input modes, phase-only: eta_1 ~ pi/4 * N = 201 (ideal)
    # Our measured eta_1 ~ 40 (much less due to drift, noise)
    # For M spots: eta_M ~ eta_1 / M approximately
    M_theory = np.array([1, 2, 3, 4, 5])
    eta_theory = single_avg / M_theory
    ax.plot(np.array([0, 1.5, 3, 4, 5]) , eta_theory, 'k--', lw=1.5, marker='o',
            markersize=5, label=r'$\eta_1/M$ scaling')
    ax.legend(fontsize=9)
    ax.axhline(y=1, color='gray', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig_enhancement_vs_targets.pdf", bbox_inches='tight', dpi=300)
    plt.savefig(f"{FIG_DIR}/fig_enhancement_vs_targets.png", bbox_inches='tight', dpi=300)
    plt.close()
    print("  fig_enhancement_vs_targets done")


# ══════════════════════════════════════════════════════════════════════════
# Run all
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating report figures...")
    fig_tm_matrix()
    fig_singular_values()
    fig_reference_drift()
    fig_single_spot_focusing()
    fig_multi_spot_focusing()
    fig_forward_model()
    fig_enhancement_vs_targets()
    
    # List generated files
    figs = sorted(os.listdir(FIG_DIR))
    print(f"\nGenerated {len(figs)} files in {FIG_DIR}/:")
    for f in figs:
        size = os.path.getsize(os.path.join(FIG_DIR, f))
        print(f"  {f}: {size/1024:.1f} KB")
