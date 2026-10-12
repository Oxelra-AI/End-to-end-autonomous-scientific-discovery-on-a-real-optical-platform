#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regenerate Complex-B complex-plane scatter figure at speckle-matched bin=10.
============================================================================
Produces a REFINED 2×2 figure showing B(0,0), B(0,1), B(1,0), B(2,3) in the
complex plane. Each panel shows all detector channels as scatter points.

Key improvements over old xor_complex_scatter.pdf (bin=25, 400 channels):
  - Speckle-matched bin=10 → 2500 channels (physically appropriate)
  - Unified axis limits across all four panels
  - Centroid marker (black star) in each panel
  - Clean layout with no colorbar clutter
  - Consistent font sizes and spacing
  - Self-pair vs cross-pair annotation
  - Per-repeat cloud shown with low alpha for stability visualization

Data source: exp_bilinear_001 (4-token XOR showcase, 432 measurements)
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe

# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════
N_TOKENS = 4
BIN_SIZE = 10          # speckle-matched (coherence FWHM ≈ 10.5 px)
ROI_H, ROI_W = 500, 500
N_CHANNELS = (ROI_H // BIN_SIZE) * (ROI_W // BIN_SIZE)  # 2500

# Pairs to show, in user-requested order
SELECTED_PAIRS = [(0, 0), (0, 1), (1, 0), (2, 3)]
PANEL_LABELS = [
    r'$B(0,0)$ — self-pair',
    r'$B(0,1)$',
    r'$B(1,0)$',
    r'$B(2,3)$',
]
PANEL_ANNOTATIONS = ['self-pair', 'cross-pair', 'cross-pair', 'cross-pair']

# ═══════════════════════════════════════════════════════════════════════
# Data loading with explicit bin_size
# ═══════════════════════════════════════════════════════════════════════
def bin_roi(roi, bin_size=BIN_SIZE):
    h, w = roi.shape
    nh, nw = h // bin_size, w // bin_size
    cropped = roi[:nh * bin_size, :nw * bin_size]
    return cropped.reshape(nh, bin_size, nw, bin_size).mean(axis=(1, 3)).flatten()


def load_and_bin(data_dir, filename, bin_size=BIN_SIZE):
    roi = np.load(os.path.join(data_dir, filename))
    return bin_roi(roi, bin_size)


def load_blank_control_data(base_dir, bin_size=BIN_SIZE):
    blanks_dir = os.path.join(base_dir, "blank_controls")
    with open(os.path.join(blanks_dir, "blank_records.json")) as f:
        records = json.load(f)
    data = {}
    for rec in records:
        key = (rec["y_token_id"], rec["phi_idx"])
        if key not in data:
            data[key] = []
        data[key].append(load_and_bin(blanks_dir, rec["file"], bin_size))
    return {k: np.array(v) for k, v in data.items()}


def load_pair_interaction_data(base_dir, bin_size=BIN_SIZE):
    pairs_dir = os.path.join(base_dir, "pair_interactions")
    with open(os.path.join(pairs_dir, "pair_records.json")) as f:
        records = json.load(f)
    data = {}
    for rec in records:
        key = (rec["x_token_id"], rec["y_token_id"], rec["phi_idx"])
        if key not in data:
            data[key] = []
        data[key].append(load_and_bin(pairs_dir, rec["file"], bin_size))
    return {k: np.array(v) for k, v in data.items()}


# ═══════════════════════════════════════════════════════════════════════
# Four-phase demodulation → B(x,y) = Q(x,y) - Q(0,y)
# ═══════════════════════════════════════════════════════════════════════
def demodulate_Q_mean(intensity_by_phase):
    """Q = (I_0 - I_pi)/4 + i*(I_{3pi/2} - I_{pi/2})/4, averaged over repeats."""
    I0 = intensity_by_phase[0].mean(axis=0)
    I1 = intensity_by_phase[1].mean(axis=0)
    I2 = intensity_by_phase[2].mean(axis=0)
    I3 = intensity_by_phase[3].mean(axis=0)
    return (I0 - I2) / 4.0 + 1j * (I3 - I1) / 4.0


def demodulate_Q_per_rep(intensity_by_phase):
    """Per-repeat demodulation."""
    n_reps = intensity_by_phase[0].shape[0]
    Q_list = []
    for r in range(n_reps):
        I0 = intensity_by_phase[0][r]
        I1 = intensity_by_phase[1][r]
        I2 = intensity_by_phase[2][r]
        I3 = intensity_by_phase[3][r]
        Q_list.append((I0 - I2) / 4.0 + 1j * (I3 - I1) / 4.0)
    return np.array(Q_list)


def compute_B_fields(blank_data, pair_data):
    """Compute mean B(x,y) and per-repeat B(x,y) for selected pairs."""
    # Blank Q(0,y) — mean
    Q_blank_mean = {}
    Q_blank_reps = {}
    for y_tid in range(N_TOKENS):
        phases = {phi: blank_data[(y_tid, phi)] for phi in range(4)}
        Q_blank_mean[y_tid] = demodulate_Q_mean(phases)
        Q_blank_reps[y_tid] = demodulate_Q_per_rep(phases)

    B_mean = {}
    B_reps = {}
    for x in range(N_TOKENS):
        for y in range(N_TOKENS):
            phases = {phi: pair_data[(x, y, phi)] for phi in range(4)}
            Q_mean = demodulate_Q_mean(phases)
            Q_reps = demodulate_Q_per_rep(phases)
            B_mean[(x, y)] = Q_mean - Q_blank_mean[y]
            B_reps[(x, y)] = Q_reps - Q_blank_reps[y]
    return B_mean, B_reps


# ═══════════════════════════════════════════════════════════════════════
# Publication-quality 2×2 figure
# ═══════════════════════════════════════════════════════════════════════
def make_refined_scatter(B_mean, B_reps, fig_path):
    """
    Create a refined 2×2 complex-plane scatter for B(0,0), B(0,1), B(1,0), B(2,3).
    """
    # ── Determine unified axis limits from all four panels ──
    all_re, all_im = [], []
    for (x, y) in SELECTED_PAIRS:
        B = B_mean[(x, y)]
        all_re.extend(np.real(B).tolist())
        all_im.extend(np.imag(B).tolist())
    
    # Use 1st–99th percentile for robust range
    re_arr = np.array(all_re)
    im_arr = np.array(all_im)
    re_lo, re_hi = np.percentile(re_arr, 1), np.percentile(re_arr, 99)
    im_lo, im_hi = np.percentile(im_arr, 1), np.percentile(im_arr, 99)
    
    # Add 15% padding and make symmetric if close to origin
    re_pad = 0.15 * (re_hi - re_lo)
    im_pad = 0.15 * (im_hi - im_lo)
    xlim = (re_lo - re_pad, re_hi + re_pad)
    ylim = (im_lo - im_pad, im_hi + im_pad)
    
    # ── Color palette ──
    self_color = '#1f77b4'     # blue for self-pair
    cross_color = '#d62728'    # red for cross-pair
    centroid_color = 'black'
    
    colors_per_panel = [self_color, cross_color, cross_color, cross_color]
    
    # ── Create figure ──
    fig, axes = plt.subplots(2, 2, figsize=(10, 9.5),
                              gridspec_kw={'hspace': 0.32, 'wspace': 0.30})
    fig.suptitle(
        r'Complex Interaction Field $B(x,y)$ in the Complex Plane'
        '\n(speckle-matched bin = 10 px, 2500 channels)',
        fontsize=13, fontweight='bold', y=0.98
    )
    
    for idx, (x, y) in enumerate(SELECTED_PAIRS):
        ax = axes[idx // 2, idx % 2]
        B = B_mean[(x, y)]
        re_B = np.real(B)
        im_B = np.imag(B)
        color = colors_per_panel[idx]
        
        # ── Per-repeat clouds (faint, to show stability) ──
        B_r = B_reps[(x, y)]
        for r in range(B_r.shape[0]):
            ax.scatter(np.real(B_r[r]), np.imag(B_r[r]),
                       s=1.5, alpha=0.08, color=color, rasterized=True)
        
        # ── Mean scatter ──
        ax.scatter(re_B, im_B, s=6, alpha=0.55, color=color,
                   edgecolors='none', rasterized=True)
        
        # ── Centroid ──
        cx, cy = re_B.mean(), im_B.mean()
        ax.plot(cx, cy, marker='*', markersize=14, color=centroid_color,
                markeredgecolor='white', markeredgewidth=0.8, zorder=10)
        
        # ── Cross-hairs at origin ──
        ax.axhline(0, color='#888888', lw=0.5, ls='-', zorder=1)
        ax.axvline(0, color='#888888', lw=0.5, ls='-', zorder=1)
        
        # ── Axis settings ──
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel(r'Re$(B)$', fontsize=11)
        ax.set_ylabel(r'Im$(B)$', fontsize=11)
        ax.tick_params(labelsize=9)
        
        # ── Panel title ──
        ax.set_title(PANEL_LABELS[idx], fontsize=12, pad=6)
        
        # ── Annotation badge ──
        ann_text = PANEL_ANNOTATIONS[idx]
        ann_color = self_color if ann_text == 'self-pair' else cross_color
        ax.text(0.97, 0.97, ann_text,
                transform=ax.transAxes, fontsize=8.5,
                ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=ann_color,
                          alpha=0.15, edgecolor=ann_color, linewidth=0.8),
                color=ann_color, fontweight='bold')
        
        # ── Centroid coordinates annotation ──
        ax.text(0.03, 0.03,
                f'centroid = ({cx:.4f}, {cy:.4f})',
                transform=ax.transAxes, fontsize=7.5,
                ha='left', va='bottom',
                color='#333333', fontstyle='italic')
    
    # ── Save ──
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  Saved refined scatter: {fig_path}  ({os.path.getsize(fig_path)} bytes)")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
def main():
    section_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    package_dir = os.path.dirname(section_dir)
    data_dir = os.path.join(package_dir, 'shared_raw_data',
                            'result4_frame_data', 'exp_bilinear_001',
                            'exp_bilinear_001')
    
    print(f"Loading XOR showcase data from: {data_dir}")
    print(f"Bin size: {BIN_SIZE} → {N_CHANNELS} channels")
    
    # Load raw data with bin=10
    blank_data = load_blank_control_data(data_dir, BIN_SIZE)
    pair_data = load_pair_interaction_data(data_dir, BIN_SIZE)
    
    print(f"  Blank controls: {len(blank_data)} keys")
    print(f"  Pair interactions: {len(pair_data)} keys")
    
    # Compute B fields
    B_mean, B_reps = compute_B_fields(blank_data, pair_data)
    
    # Report centroids for the four selected pairs
    print("\n=== B(x,y) centroids at bin=10 ===")
    for (x, y) in SELECTED_PAIRS:
        B = B_mean[(x, y)]
        cx, cy = np.real(B).mean(), np.imag(B).mean()
        mag = np.sqrt(cx**2 + cy**2)
        print(f"  B({x},{y}): centroid = ({cx:.6f}, {cy:.6f}), "
              f"|centroid| = {mag:.6f}, "
              f"Re range=[{np.real(B).min():.4f}, {np.real(B).max():.4f}], "
              f"Im range=[{np.imag(B).min():.4f}, {np.imag(B).max():.4f}]")
    
    # Report per-repeat stability for self-pair
    B00_reps = B_reps[(0, 0)]
    B00_real = np.concatenate([np.real(B00_reps), np.imag(B00_reps)], axis=1)
    corrs = []
    for i in range(B00_reps.shape[0]):
        for j in range(i+1, B00_reps.shape[0]):
            c = np.corrcoef(B00_real[i], B00_real[j])[0, 1]
            if not np.isnan(c):
                corrs.append(c)
    print(f"\n  B(0,0) repeat correlation: mean={np.mean(corrs):.4f}, "
          f"min={np.min(corrs):.4f}")
    
    # Generate figure
    fig_dir = os.path.join(section_dir, 'figures', 'phase2_refinement')
    fig_path = os.path.join(fig_dir, 'xor_complex_scatter_bin10_refined.pdf')
    make_refined_scatter(B_mean, B_reps, fig_path)
    
    print("  Supplement copy is not written by the packaged script.")
    
    print("\nDone.")


if __name__ == '__main__':
    main()
