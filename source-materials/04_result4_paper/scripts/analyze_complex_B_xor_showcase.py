#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysis: Complex-B XOR Showcase — Bilinear Physical Interaction Engine Reboot
==============================================================================
Runs on Linux. Processes data from exp_bilinear_interaction.py (old 4-token run).

REBOOT RULES (Step 164–166):
  - Primary object: complex interaction field B(x,y) ∈ C^M
  - Allowed: real-coordinate representation [Re(B), Im(B)] for numerical computation
  - FORBIDDEN: |B|, Re(B), Im(B) as standalone research objects or headline results
  - Task B redefined as true XOR/parity: label = (x + y) % 2

Core computation:
1. Bin ROIs into spatial channels (25px bins → 400 channels from 500x500)
2. Compute single-token embeddings z(x) from Group 1
3. Compute blank-control demodulated Q(0,y) from Group 2
4. Compute pair demodulated Q(x,y) from Group 3
5. Isolate complex interaction field B(x,y) = Q(x,y) - Q(0,y)
6. Compare feature families on pre-registered tasks:
   - Task A: 16-class ordered pair identification
   - Task B: XOR/parity binary relation — label = (x + y) % 2
7. Fair baselines: separate-token, digital bilinear, matched-budget

Output: figures + results JSON
"""

import os
import sys
import json
import numpy as np
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════
N_TOKENS = 4
BIN_SIZE = 25  # 500/25 = 20 bins per side → 400 channels
ROI_H, ROI_W = 500, 500
N_CHANNELS = (ROI_H // BIN_SIZE) * (ROI_W // BIN_SIZE)  # 400
PHASES = [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]
PHASE_LABELS = ["0", "pi2", "pi", "3pi2"]

# ── XOR/parity label definition ──────────────────────────────────────────
# For 4 tokens {0,1,2,3}, XOR label = (x + y) % 2
# Same-parity pairs: (0,0),(0,2),(2,0),(2,2),(1,1),(1,3),(3,1),(3,3) → label 0
# Cross-parity pairs: (0,1),(0,3),(1,0),(1,2),(2,1),(2,3),(3,0),(3,2) → label 1
# This is linearly inseparable from [z(x), z(y)] by the XOR theorem.
def xor_label(x, y):
    return (x + y) % 2


def bin_roi(roi, bin_size=BIN_SIZE):
    """Bin a 2D ROI into spatial channels by averaging bin_size x bin_size blocks."""
    h, w = roi.shape
    nh, nw = h // bin_size, w // bin_size
    cropped = roi[:nh * bin_size, :nw * bin_size]
    binned = cropped.reshape(nh, bin_size, nw, bin_size).mean(axis=(1, 3))
    return binned.flatten()


def load_and_bin(data_dir, filename):
    roi = np.load(os.path.join(data_dir, filename))
    return bin_roi(roi)


# ══════════════════════════════════════════════════════════════════════════════
# Data loading (unchanged from old script — same data format)
# ══════════════════════════════════════════════════════════════════════════════
def load_single_token_data(base_dir):
    singles_dir = os.path.join(base_dir, "singles")
    with open(os.path.join(singles_dir, "single_records.json")) as f:
        records = json.load(f)
    data = {tid: [] for tid in range(N_TOKENS)}
    for rec in records:
        vec = load_and_bin(singles_dir, rec["file"])
        data[rec["token_id"]].append(vec)
    return {tid: np.array(vecs) for tid, vecs in data.items()}


def load_blank_control_data(base_dir):
    blanks_dir = os.path.join(base_dir, "blank_controls")
    with open(os.path.join(blanks_dir, "blank_records.json")) as f:
        records = json.load(f)
    data = {}
    for rec in records:
        key = (rec["y_token_id"], rec["phi_idx"])
        if key not in data:
            data[key] = []
        vec = load_and_bin(blanks_dir, rec["file"])
        data[key].append(vec)
    return {k: np.array(v) for k, v in data.items()}


def load_pair_interaction_data(base_dir):
    pairs_dir = os.path.join(base_dir, "pair_interactions")
    with open(os.path.join(pairs_dir, "pair_records.json")) as f:
        records = json.load(f)
    data = {}
    for rec in records:
        key = (rec["x_token_id"], rec["y_token_id"], rec["phi_idx"])
        if key not in data:
            data[key] = []
        vec = load_and_bin(pairs_dir, rec["file"])
        data[key].append(vec)
    return {k: np.array(v) for k, v in data.items()}


# ══════════════════════════════════════════════════════════════════════════════
# Four-phase demodulation
# ══════════════════════════════════════════════════════════════════════════════
def demodulate_Q(intensity_by_phase):
    """Q = (I_0 - I_pi)/4 + i*(I_3pi2 - I_pi2)/4"""
    I0 = intensity_by_phase[0].mean(axis=0)
    I1 = intensity_by_phase[1].mean(axis=0)
    I2 = intensity_by_phase[2].mean(axis=0)
    I3 = intensity_by_phase[3].mean(axis=0)
    return (I0 - I2) / 4.0 + 1j * (I3 - I1) / 4.0


def compute_interaction_observables(single_data, blank_data, pair_data):
    z_single = {tid: single_data[tid].mean(axis=0) for tid in range(N_TOKENS)}
    Q_blank = {}
    for y_tid in range(N_TOKENS):
        phases_data = {phi: blank_data[(y_tid, phi)] for phi in range(4)}
        Q_blank[y_tid] = demodulate_Q(phases_data)
    Q_pair, B_pair = {}, {}
    for x in range(N_TOKENS):
        for y in range(N_TOKENS):
            phases_data = {phi: pair_data[(x, y, phi)] for phi in range(4)}
            Q_pair[(x, y)] = demodulate_Q(phases_data)
            B_pair[(x, y)] = Q_pair[(x, y)] - Q_blank[y]
    return Q_pair, Q_blank, B_pair, z_single


def compute_per_repeat_B(blank_data, pair_data):
    """Per-repeat B(x,y) for classification with proper variance."""
    Q_blank_per_rep = {}
    for y_tid in range(N_TOKENS):
        n_reps = blank_data[(y_tid, 0)].shape[0]
        Q_list = []
        for rep in range(n_reps):
            I0 = blank_data[(y_tid, 0)][rep]
            I1 = blank_data[(y_tid, 1)][rep]
            I2 = blank_data[(y_tid, 2)][rep]
            I3 = blank_data[(y_tid, 3)][rep]
            Q_list.append((I0 - I2) / 4.0 + 1j * (I3 - I1) / 4.0)
        Q_blank_per_rep[y_tid] = np.array(Q_list)

    B_per_rep = {}
    for x in range(N_TOKENS):
        for y in range(N_TOKENS):
            n_reps = pair_data[(x, y, 0)].shape[0]
            B_list = []
            for rep in range(n_reps):
                I0 = pair_data[(x, y, 0)][rep]
                I1 = pair_data[(x, y, 1)][rep]
                I2 = pair_data[(x, y, 2)][rep]
                I3 = pair_data[(x, y, 3)][rep]
                Q_xy = (I0 - I2) / 4.0 + 1j * (I3 - I1) / 4.0
                Q_0y = Q_blank_per_rep[y][rep]
                B_list.append(Q_xy - Q_0y)
            B_per_rep[(x, y)] = np.array(B_list)
    return B_per_rep


# ══════════════════════════════════════════════════════════════════════════════
# Feature construction — COMPLEX-B DISCIPLINE
# ══════════════════════════════════════════════════════════════════════════════
def build_feature_families(z_single, B_per_rep, n_reps=5):
    """
    Build feature matrices for classification.
    
    PRIMARY OBJECT: Complex-B, represented as [Re(B), Im(B)] for numerical use.
    
    Feature families:
      1. z(x) only                    — single-token baseline
      2. z(y) only                    — single-token baseline
      3. [z(x), z(y)]                — separate-token concatenation
      4. z(x)+z(y)                   — symmetric linear combination
      5. z(x)-z(y)                   — antisymmetric linear combination
      6. digital_bilinear_ew: z(x)*z(y)  — digital elementwise product (matched budget)
      7. Complex-B: [Re(B), Im(B)]   — THE PRIMARY PHYSICAL OBJECT
      8. hybrid: [z(x), z(y), Re(B), Im(B)]  — hybrid for completeness
    
    NOTE: |B| is NOT included as a standalone feature family.
          It may appear in diagnostic figures but never as a headline result.
    """
    samples = []
    pair_labels = []     # 0..15 for Task A
    xor_labels = []      # 0/1 for Task B (XOR parity)

    pair_idx_map = {}
    idx = 0
    for x in range(N_TOKENS):
        for y in range(N_TOKENS):
            pair_idx_map[(x, y)] = idx
            idx += 1

    for x in range(N_TOKENS):
        for y in range(N_TOKENS):
            pair_id = pair_idx_map[(x, y)]
            xl = xor_label(x, y)
            for rep in range(n_reps):
                zx = z_single[x]
                zy = z_single[y]
                B_xy = B_per_rep[(x, y)][rep]
                samples.append({"x": x, "y": y, "rep": rep,
                                "zx": zx, "zy": zy, "B_xy": B_xy})
                pair_labels.append(pair_id)
                xor_labels.append(xl)

    pair_labels = np.array(pair_labels)
    xor_labels = np.array(xor_labels)

    families = {}
    families["zx_only"] = np.array([s["zx"] for s in samples])
    families["zy_only"] = np.array([s["zy"] for s in samples])
    families["zx_zy_concat"] = np.array([
        np.concatenate([s["zx"], s["zy"]]) for s in samples])
    families["zx_plus_zy"] = np.array([s["zx"] + s["zy"] for s in samples])
    families["zx_minus_zy"] = np.array([s["zx"] - s["zy"] for s in samples])
    families["digital_bilinear_ew"] = np.array([
        s["zx"] * s["zy"] for s in samples])

    # PRIMARY: Complex-B as real-coordinate representation
    families["complex_B"] = np.array([
        np.concatenate([np.real(s["B_xy"]), np.imag(s["B_xy"])]) for s in samples])

    # Hybrid for completeness
    families["hybrid_zz_B"] = np.array([
        np.concatenate([s["zx"], s["zy"],
                        np.real(s["B_xy"]), np.imag(s["B_xy"])]) for s in samples])

    return families, pair_labels, xor_labels


# ══════════════════════════════════════════════════════════════════════════════
# Classification — LOO-CV ridge
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_loocv(X, y, alpha=1.0):
    from sklearn.linear_model import RidgeClassifier
    from sklearn.metrics import balanced_accuracy_score
    n = len(y)
    preds = np.zeros(n, dtype=int)
    for i in range(n):
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i)
        clf = RidgeClassifier(alpha=alpha)
        clf.fit(X_train, y_train)
        preds[i] = clf.predict(X[i:i+1])[0]
    acc = float(np.mean(preds == y))
    bal_acc = float(balanced_accuracy_score(y, preds))
    return acc, bal_acc, preds


def evaluate_compressed_budget(X, y, budget, alpha=1.0, n_random=5, seed=42):
    """Evaluate at a fixed feature budget via random subspace selection."""
    from sklearn.linear_model import RidgeClassifier
    from sklearn.metrics import balanced_accuracy_score
    n_feat = X.shape[1]
    if n_feat <= budget:
        return evaluate_loocv(X, y, alpha)
    rng = np.random.default_rng(seed)
    best_bal = -1
    best_acc, best_preds = None, None
    for _ in range(n_random):
        sel = rng.choice(n_feat, budget, replace=False)
        acc, bal, preds = evaluate_loocv(X[:, sel], y, alpha)
        if bal > best_bal:
            best_bal, best_acc, best_preds = bal, acc, preds
    return best_acc, best_bal, best_preds


def evaluate_all(families, pair_labels, xor_labels, budget=400):
    """Evaluate all families on both tasks, at full and compressed budget."""
    results = {}
    for name, X in families.items():
        if np.iscomplexobj(X):
            X = np.concatenate([np.real(X), np.imag(X)], axis=1)
        mu = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, keepdims=True)
        std[std < 1e-12] = 1.0
        X_std = (X - mu) / std

        n_feat = X_std.shape[1]
        print(f"  {name}: shape={X_std.shape}", flush=True)

        # Full budget
        acc_A, bal_A, _ = evaluate_loocv(X_std, pair_labels)
        acc_B, bal_B, _ = evaluate_loocv(X_std, xor_labels)

        entry = {
            "n_features": n_feat,
            "full_task_A_bal_acc": round(bal_A, 4),
            "full_task_B_bal_acc": round(bal_B, 4),
        }

        # Compressed budget (if features > budget)
        if n_feat > budget:
            _, cbal_A, _ = evaluate_compressed_budget(X_std, pair_labels, budget)
            _, cbal_B, _ = evaluate_compressed_budget(X_std, xor_labels, budget)
            entry["compressed_task_A_bal_acc"] = round(cbal_A, 4)
            entry["compressed_task_B_bal_acc"] = round(cbal_B, 4)
        else:
            entry["compressed_task_A_bal_acc"] = entry["full_task_A_bal_acc"]
            entry["compressed_task_B_bal_acc"] = entry["full_task_B_bal_acc"]

        results[name] = entry
        print(f"    Task A: {bal_A:.4f} (full), Task B XOR: {bal_B:.4f} (full)", flush=True)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Stability analysis — on Complex-B, not |B|
# ══════════════════════════════════════════════════════════════════════════════
def analyze_complex_B_stability(B_per_rep):
    """Assess repeat-to-repeat stability of the complex field B(x,y)."""
    stability = {}
    for (x, y), B_arr in B_per_rep.items():
        n_reps = B_arr.shape[0]
        # Correlation on real-coordinate representation
        B_real = np.concatenate([np.real(B_arr), np.imag(B_arr)], axis=1)
        corrs = []
        for i in range(n_reps):
            for j in range(i+1, n_reps):
                c = np.corrcoef(B_real[i], B_real[j])[0, 1]
                if not np.isnan(c):
                    corrs.append(c)
        # Also report magnitude and phase statistics for diagnostics
        mag = np.abs(B_arr)
        phase = np.angle(B_arr)
        stability[f"({x},{y})"] = {
            "mean_complex_corr": float(np.mean(corrs)) if corrs else 0.0,
            "min_complex_corr": float(np.min(corrs)) if corrs else 0.0,
            "mean_magnitude": float(mag.mean()),
            "mean_phase_std_rad": float(np.std(phase, axis=0).mean()),
        }
    return stability


# ══════════════════════════════════════════════════════════════════════════════
# Figures — Complex-B centered
# ══════════════════════════════════════════════════════════════════════════════
def make_figures(results, B_pair, z_single, B_per_rep, pair_data, fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(fig_dir, exist_ok=True)

    # ── Fig 1: Phase-cycle traces ─────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    phi_vals = [0, np.pi/2, np.pi, 3*np.pi/2]
    for idx, (x, y) in enumerate([(0,1), (1,2), (2,3), (0,0)]):
        ax = axes[idx // 2, idx % 2]
        for ch in [50, 100, 200, 300]:
            means = [pair_data[(x, y, phi)][:, ch].mean() for phi in range(4)]
            ax.plot(phi_vals, means, 'o-', label=f'ch={ch}')
        ax.set_title(f'Pair ({x},{y})')
        ax.set_xlabel('Phase φ (rad)')
        ax.set_ylabel('Mean intensity')
        ax.legend(fontsize=7)
    fig.suptitle('Four-phase intensity modulation (raw)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig1_phase_cycle_traces.pdf"),
                bbox_inches="tight", dpi=300)
    plt.close()

    # ── Fig 2: Complex-B field visualization ──────────────────────────
    # Show B(x,y) as 2D complex-plane scatter for selected channels
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    selected_pairs = [(0,1), (1,0), (0,0), (2,3)]
    for idx, (x, y) in enumerate(selected_pairs):
        ax = axes[idx // 2, idx % 2]
        B = B_pair[(x, y)]
        # Plot each channel as a point in the complex plane
        ax.scatter(np.real(B), np.imag(B), s=8, alpha=0.6, c=np.arange(len(B)),
                   cmap='viridis')
        ax.axhline(0, color='gray', lw=0.5)
        ax.axvline(0, color='gray', lw=0.5)
        ax.set_xlabel('Re(B)')
        ax.set_ylabel('Im(B)')
        ax.set_title(f'B({x},{y}) — {len(B)} channels in complex plane')
        ax.set_aspect('equal')
    fig.suptitle('Complex interaction field B(x,y) — channel distribution', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig2_complex_B_scatter.pdf"),
                bbox_inches="tight", dpi=300)
    plt.close()

    # ── Fig 3: Complex-B channel profiles ─────────────────────────────
    fig, axes = plt.subplots(4, 4, figsize=(16, 14))
    for x in range(N_TOKENS):
        for y in range(N_TOKENS):
            ax = axes[x, y]
            B = B_pair[(x, y)]
            ax.plot(np.real(B), label='Re(B)', alpha=0.7, lw=0.8)
            ax.plot(np.imag(B), label='Im(B)', alpha=0.7, lw=0.8)
            ax.set_title(f'B({x},{y})', fontsize=9)
            if x == 3:
                ax.set_xlabel('Channel', fontsize=8)
            if y == 0:
                ax.set_ylabel('Value', fontsize=8)
            if x == 0 and y == 0:
                ax.legend(fontsize=6)
    fig.suptitle('Complex interaction field profiles — all 16 ordered pairs', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig3_complex_B_profiles_all_pairs.pdf"),
                bbox_inches="tight", dpi=300)
    plt.close()

    # ── Fig 4: Complex-B repeat stability ─────────────────────────────
    stability = analyze_complex_B_stability(B_per_rep)
    pairs_sorted = sorted(stability.keys())
    corrs = [stability[p]["mean_complex_corr"] for p in pairs_sorted]

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ['#2ecc71' if c > 0.9 else '#e74c3c' for c in corrs]
    ax.bar(range(len(pairs_sorted)), corrs, color=colors)
    ax.set_xticks(range(len(pairs_sorted)))
    ax.set_xticklabels(pairs_sorted, rotation=45, fontsize=7)
    ax.set_ylabel('Mean repeat correlation (complex coordinate)')
    ax.set_title('Complex-B repeat-to-repeat stability')
    ax.axhline(0.9, color='r', ls='--', lw=0.8, label='threshold=0.9')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig4_complex_B_stability.pdf"),
                bbox_inches="tight", dpi=300)
    plt.close()

    # ── Fig 5: XOR showcase — classification comparison ───────────────
    family_order = ["zx_only", "zy_only", "zx_zy_concat", "zx_plus_zy",
                    "zx_minus_zy", "digital_bilinear_ew", "complex_B", "hybrid_zz_B"]
    family_order = [f for f in family_order if f in results]

    task_A_vals = [results[f]["compressed_task_A_bal_acc"] for f in family_order]
    task_B_vals = [results[f]["compressed_task_B_bal_acc"] for f in family_order]

    x_pos = np.arange(len(family_order))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x_pos - width/2, task_A_vals, width,
                   label='Task A (16-class pair ID)', color='#3498db')
    bars2 = ax.bar(x_pos + width/2, task_B_vals, width,
                   label='Task B (XOR parity)', color='#e67e22')
    ax.set_ylabel('Balanced Accuracy (matched budget)')
    ax.set_title('Complex-B XOR Showcase — Feature Family Comparison')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(family_order, rotation=35, ha='right', fontsize=9)
    ax.legend(loc='upper left')
    ax.axhline(1.0/16, color='gray', ls=':', alpha=0.5, label='Task A chance')
    ax.axhline(0.5, color='gray', ls='--', alpha=0.3)

    for bar, val in zip(bars1, task_A_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=7)
    for bar, val in zip(bars2, task_B_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig5_xor_showcase_comparison.pdf"),
                bbox_inches="tight", dpi=300)
    plt.close()

    # ── Fig 6: XOR class structure in Complex-B space ─────────────────
    # 2D PCA projection of Complex-B, colored by XOR label
    from sklearn.decomposition import PCA

    B_all = []
    xor_all = []
    pair_all = []
    for x in range(N_TOKENS):
        for y in range(N_TOKENS):
            B_reps = B_per_rep[(x, y)]
            for rep in range(B_reps.shape[0]):
                B_vec = np.concatenate([np.real(B_reps[rep]),
                                        np.imag(B_reps[rep])])
                B_all.append(B_vec)
                xor_all.append(xor_label(x, y))
                pair_all.append(f"({x},{y})")
    B_all = np.array(B_all)
    xor_all = np.array(xor_all)

    pca = PCA(n_components=2)
    B_2d = pca.fit_transform(B_all)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Color by XOR label
    for label, color, name in [(0, '#3498db', 'Same parity'), (1, '#e74c3c', 'Cross parity')]:
        mask = xor_all == label
        ax1.scatter(B_2d[mask, 0], B_2d[mask, 1], c=color, s=20, alpha=0.7, label=name)
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax1.set_title('Complex-B PCA — colored by XOR parity')
    ax1.legend()

    # Color by pair identity
    unique_pairs = sorted(set(pair_all))
    cmap = plt.cm.get_cmap('tab20', len(unique_pairs))
    for i, p in enumerate(unique_pairs):
        mask = np.array([pp == p for pp in pair_all])
        ax2.scatter(B_2d[mask, 0], B_2d[mask, 1], c=[cmap(i)], s=20, alpha=0.7,
                    label=p if i < 8 else None)
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax2.set_title('Complex-B PCA — colored by pair identity')
    ax2.legend(fontsize=6, ncol=2)

    fig.suptitle('XOR structure in Complex-B space', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig6_xor_structure_pca.pdf"),
                bbox_inches="tight", dpi=300)
    plt.close()

    print(f"  Saved 6 figures to {fig_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Complex-B XOR Showcase Analysis (Bilinear Reboot)")
    parser.add_argument("--data-dir", required=True, help="Experiment data directory")
    parser.add_argument("--output-dir", required=True, help="Analysis output directory")
    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    print("=" * 60)
    print("Complex-B XOR Showcase — Bilinear Interaction Engine Reboot")
    print("=" * 60)

    # Load metadata
    with open(os.path.join(data_dir, "metadata.json")) as f:
        metadata = json.load(f)
    print(f"Experiment: {metadata['experiment']}")
    print(f"Total measurements: {metadata['measurement_counts']['total']}")

    # Load data
    print("\nLoading data...")
    single_data = load_single_token_data(data_dir)
    blank_data = load_blank_control_data(data_dir)
    pair_data = load_pair_interaction_data(data_dir)
    print(f"  Singles: {sum(v.shape[0] for v in single_data.values())}")
    print(f"  Blanks: {sum(v.shape[0] for v in blank_data.values())}")
    print(f"  Pairs: {sum(v.shape[0] for v in pair_data.values())}")

    # Compute observables
    print("\nComputing complex interaction fields...")
    Q_pair, Q_blank, B_pair, z_single = compute_interaction_observables(
        single_data, blank_data, pair_data)
    B_per_rep = compute_per_repeat_B(blank_data, pair_data)

    # Phase-cycle validation
    print("\nPhase-cycle validation (pair (0,1), channel 100):")
    for phi in range(4):
        vals = pair_data[(0, 1, phi)][:, 100]
        print(f"  phi={PHASE_LABELS[phi]}: mean={vals.mean():.4f}, std={vals.std():.4f}")

    # Complex-B stability
    print("\nComplex-B stability analysis:")
    stability = analyze_complex_B_stability(B_per_rep)
    mean_corrs = [s["mean_complex_corr"] for s in stability.values()]
    print(f"  Mean complex-coordinate repeat correlation: {np.mean(mean_corrs):.4f}")
    print(f"  Min complex-coordinate repeat correlation: {np.min(mean_corrs):.4f}")

    # Build features — COMPLEX-B DISCIPLINE
    print("\nBuilding feature families (Complex-B discipline)...")
    families, pair_labels, xor_labels = build_feature_families(z_single, B_per_rep)
    for name, X in families.items():
        print(f"  {name}: shape={X.shape}")

    # XOR label verification
    print(f"\nXOR label distribution: {np.bincount(xor_labels)} "
          f"(class 0: same parity, class 1: cross parity)")

    # Evaluate
    print("\nEvaluating classification (LOO-CV, matched budget=400)...")
    results = evaluate_all(families, pair_labels, xor_labels, budget=400)

    # Generate figures
    print("\nGenerating figures...")
    make_figures(results, B_pair, z_single, B_per_rep, pair_data, fig_dir)

    # Save results
    output_results = {
        "experiment": "complex_B_xor_showcase_reboot",
        "primary_object": "complex interaction field B(x,y) = conj(TE_x) * (TE_y)",
        "object_discipline": "Complex-B only; |B| not used as headline",
        "task_B_definition": "XOR parity: label = (x + y) % 2",
        "metadata_summary": {
            "total_measurements": metadata["measurement_counts"]["total"],
            "total_time_sec": metadata["timing"]["total_sec"],
        },
        "complex_B_stability": stability,
        "classification_results": results,
        "n_samples": int(len(pair_labels)),
        "n_channels": N_CHANNELS,
        "xor_label_counts": np.bincount(xor_labels).tolist(),
    }

    results_path = os.path.join(output_dir, "complex_B_xor_results.json")
    with open(results_path, "w") as f:
        json.dump(output_results, f, indent=2)

    # Print key comparison
    print("\n" + "=" * 60)
    print("KEY XOR SHOWCASE RESULTS (matched budget = 400 features)")
    print("=" * 60)
    print(f"\nTask A (16-class pair identification):")
    for name in sorted(results.keys(),
                       key=lambda n: results[n]["compressed_task_A_bal_acc"], reverse=True):
        r = results[name]
        print(f"  {name:25s}  {r['compressed_task_A_bal_acc']:.4f}  ({r['n_features']} feat)")

    print(f"\nTask B (XOR parity — the showcase task):")
    for name in sorted(results.keys(),
                       key=lambda n: results[n]["compressed_task_B_bal_acc"], reverse=True):
        r = results[name]
        print(f"  {name:25s}  {r['compressed_task_B_bal_acc']:.4f}  ({r['n_features']} feat)")

    # The critical XOR comparison
    cB = results.get("complex_B", {})
    concat = results.get("zx_zy_concat", {})
    dig = results.get("digital_bilinear_ew", {})
    print(f"\n--- CRITICAL XOR COMPARISON ---")
    print(f"  [z(x),z(y)] concat on XOR:    {concat.get('compressed_task_B_bal_acc', 'N/A')}")
    print(f"  Digital bilinear on XOR:       {dig.get('compressed_task_B_bal_acc', 'N/A')}")
    print(f"  Complex-B on XOR:              {cB.get('compressed_task_B_bal_acc', 'N/A')}")
    print(f"  Complex-B on pair ID:          {cB.get('compressed_task_A_bal_acc', 'N/A')}")

    print(f"\nResults saved to: {results_path}")
    print(f"Figures saved to: {fig_dir}")


if __name__ == "__main__":
    main()
