#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysis: Bilinear Token Interaction Engine — Proof-of-Principle
================================================================
Runs on Linux. Processes data from exp_bilinear_interaction.py.

Core computation:
1. Bin ROIs into spatial channels (25px bins → 400 channels from 500x500)
2. Compute single-token embeddings z(x) from Group 1
3. Compute blank-control demodulated Q(0,y) from Group 2
4. Compute pair demodulated Q(x,y) from Group 3
5. Isolate interaction observable B(x,y) = Q(x,y) - Q(0,y)
6. Compare feature families on pre-registered tasks:
   - Task A: 16-class ordered pair identification
   - Task B: binary relation (successor/predecessor vs other)
7. Include digital bilinear baseline controls

Output: figures + results JSON + assessment markdown
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

# Pre-registered binary relation
POSITIVE_PAIRS = {(0,1), (1,2), (2,3), (3,0), (1,0), (2,1), (3,2), (0,3)}


def bin_roi(roi, bin_size=BIN_SIZE):
    """Bin a 2D ROI into spatial channels by averaging bin_size x bin_size blocks."""
    h, w = roi.shape
    nh, nw = h // bin_size, w // bin_size
    cropped = roi[:nh * bin_size, :nw * bin_size]
    binned = cropped.reshape(nh, bin_size, nw, bin_size).mean(axis=(1, 3))
    return binned.flatten()


def load_and_bin(data_dir, filename):
    """Load a .npy ROI and bin it to channel vector."""
    roi = np.load(os.path.join(data_dir, filename))
    return bin_roi(roi)


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════
def load_single_token_data(base_dir):
    """Load Group 1: single-token baselines. Returns dict: tid -> (n_reps, n_channels)."""
    singles_dir = os.path.join(base_dir, "singles")
    with open(os.path.join(singles_dir, "single_records.json")) as f:
        records = json.load(f)

    data = {tid: [] for tid in range(N_TOKENS)}
    for rec in records:
        vec = load_and_bin(singles_dir, rec["file"])
        data[rec["token_id"]].append(vec)

    return {tid: np.array(vecs) for tid, vecs in data.items()}


def load_blank_control_data(base_dir):
    """Load Group 2: blank-control phase cycles.
    Returns dict: (y_tid, phi_idx) -> (n_reps, n_channels)."""
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
    """Load Group 3: pair interaction phase cycles.
    Returns dict: (x_tid, y_tid, phi_idx) -> (n_reps, n_channels)."""
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
# Core computation: four-phase demodulation
# ══════════════════════════════════════════════════════════════════════════════
def demodulate_Q(intensity_by_phase):
    """
    Four-phase demodulation to extract complex Q.
    intensity_by_phase: dict phi_idx -> (n_reps, n_channels) intensity arrays
    Returns: complex Q vector (n_channels,) averaged over repeats.

    Q = (I_0 - I_pi)/4 + i*(I_3pi2 - I_pi2)/4
    """
    I0 = intensity_by_phase[0].mean(axis=0)      # phi = 0
    I1 = intensity_by_phase[1].mean(axis=0)      # phi = pi/2
    I2 = intensity_by_phase[2].mean(axis=0)      # phi = pi
    I3 = intensity_by_phase[3].mean(axis=0)      # phi = 3pi/2

    Q_real = (I0 - I2) / 4.0
    Q_imag = (I3 - I1) / 4.0
    return Q_real + 1j * Q_imag


def compute_interaction_observables(single_data, blank_data, pair_data):
    """
    Compute all interaction observables B(x,y) for all ordered pairs.

    Returns:
        Q_pair: dict (x,y) -> complex Q(x,y) vector
        Q_blank: dict y -> complex Q(0,y) vector
        B_pair: dict (x,y) -> complex B(x,y) = Q(x,y) - Q(0,y) vector
        z_single: dict tid -> mean single-token embedding (real)
    """
    # Single-token mean embeddings
    z_single = {}
    for tid in range(N_TOKENS):
        z_single[tid] = single_data[tid].mean(axis=0)

    # Blank-control demodulation: Q(0,y) for each y
    Q_blank = {}
    for y_tid in range(N_TOKENS):
        phases_data = {}
        for phi_idx in range(4):
            phases_data[phi_idx] = blank_data[(y_tid, phi_idx)]
        Q_blank[y_tid] = demodulate_Q(phases_data)

    # Pair demodulation: Q(x,y) for each (x,y)
    Q_pair = {}
    B_pair = {}
    for x_tid in range(N_TOKENS):
        for y_tid in range(N_TOKENS):
            phases_data = {}
            for phi_idx in range(4):
                phases_data[phi_idx] = pair_data[(x_tid, y_tid, phi_idx)]
            Q_pair[(x_tid, y_tid)] = demodulate_Q(phases_data)

            # Isolate interaction: B(x,y) = Q(x,y) - Q(0,y)
            B_pair[(x_tid, y_tid)] = Q_pair[(x_tid, y_tid)] - Q_blank[y_tid]

    return Q_pair, Q_blank, B_pair, z_single


# ══════════════════════════════════════════════════════════════════════════════
# Per-repeat interaction observables (for stability and classification)
# ══════════════════════════════════════════════════════════════════════════════
def compute_per_repeat_B(blank_data, pair_data):
    """
    Compute B(x,y) for each individual repeat (not averaged).
    Uses per-repeat four-phase demodulation.

    Returns: dict (x,y) -> (n_reps, n_channels) complex array
    """
    # Per-repeat blank Q(0,y)
    Q_blank_per_rep = {}
    for y_tid in range(N_TOKENS):
        n_reps = blank_data[(y_tid, 0)].shape[0]
        Q_list = []
        for rep in range(n_reps):
            I0 = blank_data[(y_tid, 0)][rep]
            I1 = blank_data[(y_tid, 1)][rep]
            I2 = blank_data[(y_tid, 2)][rep]
            I3 = blank_data[(y_tid, 3)][rep]
            Q_r = (I0 - I2) / 4.0
            Q_i = (I3 - I1) / 4.0
            Q_list.append(Q_r + 1j * Q_i)
        Q_blank_per_rep[y_tid] = np.array(Q_list)

    # Per-repeat pair Q(x,y) and B(x,y)
    B_per_rep = {}
    for x_tid in range(N_TOKENS):
        for y_tid in range(N_TOKENS):
            n_reps = pair_data[(x_tid, y_tid, 0)].shape[0]
            B_list = []
            for rep in range(n_reps):
                I0 = pair_data[(x_tid, y_tid, 0)][rep]
                I1 = pair_data[(x_tid, y_tid, 1)][rep]
                I2 = pair_data[(x_tid, y_tid, 2)][rep]
                I3 = pair_data[(x_tid, y_tid, 3)][rep]
                Q_r = (I0 - I2) / 4.0
                Q_i = (I3 - I1) / 4.0
                Q_xy = Q_r + 1j * Q_i
                Q_0y = Q_blank_per_rep[y_tid][rep]
                B_list.append(Q_xy - Q_0y)
            B_per_rep[(x_tid, y_tid)] = np.array(B_list)

    return B_per_rep


# ══════════════════════════════════════════════════════════════════════════════
# Feature construction for classification
# ══════════════════════════════════════════════════════════════════════════════
def build_feature_families(z_single, B_per_rep, single_data, n_reps=5):
    """
    Build feature matrices for each feature family across all pair-repeat samples.

    For each ordered pair (x,y) and each repeat r, construct:
      1. z(x) only
      2. z(y) only
      3. [z(x), z(y)] concatenation
      4. z(x)+z(y)
      5. z(x)-z(y)
      6. digital bilinear: elementwise z(x)*z(y)
      7. digital bilinear random sketch: W @ (z(x) kron z(y)) with random W
      8. physical |B(x,y)| (magnitude)
      9. physical [Re(B), Im(B)]
     10. hybrid [z(x), z(y), |B(x,y)|]
     11. hybrid [z(x), z(y), Re(B), Im(B)]

    Returns: dict family_name -> (n_samples, n_features) array, plus labels arrays
    """
    samples = []
    pair_labels = []  # 0..15 for Task A
    binary_labels = []  # 0/1 for Task B

    pair_idx_map = {}
    idx = 0
    for x in range(N_TOKENS):
        for y in range(N_TOKENS):
            pair_idx_map[(x, y)] = idx
            idx += 1

    for x_tid in range(N_TOKENS):
        for y_tid in range(N_TOKENS):
            pair_id = pair_idx_map[(x_tid, y_tid)]
            bl = 1 if (x_tid, y_tid) in POSITIVE_PAIRS else 0

            for rep in range(n_reps):
                # Use mean single-token embedding (averaged over all single repeats)
                zx = z_single[x_tid]
                zy = z_single[y_tid]

                # Per-repeat interaction
                B_xy = B_per_rep[(x_tid, y_tid)][rep]

                samples.append({
                    "x_tid": x_tid,
                    "y_tid": y_tid,
                    "rep": rep,
                    "zx": zx,
                    "zy": zy,
                    "B_xy": B_xy,
                })
                pair_labels.append(pair_id)
                binary_labels.append(bl)

    pair_labels = np.array(pair_labels)
    binary_labels = np.array(binary_labels)
    n_samples = len(samples)

    # Build feature matrices
    families = {}

    # 1. z(x) only
    families["zx_only"] = np.array([s["zx"] for s in samples])

    # 2. z(y) only
    families["zy_only"] = np.array([s["zy"] for s in samples])

    # 3. [z(x), z(y)]
    families["zx_zy_concat"] = np.array([np.concatenate([s["zx"], s["zy"]]) for s in samples])

    # 4. z(x)+z(y)
    families["zx_plus_zy"] = np.array([s["zx"] + s["zy"] for s in samples])

    # 5. z(x)-z(y)
    families["zx_minus_zy"] = np.array([s["zx"] - s["zy"] for s in samples])

    # 6. digital bilinear: elementwise product
    families["digital_bilinear_ew"] = np.array([s["zx"] * s["zy"] for s in samples])

    # 7. digital bilinear random sketch (project outer product to 400 dims)
    rng = np.random.default_rng(12345)
    W_sketch = rng.standard_normal((N_CHANNELS, N_CHANNELS)) / np.sqrt(N_CHANNELS)
    families["digital_bilinear_sketch"] = np.array([
        W_sketch @ (s["zx"] * s["zy"]) for s in samples
    ])

    # 8. physical |B(x,y)|
    families["physical_B_mag"] = np.array([np.abs(s["B_xy"]) for s in samples])

    # 9. physical [Re(B), Im(B)]
    families["physical_B_complex"] = np.array([
        np.concatenate([np.real(s["B_xy"]), np.imag(s["B_xy"])]) for s in samples
    ])

    # 10. hybrid [z(x), z(y), |B|]
    families["hybrid_mag"] = np.array([
        np.concatenate([s["zx"], s["zy"], np.abs(s["B_xy"])]) for s in samples
    ])

    # 11. hybrid [z(x), z(y), Re(B), Im(B)]
    families["hybrid_complex"] = np.array([
        np.concatenate([s["zx"], s["zy"], np.real(s["B_xy"]), np.imag(s["B_xy"])])
        for s in samples
    ])

    return families, pair_labels, binary_labels


# ══════════════════════════════════════════════════════════════════════════════
# Classification evaluation
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_loocv(X, y, alpha=1.0):
    """
    Leave-one-out cross-validation with ridge regression.
    For multi-class: one-vs-rest encoding.
    Returns: accuracy, balanced_accuracy, predictions
    """
    from sklearn.linear_model import RidgeClassifier
    from sklearn.metrics import balanced_accuracy_score

    n = len(y)
    preds = np.zeros(n, dtype=int)

    for i in range(n):
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i)
        X_test = X[i:i+1]

        clf = RidgeClassifier(alpha=alpha)
        clf.fit(X_train, y_train)
        preds[i] = clf.predict(X_test)[0]

    acc = np.mean(preds == y)
    bal_acc = balanced_accuracy_score(y, preds)
    return float(acc), float(bal_acc), preds


def evaluate_all_families(families, pair_labels, binary_labels):
    """Evaluate all feature families on both tasks."""
    results = {}

    for name, X in families.items():
        # Handle complex features by taking real part if needed
        if np.iscomplexobj(X):
            X = np.concatenate([np.real(X), np.imag(X)], axis=1)

        # Standardize
        mu = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, keepdims=True)
        std[std < 1e-12] = 1.0
        X_std = (X - mu) / std

        print(f"  Evaluating {name} ... shape={X_std.shape}", flush=True)

        # Task A: 16-class pair identification
        acc_A, bal_A, _ = evaluate_loocv(X_std, pair_labels, alpha=1.0)

        # Task B: binary relation
        acc_B, bal_B, _ = evaluate_loocv(X_std, binary_labels, alpha=1.0)

        results[name] = {
            "n_features": int(X_std.shape[1]),
            "task_A_accuracy": round(acc_A, 4),
            "task_A_balanced_accuracy": round(bal_A, 4),
            "task_B_accuracy": round(acc_B, 4),
            "task_B_balanced_accuracy": round(bal_B, 4),
        }
        print(f"    Task A bal_acc={bal_A:.4f}, Task B bal_acc={bal_B:.4f}", flush=True)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Interaction stability analysis
# ══════════════════════════════════════════════════════════════════════════════
def analyze_interaction_stability(B_per_rep):
    """Assess repeat-to-repeat stability of B(x,y)."""
    stability = {}
    for (x, y), B_arr in B_per_rep.items():
        # B_arr: (n_reps, n_channels) complex
        mag = np.abs(B_arr)
        mean_mag = mag.mean(axis=0)
        std_mag = mag.std(axis=0)
        cv = np.where(mean_mag > 1e-10, std_mag / mean_mag, 0.0)

        # Cross-repeat correlation of magnitude
        n_reps = B_arr.shape[0]
        corrs = []
        for i in range(n_reps):
            for j in range(i+1, n_reps):
                c = np.corrcoef(mag[i], mag[j])[0, 1]
                if not np.isnan(c):
                    corrs.append(c)

        stability[f"({x},{y})"] = {
            "mean_magnitude": float(mean_mag.mean()),
            "mean_cv": float(cv[mean_mag > 1e-10].mean()) if (mean_mag > 1e-10).any() else 0.0,
            "mean_repeat_corr": float(np.mean(corrs)) if corrs else 0.0,
        }

    return stability


# ══════════════════════════════════════════════════════════════════════════════
# Phase-cycle validation
# ══════════════════════════════════════════════════════════════════════════════
def validate_phase_cycle(pair_data, x_tid=0, y_tid=1, channel_idx=100):
    """
    Check that intensity varies sinusoidally with phase for a representative
    pair and channel. Returns phase-intensity trace.
    """
    traces = {}
    for phi_idx in range(4):
        intensities = pair_data[(x_tid, y_tid, phi_idx)][:, channel_idx]
        traces[PHASE_LABELS[phi_idx]] = {
            "mean": float(intensities.mean()),
            "std": float(intensities.std()),
            "values": intensities.tolist(),
        }
    return traces


# ══════════════════════════════════════════════════════════════════════════════
# Figures
# ══════════════════════════════════════════════════════════════════════════════
def make_figures(results, B_pair, z_single, Q_pair, Q_blank,
                 B_per_rep, pair_data, fig_dir):
    """Generate all analysis figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(fig_dir, exist_ok=True)

    # ── Fig 1: Phase-cycle traces for representative pairs ────────────
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    phi_vals = [0, np.pi/2, np.pi, 3*np.pi/2]
    for idx, (x, y) in enumerate([(0,1), (1,2), (2,3), (0,0)]):
        ax = axes[idx // 2, idx % 2]
        for ch in [50, 100, 200, 300]:
            means = []
            for phi_idx in range(4):
                means.append(pair_data[(x, y, phi_idx)][:, ch].mean())
            ax.plot(phi_vals, means, 'o-', label=f'ch={ch}')
        ax.set_title(f'Pair ({x},{y})')
        ax.set_xlabel('Phase (rad)')
        ax.set_ylabel('Mean intensity')
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig_phase_cycle_traces.png"), dpi=200)
    plt.close()

    # ── Fig 2: |B(x,y)| heatmap across pairs ─────────────────────────
    B_mag_matrix = np.zeros((N_TOKENS, N_TOKENS))
    for x in range(N_TOKENS):
        for y in range(N_TOKENS):
            B_mag_matrix[x, y] = np.abs(B_pair[(x, y)]).mean()

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(B_mag_matrix, cmap='viridis')
    ax.set_xlabel('Token y')
    ax.set_ylabel('Token x')
    ax.set_title('Mean |B(x,y)| across channels')
    ax.set_xticks(range(N_TOKENS))
    ax.set_yticks(range(N_TOKENS))
    for i in range(N_TOKENS):
        for j in range(N_TOKENS):
            ax.text(j, i, f'{B_mag_matrix[i,j]:.3f}', ha='center', va='center',
                    color='white' if B_mag_matrix[i,j] < B_mag_matrix.max()*0.7 else 'black',
                    fontsize=9)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig_B_magnitude_heatmap.png"), dpi=200)
    plt.close()

    # ── Fig 3: Interaction stability (repeat correlation) ─────────────
    stability = analyze_interaction_stability(B_per_rep)
    pairs_sorted = sorted(stability.keys())
    corrs = [stability[p]["mean_repeat_corr"] for p in pairs_sorted]
    cvs = [stability[p]["mean_cv"] for p in pairs_sorted]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.bar(range(len(pairs_sorted)), corrs)
    ax1.set_xticks(range(len(pairs_sorted)))
    ax1.set_xticklabels(pairs_sorted, rotation=45, fontsize=7)
    ax1.set_ylabel('Mean repeat correlation')
    ax1.set_title('B(x,y) repeat-to-repeat stability')
    ax1.axhline(0.9, color='r', ls='--', label='threshold=0.9')
    ax1.legend()

    ax2.bar(range(len(pairs_sorted)), cvs)
    ax2.set_xticks(range(len(pairs_sorted)))
    ax2.set_xticklabels(pairs_sorted, rotation=45, fontsize=7)
    ax2.set_ylabel('Mean CV')
    ax2.set_title('B(x,y) coefficient of variation')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig_B_stability.png"), dpi=200)
    plt.close()

    # ── Fig 4: Classification comparison bar chart ────────────────────
    family_names = list(results.keys())
    task_A_vals = [results[n]["task_A_balanced_accuracy"] for n in family_names]
    task_B_vals = [results[n]["task_B_balanced_accuracy"] for n in family_names]

    x_pos = np.arange(len(family_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x_pos - width/2, task_A_vals, width, label='Task A (16-class pair ID)')
    bars2 = ax.bar(x_pos + width/2, task_B_vals, width, label='Task B (binary relation)')
    ax.set_ylabel('Balanced Accuracy')
    ax.set_title('Feature Family Comparison')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(family_names, rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.axhline(1.0/16, color='gray', ls=':', label='Task A chance')
    ax.axhline(0.5, color='gray', ls='--', alpha=0.5)

    for bar, val in zip(bars1, task_A_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=6)
    for bar, val in zip(bars2, task_B_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=6)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig_classification_comparison.png"), dpi=200)
    plt.close()

    # ── Fig 5: B(x,y) channel profiles for selected pairs ────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for idx, (x, y) in enumerate([(0,1), (1,0), (0,0), (2,3)]):
        ax = axes[idx // 2, idx % 2]
        B = B_pair[(x, y)]
        ax.plot(np.real(B), label='Re(B)', alpha=0.7)
        ax.plot(np.imag(B), label='Im(B)', alpha=0.7)
        ax.plot(np.abs(B), label='|B|', alpha=0.7)
        ax.set_title(f'B({x},{y}) channel profile')
        ax.set_xlabel('Channel index')
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig_B_channel_profiles.png"), dpi=200)
    plt.close()

    print(f"  Saved 5 figures to {fig_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze bilinear interaction data")
    parser.add_argument("--data-dir", required=True, help="Experiment data directory")
    parser.add_argument("--output-dir", required=True, help="Analysis output directory")
    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    print("=" * 60)
    print("Bilinear Interaction Engine — Analysis")
    print("=" * 60)

    # Load metadata
    with open(os.path.join(data_dir, "metadata.json")) as f:
        metadata = json.load(f)
    print(f"Experiment: {metadata['experiment']}")
    print(f"Total measurements: {metadata['measurement_counts']['total']}")

    # Load all data groups
    print("\nLoading data...")
    single_data = load_single_token_data(data_dir)
    blank_data = load_blank_control_data(data_dir)
    pair_data = load_pair_interaction_data(data_dir)

    print(f"  Singles: {sum(v.shape[0] for v in single_data.values())} measurements")
    print(f"  Blanks: {sum(v.shape[0] for v in blank_data.values())} measurements")
    print(f"  Pairs: {sum(v.shape[0] for v in pair_data.values())} measurements")

    # Compute interaction observables (averaged)
    print("\nComputing interaction observables...")
    Q_pair, Q_blank, B_pair, z_single = compute_interaction_observables(
        single_data, blank_data, pair_data
    )

    # Compute per-repeat B for classification
    print("Computing per-repeat B(x,y)...")
    B_per_rep = compute_per_repeat_B(blank_data, pair_data)

    # Validate phase cycle
    print("\nPhase-cycle validation...")
    phase_trace = validate_phase_cycle(pair_data, x_tid=0, y_tid=1, channel_idx=100)
    for pl, tr in phase_trace.items():
        print(f"  phi={pl}: mean={tr['mean']:.4f}, std={tr['std']:.4f}")

    # Interaction stability
    print("\nInteraction stability analysis...")
    stability = analyze_interaction_stability(B_per_rep)
    mean_corrs = [s["mean_repeat_corr"] for s in stability.values()]
    print(f"  Mean repeat correlation across all pairs: {np.mean(mean_corrs):.4f}")
    print(f"  Min repeat correlation: {np.min(mean_corrs):.4f}")

    # Build feature families
    print("\nBuilding feature families...")
    families, pair_labels, binary_labels = build_feature_families(
        z_single, B_per_rep, single_data
    )
    for name, X in families.items():
        print(f"  {name}: shape={X.shape}")

    # Evaluate classification
    print("\nEvaluating classification (LOO-CV)...")
    results = evaluate_all_families(families, pair_labels, binary_labels)

    # Generate figures
    print("\nGenerating figures...")
    make_figures(results, B_pair, z_single, Q_pair, Q_blank,
                 B_per_rep, pair_data, fig_dir)

    # Save results
    output_results = {
        "metadata_summary": {
            "experiment": metadata["experiment"],
            "total_measurements": metadata["measurement_counts"]["total"],
            "total_time_sec": metadata["timing"]["total_sec"],
        },
        "phase_cycle_validation": phase_trace,
        "interaction_stability": stability,
        "classification_results": results,
        "n_samples": len(pair_labels),
        "n_channels": N_CHANNELS,
        "task_A_classes": 16,
        "task_B_positive_fraction": float(binary_labels.mean()),
    }

    results_path = os.path.join(output_dir, "bilinear_analysis_results.json")
    with open(results_path, "w") as f:
        json.dump(output_results, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"\nTask A (16-class pair identification):")
    for name in sorted(results.keys(), key=lambda n: results[n]["task_A_balanced_accuracy"], reverse=True):
        r = results[name]
        print(f"  {name:30s}  bal_acc={r['task_A_balanced_accuracy']:.4f}  "
              f"({r['n_features']} features)")

    print(f"\nTask B (binary relation):")
    for name in sorted(results.keys(), key=lambda n: results[n]["task_B_balanced_accuracy"], reverse=True):
        r = results[name]
        print(f"  {name:30s}  bal_acc={r['task_B_balanced_accuracy']:.4f}  "
              f"({r['n_features']} features)")

    # Key comparison for the main scientific question
    phys_B = results.get("physical_B_complex", {}).get("task_A_balanced_accuracy", 0)
    dig_ew = results.get("digital_bilinear_ew", {}).get("task_A_balanced_accuracy", 0)
    concat = results.get("zx_zy_concat", {}).get("task_A_balanced_accuracy", 0)
    hybrid = results.get("hybrid_complex", {}).get("task_A_balanced_accuracy", 0)

    print(f"\n--- KEY COMPARISON (Task A) ---")
    print(f"  Separate tokens [z(x),z(y)]:  {concat:.4f}")
    print(f"  Digital bilinear (ew):         {dig_ew:.4f}")
    print(f"  Physical B(x,y):               {phys_B:.4f}")
    print(f"  Hybrid [z,z,B]:                {hybrid:.4f}")
    print(f"  Physical B minus digital bilinear: {phys_B - dig_ew:+.4f}")
    print(f"  Hybrid minus concat:               {hybrid - concat:+.4f}")

    print(f"\nResults saved to: {results_path}")
    print(f"Figures saved to: {fig_dir}")


if __name__ == "__main__":
    main()
