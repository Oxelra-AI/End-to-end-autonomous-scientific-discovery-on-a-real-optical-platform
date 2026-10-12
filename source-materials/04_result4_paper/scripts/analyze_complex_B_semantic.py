#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysis: Complex-B Semantic Benchmark — Bilinear Physical Interaction Engine
=============================================================================
Track B analysis. Processes data from exp_complex_B_semantic.py.

REBOOT RULES:
  - Primary object: complex interaction field B(x,y) ∈ C^M
  - Allowed: real-coordinate representation [Re(B), Im(B)] for numerical work
  - FORBIDDEN: |B| as standalone research object or headline result

Semantic tasks:
  Task A: 64-class ordered pair identification (x,y)
  Task B: binary same-category vs different-category
  Task C: 4-class category-pair identification (cat_x, cat_y)
  Task D: cross-split generalization (train on some pairs, test on held-out)

Feature families:
  1. z(x) only
  2. z(y) only
  3. [z(x), z(y)] concatenation
  4. z(x)+z(y) symmetric
  5. z(x)-z(y) antisymmetric
  6. digital_bilinear_ew: z(x)*z(y)
  7. complex_B: [Re(B), Im(B)]   — PRIMARY
  8. hybrid: [z(x), z(y), Re(B), Im(B)]

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
N_TOKENS = 8
BIN_SIZE = 25
ROI_H, ROI_W = 500, 500
N_CHANNELS = (ROI_H // BIN_SIZE) * (ROI_W // BIN_SIZE)  # 400
PHASES = [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]
PHASE_LABELS = ["0", "pi2", "pi", "3pi2"]

SEMANTIC_TOKENS = [
    {"id": 0, "word": "cat",    "category": 0, "category_name": "animals"},
    {"id": 1, "word": "dog",    "category": 0, "category_name": "animals"},
    {"id": 2, "word": "hammer", "category": 1, "category_name": "tools"},
    {"id": 3, "word": "saw",    "category": 1, "category_name": "tools"},
    {"id": 4, "word": "apple",  "category": 2, "category_name": "fruits"},
    {"id": 5, "word": "grape",  "category": 2, "category_name": "fruits"},
    {"id": 6, "word": "car",    "category": 3, "category_name": "vehicles"},
    {"id": 7, "word": "train",  "category": 3, "category_name": "vehicles"},
]


def bin_roi(roi, bin_size=BIN_SIZE):
    h, w = roi.shape
    nh, nw = h // bin_size, w // bin_size
    cropped = roi[:nh * bin_size, :nw * bin_size]
    return cropped.reshape(nh, bin_size, nw, bin_size).mean(axis=(1, 3)).flatten()


def load_and_bin(data_dir, filename):
    return bin_roi(np.load(os.path.join(data_dir, filename)))


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
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
        data[key].append(load_and_bin(blanks_dir, rec["file"]))
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
        data[key].append(load_and_bin(pairs_dir, rec["file"]))
    return {k: np.array(v) for k, v in data.items()}


# ══════════════════════════════════════════════════════════════════════════════
# Four-phase demodulation and Complex-B extraction
# ══════════════════════════════════════════════════════════════════════════════
def demodulate_Q(intensity_by_phase):
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


def compute_per_repeat_B(blank_data, pair_data, n_reps=5):
    Q_blank_per_rep = {}
    for y_tid in range(N_TOKENS):
        n = blank_data[(y_tid, 0)].shape[0]
        Q_list = []
        for rep in range(n):
            I0 = blank_data[(y_tid, 0)][rep]
            I1 = blank_data[(y_tid, 1)][rep]
            I2 = blank_data[(y_tid, 2)][rep]
            I3 = blank_data[(y_tid, 3)][rep]
            Q_list.append((I0 - I2) / 4.0 + 1j * (I3 - I1) / 4.0)
        Q_blank_per_rep[y_tid] = np.array(Q_list)

    B_per_rep = {}
    for x in range(N_TOKENS):
        for y in range(N_TOKENS):
            n = pair_data[(x, y, 0)].shape[0]
            B_list = []
            for rep in range(n):
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
# Complex-B stability
# ══════════════════════════════════════════════════════════════════════════════
def analyze_complex_B_stability(B_per_rep):
    stability = {}
    all_corrs = []
    for (x, y), B_arr in B_per_rep.items():
        B_real = np.concatenate([np.real(B_arr), np.imag(B_arr)], axis=1)
        corrs = []
        for i in range(B_arr.shape[0]):
            for j in range(i+1, B_arr.shape[0]):
                c = np.corrcoef(B_real[i], B_real[j])[0, 1]
                if not np.isnan(c):
                    corrs.append(c)
        mc = float(np.mean(corrs)) if corrs else 0.0
        stability[f"({x},{y})"] = {"mean_complex_corr": mc}
        all_corrs.append(mc)
    return stability, float(np.mean(all_corrs)), float(np.min(all_corrs))


# ══════════════════════════════════════════════════════════════════════════════
# Feature construction — COMPLEX-B DISCIPLINE
# ══════════════════════════════════════════════════════════════════════════════
def build_feature_families(z_single, B_per_rep, n_reps=5):
    samples = []
    pair_labels = []       # 0..63 for Task A (ordered pair ID)
    cat_binary_labels = [] # 0/1 for Task B (same vs different category)
    cat_pair_labels = []   # 0..15 for Task C (category pair)

    pair_idx = 0
    for x in range(N_TOKENS):
        for y in range(N_TOKENS):
            cat_x = SEMANTIC_TOKENS[x]["category"]
            cat_y = SEMANTIC_TOKENS[y]["category"]
            same_cat = int(cat_x == cat_y)
            cat_pair_id = cat_x * 4 + cat_y  # 0..15

            for rep in range(n_reps):
                samples.append({
                    "x": x, "y": y, "rep": rep,
                    "zx": z_single[x], "zy": z_single[y],
                    "B_xy": B_per_rep[(x, y)][rep],
                })
                pair_labels.append(pair_idx)
                cat_binary_labels.append(same_cat)
                cat_pair_labels.append(cat_pair_id)
            pair_idx += 1

    pair_labels = np.array(pair_labels)
    cat_binary_labels = np.array(cat_binary_labels)
    cat_pair_labels = np.array(cat_pair_labels)

    families = {}
    families["zx_only"] = np.array([s["zx"] for s in samples])
    families["zy_only"] = np.array([s["zy"] for s in samples])
    families["zx_zy_concat"] = np.array([
        np.concatenate([s["zx"], s["zy"]]) for s in samples])
    families["zx_plus_zy"] = np.array([s["zx"] + s["zy"] for s in samples])
    families["zx_minus_zy"] = np.array([s["zx"] - s["zy"] for s in samples])
    families["digital_bilinear_ew"] = np.array([
        s["zx"] * s["zy"] for s in samples])
    families["complex_B"] = np.array([
        np.concatenate([np.real(s["B_xy"]), np.imag(s["B_xy"])]) for s in samples])
    families["hybrid_zz_B"] = np.array([
        np.concatenate([s["zx"], s["zy"],
                        np.real(s["B_xy"]), np.imag(s["B_xy"])]) for s in samples])

    return families, pair_labels, cat_binary_labels, cat_pair_labels


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
    return float(balanced_accuracy_score(y, preds)), preds


def standardize(X):
    mu = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std[std < 1e-12] = 1.0
    return (X - mu) / std


def evaluate_all(families, pair_labels, cat_binary_labels, cat_pair_labels):
    results = {}
    for name, X in families.items():
        if np.iscomplexobj(X):
            X = np.concatenate([np.real(X), np.imag(X)], axis=1)
        X_std = standardize(X)
        n_feat = X_std.shape[1]
        print(f"  {name}: shape={X_std.shape}", flush=True)

        bal_A, _ = evaluate_loocv(X_std, pair_labels)
        bal_B, _ = evaluate_loocv(X_std, cat_binary_labels)
        bal_C, _ = evaluate_loocv(X_std, cat_pair_labels)

        results[name] = {
            "n_features": n_feat,
            "task_A_pair_id": round(bal_A, 4),
            "task_B_same_cat": round(bal_B, 4),
            "task_C_cat_pair": round(bal_C, 4),
        }
        print(f"    A(pair_id)={bal_A:.4f}  B(same_cat)={bal_B:.4f}  "
              f"C(cat_pair)={bal_C:.4f}", flush=True)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Cross-split generalization (Task D)
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_cross_split(families, cat_binary_labels, pair_labels, n_reps=5):
    """
    Hold out all samples from 16 specific ordered pairs (25% of 64).
    Train on remaining 48 pairs, test on held-out 16.
    Tests whether semantic category relation generalizes to unseen pairs.
    """
    from sklearn.linear_model import RidgeClassifier
    from sklearn.metrics import balanced_accuracy_score

    # Select 16 held-out pairs: 1 per pair of categories (4 same + 12 cross)
    rng = np.random.default_rng(42)
    held_out_pairs = set()
    for cx in range(4):
        for cy in range(4):
            tokens_x = [t["id"] for t in SEMANTIC_TOKENS if t["category"] == cx]
            tokens_y = [t["id"] for t in SEMANTIC_TOKENS if t["category"] == cy]
            # pick one random pair from this category combination
            tx = rng.choice(tokens_x)
            ty = rng.choice(tokens_y)
            held_out_pairs.add((int(tx), int(ty)))

    # Build masks
    n_samples = len(cat_binary_labels)
    pair_class_per_sample = pair_labels  # 0..63, each appears n_reps times
    # map pair_class -> (x, y)
    pair_class_to_xy = {}
    idx = 0
    for x in range(N_TOKENS):
        for y in range(N_TOKENS):
            pair_class_to_xy[idx] = (x, y)
            idx += 1

    test_mask = np.zeros(n_samples, dtype=bool)
    for i in range(n_samples):
        xy = pair_class_to_xy[pair_labels[i]]
        if xy in held_out_pairs:
            test_mask[i] = True
    train_mask = ~test_mask

    results = {}
    for name, X in families.items():
        if np.iscomplexobj(X):
            X = np.concatenate([np.real(X), np.imag(X)], axis=1)
        X_std = standardize(X)

        X_train, y_train = X_std[train_mask], cat_binary_labels[train_mask]
        X_test, y_test = X_std[test_mask], cat_binary_labels[test_mask]

        clf = RidgeClassifier(alpha=1.0)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        bal = float(balanced_accuracy_score(y_test, preds))

        results[name] = {
            "cross_split_same_cat_bal_acc": round(bal, 4),
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
        }
        print(f"  {name:25s} cross-split same_cat: {bal:.4f} "
              f"(train={train_mask.sum()}, test={test_mask.sum()})", flush=True)

    return results, sorted(list(held_out_pairs))


# ══════════════════════════════════════════════════════════════════════════════
# Figures
# ══════════════════════════════════════════════════════════════════════════════
def make_figures(results, cross_results, B_pair, z_single, B_per_rep,
                 pair_data, stability_info, fig_dir, cmos1_dir=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(fig_dir, exist_ok=True)

    # ── Fig 1: Phase-cycle traces for representative semantic pairs ───
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    phi_vals = [0, np.pi/2, np.pi, 3*np.pi/2]
    rep_pairs = [(0,1,"cat-dog"), (2,3,"hammer-saw"),
                 (0,2,"cat-hammer"), (4,6,"apple-car"),
                 (0,0,"cat-cat"), (6,7,"car-train")]
    for idx, (x, y, label) in enumerate(rep_pairs):
        ax = axes[idx // 3, idx % 3]
        for ch in [50, 100, 200, 300]:
            try:
                means = [pair_data[(x, y, phi)][:, ch].mean() for phi in range(4)]
                ax.plot(phi_vals, means, 'o-', label=f'ch={ch}', markersize=4)
            except KeyError:
                pass
        ax.set_title(f'({label})', fontsize=10)
        ax.set_xlabel('Phase φ')
        ax.set_ylabel('Intensity')
        if idx == 0:
            ax.legend(fontsize=6)
    fig.suptitle('Four-phase intensity modulation — semantic pairs', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig1_phase_cycles_semantic.pdf"),
                bbox_inches="tight", dpi=300)
    plt.close()

    # ── Fig 2: Complex-B field scatter — same-cat vs cross-cat ────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    same_pairs = [(0,1), (2,3), (4,5)]
    cross_pairs = [(0,2), (4,6), (1,7)]
    for idx, ((sx, sy), (cx, cy)) in enumerate(zip(same_pairs, cross_pairs)):
        ax_s = axes[0, idx]
        B_s = B_pair[(sx, sy)]
        ax_s.scatter(np.real(B_s), np.imag(B_s), s=6, alpha=0.5, c='#2ecc71')
        ax_s.set_title(f'Same: ({SEMANTIC_TOKENS[sx]["word"]},{SEMANTIC_TOKENS[sy]["word"]})',
                       fontsize=9)
        ax_s.set_xlabel('Re(B)')
        ax_s.set_ylabel('Im(B)')
        ax_s.set_aspect('equal')
        ax_s.axhline(0, color='gray', lw=0.3)
        ax_s.axvline(0, color='gray', lw=0.3)

        ax_c = axes[1, idx]
        B_c = B_pair[(cx, cy)]
        ax_c.scatter(np.real(B_c), np.imag(B_c), s=6, alpha=0.5, c='#e74c3c')
        ax_c.set_title(f'Cross: ({SEMANTIC_TOKENS[cx]["word"]},{SEMANTIC_TOKENS[cy]["word"]})',
                       fontsize=9)
        ax_c.set_xlabel('Re(B)')
        ax_c.set_ylabel('Im(B)')
        ax_c.set_aspect('equal')
        ax_c.axhline(0, color='gray', lw=0.3)
        ax_c.axvline(0, color='gray', lw=0.3)

    fig.suptitle('Complex-B channels: same-category (top) vs cross-category (bottom)',
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig2_complex_B_same_vs_cross.pdf"),
                bbox_inches="tight", dpi=300)
    plt.close()

    # ── Fig 3: Classification comparison — all tasks ──────────────────
    family_order = ["zx_only", "zy_only", "zx_zy_concat", "zx_plus_zy",
                    "zx_minus_zy", "digital_bilinear_ew", "complex_B", "hybrid_zz_B"]
    family_order = [f for f in family_order if f in results]

    task_A = [results[f]["task_A_pair_id"] for f in family_order]
    task_B = [results[f]["task_B_same_cat"] for f in family_order]
    task_C = [results[f]["task_C_cat_pair"] for f in family_order]

    x_pos = np.arange(len(family_order))
    width = 0.25

    fig, ax = plt.subplots(figsize=(15, 7))
    ax.bar(x_pos - width, task_A, width, label='Task A: pair ID', color='#3498db')
    ax.bar(x_pos, task_B, width, label='Task B: same category', color='#e67e22')
    ax.bar(x_pos + width, task_C, width, label='Task C: category pair', color='#2ecc71')
    ax.set_ylabel('Balanced Accuracy')
    ax.set_title('Semantic Complex-B Benchmark — Feature Family Comparison')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(family_order, rotation=35, ha='right', fontsize=9)
    ax.legend()
    ax.axhline(0.5, color='gray', ls='--', alpha=0.3)

    for i, (a, b, c) in enumerate(zip(task_A, task_B, task_C)):
        ax.text(i - width, a + 0.01, f'{a:.3f}', ha='center', fontsize=6)
        ax.text(i, b + 0.01, f'{b:.3f}', ha='center', fontsize=6)
        ax.text(i + width, c + 0.01, f'{c:.3f}', ha='center', fontsize=6)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig3_semantic_benchmark_comparison.pdf"),
                bbox_inches="tight", dpi=300)
    plt.close()

    # ── Fig 4: Cross-split generalization ─────────────────────────────
    if cross_results:
        cross_vals = [cross_results[f]["cross_split_same_cat_bal_acc"]
                      for f in family_order if f in cross_results]
        fig, ax = plt.subplots(figsize=(12, 5))
        bars = ax.bar(range(len(family_order)), cross_vals,
                      color=['#e74c3c' if f == 'complex_B' else '#95a5a6'
                             for f in family_order])
        ax.set_xticks(range(len(family_order)))
        ax.set_xticklabels(family_order, rotation=35, ha='right', fontsize=9)
        ax.set_ylabel('Balanced Accuracy')
        ax.set_title('Cross-split generalization: same-category on held-out pairs')
        ax.axhline(0.5, color='gray', ls='--', alpha=0.3)
        for i, v in enumerate(cross_vals):
            ax.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=7)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "fig4_cross_split_generalization.pdf"),
                    bbox_inches="tight", dpi=300)
        plt.close()

    # ── Fig 5: Complex-B stability ────────────────────────────────────
    stab_dict = stability_info[0]
    pairs_sorted = sorted(stab_dict.keys())
    corrs = [stab_dict[p]["mean_complex_corr"] for p in pairs_sorted]

    fig, ax = plt.subplots(figsize=(16, 5))
    colors = ['#2ecc71' if c > 0.9 else '#e74c3c' for c in corrs]
    ax.bar(range(len(pairs_sorted)), corrs, color=colors, width=0.8)
    ax.set_xticks(range(len(pairs_sorted)))
    ax.set_xticklabels(pairs_sorted, rotation=90, fontsize=5)
    ax.set_ylabel('Mean repeat correlation')
    ax.set_title(f'Complex-B stability (mean={stability_info[1]:.4f}, '
                 f'min={stability_info[2]:.4f})')
    ax.axhline(0.9, color='r', ls='--', lw=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig5_semantic_stability.pdf"),
                bbox_inches="tight", dpi=300)
    plt.close()

    # ── Fig 6: PCA of Complex-B colored by category pair ──────────────
    from sklearn.decomposition import PCA
    B_all, cat_bin_all, cat_pair_all = [], [], []
    for x in range(N_TOKENS):
        for y in range(N_TOKENS):
            cx = SEMANTIC_TOKENS[x]["category"]
            cy = SEMANTIC_TOKENS[y]["category"]
            same = int(cx == cy)
            cpid = cx * 4 + cy
            B_reps = B_per_rep[(x, y)]
            for rep in range(B_reps.shape[0]):
                B_vec = np.concatenate([np.real(B_reps[rep]),
                                        np.imag(B_reps[rep])])
                B_all.append(B_vec)
                cat_bin_all.append(same)
                cat_pair_all.append(cpid)
    B_all = np.array(B_all)
    cat_bin_all = np.array(cat_bin_all)

    pca = PCA(n_components=2)
    B_2d = pca.fit_transform(B_all)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    for label, color, name in [(1, '#2ecc71', 'Same category'),
                                (0, '#e74c3c', 'Different category')]:
        mask = cat_bin_all == label
        ax1.scatter(B_2d[mask, 0], B_2d[mask, 1], c=color, s=8, alpha=0.5, label=name)
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax1.set_title('Complex-B PCA — same vs different category')
    ax1.legend()

    # Color by category pair
    unique_cpids = sorted(set(cat_pair_all))
    cmap = plt.colormaps.get_cmap('tab20')
    for i, cpid in enumerate(unique_cpids):
        mask = np.array([c == cpid for c in cat_pair_all])
        cx, cy = cpid // 4, cpid % 4
        cat_names = ["ani", "tool", "fruit", "veh"]
        label_str = f"{cat_names[cx]}-{cat_names[cy]}"
        ax2.scatter(B_2d[mask, 0], B_2d[mask, 1],
                    c=[cmap(i / len(unique_cpids))], s=8, alpha=0.5,
                    label=label_str if i < 10 else None)
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax2.set_title('Complex-B PCA — category pairs')
    ax2.legend(fontsize=5, ncol=2)

    fig.suptitle('Semantic structure in Complex-B space', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig6_semantic_pca.pdf"),
                bbox_inches="tight", dpi=300)
    plt.close()

    # ── Fig 7: CMOS1 documentation montage (if available) ─────────────
    if cmos1_dir and os.path.exists(cmos1_dir):
        cmos1_files = sorted([f for f in os.listdir(cmos1_dir) if f.endswith('.npy')])
        if cmos1_files:
            n = min(len(cmos1_files), 8)
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            for i in range(n):
                ax = axes[i // 4, i % 4]
                roi = np.load(os.path.join(cmos1_dir, cmos1_files[i]))
                ax.imshow(roi, cmap='hot', aspect='auto')
                word = SEMANTIC_TOKENS[i]["word"] if i < N_TOKENS else f"t{i}"
                ax.set_title(f'{word}', fontsize=10)
                ax.axis('off')
            fig.suptitle('CMOS1 documentation: SLM-generated token fields', fontsize=13)
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, "fig7_cmos1_token_fields.pdf"),
                        bbox_inches="tight", dpi=300)
            plt.close()

    print(f"  Saved figures to {fig_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Complex-B Semantic Benchmark Analysis")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    print("=" * 60)
    print("Complex-B Semantic Benchmark — Analysis")
    print("=" * 60)

    with open(os.path.join(data_dir, "metadata.json")) as f:
        metadata = json.load(f)
    print(f"Experiment: {metadata['experiment']}")
    print(f"Total measurements: {metadata['measurement_counts']['total']}")

    print("\nLoading data...")
    single_data = load_single_token_data(data_dir)
    blank_data = load_blank_control_data(data_dir)
    pair_data = load_pair_interaction_data(data_dir)
    print(f"  Singles: {sum(v.shape[0] for v in single_data.values())}")
    print(f"  Blanks: {sum(v.shape[0] for v in blank_data.values())}")
    print(f"  Pairs: {sum(v.shape[0] for v in pair_data.values())}")

    print("\nComputing complex interaction fields...")
    Q_pair, Q_blank, B_pair, z_single = compute_interaction_observables(
        single_data, blank_data, pair_data)
    B_per_rep = compute_per_repeat_B(blank_data, pair_data)

    print("\nComplex-B stability:")
    stability_info = analyze_complex_B_stability(B_per_rep)
    print(f"  Mean: {stability_info[1]:.4f}, Min: {stability_info[2]:.4f}")

    print("\nBuilding feature families...")
    families, pair_labels, cat_binary, cat_pair = build_feature_families(
        z_single, B_per_rep)
    for name, X in families.items():
        print(f"  {name}: shape={X.shape}")

    print(f"\nCategory binary balance: {np.bincount(cat_binary)}")

    print("\nEvaluating all tasks (LOO-CV)...")
    results = evaluate_all(families, pair_labels, cat_binary, cat_pair)

    print("\nCross-split generalization (Task D)...")
    cross_results, held_out = evaluate_cross_split(
        families, cat_binary, pair_labels)

    print("\nGenerating figures...")
    cmos1_dir = os.path.join(data_dir, "cmos1_documentation")
    make_figures(results, cross_results, B_pair, z_single, B_per_rep,
                 pair_data, stability_info, fig_dir, cmos1_dir)

    output_results = {
        "experiment": "complex_B_semantic_benchmark",
        "primary_object": "complex interaction field B(x,y)",
        "object_discipline": "|B| NOT used as headline",
        "semantic_tokens": SEMANTIC_TOKENS,
        "metadata_summary": {
            "total_measurements": metadata["measurement_counts"]["total"],
        },
        "complex_B_stability": {
            "mean_corr": stability_info[1],
            "min_corr": stability_info[2],
        },
        "classification_results": results,
        "cross_split_results": cross_results,
        "held_out_pairs": held_out,
        "n_samples": int(len(pair_labels)),
        "cat_binary_balance": np.bincount(cat_binary).tolist(),
    }

    results_path = os.path.join(output_dir, "complex_B_semantic_results.json")
    with open(results_path, "w") as f:
        json.dump(output_results, f, indent=2)

    print("\n" + "=" * 60)
    print("KEY SEMANTIC BENCHMARK RESULTS")
    print("=" * 60)
    for name in ["zx_zy_concat", "digital_bilinear_ew", "complex_B", "hybrid_zz_B"]:
        if name in results:
            r = results[name]
            cr = cross_results.get(name, {})
            print(f"  {name:25s}  A={r['task_A_pair_id']:.4f}  "
                  f"B={r['task_B_same_cat']:.4f}  C={r['task_C_cat_pair']:.4f}  "
                  f"D(cross)={cr.get('cross_split_same_cat_bal_acc', 'N/A')}")

    print(f"\nResults: {results_path}")
    print(f"Figures: {fig_dir}")


if __name__ == "__main__":
    main()
