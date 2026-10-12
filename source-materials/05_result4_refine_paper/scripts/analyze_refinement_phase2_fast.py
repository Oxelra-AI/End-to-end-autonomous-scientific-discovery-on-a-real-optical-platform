#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase-2 Refinement Analysis — FAST version
============================================
Operator note: the 64-pair / 8-token benchmark can be sensitive to output
dimensionality and baseline choice. This version:
  1. Speckle coherence size (already done, reuse result)
  2. Multi-bin reanalysis with stratified 5-fold CV (fast)
  3. Pseudo-B random-T control — tests whether random T also achieves 1.0
  4. PCA fairness — project ALL families to same low dimension
  5. Figure generation for paper/report

Key change: use StratifiedKFold(5) instead of LOO-CV. With 320 samples
this is ~64× faster.
"""

import os, sys, json, time
import numpy as np
from pathlib import Path
from scipy.signal import fftconvolve
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 10, 'figure.dpi': 150})

# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════
N_TOKENS = 8
ROI_H, ROI_W = 500, 500
PHASE_FILE_LABELS = ["0", "pi2", "pi", "3pi2"]

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

CAT_COLORS = {0: '#e74c3c', 1: '#3498db', 2: '#2ecc71', 3: '#9b59b6'}
CAT_NAMES = ['animals', 'tools', 'fruits', 'vehicles']


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════
def bin_roi(roi, bin_size):
    h, w = roi.shape
    nh, nw = h // bin_size, w // bin_size
    cropped = roi[:nh * bin_size, :nw * bin_size]
    return cropped.reshape(nh, bin_size, nw, bin_size).mean(axis=(1, 3)).flatten()


def load_all_data_binned(data_dir, bin_size):
    singles_dir = os.path.join(data_dir, "singles")
    blanks_dir = os.path.join(data_dir, "blank_controls")
    pairs_dir = os.path.join(data_dir, "pair_interactions")

    single_data = {}
    for tid in range(N_TOKENS):
        vecs = []
        for rep in range(5):
            fname = f"single_t{tid}_rep{rep:02d}.npy"
            vecs.append(bin_roi(np.load(os.path.join(singles_dir, fname)), bin_size))
        single_data[tid] = np.array(vecs)

    blank_data = {}
    for y_tid in range(N_TOKENS):
        for phi_idx in range(4):
            vecs = []
            plabel = PHASE_FILE_LABELS[phi_idx]
            for rep in range(5):
                fname = f"blank_y{y_tid}_phi{plabel}_rep{rep:02d}.npy"
                vecs.append(bin_roi(np.load(os.path.join(blanks_dir, fname)), bin_size))
            blank_data[(y_tid, phi_idx)] = np.array(vecs)

    pair_data = {}
    for x in range(N_TOKENS):
        for y in range(N_TOKENS):
            for phi_idx in range(4):
                vecs = []
                plabel = PHASE_FILE_LABELS[phi_idx]
                for rep in range(5):
                    fname = f"pair_x{x}_y{y}_phi{plabel}_rep{rep:02d}.npy"
                    vecs.append(bin_roi(np.load(os.path.join(pairs_dir, fname)), bin_size))
                pair_data[(x, y, phi_idx)] = np.array(vecs)

    return single_data, blank_data, pair_data


def compute_B_per_repeat(blank_data, pair_data, n_reps=5):
    Q_blank_per_rep = {}
    for y_tid in range(N_TOKENS):
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


def build_features_and_labels(single_data, B_per_rep, n_reps=5):
    z_mean = {tid: single_data[tid].mean(axis=0) for tid in range(N_TOKENS)}
    features = {"zx_zy_concat": [], "digital_bilinear_ew": [], "complex_B": []}
    pid_labels, sc_labels, cp_labels, pair_info = [], [], [], []

    for x in range(N_TOKENS):
        for y in range(N_TOKENS):
            cx = SEMANTIC_TOKENS[x]["category"]
            cy = SEMANTIC_TOKENS[y]["category"]
            same = int(cx == cy)
            cpid = cx * 4 + cy
            pid = x * N_TOKENS + y
            B_reps = B_per_rep[(x, y)]
            for rep in range(min(n_reps, B_reps.shape[0])):
                zx, zy = z_mean[x], z_mean[y]
                features["zx_zy_concat"].append(np.concatenate([zx, zy]))
                features["digital_bilinear_ew"].append(zx * zy)
                features["complex_B"].append(
                    np.concatenate([np.real(B_reps[rep]), np.imag(B_reps[rep])]))
                pid_labels.append(pid)
                sc_labels.append(same)
                cp_labels.append(cpid)
                pair_info.append((x, y))

    return ({k: np.array(v) for k, v in features.items()},
            np.array(pid_labels), np.array(sc_labels),
            np.array(cp_labels), pair_info)


# ══════════════════════════════════════════════════════════════════════════════
# Fast classification: StratifiedKFold(5)
# ══════════════════════════════════════════════════════════════════════════════
def skf_balanced_accuracy(X, y, n_splits=5):
    clf = RidgeClassifier(alpha=1.0)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=skf, scoring='balanced_accuracy')
    return float(scores.mean())


def cross_split_eval(X, y_same_cat, pair_info):
    unique_cats = sorted(set(SEMANTIC_TOKENS[i]["category"] for i in range(N_TOKENS)))
    accs = []
    for held_cat in unique_cats:
        train_mask, test_mask = [], []
        for i, (x, y) in enumerate(pair_info):
            cx = SEMANTIC_TOKENS[x]["category"]
            cy = SEMANTIC_TOKENS[y]["category"]
            if cx == held_cat or cy == held_cat:
                test_mask.append(i)
            else:
                train_mask.append(i)
        if len(train_mask) < 2 or len(test_mask) < 2:
            continue
        train_mask, test_mask = np.array(train_mask), np.array(test_mask)
        clf = RidgeClassifier(alpha=1.0)
        clf.fit(X[train_mask], y_same_cat[train_mask])
        pred = clf.predict(X[test_mask])
        accs.append(balanced_accuracy_score(y_same_cat[test_mask], pred))
    return float(np.mean(accs)) if accs else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# PART 1: Speckle Coherence Size
# ══════════════════════════════════════════════════════════════════════════════
def estimate_speckle_coherence(data_dir, fig_dir):
    print("=" * 70)
    print("PART 1: Speckle Coherence Size Estimation")
    print("=" * 70)
    singles_dir = os.path.join(data_dir, "singles")
    pairs_dir = os.path.join(data_dir, "pair_interactions")

    all_rois = []
    for tid in range(N_TOKENS):
        all_rois.append(np.load(os.path.join(singles_dir, f"single_t{tid}_rep00.npy")))
    for x in range(4):
        for y in range(4):
            fp = os.path.join(pairs_dir, f"pair_x{x}_y{y}_phi0_rep00.npy")
            if os.path.exists(fp):
                all_rois.append(np.load(fp))

    coherence_h, coherence_w = [], []
    acf_examples = []

    for roi in all_rois:
        fluct = roi - roi.mean()
        acf = fftconvolve(fluct, fluct[::-1, ::-1], mode='full')
        c = (acf.shape[0] // 2, acf.shape[1] // 2)
        peak = acf[c[0], c[1]]
        if peak <= 0:
            continue
        acf_norm = acf / peak

        def find_fwhm(profile):
            ci = len(profile) // 2
            r = ci
            while r < len(profile) - 1 and profile[r] > 0.5:
                r += 1
            l = ci
            while l > 0 and profile[l] > 0.5:
                l -= 1
            return r - l

        coherence_h.append(find_fwhm(acf_norm[:, c[1]]))
        coherence_w.append(find_fwhm(acf_norm[c[0], :]))
        if len(acf_examples) < 4:
            acf_examples.append(acf_norm)

    mh, mw = np.mean(coherence_h), np.mean(coherence_w)
    sh, sw = np.std(coherence_h), np.std(coherence_w)
    mean_coh = (mh + mw) / 2.0

    print(f"  FWHM vertical:   {mh:.1f} ± {sh:.1f} px")
    print(f"  FWHM horizontal: {mw:.1f} ± {sw:.1f} px")
    print(f"  Mean coherence:  {mean_coh:.1f} px")
    print(f"  bin=25 / speckle = {25.0/mean_coh:.2f}x (OVERSAMPLING)")

    # ── Figure: 4-panel ACF + profiles ──
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    crop = 30
    for idx in range(min(4, len(acf_examples))):
        acf_n = acf_examples[idx]
        c = (acf_n.shape[0] // 2, acf_n.shape[1] // 2)
        acf_crop = acf_n[c[0]-crop:c[0]+crop, c[1]-crop:c[1]+crop]

        ax = axes[0, idx]
        im = ax.imshow(acf_crop, cmap='hot', extent=[-crop, crop, -crop, crop],
                        vmin=-0.1, vmax=1.0)
        ax.set_title(f'ROI {idx}', fontsize=9)
        ax.set_xlabel('Δx (px)'); ax.set_ylabel('Δy (px)')
        plt.colorbar(im, ax=ax, fraction=0.046)

        ax2 = axes[1, idx]
        xax = np.arange(-crop, crop)
        ax2.plot(xax, acf_n[c[0], c[1]-crop:c[1]+crop], 'b-', lw=1.5, label='H')
        ax2.plot(xax, acf_n[c[0]-crop:c[0]+crop, c[1]], 'r-', lw=1.5, label='V')
        ax2.axhline(0.5, color='gray', ls='--', lw=0.8)
        ax2.axhline(0, color='gray', ls=':', lw=0.5)
        ax2.set_xlabel('Lag (px)'); ax2.set_ylabel('ACF')
        ax2.set_ylim(-0.2, 1.1)
        ax2.legend(fontsize=7)

    fig.suptitle(f'Speckle Coherence: FWHM ≈ {mean_coh:.1f} px  |  '
                 f'bin=25 is {25.0/mean_coh:.1f}× oversampling', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig_speckle_coherence.pdf"),
                bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  Saved fig_speckle_coherence.pdf")

    # ── Figure: histogram of FWHM values ──
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(coherence_h, bins=15, alpha=0.6, color='red', label=f'Vertical (μ={mh:.1f})')
    ax.hist(coherence_w, bins=15, alpha=0.6, color='blue', label=f'Horizontal (μ={mw:.1f})')
    ax.axvline(25, color='black', ls='--', lw=2, label='Current bin=25')
    ax.axvline(10, color='green', ls='--', lw=2, label='Proposed bin=10')
    ax.set_xlabel('FWHM (pixels)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Speckle Coherence Sizes')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig_speckle_fwhm_histogram.pdf"),
                bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  Saved fig_speckle_fwhm_histogram.pdf")

    return {"fwhm_v_mean": round(float(mh), 2), "fwhm_v_std": round(float(sh), 2),
            "fwhm_h_mean": round(float(mw), 2), "fwhm_h_std": round(float(sw), 2),
            "fwhm_mean": round(float(mean_coh), 2), "n_rois": len(coherence_h),
            "bin25_over_speckle": round(25.0/mean_coh, 3)}


# ══════════════════════════════════════════════════════════════════════════════
# PART 2: Multi-Bin Reanalysis
# ══════════════════════════════════════════════════════════════════════════════
def multi_bin_analysis(data_dir, fig_dir):
    print("\n" + "=" * 70)
    print("PART 2: Multi-Bin Reanalysis")
    print("=" * 70)

    bin_sizes = [5, 10, 12, 16, 20, 25, 50]
    results = {}
    cached_B = {}

    for bs in bin_sizes:
        t0 = time.time()
        n_ch = (ROI_H // bs) * (ROI_W // bs)
        print(f"\n  bin={bs}, channels={n_ch}, B_dim={2*n_ch}")

        single_data, blank_data, pair_data = load_all_data_binned(data_dir, bs)
        B_per_rep = compute_B_per_repeat(blank_data, pair_data)
        cached_B[bs] = B_per_rep
        features, pid, sc, cp, pinfo = build_features_and_labels(single_data, B_per_rep)

        res = {"bin_size": bs, "n_channels": n_ch}
        for fname, X in features.items():
            tA = skf_balanced_accuracy(X, pid)
            tB = skf_balanced_accuracy(X, sc)
            tC = skf_balanced_accuracy(X, cp)
            tD = cross_split_eval(X, sc, pinfo)
            res[fname] = {"dim": X.shape[1],
                          "A": round(tA, 4), "B": round(tB, 4),
                          "C": round(tC, 4), "D": round(tD, 4)}
            print(f"    {fname:25s} dim={X.shape[1]:5d}  "
                  f"A={tA:.4f} B={tB:.4f} C={tC:.4f} D={tD:.4f}")

        # Self-pair B(x,x) diagnostics
        self_stats = []
        for x in range(N_TOKENS):
            Bxx = B_per_rep[(x, x)].mean(axis=0)
            cr, ci = float(np.mean(np.real(Bxx))), float(np.mean(np.imag(Bxx)))
            mag = float(np.sqrt(cr**2 + ci**2))
            self_stats.append({"token": SEMANTIC_TOKENS[x]["word"],
                               "re": round(cr, 6), "im": round(ci, 6),
                               "mag": round(mag, 6),
                               "phase_deg": round(float(np.degrees(np.arctan2(ci, cr))), 2)})
        res["self_pair_Bxx"] = self_stats
        results[bs] = res
        print(f"    [{time.time()-t0:.1f}s]")

    # ── Figure: 4-task performance vs bin size ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    tasks = [("A", "Task A: 64-class Pair ID"), ("B", "Task B: Same Category"),
             ("C", "Task C: Category Pair"), ("D", "Task D: Cross-Split")]
    for tidx, (tk, title) in enumerate(tasks):
        ax = axes[tidx // 2, tidx % 2]
        for fname, color, marker, ls in [
            ("zx_zy_concat", "#3498db", "o", "-"),
            ("digital_bilinear_ew", "#e67e22", "s", "-"),
            ("complex_B", "#e74c3c", "D", "-"),
        ]:
            vals = [results[bs][fname][tk] for bs in bin_sizes]
            ax.plot(bin_sizes, vals, f'{ls}{marker}', color=color, label=fname,
                    markersize=7, lw=2)
        ax.axhline(0.5, color='gray', ls='--', lw=0.8, alpha=0.5)
        ax.axvline(10, color='green', ls=':', lw=1.5, alpha=0.7, label='speckle-matched')
        ax.axvline(25, color='black', ls=':', lw=1.5, alpha=0.7, label='original bin')
        ax.set_xlabel('Bin Size (pixels)')
        ax.set_ylabel('Balanced Accuracy')
        ax.set_title(title)
        ax.set_ylim(-0.05, 1.15)
        if tidx == 0:
            ax.legend(fontsize=7, loc='lower right')
    fig.suptitle('Multi-Bin Reanalysis: Structural Separation vs Detector Binning', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig_multi_bin_tasks.pdf"), bbox_inches="tight", dpi=300)
    plt.close()
    print(f"\n  Saved fig_multi_bin_tasks.pdf")

    # ── Figure: Self-pair B(x,x) complex-plane across bin sizes ──
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    plot_bins = [bs for bs in bin_sizes if bs in cached_B]
    for bidx, bs in enumerate(plot_bins[:8]):
        ax = axes[bidx // 4, bidx % 4]
        B_loc = cached_B[bs]
        # Self-pairs: colored stars
        for x in range(N_TOKENS):
            Bxx = B_loc[(x, x)].mean(axis=0)
            cr, ci = np.real(Bxx).mean(), np.imag(Bxx).mean()
            cat = SEMANTIC_TOKENS[x]["category"]
            ax.scatter(cr, ci, s=120, marker='*', color=CAT_COLORS[cat], zorder=5,
                       edgecolors='black', linewidths=0.5)
            ax.annotate(SEMANTIC_TOKENS[x]["word"], (cr, ci), fontsize=6,
                        textcoords="offset points", xytext=(4, 4))
        # Cross-pairs: small gray dots
        for x in range(N_TOKENS):
            for y in range(N_TOKENS):
                if x == y:
                    continue
                Bxy = B_loc[(x, y)].mean(axis=0)
                ax.scatter(np.real(Bxy).mean(), np.imag(Bxy).mean(),
                           s=8, color='gray', alpha=0.3)
        ax.axhline(0, color='gray', lw=0.3)
        ax.axvline(0, color='gray', lw=0.3)
        n_ch = (ROI_H // bs) * (ROI_W // bs)
        ax.set_title(f'bin={bs} ({n_ch} ch)', fontsize=10)
        ax.set_xlabel('⟨Re(B)⟩'); ax.set_ylabel('⟨Im(B)⟩')
        ax.set_aspect('equal')
    for bidx in range(len(plot_bins), 8):
        axes[bidx // 4, bidx % 4].set_visible(False)
    fig.suptitle('Self-pair B(x,x) centroids (★) vs cross-pairs (•) across bin sizes', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig_self_pair_complex_plane.pdf"),
                bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  Saved fig_self_pair_complex_plane.pdf")

    # ── Figure: Feature dimension vs bin size ──
    fig, ax = plt.subplots(figsize=(8, 4))
    for fname, color in [("zx_zy_concat", "#3498db"),
                          ("digital_bilinear_ew", "#e67e22"),
                          ("complex_B", "#e74c3c")]:
        dims = [results[bs][fname]["dim"] for bs in bin_sizes]
        ax.plot(bin_sizes, dims, 'o-', color=color, label=fname, lw=2)
    ax.axvline(10, color='green', ls=':', lw=1.5, label='speckle-matched')
    ax.set_xlabel('Bin Size (pixels)')
    ax.set_ylabel('Feature Dimension')
    ax.set_title('Feature Dimensionality vs Bin Size')
    ax.legend()
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig_dimension_vs_bin.pdf"), bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  Saved fig_dimension_vs_bin.pdf")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# PART 3: Pseudo-B Random-T Control
# ══════════════════════════════════════════════════════════════════════════════
def pseudo_B_control(data_dir, fig_dir, n_seeds=20):
    print("\n" + "=" * 70)
    print("PART 3: Pseudo-B Random-T Control")
    print("=" * 70)

    # Token input fields (36 complex amplitudes each, from superpixel phases)
    def make_token_input(tid):
        seed = 10000 + tid * 7919
        rng = np.random.default_rng(seed)
        phases = rng.uniform(0, 2 * np.pi, (6, 6))
        return np.exp(1j * phases).flatten()  # 36 complex

    token_inputs = [make_token_input(tid) for tid in range(N_TOKENS)]
    input_dim = 36

    # Test multiple output dimensions to match various bin sizes
    test_configs = [
        {"name": "dim=100 (bin≈50)", "n_out": 100},
        {"name": "dim=400 (bin=25)", "n_out": 400},
        {"name": "dim=2500 (bin=10)", "n_out": 2500},
    ]

    all_results = {}

    for cfg in test_configs:
        n_out = cfg["n_out"]
        print(f"\n  Config: {cfg['name']}, n_out={n_out}")
        seed_results = []

        for seed_idx in range(n_seeds):
            rng = np.random.default_rng(42000 + seed_idx)
            T_rand = (rng.standard_normal((n_out, input_dim)) +
                      1j * rng.standard_normal((n_out, input_dim))) / np.sqrt(2 * input_dim)

            z_pseudo = {tid: T_rand @ token_inputs[tid] for tid in range(N_TOKENS)}

            # Build features for all 64 pairs × 5 pseudo-repeats
            feat_concat, feat_digbil, feat_pseudoB = [], [], []
            pid_labels, sc_labels, cp_labels = [], [], []

            for x in range(N_TOKENS):
                for y in range(N_TOKENS):
                    cx = SEMANTIC_TOKENS[x]["category"]
                    cy = SEMANTIC_TOKENS[y]["category"]
                    pid = x * N_TOKENS + y
                    same = int(cx == cy)
                    cpid = cx * 4 + cy

                    zx_r = np.concatenate([np.real(z_pseudo[x]), np.imag(z_pseudo[x])])
                    zy_r = np.concatenate([np.real(z_pseudo[y]), np.imag(z_pseudo[y])])
                    Bxy = np.conj(z_pseudo[x]) * z_pseudo[y]
                    B_r = np.concatenate([np.real(Bxy), np.imag(Bxy)])

                    for rep in range(5):
                        noise = rng.standard_normal(B_r.shape) * 0.01 * np.std(B_r)
                        feat_concat.append(np.concatenate([zx_r, zy_r]))
                        feat_digbil.append(np.abs(zx_r[:n_out]) * np.abs(zy_r[:n_out]))
                        feat_pseudoB.append(B_r + noise)
                        pid_labels.append(pid)
                        sc_labels.append(same)
                        cp_labels.append(cpid)

            feat_concat = np.array(feat_concat)
            feat_digbil = np.array(feat_digbil)
            feat_pseudoB = np.array(feat_pseudoB)
            pid_labels = np.array(pid_labels)
            sc_labels = np.array(sc_labels)
            cp_labels = np.array(cp_labels)

            res = {}
            for fn, X in [("concat", feat_concat), ("dig_bilinear", feat_digbil),
                           ("pseudo_B", feat_pseudoB)]:
                tA = skf_balanced_accuracy(X, pid_labels)
                tB = skf_balanced_accuracy(X, sc_labels)
                tC = skf_balanced_accuracy(X, cp_labels)
                res[fn] = {"A": tA, "B": tB, "C": tC}
            seed_results.append(res)

            if seed_idx < 3:
                print(f"    seed={seed_idx}: pseudo_B A={res['pseudo_B']['A']:.4f} "
                      f"B={res['pseudo_B']['B']:.4f} C={res['pseudo_B']['C']:.4f}")

        # Aggregate
        agg = {}
        for fn in ["concat", "dig_bilinear", "pseudo_B"]:
            for tk in ["A", "B", "C"]:
                vals = [r[fn][tk] for r in seed_results]
                agg[f"{fn}_{tk}"] = {"mean": round(np.mean(vals), 4),
                                      "std": round(np.std(vals), 4)}
        all_results[cfg["name"]] = agg

        print(f"  Summary for {cfg['name']}:")
        for fn in ["concat", "dig_bilinear", "pseudo_B"]:
            a = agg[f"{fn}_A"]; b = agg[f"{fn}_B"]; c = agg[f"{fn}_C"]
            print(f"    {fn:15s}  A={a['mean']:.4f}±{a['std']:.4f}  "
                  f"B={b['mean']:.4f}±{b['std']:.4f}  C={c['mean']:.4f}±{c['std']:.4f}")

    # ── Figure: Pseudo-B control bar chart ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    task_labels = ["Task A: Pair ID", "Task B: Same Category", "Task C: Category Pair"]

    for tidx, (tk, tlabel) in enumerate(zip(["A", "B", "C"], task_labels)):
        ax = axes[tidx]
        x_pos = np.arange(len(test_configs))
        width = 0.2

        for fidx, (fn, color, label) in enumerate([
            ("concat", "#3498db", "Pseudo Concat"),
            ("dig_bilinear", "#e67e22", "Pseudo Dig.Bilin."),
            ("pseudo_B", "#9b59b6", "Pseudo-B (random T)"),
        ]):
            means = [all_results[cfg["name"]][f"{fn}_{tk}"]["mean"] for cfg in test_configs]
            stds = [all_results[cfg["name"]][f"{fn}_{tk}"]["std"] for cfg in test_configs]
            ax.bar(x_pos + fidx * width, means, width, yerr=stds,
                   color=color, label=label, alpha=0.8, capsize=3)

        ax.axhline(1.0, color='red', ls='-', lw=2, alpha=0.7, label='Real Complex-B')
        ax.set_xticks(x_pos + width)
        ax.set_xticklabels([c["name"] for c in test_configs], fontsize=8, rotation=15)
        ax.set_ylabel('Balanced Accuracy')
        ax.set_title(tlabel)
        ax.set_ylim(0, 1.15)
        ax.axhline(0.5, color='gray', ls='--', lw=0.5)
        if tidx == 0:
            ax.legend(fontsize=7, loc='lower left')

    fig.suptitle('Pseudo-B Control: Random T vs Real Optical T', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig_pseudo_B_control.pdf"), bbox_inches="tight", dpi=300)
    plt.close()
    print(f"\n  Saved fig_pseudo_B_control.pdf")

    # ── Figure: Boxplot distribution across seeds ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for tidx, (tk, tlabel) in enumerate(zip(["A", "B", "C"], task_labels)):
        ax = axes[tidx]
        # For the dim=400 config (matching original bin=25)
        cfg_name = test_configs[1]["name"]  # dim=400
        data_box = []
        labels_box = []
        colors_box = []
        for fn, color, label in [("concat", "#3498db", "Concat"),
                                  ("dig_bilinear", "#e67e22", "Dig.Bilin."),
                                  ("pseudo_B", "#9b59b6", "Pseudo-B")]:
            # Need raw seed values — recompute from all_results structure
            # Actually we only stored mean/std. Let me use the bar chart instead.
            pass

        # Simpler: just show means with error bars
        for fidx, (fn, color, label) in enumerate([
            ("concat", "#3498db", "Concat"),
            ("dig_bilinear", "#e67e22", "Dig.Bilin."),
            ("pseudo_B", "#9b59b6", "Pseudo-B"),
        ]):
            for cidx, cfg in enumerate(test_configs):
                v = all_results[cfg["name"]][f"{fn}_{tk}"]
                ax.errorbar(cidx + fidx * 0.25, v["mean"], yerr=v["std"],
                            fmt='o', color=color, markersize=8, capsize=4)
        ax.axhline(1.0, color='red', ls='-', lw=2, alpha=0.5)
        ax.set_title(tlabel)
        ax.set_ylabel('Balanced Accuracy')
        ax.set_ylim(0, 1.15)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig_pseudo_B_scatter.pdf"), bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  Saved fig_pseudo_B_scatter.pdf")

    return all_results


# ══════════════════════════════════════════════════════════════════════════════
# PART 4: PCA Fairness Control
# ══════════════════════════════════════════════════════════════════════════════
def pca_fairness_control(data_dir, fig_dir, bin_size=25):
    print("\n" + "=" * 70)
    print("PART 4: PCA Fairness Control (bin={})".format(bin_size))
    print("=" * 70)

    single_data, blank_data, pair_data = load_all_data_binned(data_dir, bin_size)
    B_per_rep = compute_B_per_repeat(blank_data, pair_data)
    features, pid, sc, cp, pinfo = build_features_and_labels(single_data, B_per_rep)

    # Operator insight: task is simple, even a few features might suffice
    pca_dims = [2, 3, 5, 10, 25, 50, 100, 200]
    results = {}

    print(f"  Original dims: " +
          ", ".join(f"{k}={v.shape[1]}" for k, v in features.items()))

    for d in pca_dims:
        results[d] = {}
        for fname, X in features.items():
            actual_d = min(d, X.shape[1], X.shape[0] - 1)
            pca = PCA(n_components=actual_d)
            X_pca = pca.fit_transform(X)
            var_exp = pca.explained_variance_ratio_.sum()

            tA = skf_balanced_accuracy(X_pca, pid)
            tB = skf_balanced_accuracy(X_pca, sc)
            tC = skf_balanced_accuracy(X_pca, cp)
            tD = cross_split_eval(X_pca, sc, pinfo)

            results[d][fname] = {"actual_dim": actual_d,
                                  "var_explained": round(float(var_exp), 4),
                                  "A": round(tA, 4), "B": round(tB, 4),
                                  "C": round(tC, 4), "D": round(tD, 4)}
            print(f"  d={d:3d} {fname:25s} act={actual_d:3d} var={var_exp:.3f}  "
                  f"A={tA:.4f} B={tB:.4f} C={tC:.4f} D={tD:.4f}")

    # Original (no PCA)
    results["original"] = {}
    for fname, X in features.items():
        tA = skf_balanced_accuracy(X, pid)
        tB = skf_balanced_accuracy(X, sc)
        tC = skf_balanced_accuracy(X, cp)
        tD = cross_split_eval(X, sc, pinfo)
        results["original"][fname] = {"dim": X.shape[1],
                                       "A": round(tA, 4), "B": round(tB, 4),
                                       "C": round(tC, 4), "D": round(tD, 4)}

    # Also do speckle-matched bin=10
    print(f"\n  === Also testing bin=10 (speckle-matched) ===")
    sd10, bd10, pd10 = load_all_data_binned(data_dir, 10)
    Bp10 = compute_B_per_repeat(bd10, pd10)
    feat10, pid10, sc10, cp10, pi10 = build_features_and_labels(sd10, Bp10)
    results["bin10_original"] = {}
    for fname, X in feat10.items():
        tA = skf_balanced_accuracy(X, pid10)
        tB = skf_balanced_accuracy(X, sc10)
        tC = skf_balanced_accuracy(X, cp10)
        tD = cross_split_eval(X, sc10, pi10)
        results["bin10_original"][fname] = {"dim": X.shape[1],
                                             "A": round(tA, 4), "B": round(tB, 4),
                                             "C": round(tC, 4), "D": round(tD, 4)}
        print(f"  bin10 {fname:25s} dim={X.shape[1]:5d}  "
              f"A={tA:.4f} B={tB:.4f} C={tC:.4f} D={tD:.4f}")

    # PCA on bin=10 features too
    for d in pca_dims:
        results[f"bin10_d{d}"] = {}
        for fname, X in feat10.items():
            actual_d = min(d, X.shape[1], X.shape[0] - 1)
            pca = PCA(n_components=actual_d)
            X_pca = pca.fit_transform(X)
            tA = skf_balanced_accuracy(X_pca, pid10)
            tB = skf_balanced_accuracy(X_pca, sc10)
            tC = skf_balanced_accuracy(X_pca, cp10)
            tD = cross_split_eval(X_pca, sc10, pi10)
            results[f"bin10_d{d}"][fname] = {
                "actual_dim": actual_d,
                "A": round(tA, 4), "B": round(tB, 4),
                "C": round(tC, 4), "D": round(tD, 4)}

    # ── Figure: PCA fairness — 4 tasks ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    tasks = [("A", "Task A: Pair ID"), ("B", "Task B: Same Category"),
             ("C", "Task C: Category Pair"), ("D", "Task D: Cross-Split")]

    for tidx, (tk, title) in enumerate(tasks):
        ax = axes[tidx // 2, tidx % 2]
        for fname, color, marker in [
            ("zx_zy_concat", "#3498db", "o"),
            ("digital_bilinear_ew", "#e67e22", "s"),
            ("complex_B", "#e74c3c", "D"),
        ]:
            # bin=25 PCA
            vals25 = [results[d][fname][tk] for d in pca_dims]
            orig25 = results["original"][fname][tk]
            ax.plot(pca_dims, vals25, f'-{marker}', color=color, label=f'{fname} (bin25)',
                    markersize=6, lw=2)
            ax.scatter([features[fname].shape[1]], [orig25], marker='*',
                       s=150, color=color, zorder=5, edgecolors='black')

            # bin=10 PCA
            vals10 = [results[f"bin10_d{d}"][fname][tk] for d in pca_dims]
            ax.plot(pca_dims, vals10, f'--{marker}', color=color,
                    markersize=5, lw=1.5, alpha=0.6)

        ax.axhline(0.5, color='gray', ls='--', lw=0.8, alpha=0.5)
        ax.set_xlabel('PCA Dimension')
        ax.set_ylabel('Balanced Accuracy')
        ax.set_title(title)
        ax.set_ylim(-0.05, 1.15)
        if tidx == 0:
            ax.legend(fontsize=6, loc='lower right')

    fig.suptitle('PCA Fairness: Performance at Equal Dimensions\n'
                 '(solid=bin25, dashed=bin10, ★=original full dim)', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig_pca_fairness.pdf"), bbox_inches="tight", dpi=300)
    plt.close()
    print(f"\n  Saved fig_pca_fairness.pdf")

    # ── Figure: Critical question — at matched dim, does separation hold? ──
    fig, ax = plt.subplots(figsize=(12, 6))
    # At d=5 (very low), show all 3 families × 3 tasks
    d_test = 5
    families = ["zx_zy_concat", "digital_bilinear_ew", "complex_B"]
    x_pos = np.arange(3)
    width = 0.25
    for fidx, fname in enumerate(families):
        vals = [results[d_test][fname][tk] for tk in ["A", "B", "C"]]
        bars = ax.bar(x_pos + fidx * width, vals, width, label=fname, alpha=0.8)
        for i, v in enumerate(vals):
            ax.text(x_pos[i] + fidx * width, v + 0.02, f'{v:.3f}', ha='center', fontsize=8)
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels(["Task A: Pair ID", "Task B: Same Cat", "Task C: Cat Pair"])
    ax.set_ylabel('Balanced Accuracy')
    ax.set_title(f'Critical Test: Structural Separation at Only d={d_test} PCA Dimensions')
    ax.axhline(0.5, color='gray', ls='--', lw=0.8)
    ax.set_ylim(0, 1.15)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig_critical_low_dim_test.pdf"),
                bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  Saved fig_critical_low_dim_test.pdf")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    section_dir = os.path.dirname(base)
    package_dir = os.path.dirname(section_dir)
    data_dir = os.path.join(package_dir, "shared_raw_data", "result4_frame_data",
                            "exp_complex_B_semantic")
    fig_dir = os.path.join(section_dir, "figures", "phase2_refinement")
    out_dir = os.path.join(section_dir, "source_data", "processed",
                           "phase2_refinement_results")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    t_total = time.time()
    all_results = {}

    # Part 1: Speckle coherence
    all_results["speckle_coherence"] = estimate_speckle_coherence(data_dir, fig_dir)

    # Part 2: Multi-bin
    mb = multi_bin_analysis(data_dir, fig_dir)
    all_results["multi_bin"] = {str(k): v for k, v in mb.items()}

    # Part 3: Pseudo-B control
    all_results["pseudo_B_control"] = pseudo_B_control(data_dir, fig_dir, n_seeds=20)

    # Part 4: PCA fairness
    pca = pca_fairness_control(data_dir, fig_dir, bin_size=25)
    all_results["pca_fairness"] = {str(k): v for k, v in pca.items()}

    # Save
    out_path = os.path.join(out_dir, "phase2_refinement_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    elapsed = time.time() - t_total
    print(f"\n{'='*70}")
    print(f"ALL DONE in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Results: {out_path}")
    print(f"Figures: {fig_dir}/")
    print(f"{'='*70}")
