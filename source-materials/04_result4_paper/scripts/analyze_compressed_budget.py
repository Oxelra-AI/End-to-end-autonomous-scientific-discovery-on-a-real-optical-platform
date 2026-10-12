#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysis: Equal-Feature-Budget Comparison for Bilinear Interaction Engine
=========================================================================
Runs on Linux. Uses existing data from exp_bilinear_001.

Core question:
  Under matched feature budgets k in {5, 10, 20, 50, 100, 200},
  does physical B(x,y) provide more compact task-relevant information
  than separate-token features and digital bilinear features?

Feature families compared at each budget k:
  1. top-k physical |B| channels (F-score ranked)
  2. top-k physical |B| channels (variance ranked)
  3. top-k from [z(x),z(y)] concat (F-score ranked)
  4. top-k digital bilinear elementwise (F-score ranked)
  5. random k channels from physical |B|  (50 trials)
  6. random k channels from [z(x),z(y)]  (50 trials)
  7. random k digital bilinear features   (50 trials)

Both Task A (16-class pair ID) and Task B (binary relation) are evaluated.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
# Configuration — must match analyze_bilinear_interaction.py
# ══════════════════════════════════════════════════════════════════════════════
N_TOKENS = 4
BIN_SIZE = 25
ROI_H, ROI_W = 500, 500
N_CHANNELS = (ROI_H // BIN_SIZE) * (ROI_W // BIN_SIZE)  # 400
PHASES = [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]
POSITIVE_PAIRS = {(0,1), (1,2), (2,3), (3,0), (1,0), (2,1), (3,2), (0,3)}

BUDGETS = [5, 10, 20, 50, 100, 200]
N_RANDOM_TRIALS = 50


def bin_roi(roi, bin_size=BIN_SIZE):
    h, w = roi.shape
    nh, nw = h // bin_size, w // bin_size
    cropped = roi[:nh * bin_size, :nw * bin_size]
    return cropped.reshape(nh, bin_size, nw, bin_size).mean(axis=(1, 3)).flatten()


def load_and_bin(data_dir, filename):
    return bin_roi(np.load(os.path.join(data_dir, filename)))


# ══════════════════════════════════════════════════════════════════════════════
# Data loading (reuse logic from main analysis)
# ══════════════════════════════════════════════════════════════════════════════
def load_single_token_data(base_dir):
    singles_dir = os.path.join(base_dir, "singles")
    with open(os.path.join(singles_dir, "single_records.json")) as f:
        records = json.load(f)
    data = {tid: [] for tid in range(N_TOKENS)}
    for rec in records:
        data[rec["token_id"]].append(load_and_bin(singles_dir, rec["file"]))
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
# Compute per-repeat B(x,y)
# ══════════════════════════════════════════════════════════════════════════════
def compute_per_repeat_B(blank_data, pair_data):
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
    for x_tid in range(N_TOKENS):
        for y_tid in range(N_TOKENS):
            n_reps = pair_data[(x_tid, y_tid, 0)].shape[0]
            B_list = []
            for rep in range(n_reps):
                I0 = pair_data[(x_tid, y_tid, 0)][rep]
                I1 = pair_data[(x_tid, y_tid, 1)][rep]
                I2 = pair_data[(x_tid, y_tid, 2)][rep]
                I3 = pair_data[(x_tid, y_tid, 3)][rep]
                Q_xy = (I0 - I2) / 4.0 + 1j * (I3 - I1) / 4.0
                B_list.append(Q_xy - Q_blank_per_rep[y_tid][rep])
            B_per_rep[(x_tid, y_tid)] = np.array(B_list)
    return B_per_rep


# ══════════════════════════════════════════════════════════════════════════════
# Build full feature matrices
# ══════════════════════════════════════════════════════════════════════════════
def build_full_features(single_data, B_per_rep, n_reps=5):
    z_single = {tid: single_data[tid].mean(axis=0) for tid in range(N_TOKENS)}

    samples_B_mag = []      # |B(x,y)|, shape will be (80, 400)
    samples_concat = []     # [z(x), z(y)], shape will be (80, 800)
    samples_dig_ew = []     # z(x)*z(y), shape will be (80, 400)
    pair_labels = []
    binary_labels = []

    pair_idx = 0
    for x_tid in range(N_TOKENS):
        for y_tid in range(N_TOKENS):
            bl = 1 if (x_tid, y_tid) in POSITIVE_PAIRS else 0
            for rep in range(n_reps):
                zx = z_single[x_tid]
                zy = z_single[y_tid]
                B_xy = B_per_rep[(x_tid, y_tid)][rep]

                samples_B_mag.append(np.abs(B_xy))
                samples_concat.append(np.concatenate([zx, zy]))
                samples_dig_ew.append(zx * zy)
                pair_labels.append(pair_idx)
                binary_labels.append(bl)
            pair_idx += 1

    return (np.array(samples_B_mag), np.array(samples_concat),
            np.array(samples_dig_ew), np.array(pair_labels), np.array(binary_labels))


# ══════════════════════════════════════════════════════════════════════════════
# Feature selection helpers
# ══════════════════════════════════════════════════════════════════════════════
def f_score_ranking(X, y):
    """Compute per-feature ANOVA F-score for multi-class y. Return sorted indices."""
    from sklearn.feature_selection import f_classif
    scores, _ = f_classif(X, y)
    scores = np.nan_to_num(scores, nan=0.0)
    return np.argsort(scores)[::-1]


def variance_ranking(X):
    """Rank features by variance (descending)."""
    return np.argsort(X.var(axis=0))[::-1]


# ══════════════════════════════════════════════════════════════════════════════
# LOO-CV evaluation
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

    return float(balanced_accuracy_score(y, preds))


def standardize(X):
    mu = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std[std < 1e-12] = 1.0
    return (X - mu) / std


# ══════════════════════════════════════════════════════════════════════════════
# Main compressed-budget comparison
# ══════════════════════════════════════════════════════════════════════════════
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    print("=" * 60)
    print("Compressed-Budget Feature Comparison")
    print("=" * 60)

    # Load data
    print("Loading data...")
    single_data = load_single_token_data(data_dir)
    blank_data = load_blank_control_data(data_dir)
    pair_data = load_pair_interaction_data(data_dir)

    print("Computing per-repeat B(x,y)...")
    B_per_rep = compute_per_repeat_B(blank_data, pair_data)

    print("Building full feature matrices...")
    X_B_mag, X_concat, X_dig_ew, pair_labels, binary_labels = \
        build_full_features(single_data, B_per_rep)

    print(f"  X_B_mag:   {X_B_mag.shape}")
    print(f"  X_concat:  {X_concat.shape}")
    print(f"  X_dig_ew:  {X_dig_ew.shape}")
    print(f"  pair_labels: {len(pair_labels)} samples, {len(set(pair_labels))} classes")
    print(f"  binary_labels: {binary_labels.sum()}/{len(binary_labels)} positive")

    # Compute rankings on full data (for deterministic selection)
    # For Task A rankings
    print("\nComputing feature rankings (Task A = pair labels)...")
    rank_B_fscore_A = f_score_ranking(X_B_mag, pair_labels)
    rank_B_var = variance_ranking(X_B_mag)
    rank_concat_fscore_A = f_score_ranking(X_concat, pair_labels)
    rank_dig_fscore_A = f_score_ranking(X_dig_ew, pair_labels)

    # For Task B rankings
    print("Computing feature rankings (Task B = binary labels)...")
    rank_B_fscore_B = f_score_ranking(X_B_mag, binary_labels)
    rank_concat_fscore_B = f_score_ranking(X_concat, binary_labels)
    rank_dig_fscore_B = f_score_ranking(X_dig_ew, binary_labels)

    rng = np.random.default_rng(54321)

    results = {}

    for k in BUDGETS:
        print(f"\n{'='*40}")
        print(f"  Budget k = {k}")
        print(f"{'='*40}")

        res_k = {}

        # ── Deterministic selections ──────────────────────────────────

        # Task A evaluations
        for name, X_full, rank in [
            ("phys_B_fscore", X_B_mag, rank_B_fscore_A),
            ("phys_B_var", X_B_mag, rank_B_var),
            ("concat_fscore", X_concat, rank_concat_fscore_A),
            ("dig_ew_fscore", X_dig_ew, rank_dig_fscore_A),
        ]:
            sel = rank[:k]
            X_sel = standardize(X_full[:, sel])
            ba_A = evaluate_loocv(X_sel, pair_labels)
            ba_B = evaluate_loocv(X_sel, binary_labels)
            res_k[name] = {
                "task_A_bal_acc": round(ba_A, 4),
                "task_B_bal_acc": round(ba_B, 4),
                "n_features": k,
                "selection": "deterministic",
            }
            print(f"  {name:25s}  A={ba_A:.4f}  B={ba_B:.4f}")

        # Task B-optimized selection (use Task B F-scores)
        for name, X_full, rank in [
            ("phys_B_fscore_B", X_B_mag, rank_B_fscore_B),
            ("concat_fscore_B", X_concat, rank_concat_fscore_B),
            ("dig_ew_fscore_B", X_dig_ew, rank_dig_fscore_B),
        ]:
            sel = rank[:k]
            X_sel = standardize(X_full[:, sel])
            ba_A = evaluate_loocv(X_sel, pair_labels)
            ba_B = evaluate_loocv(X_sel, binary_labels)
            res_k[name] = {
                "task_A_bal_acc": round(ba_A, 4),
                "task_B_bal_acc": round(ba_B, 4),
                "n_features": k,
                "selection": "deterministic_taskB",
            }
            print(f"  {name:25s}  A={ba_A:.4f}  B={ba_B:.4f}")

        # ── Random selections (50 trials each) ───────────────────────
        for name, X_full, n_feat_total in [
            ("phys_B_random", X_B_mag, X_B_mag.shape[1]),
            ("concat_random", X_concat, X_concat.shape[1]),
            ("dig_ew_random", X_dig_ew, X_dig_ew.shape[1]),
        ]:
            ba_A_trials = []
            ba_B_trials = []
            for trial in range(N_RANDOM_TRIALS):
                sel = rng.choice(n_feat_total, size=min(k, n_feat_total), replace=False)
                X_sel = standardize(X_full[:, sel])
                ba_A_trials.append(evaluate_loocv(X_sel, pair_labels))
                ba_B_trials.append(evaluate_loocv(X_sel, binary_labels))

            res_k[name] = {
                "task_A_bal_acc_mean": round(float(np.mean(ba_A_trials)), 4),
                "task_A_bal_acc_std": round(float(np.std(ba_A_trials)), 4),
                "task_B_bal_acc_mean": round(float(np.mean(ba_B_trials)), 4),
                "task_B_bal_acc_std": round(float(np.std(ba_B_trials)), 4),
                "n_features": k,
                "n_trials": N_RANDOM_TRIALS,
                "selection": "random",
            }
            print(f"  {name:25s}  A={np.mean(ba_A_trials):.4f}±{np.std(ba_A_trials):.4f}  "
                  f"B={np.mean(ba_B_trials):.4f}±{np.std(ba_B_trials):.4f}")

        results[str(k)] = res_k

    # ══════════════════════════════════════════════════════════════════════
    # Generate figures
    # ══════════════════════════════════════════════════════════════════════
    print("\nGenerating figures...")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ── Fig 1: Task A accuracy vs budget ──────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for task_key, task_label, ax in [
        ("task_A", "Task A (16-class pair ID)", ax1),
        ("task_B", "Task B (binary relation)", ax2),
    ]:
        # Deterministic lines
        for name, color, marker, ls in [
            ("phys_B_fscore", "red", "o", "-"),
            ("concat_fscore", "blue", "s", "-"),
            ("dig_ew_fscore", "green", "^", "-"),
            ("phys_B_var", "red", "o", "--"),
        ]:
            vals = []
            for k in BUDGETS:
                r = results[str(k)].get(name, {})
                vals.append(r.get(f"{task_key}_bal_acc", 0))
            label = name
            ax.plot(BUDGETS, vals, f'{marker}{ls}', color=color, label=label, markersize=6)

        # Random bands
        for name, color in [
            ("phys_B_random", "red"),
            ("concat_random", "blue"),
            ("dig_ew_random", "green"),
        ]:
            means = []
            stds = []
            for k in BUDGETS:
                r = results[str(k)].get(name, {})
                means.append(r.get(f"{task_key}_bal_acc_mean", 0))
                stds.append(r.get(f"{task_key}_bal_acc_std", 0))
            means = np.array(means)
            stds = np.array(stds)
            ax.fill_between(BUDGETS, means - stds, means + stds,
                           alpha=0.15, color=color)
            ax.plot(BUDGETS, means, ':', color=color, alpha=0.6,
                   label=f"{name} (mean±std)")

        ax.set_xlabel("Feature budget k")
        ax.set_ylabel("Balanced accuracy")
        ax.set_title(task_label)
        ax.set_xscale("log")
        ax.set_xticks(BUDGETS)
        ax.set_xticklabels([str(b) for b in BUDGETS])
        ax.legend(fontsize=6, loc="lower right")
        ax.axhline(1/16, color="gray", ls=":", alpha=0.5, label="chance (A)")
        ax.axhline(0.5, color="gray", ls="--", alpha=0.3)
        ax.set_ylim(-0.05, 1.1)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig_budget_comparison_both_tasks.png"), dpi=200)
    plt.close()
    print("  Saved fig_budget_comparison_both_tasks.png")

    # ── Fig 2: Task B with Task-B-optimized selection ─────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, color, marker in [
        ("phys_B_fscore_B", "red", "o"),
        ("concat_fscore_B", "blue", "s"),
        ("dig_ew_fscore_B", "green", "^"),
    ]:
        vals = []
        for k in BUDGETS:
            r = results[str(k)].get(name, {})
            vals.append(r.get("task_B_bal_acc", 0))
        ax.plot(BUDGETS, vals, f'{marker}-', color=color, label=name, markersize=7)

    # Add random bands
    for name, color in [
        ("phys_B_random", "red"),
        ("dig_ew_random", "green"),
    ]:
        means = []
        stds = []
        for k in BUDGETS:
            r = results[str(k)].get(name, {})
            means.append(r.get("task_B_bal_acc_mean", 0))
            stds.append(r.get("task_B_bal_acc_std", 0))
        means = np.array(means)
        stds = np.array(stds)
        ax.fill_between(BUDGETS, means - stds, means + stds, alpha=0.15, color=color)

    ax.set_xlabel("Feature budget k")
    ax.set_ylabel("Balanced accuracy (Task B)")
    ax.set_title("Task B — Feature-Budget-Optimized Selection")
    ax.set_xscale("log")
    ax.set_xticks(BUDGETS)
    ax.set_xticklabels([str(b) for b in BUDGETS])
    ax.legend(fontsize=7)
    ax.axhline(0.5, color="gray", ls="--", alpha=0.5)
    ax.set_ylim(-0.05, 1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig_taskB_optimized_budget.png"), dpi=200)
    plt.close()
    print("  Saved fig_taskB_optimized_budget.png")

    # ── Fig 3: Advantage heatmap (physical B vs best competitor) ──────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for task_key, task_label, ax in [
        ("task_A", "Task A advantage", ax1),
        ("task_B", "Task B advantage", ax2),
    ]:
        adv_vs_concat = []
        adv_vs_dig = []
        for k in BUDGETS:
            phys = results[str(k)]["phys_B_fscore"].get(f"{task_key}_bal_acc", 0)
            conc = results[str(k)]["concat_fscore"].get(f"{task_key}_bal_acc", 0)
            dig = results[str(k)]["dig_ew_fscore"].get(f"{task_key}_bal_acc", 0)
            adv_vs_concat.append(phys - conc)
            adv_vs_dig.append(phys - dig)

        x = np.arange(len(BUDGETS))
        w = 0.35
        ax.bar(x - w/2, adv_vs_concat, w, label="phys_B - concat", color="steelblue")
        ax.bar(x + w/2, adv_vs_dig, w, label="phys_B - dig_ew", color="coral")
        ax.set_xticks(x)
        ax.set_xticklabels([str(b) for b in BUDGETS])
        ax.set_xlabel("Feature budget k")
        ax.set_ylabel("Balanced accuracy advantage")
        ax.set_title(task_label)
        ax.legend(fontsize=8)
        ax.axhline(0, color="black", lw=0.5)
        ax.axhline(0.05, color="red", ls="--", alpha=0.5, label="5pp threshold")

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig_advantage_heatmap.png"), dpi=200)
    plt.close()
    print("  Saved fig_advantage_heatmap.png")

    # ══════════════════════════════════════════════════════════════════════
    # Save results
    # ══════════════════════════════════════════════════════════════════════
    results_path = os.path.join(output_dir, "compressed_budget_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # ══════════════════════════════════════════════════════════════════════
    # Print summary table
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("SUMMARY: Physical B advantage over competitors at each budget")
    print("=" * 70)
    print(f"{'k':>5}  {'phys_B(A)':>10}  {'concat(A)':>10}  {'dig_ew(A)':>10}  "
          f"{'adv_vs_C':>10}  {'adv_vs_D':>10}  |  "
          f"{'phys_B(B)':>10}  {'dig_ew(B)':>10}  {'adv_B':>10}")
    print("-" * 110)

    for k in BUDGETS:
        rk = results[str(k)]
        pA = rk["phys_B_fscore"]["task_A_bal_acc"]
        cA = rk["concat_fscore"]["task_A_bal_acc"]
        dA = rk["dig_ew_fscore"]["task_A_bal_acc"]
        pB = rk["phys_B_fscore"]["task_B_bal_acc"]
        dB = rk["dig_ew_fscore"]["task_B_bal_acc"]
        print(f"{k:>5}  {pA:>10.4f}  {cA:>10.4f}  {dA:>10.4f}  "
              f"{pA-cA:>+10.4f}  {pA-dA:>+10.4f}  |  "
              f"{pB:>10.4f}  {dB:>10.4f}  {pB-dB:>+10.4f}")

    # Thresholded comparison summary
    any_advantage = False
    for k in BUDGETS:
        rk = results[str(k)]
        pA = rk["phys_B_fscore"]["task_A_bal_acc"]
        cA = rk["concat_fscore"]["task_A_bal_acc"]
        dA = rk["dig_ew_fscore"]["task_A_bal_acc"]
        pB = rk["phys_B_fscore"]["task_B_bal_acc"]
        dB = rk["dig_ew_fscore"]["task_B_bal_acc"]
        if (pA - cA >= 0.05 and pA - dA >= 0.05) or (pB - dB >= 0.05):
            any_advantage = True
            break

    print("\n" + "=" * 70)
    if any_advantage:
        print("SUMMARY: Physical B shows a >= 5 percentage-point margin "
              "in at least one tested budget/task comparison.")
    else:
        print("SUMMARY: Physical B does not show a consistent >= 5 "
              "percentage-point margin across the tested budget/task comparisons.")
    print("=" * 70)

    print(f"\nResults saved to: {results_path}")
    print(f"Figures saved to: {fig_dir}")


if __name__ == "__main__":
    main()
