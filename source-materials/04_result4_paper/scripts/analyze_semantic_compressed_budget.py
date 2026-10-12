#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compressed-budget reanalysis — Semantic Complex-B Benchmark
============================================================
Runs on existing physical data (no new experiment needed).

Tests whether the structural separation (Complex-B = joint encoding,
concat = identity only, digital bilinear = relation only) survives
under matched feature budgets k = {10, 25, 50, 100, 200, 400}.

Tasks:
  A: 64-class ordered pair identification
  B: binary same-category vs different-category
  C: 16-class category-pair identification

Complex-B discipline: primary object is B(x,y) as [Re(B), Im(B)].
"""

import os
import sys
import json
import numpy as np

N_TOKENS = 8
BIN_SIZE = 25
PHASES = [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]
BUDGETS = [10, 25, 50, 100, 200, 400]

SEMANTIC_TOKENS = [
    {"id": 0, "word": "cat",    "category": 0},
    {"id": 1, "word": "dog",    "category": 0},
    {"id": 2, "word": "hammer", "category": 1},
    {"id": 3, "word": "saw",    "category": 1},
    {"id": 4, "word": "apple",  "category": 2},
    {"id": 5, "word": "grape",  "category": 2},
    {"id": 6, "word": "car",    "category": 3},
    {"id": 7, "word": "train",  "category": 3},
]


def bin_roi(roi, bin_size=BIN_SIZE):
    h, w = roi.shape
    nh, nw = h // bin_size, w // bin_size
    return roi[:nh*bin_size, :nw*bin_size].reshape(nh, bin_size, nw, bin_size).mean(axis=(1, 3)).flatten()


def load_and_bin(data_dir, filename):
    return bin_roi(np.load(os.path.join(data_dir, filename)))


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


def compute_per_repeat_B(blank_data, pair_data):
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


def build_feature_families(single_data, B_per_rep, n_reps=5):
    z_single = {tid: single_data[tid].mean(axis=0) for tid in range(N_TOKENS)}

    samples = []
    pair_labels = []
    cat_binary_labels = []
    cat_pair_labels = []

    pair_idx = 0
    for x in range(N_TOKENS):
        for y in range(N_TOKENS):
            cat_x = SEMANTIC_TOKENS[x]["category"]
            cat_y = SEMANTIC_TOKENS[y]["category"]
            same_cat = int(cat_x == cat_y)
            cat_pair_id = cat_x * 4 + cat_y

            for rep in range(n_reps):
                samples.append({
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

    return families, pair_labels, cat_binary_labels, cat_pair_labels


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


def evaluate_at_budget(X, y, budget, n_random=10, seed=42):
    """Evaluate at fixed feature budget via random subspace, averaging over trials."""
    from sklearn.linear_model import RidgeClassifier
    from sklearn.metrics import balanced_accuracy_score
    n_feat = X.shape[1]
    if n_feat <= budget:
        return evaluate_loocv(X, y)
    rng = np.random.default_rng(seed)
    accs = []
    for trial in range(n_random):
        sel = rng.choice(n_feat, budget, replace=False)
        acc = evaluate_loocv(X[:, sel], y)
        accs.append(acc)
    return float(np.mean(accs))


def standardize(X):
    mu = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std[std < 1e-12] = 1.0
    return (X - mu) / std


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    fig_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    print("=" * 60)
    print("Compressed-Budget Reanalysis — Semantic Complex-B")
    print("=" * 60)

    # Load data
    print("Loading data...")
    single_data = load_single_token_data(args.data_dir)
    blank_data = load_blank_control_data(args.data_dir)
    pair_data = load_pair_interaction_data(args.data_dir)

    print("Computing Complex-B fields...")
    B_per_rep = compute_per_repeat_B(blank_data, pair_data)

    print("Building feature families...")
    families, pair_labels, cat_binary, cat_pair = build_feature_families(
        single_data, B_per_rep)

    # Focus on the key families for compressed-budget comparison
    key_families = ["zx_only", "zy_only", "zx_zy_concat", "zx_minus_zy",
                    "digital_bilinear_ew", "complex_B"]

    results = {}
    for name in key_families:
        X = families[name]
        if np.iscomplexobj(X):
            X = np.concatenate([np.real(X), np.imag(X)], axis=1)
        X_std = standardize(X)
        n_feat = X_std.shape[1]

        results[name] = {"n_features": n_feat, "budgets": {}}
        print(f"\n{name} ({n_feat} features):")

        for k in BUDGETS:
            if k > n_feat:
                # Use full features if budget exceeds available
                k_eff = n_feat
            else:
                k_eff = k

            print(f"  k={k} (eff={k_eff})...", end=" ", flush=True)

            bal_A = evaluate_at_budget(X_std, pair_labels, k_eff)
            bal_B = evaluate_at_budget(X_std, cat_binary, k_eff)
            bal_C = evaluate_at_budget(X_std, cat_pair, k_eff)

            results[name]["budgets"][str(k)] = {
                "k_effective": k_eff,
                "task_A_pair_id": round(bal_A, 4),
                "task_B_same_cat": round(bal_B, 4),
                "task_C_cat_pair": round(bal_C, 4),
            }
            print(f"A={bal_A:.4f}  B={bal_B:.4f}  C={bal_C:.4f}")

    # Save results
    results_path = os.path.join(args.output_dir, "compressed_budget_semantic_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # ── Generate figures ──────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Fig 1: Task A (pair ID) vs budget
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    task_keys = [
        ("task_A_pair_id", "Task A: 64-class Pair ID"),
        ("task_B_same_cat", "Task B: Same-Category (binary)"),
        ("task_C_cat_pair", "Task C: 16-class Category-Pair"),
    ]

    colors = {
        "zx_only": "#95a5a6",
        "zy_only": "#bdc3c7",
        "zx_zy_concat": "#3498db",
        "zx_minus_zy": "#9b59b6",
        "digital_bilinear_ew": "#e67e22",
        "complex_B": "#e74c3c",
    }
    markers = {
        "zx_only": "v",
        "zy_only": "^",
        "zx_zy_concat": "s",
        "zx_minus_zy": "D",
        "digital_bilinear_ew": "o",
        "complex_B": "*",
    }

    for ax_idx, (task_key, task_title) in enumerate(task_keys):
        ax = axes[ax_idx]
        for name in key_families:
            vals = []
            ks = []
            for k in BUDGETS:
                entry = results[name]["budgets"][str(k)]
                vals.append(entry[task_key])
                ks.append(entry["k_effective"])
            ax.plot(BUDGETS, vals, marker=markers[name], color=colors[name],
                    label=name, linewidth=1.5, markersize=7)
        ax.set_xlabel("Feature budget k")
        ax.set_ylabel("Balanced Accuracy")
        ax.set_title(task_title, fontsize=11)
        ax.set_xscale("log")
        ax.set_xticks(BUDGETS)
        ax.set_xticklabels([str(k) for k in BUDGETS])
        if task_key == "task_B_same_cat":
            ax.axhline(0.5, color='gray', ls='--', alpha=0.3, label='chance')
        if ax_idx == 0:
            ax.legend(fontsize=7, loc='lower right')

    fig.suptitle("Compressed-Budget Analysis — Semantic Complex-B Benchmark",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig_compressed_budget_semantic.pdf"),
                bbox_inches="tight", dpi=300)
    plt.savefig(os.path.join(fig_dir, "fig_compressed_budget_semantic.png"),
                bbox_inches="tight", dpi=300)
    plt.close()

    # Fig 2: Summary heatmap at k=100 (matched budget)
    fig, ax = plt.subplots(figsize=(10, 5))
    k_ref = "100"
    family_names = key_families
    task_names = ["task_A_pair_id", "task_B_same_cat", "task_C_cat_pair"]
    task_display = ["Pair ID (A)", "Same-Cat (B)", "Cat-Pair (C)"]

    matrix = np.zeros((len(family_names), len(task_names)))
    for i, name in enumerate(family_names):
        for j, tk in enumerate(task_names):
            matrix[i, j] = results[name]["budgets"][k_ref][tk]

    im = ax.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(len(task_display)))
    ax.set_xticklabels(task_display, fontsize=10)
    ax.set_yticks(range(len(family_names)))
    ax.set_yticklabels(family_names, fontsize=10)
    for i in range(len(family_names)):
        for j in range(len(task_names)):
            ax.text(j, i, f"{matrix[i,j]:.3f}", ha='center', va='center',
                    fontsize=10, fontweight='bold',
                    color='white' if matrix[i,j] < 0.4 else 'black')
    plt.colorbar(im, ax=ax, label='Balanced Accuracy')
    ax.set_title(f"Matched-Budget Comparison at k={k_ref} features", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig_matched_budget_heatmap.pdf"),
                bbox_inches="tight", dpi=300)
    plt.savefig(os.path.join(fig_dir, "fig_matched_budget_heatmap.png"),
                bbox_inches="tight", dpi=300)
    plt.close()

    # Print summary
    print("\n" + "=" * 60)
    print("COMPRESSED-BUDGET SUMMARY (k=100, matched)")
    print("=" * 60)
    for name in key_families:
        e = results[name]["budgets"]["100"]
        print(f"  {name:25s}  A={e['task_A_pair_id']:.4f}  "
              f"B={e['task_B_same_cat']:.4f}  C={e['task_C_cat_pair']:.4f}")

    print(f"\nKey structural separation check at k=100:")
    cB = results["complex_B"]["budgets"]["100"]
    cc = results["zx_zy_concat"]["budgets"]["100"]
    db = results["digital_bilinear_ew"]["budgets"]["100"]
    print(f"  concat:           A={cc['task_A_pair_id']:.4f}  B={cc['task_B_same_cat']:.4f}")
    print(f"  digital_bilinear: A={db['task_A_pair_id']:.4f}  B={db['task_B_same_cat']:.4f}")
    print(f"  complex_B:        A={cB['task_A_pair_id']:.4f}  B={cB['task_B_same_cat']:.4f}")
    print(f"  → Complex-B is joint encoder: {'YES' if cB['task_A_pair_id'] > 0.5 and cB['task_B_same_cat'] > 0.5 else 'NO'}")
    print(f"  → Concat fails relation:      {'YES' if cc['task_B_same_cat'] <= 0.55 else 'NO'}")
    print(f"  → Dig bilinear fails identity: {'YES' if db['task_A_pair_id'] < 0.5 else 'NO'}")

    print(f"\nResults: {results_path}")
    print(f"Figures: {fig_dir}")


if __name__ == "__main__":
    main()
