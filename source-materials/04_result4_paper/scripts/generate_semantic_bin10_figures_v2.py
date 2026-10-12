#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Proper bin=10 semantic benchmark figure regeneration.
Uses explicit bin_size=10 loaders, not monkey-patching defaults.
"""
import os, json, numpy as np
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from analyze_refinement_phase2_fast import (
    load_all_data_binned, compute_B_per_repeat, build_features_and_labels,
    skf_balanced_accuracy, cross_split_eval, SEMANTIC_TOKENS, N_TOKENS, ROI_H, ROI_W
)


def analyze_complex_B_stability(B_per_rep):
    stability = {}
    all_corrs = []
    for (x, y), B_arr in B_per_rep.items():
        B_real = np.concatenate([np.real(B_arr), np.imag(B_arr)], axis=1)
        corrs = []
        for i in range(B_arr.shape[0]):
            for j in range(i + 1, B_arr.shape[0]):
                c = np.corrcoef(B_real[i], B_real[j])[0, 1]
                if not np.isnan(c):
                    corrs.append(c)
        mc = float(np.mean(corrs)) if corrs else 0.0
        stability[f"({x},{y})"] = {"mean_complex_corr": mc}
        all_corrs.append(mc)
    return stability, float(np.mean(all_corrs)), float(np.min(all_corrs))


def make_figures_bin10(results, cross_results, B_pair_mean, B_per_rep, pair_data, fig_dir):
    os.makedirs(fig_dir, exist_ok=True)

    # Fig1 phase cycles
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    phi_vals = [0, np.pi/2, np.pi, 3*np.pi/2]
    rep_pairs = [(0,1,"cat-dog"), (2,3,"hammer-saw"), (0,2,"cat-hammer"),
                 (4,6,"apple-car"), (0,0,"cat-cat"), (6,7,"car-train")]
    for idx, (x, y, label) in enumerate(rep_pairs):
        ax = axes[idx // 3, idx % 3]
        for ch in [100, 500, 1000, 1500]:
            if ch < pair_data[(x, y, 0)].shape[1]:
                means = [pair_data[(x, y, phi)][:, ch].mean() for phi in range(4)]
                ax.plot(phi_vals, means, 'o-', label=f'ch={ch}', markersize=4)
        ax.set_title(f'({label})', fontsize=10)
        ax.set_xlabel('Phase φ')
        ax.set_ylabel('Intensity')
        if idx == 0:
            ax.legend(fontsize=6)
    fig.suptitle('Four-phase intensity modulation — semantic pairs (bin=10)', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'fig1_phase_cycles_semantic_bin10.pdf'), bbox_inches='tight', dpi=300)
    plt.close()

    # Fig2 same vs cross complex plane
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    same_pairs = [(0,1), (2,3), (4,5)]
    cross_pairs = [(0,2), (4,6), (1,7)]
    for idx, ((sx, sy), (cx, cy)) in enumerate(zip(same_pairs, cross_pairs)):
        ax_s = axes[0, idx]
        Bs = B_pair_mean[(sx, sy)]
        ax_s.scatter(np.real(Bs), np.imag(Bs), s=3, alpha=0.4, c='#2ecc71')
        ax_s.set_title(f'Same: ({SEMANTIC_TOKENS[sx]["word"]},{SEMANTIC_TOKENS[sy]["word"]})', fontsize=9)
        ax_s.set_xlabel('Re(B)'); ax_s.set_ylabel('Im(B)'); ax_s.set_aspect('equal')
        ax_s.axhline(0, color='gray', lw=0.3); ax_s.axvline(0, color='gray', lw=0.3)
        ax_c = axes[1, idx]
        Bc = B_pair_mean[(cx, cy)]
        ax_c.scatter(np.real(Bc), np.imag(Bc), s=3, alpha=0.4, c='#e74c3c')
        ax_c.set_title(f'Cross: ({SEMANTIC_TOKENS[cx]["word"]},{SEMANTIC_TOKENS[cy]["word"]})', fontsize=9)
        ax_c.set_xlabel('Re(B)'); ax_c.set_ylabel('Im(B)'); ax_c.set_aspect('equal')
        ax_c.axhline(0, color='gray', lw=0.3); ax_c.axvline(0, color='gray', lw=0.3)
    fig.suptitle('Complex-B channels: same-category vs cross-category (bin=10)', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'fig2_complex_B_same_vs_cross_bin10.pdf'), bbox_inches='tight', dpi=300)
    plt.close()

    # Fig3 classification bars
    family_order = ["zx_zy_concat", "digital_bilinear_ew", "complex_B"]
    task_A = [results[f]['task_A_pair_id'] for f in family_order]
    task_B = [results[f]['task_B_same_cat'] for f in family_order]
    task_C = [results[f]['task_C_cat_pair'] for f in family_order]
    x_pos = np.arange(len(family_order)); width = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x_pos - width, task_A, width, label='Task A', color='#3498db')
    ax.bar(x_pos, task_B, width, label='Task B', color='#e67e22')
    ax.bar(x_pos + width, task_C, width, label='Task C', color='#2ecc71')
    ax.set_ylabel('Balanced Accuracy')
    ax.set_title('Semantic Complex-B Benchmark — bin=10')
    ax.set_xticks(x_pos); ax.set_xticklabels(family_order, rotation=20)
    ax.legend(); ax.axhline(0.5, color='gray', ls='--', alpha=0.3)
    for i, (a, b, c) in enumerate(zip(task_A, task_B, task_C)):
        ax.text(i - width, a + 0.01, f'{a:.3f}', ha='center', fontsize=7)
        ax.text(i, b + 0.01, f'{b:.3f}', ha='center', fontsize=7)
        ax.text(i + width, c + 0.01, f'{c:.3f}', ha='center', fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'fig3_semantic_benchmark_comparison_bin10.pdf'), bbox_inches='tight', dpi=300)
    plt.close()

    # Fig4 cross-split
    cross_vals = [cross_results[f]['cross_split_same_cat_bal_acc'] for f in family_order]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(family_order)), cross_vals, color=['#95a5a6', '#95a5a6', '#e74c3c'])
    ax.set_xticks(range(len(family_order))); ax.set_xticklabels(family_order, rotation=20)
    ax.set_ylabel('Balanced Accuracy')
    ax.set_title('Cross-split generalization (bin=10)')
    ax.axhline(0.5, color='gray', ls='--', alpha=0.3)
    for i, v in enumerate(cross_vals):
        ax.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'fig4_cross_split_generalization_bin10.pdf'), bbox_inches='tight', dpi=300)
    plt.close()

    # Fig5 stability
    stab_dict, mean_corr, min_corr = analyze_complex_B_stability(B_per_rep)
    pairs_sorted = sorted(stab_dict.keys())
    corrs = [stab_dict[p]['mean_complex_corr'] for p in pairs_sorted]
    fig, ax = plt.subplots(figsize=(16, 5))
    colors = ['#2ecc71' if c > 0.9 else '#e74c3c' for c in corrs]
    ax.bar(range(len(pairs_sorted)), corrs, color=colors, width=0.8)
    ax.set_xticks(range(len(pairs_sorted))); ax.set_xticklabels(pairs_sorted, rotation=90, fontsize=5)
    ax.set_ylabel('Mean repeat correlation')
    ax.set_title(f'Complex-B stability (bin=10, mean={mean_corr:.4f}, min={min_corr:.4f})')
    ax.axhline(0.9, color='r', ls='--', lw=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'fig5_semantic_stability_bin10.pdf'), bbox_inches='tight', dpi=300)
    plt.close()

    # Fig6 PCA
    B_all, cat_bin_all, cat_pair_all = [], [], []
    for x in range(N_TOKENS):
        for y in range(N_TOKENS):
            cx = SEMANTIC_TOKENS[x]['category']; cy = SEMANTIC_TOKENS[y]['category']
            same = int(cx == cy); cpid = cx * 4 + cy
            B_reps = B_per_rep[(x, y)]
            for rep in range(B_reps.shape[0]):
                B_all.append(np.concatenate([np.real(B_reps[rep]), np.imag(B_reps[rep])]))
                cat_bin_all.append(same); cat_pair_all.append(cpid)
    B_all = np.array(B_all); cat_bin_all = np.array(cat_bin_all); cat_pair_all = np.array(cat_pair_all)
    pca = PCA(n_components=2); B_2d = pca.fit_transform(B_all)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    for label, color, name in [(1, '#2ecc71', 'Same category'), (0, '#e74c3c', 'Different category')]:
        mask = cat_bin_all == label
        ax1.scatter(B_2d[mask, 0], B_2d[mask, 1], c=color, s=8, alpha=0.5, label=name)
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)'); ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax1.set_title('Complex-B PCA — same vs different (bin=10)'); ax1.legend()
    cmap = plt.colormaps.get_cmap('tab20')
    unique_cpids = sorted(set(cat_pair_all))
    for i, cpid in enumerate(unique_cpids):
        mask = np.array([c == cpid for c in cat_pair_all])
        cx, cy = cpid // 4, cpid % 4
        ax2.scatter(B_2d[mask, 0], B_2d[mask, 1], c=[cmap(i / len(unique_cpids))], s=8, alpha=0.5)
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)'); ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax2.set_title('Complex-B PCA — category pairs (bin=10)')
    fig.suptitle('Semantic structure in Complex-B space (bin=10)', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'fig6_semantic_pca_bin10.pdf'), bbox_inches='tight', dpi=300)
    plt.close()


def main():
    section_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    package_dir = os.path.dirname(section_dir)
    data_dir = os.path.join(package_dir, 'shared_raw_data', 'result4_frame_data',
                            'exp_complex_B_semantic')
    out_dir = os.path.join(section_dir, 'source_data', 'processed',
                           'complex_B_semantic_results_bin10_v2')
    fig_dir = os.path.join(section_dir, 'figures', 'semantic_bin10_v2')
    os.makedirs(fig_dir, exist_ok=True)
    bin_size = 10

    single_data, blank_data, pair_data = load_all_data_binned(data_dir, bin_size)
    B_per_rep = compute_B_per_repeat(blank_data, pair_data)
    B_pair_mean = {(x, y): B_per_rep[(x, y)].mean(axis=0) for x in range(N_TOKENS) for y in range(N_TOKENS)}

    features, pair_labels, cat_binary, cat_pair, pair_info = build_features_and_labels(single_data, B_per_rep)

    results = {}
    for name, X in features.items():
        results[name] = {
            'n_features': int(X.shape[1]),
            'task_A_pair_id': round(skf_balanced_accuracy(X, pair_labels), 4),
            'task_B_same_cat': round(skf_balanced_accuracy(X, cat_binary), 4),
            'task_C_cat_pair': round(skf_balanced_accuracy(X, cat_pair), 4),
        }
    cross_results = {}
    for name, X in features.items():
        cross_results[name] = {
            'cross_split_same_cat_bal_acc': round(cross_split_eval(X, cat_binary, pair_info), 4)
        }

    make_figures_bin10(results, cross_results, B_pair_mean, B_per_rep, pair_data, fig_dir)

    payload = {
        'bin_size': 10,
        'n_channels': int((ROI_H // bin_size) * (ROI_W // bin_size)),
        'classification_results': results,
        'cross_split_results': cross_results,
    }
    with open(os.path.join(out_dir, 'complex_B_semantic_results_bin10_v2.json'), 'w') as f:
        json.dump(payload, f, indent=2)
    print('Saved proper bin=10 figures/results to', out_dir)


if __name__ == '__main__':
    main()
