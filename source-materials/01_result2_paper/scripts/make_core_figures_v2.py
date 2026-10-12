import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

root = Path(__file__).resolve().parents[1]
data = root / 'source_data'
figdir = root / 'figures'
figdir.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
})

# Existing scaling summary preserved from earlier evidence-bounded runs
Ns = np.array([16, 64, 256])
mean_enh = np.array([1.452019, 6.09, 16.76])
max_enh = np.array([3.321549, 15.78, 25.59])
brightest_rate = np.array([0/4, 1/6, 6/9])
fig, ax1 = plt.subplots(figsize=(5.6, 4.2))
ax1.plot(Ns, mean_enh, 'o-', lw=2, ms=7, label='Mean enhancement', color='#1f77b4')
ax1.plot(Ns, max_enh, 's--', lw=2, ms=6, label='Max enhancement', color='#d62728')
ax1.set_xscale('log', base=2)
ax1.set_xticks(Ns)
ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax1.set_xlabel('Controlled input modes')
ax1.set_ylabel('Enhancement vs random baseline')
ax1.grid(True, alpha=0.25)
ax2 = ax1.twinx()
ax2.plot(Ns, brightest_rate, 'd-.', lw=2, ms=6, color='#2ca02c', label='Brightest-bin success')
ax2.set_ylabel('Brightest-bin success ratio')
ax2.set_ylim(0, 1.0)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=False)
fig.tight_layout()
fig.savefig(figdir / 'tm_scaling_summary.pdf', bbox_inches='tight')
plt.close(fig)

# v6 singular spectrum
sv = np.load(data / 'singular_values.npy')
fig, ax = plt.subplots(figsize=(5.4, 4.0))
ax.semilogy(np.arange(1, len(sv)+1), sv, lw=2, color='#9467bd')
ax.set_xlabel('Singular-value index')
ax.set_ylabel('Singular value')
ax.set_title('Full 256x256 TM singular spectrum (v6)')
ax.grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig(figdir / 'tm_singular_spectrum_v6.pdf', bbox_inches='tight')
plt.close(fig)

# v6 focus maps with target markers from metadata
with open(data / 'metadata.json', 'r', encoding='utf-8') as f:
    meta = json.load(f)
focus_lookup = {r['target']: r for r in meta['focus_results']}
show_targets = [0, 255]
fig, axs = plt.subplots(1, 2, figsize=(8.2, 3.8))
for ax, target in zip(axs, show_targets):
    arr = np.load(data / f'focus_target{target}_modes.npy').reshape(16, 16)
    im = ax.imshow(arr, cmap='inferno')
    ty, tx = divmod(target, 16)
    ay, axp = divmod(focus_lookup[target]['actual_peak_mode'], 16)
    ax.scatter([tx], [ty], marker='s', s=70, facecolors='none', edgecolors='cyan', linewidths=1.8, label='Target')
    ax.scatter([axp], [ay], marker='x', s=55, c='lime', linewidths=1.8, label='Measured peak')
    ax.set_title(f'Target mode {target}')
    ax.set_xlabel('Output mode x')
    ax.set_ylabel('Output mode y')
    ax.legend(loc='upper right', frameon=True)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.tight_layout()
fig.savefig(figdir / 'tm_focus_maps_v6_marked.pdf', bbox_inches='tight')
plt.close(fig)

# New causal isolation figure: v5 vs v6 replay strategy comparison
labels = ['Mean enh.', 'Max enh.', 'Brightest-bin\nsuccess']
v5 = np.array([17.04, 33.51, 0/9])
v6 = np.array([17.65604098396905, 24.923296749310428, 6/9])
x = np.arange(len(labels))
width = 0.34
fig, ax = plt.subplots(figsize=(5.8, 4.2))
ax.bar(x - width/2, v5, width, label='v5: phase-only replay', color='#ff7f0e')
ax.bar(x + width/2, v6, width, label='v6: amplitude+phase replay', color='#1f77b4')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('Metric value')
ax.set_title('Replay strategy isolation under fixed ring-gap geometry')
ax.legend(frameon=False)
ax.grid(True, axis='y', alpha=0.25)
fig.tight_layout()
fig.savefig(figdir / 'tm_replay_strategy_comparison.pdf', bbox_inches='tight')
plt.close(fig)

print('Saved updated figures to', figdir)
