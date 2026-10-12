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

# Figure 1: scaling summary
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

# Figure 2: full-TM singular spectrum
sv = np.load(data / 'singular_values.npy')
fig, ax = plt.subplots(figsize=(5.4, 4.0))
ax.semilogy(np.arange(1, len(sv)+1), sv, lw=2, color='#9467bd')
ax.set_xlabel('Singular-value index')
ax.set_ylabel('Singular value')
ax.set_title('Full 256×256 TM singular spectrum')
ax.grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig(figdir / 'tm_singular_spectrum.pdf', bbox_inches='tight')
plt.close(fig)

# Figure 3: focusing maps
f0 = np.load(data / 'focus_target0_modes.npy').reshape(16,16)
f192 = np.load(data / 'focus_target192_modes.npy').reshape(16,16)
fig, axs = plt.subplots(1, 2, figsize=(8.0, 3.6))
for ax, arr, title in zip(axs, [f0, f192], ['Target mode 0', 'Target mode 192']):
    im = ax.imshow(arr, cmap='inferno')
    ax.set_title(title)
    ax.set_xlabel('Output mode x')
    ax.set_ylabel('Output mode y')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.tight_layout()
fig.savefig(figdir / 'tm_focus_maps.pdf', bbox_inches='tight')
plt.close(fig)

# Figure 4: pattern prediction example
pred = np.load(data / 'output_center_spot.npy')
targ = np.load(data / 'target_center_spot.npy')
fig, axs = plt.subplots(1, 2, figsize=(6.8, 3.2))
for ax, arr, title in zip(axs, [targ, pred], ['Target pattern', 'Measured output']):
    im = ax.imshow(arr, cmap='viridis', vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xlabel('Mode x')
    ax.set_ylabel('Mode y')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.tight_layout()
fig.savefig(figdir / 'tm_pattern_prediction_center_spot.pdf', bbox_inches='tight')
plt.close(fig)

print('Saved figures to', figdir)
