#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fix the reference-amplitude compensation direction.

Physics: K_obs[m,n] = conj(s_m) * K[m,n]
  => |K_obs[m,n]| = |s_m| * |K[m,n]|
  => row m is scaled by |s_m|
  => to remove |s_m|, compute row-mean amplitude and divide each row.

Previous (incorrect): axis=0 (column-wise), dividing columns
Corrected: axis=1 (row-wise), dividing rows
"""

import os
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_TM = f"{BASE}/source_data/tm_256"
FIG_DIR = f"{BASE}/figures"
PAPER_FIG = FIG_DIR

# Load the original K_obs
K_obs = np.load(f"{DATA_TM}/K_obs_canonical.npy")
print(f"K_obs shape: {K_obs.shape}")  # (256, 256)
print(f"K_obs mean|K|: {np.mean(np.abs(K_obs)):.6f}")

# ── CORRECT filtering: row-wise ──
K_amp = np.abs(K_obs)

# Row-wise: mean over columns (axis=1) → one value per row (output mode m)
S_row = np.mean(K_amp, axis=1)  # shape (256,)
S_row_safe = np.where(S_row > 1e-10, S_row, 1e-10)
K_filt_row = K_obs / S_row_safe[:, np.newaxis]

# Column-wise (OLD, incorrect): mean over rows (axis=0) → one value per column
S_col = np.mean(K_amp, axis=0)  # shape (256,)
S_col_safe = np.where(S_col > 1e-10, S_col, 1e-10)
K_filt_col = K_obs / S_col_safe[np.newaxis, :]

print(f"\nRow-wise S_row: mean={np.mean(S_row):.6f}, std={np.std(S_row):.6f}, CV={np.std(S_row)/np.mean(S_row):.3f}")
print(f"Col-wise S_col: mean={np.mean(S_col):.6f}, std={np.std(S_col):.6f}, CV={np.std(S_col)/np.mean(S_col):.3f}")

# The one with higher CV is the one that captures the |s_m| variation
# Row-wise should have higher CV since |s_m| varies by output mode

# SVD of both
_, sv_row, _ = np.linalg.svd(K_filt_row, full_matrices=False)
_, sv_col, _ = np.linalg.svd(K_filt_col, full_matrices=False)
_, sv_raw, _ = np.linalg.svd(K_obs, full_matrices=False)

sv_row_norm = sv_row / np.sqrt(np.sum(sv_row**2))
sv_col_norm = sv_col / np.sqrt(np.sum(sv_col**2))
sv_raw_norm = sv_raw / np.sqrt(np.sum(sv_raw**2))

print(f"\nSV condition numbers:")
print(f"  Raw:      {sv_raw[0]/sv_raw[-1]:.1f}")
print(f"  Row-filt: {sv_row[0]/sv_row[-1]:.1f}")
print(f"  Col-filt: {sv_col[0]/sv_col[-1]:.1f}")

# Save corrected files
np.save(f"{DATA_TM}/K_filtered.npy", K_filt_row)
np.save(f"{DATA_TM}/sv_filtered_normalized.npy", sv_row_norm)
print(f"\nSaved corrected K_filtered.npy and sv_filtered_normalized.npy")

# Also save the old column-wise for comparison
np.save(f"{DATA_TM}/K_filtered_col_old.npy", K_filt_col)
np.save(f"{DATA_TM}/sv_filtered_col_old.npy", sv_col_norm)

# ── Regenerate figures ──
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ═══ Figure: TM matrix (updated panel c with correct filtering) ═══
from matplotlib.gridspec import GridSpec

KKd = np.load(f"{DATA_TM}/KKdagger.npy")
KKd_norm = np.load(f"{DATA_TM}/KKdagger_normalized.npy")

fig = plt.figure(figsize=(14, 10))
gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

# (a) Amplitude
ax = fig.add_subplot(gs[0, 0])
im = ax.imshow(np.abs(K_obs), cmap='hot', aspect='equal', interpolation='nearest')
ax.set_title(r'(a) $|K_{\mathrm{obs}}|$', fontsize=11)
ax.set_xlabel('Input mode'); ax.set_ylabel('Output mode')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# (b) Phase
ax = fig.add_subplot(gs[0, 1])
im = ax.imshow(np.angle(K_obs), cmap='hsv', aspect='equal', vmin=-np.pi, vmax=np.pi,
               interpolation='nearest')
ax.set_title(r'(b) Phase of $K_{\mathrm{obs}}$', fontsize=11)
ax.set_xlabel('Input mode')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='rad')

# (c) CORRECTED: Row-wise filtered amplitude
ax = fig.add_subplot(gs[0, 2])
im = ax.imshow(np.abs(K_filt_row), cmap='hot', aspect='equal', interpolation='nearest')
ax.set_title(r'(c) $|K_{\mathrm{filtered}}|$ (row-normalised)', fontsize=11)
ax.set_xlabel('Input mode')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# (d) |KK†|
ax = fig.add_subplot(gs[1, 0])
im = ax.imshow(np.abs(KKd), cmap='hot', aspect='equal', interpolation='nearest')
ax.set_title(r'(d) $|K K^\dagger|$', fontsize=11)
ax.set_xlabel('Output mode'); ax.set_ylabel('Output mode')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# (e) Normalized |K_norm K_norm†|/N
ax = fig.add_subplot(gs[1, 1])
im = ax.imshow(np.abs(KKd_norm)/256, cmap='hot', aspect='equal', interpolation='nearest')
ax.set_title(r'(e) $|K_{\phi}K_{\phi}^\dagger|/N$', fontsize=11)
ax.set_xlabel('Output mode')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# (f) Diagonal / off-diagonal line cuts
ax = fig.add_subplot(gs[1, 2])
diag_KKd = np.abs(np.diag(KKd))
diag_norm = np.abs(np.diag(KKd_norm)) / 256
row128 = np.abs(KKd[128, :])
row128_norm = np.abs(KKd_norm[128, :]) / 256

ax.plot(diag_KKd / np.max(diag_KKd), 'r-', lw=1.2, label=r'Diag $|KK^\dagger|$')
ax.plot(diag_norm / np.max(diag_norm), 'b-', lw=1.2, label=r'Diag $|K_\phi K_\phi^\dagger|/N$')
ax.plot(row128 / np.max(diag_KKd), 'r--', lw=0.8, alpha=0.5, label='Row 128 (off-diag)')
ax.plot(row128_norm / np.max(diag_norm), 'b--', lw=0.8, alpha=0.5, label='Row 128 norm')
ax.set_xlabel('Mode index')
ax.set_ylabel('Normalized value')
ax.set_title('(f) Diagonal vs off-diagonal', fontsize=11)
ax.legend(fontsize=7, loc='upper right')

plt.savefig(f"{FIG_DIR}/fig_tm_matrix.pdf", bbox_inches='tight', dpi=300)
plt.savefig(f"{FIG_DIR}/fig_tm_matrix.png", bbox_inches='tight', dpi=300)
plt.close()
print("  fig_tm_matrix regenerated")

# ═══ Figure: Singular values (updated with correct filtering) ═══
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

ax = axes[0]
ax.hist(sv_raw_norm, bins=25, density=True, alpha=0.6, color='steelblue',
        label=r'Raw $K_{\mathrm{obs}}$', edgecolor='navy')
ax.hist(sv_row_norm, bins=25, density=True, alpha=0.6, color='coral',
        label=r'Row-filtered $K$', edgecolor='darkred')
lam = np.linspace(0, 2.2, 300)
qc = np.sqrt(np.maximum(4 - lam**2, 0)) / np.pi
ax.plot(lam, qc, 'k-', lw=2, label='Quarter-circle law')
ax.set_xlabel('Normalized singular value')
ax.set_ylabel('Probability density')
ax.set_title('(a) Singular value distribution')
ax.legend(fontsize=8)
ax.set_xlim(0, 2.5)

ax = axes[1]
ax.plot(np.arange(1, len(sv_raw_norm)+1), sv_raw_norm, 'b.-', markersize=2, lw=0.8, label='Raw')
ax.plot(np.arange(1, len(sv_row_norm)+1), sv_row_norm, 'r.-', markersize=2, lw=0.8, label='Row-filtered')
ax.set_xlabel('Singular value index')
ax.set_ylabel('Normalized singular value')
ax.set_title('(b) Sorted singular values')
ax.legend(fontsize=8)
ax.set_yscale('log')

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig_singular_values.pdf", bbox_inches='tight', dpi=300)
plt.savefig(f"{FIG_DIR}/fig_singular_values.png", bbox_inches='tight', dpi=300)
plt.close()
print("  fig_singular_values regenerated")

print("\nDone. All figures and data corrected.")
