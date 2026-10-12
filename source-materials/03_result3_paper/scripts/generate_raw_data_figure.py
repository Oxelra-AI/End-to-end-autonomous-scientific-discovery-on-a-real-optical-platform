#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

section_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
base = os.path.join(section_dir, 'source_data', 'exp_transport_v3', 'exp_transport_v3')
figdir = os.path.join(section_dir, 'figures')
os.makedirs(figdir, exist_ok=True)

K_obs = np.load(os.path.join(base, 'K_obs.npy'))
ref_vec = np.load(os.path.join(base, 'ref_vec.npy'))

bh = bw = 125
ref_img = ref_vec.reshape(bh, bw)

# Show magnitude of first 4 TM columns as representative raw measured channels
cols = [0, 1, 2, 3]
mag_imgs = [np.abs(K_obs[:, c]).reshape(bh, bw) for c in cols]

fig, axes = plt.subplots(2, 3, figsize=(11, 7))

im0 = axes[0, 0].imshow(ref_img, cmap='inferno')
axes[0, 0].set_title('Reference baseline $r_m$')
axes[0, 0].set_xlabel('Output x-bin')
axes[0, 0].set_ylabel('Output y-bin')
plt.colorbar(im0, ax=axes[0,0], fraction=0.046, pad=0.04)

for idx, c in enumerate(cols[:2]):
    ax = axes[0, idx+1]
    im = ax.imshow(mag_imgs[idx], cmap='viridis')
    ax.set_title(f'$|K_{{obs}}[:,{c}]|$')
    ax.set_xlabel('Output x-bin')
    ax.set_ylabel('Output y-bin')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

for idx, c in enumerate(cols[2:]):
    ax = axes[1, idx]
    im = ax.imshow(mag_imgs[idx+2], cmap='viridis')
    ax.set_title(f'$|K_{{obs}}[:,{c}]|$')
    ax.set_xlabel('Output x-bin')
    ax.set_ylabel('Output y-bin')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# sixth panel: histogram of reference values
ax = axes[1, 2]
ax.hist(ref_vec, bins=50, color='#607D8B', edgecolor='white')
ax.axvline(np.percentile(ref_vec, 25), color='red', linestyle='--', label='25th percentile')
ax.set_title('Reference baseline distribution')
ax.set_xlabel('$r_m$')
ax.set_ylabel('Count')
ax.legend(fontsize=8)

plt.tight_layout()
for ext in ['pdf', 'png']:
    plt.savefig(os.path.join(figdir, f'v3_raw_data_examples.{ext}'), dpi=300, bbox_inches='tight')
print('Saved raw data figure')
