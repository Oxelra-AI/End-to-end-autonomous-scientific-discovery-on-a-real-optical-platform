#!/usr/bin/env python3
"""
Visualize power-distribution permutohedra for selected comparable and incomparable
spectrum pairs, projected onto the probability simplex.

Produces report figures for:
1. PRL n=3 example (rho_a through rho_e) — reproducing PRL Fig. 2(d)
2. 4-port comparable pair — showing strict nesting
3. 4-port incomparable pair (Hand-B) — showing partial overlap
4. Combined 4-port panel with both pairs

Output: PDF + PNG figures in this report section's figures folder.
"""

import numpy as np
from itertools import permutations
from scipy.spatial import ConvexHull
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import os

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures')
os.makedirs(FIGDIR, exist_ok=True)

# ============================================================
# Geometry utilities
# ============================================================

def permutohedron_vertices(lam):
    """All distinct permutations of lambda (sorted desc)."""
    lam_sorted = tuple(sorted(lam, reverse=True))
    verts = set()
    for p in permutations(lam_sorted):
        verts.add(p)
    return np.array(list(verts))

def barycentric_to_2d(pts):
    """
    Convert n=3 barycentric coordinates (on simplex d1+d2+d3=1)
    to 2D Cartesian for plotting. Uses equilateral triangle projection.
    """
    # Vertices of equilateral triangle: (0,0), (1,0), (0.5, sqrt(3)/2)
    # Point (d1,d2,d3) maps to d1*(0,0) + d2*(1,0) + d3*(0.5, sqrt(3)/2)
    x = pts[:, 1] + 0.5 * pts[:, 2]
    y = (np.sqrt(3) / 2) * pts[:, 2]
    return np.column_stack([x, y])

def project_4d_to_3d_simplex(pts):
    """
    Project 4D points on hyperplane sum=1 to 3D coordinates.
    Uses first 3 components (d1,d2,d3); d4 = 1-d1-d2-d3.
    Then further project to 2D using a chosen viewing angle.
    """
    return pts[:, :3]

def simplex_3d_to_2d(pts3d):
    """
    Map 3D simplex coordinates (d1,d2,d3) where d4=1-sum to a nice 2D projection.
    Use the standard ternary-like projection for 4-simplex slices.
    We project along the (1,1,1,1) direction onto the orthogonal hyperplane.
    """
    # Basis vectors for the hyperplane sum(x)=1 in R^4
    # e1 = (1,-1,0,0)/sqrt(2), e2 = (1,1,-2,0)/sqrt(6), e3 = (1,1,1,-3)/sqrt(12)
    n = pts3d.shape[0]
    pts4d = np.column_stack([pts3d, 1 - np.sum(pts3d, axis=1)])
    
    e1 = np.array([1, -1, 0, 0]) / np.sqrt(2)
    e2 = np.array([1, 1, -2, 0]) / np.sqrt(6)
    # For 2D we use e1 and e2
    x = pts4d @ e1
    y = pts4d @ e2
    return np.column_stack([x, y])

def ordered_hull_2d(pts_2d):
    """Return convex hull vertices in order for polygon plotting."""
    if len(pts_2d) < 3:
        return pts_2d
    try:
        hull = ConvexHull(pts_2d)
        return pts_2d[hull.vertices]
    except:
        return pts_2d

# ============================================================
# Figure 1: PRL n=3 example — rho_a through rho_e
# ============================================================

def fig_prl_n3():
    spectra = {
        r'$\rho_a$': np.array([1/3, 1/3, 1/3]),
        r'$\rho_b$': np.array([0.60, 0.30, 0.10]),
        r'$\rho_c$': np.array([0.80, 0.15, 0.05]),
        r'$\rho_d$': np.array([1.00, 0.00, 0.00]),
        r'$\rho_e$': np.array([0.55, 0.45, 0.00]),
    }
    
    colors = {
        r'$\rho_a$': '#2196F3',  # blue
        r'$\rho_b$': '#4CAF50',  # green
        r'$\rho_c$': '#FF9800',  # orange
        r'$\rho_d$': '#F44336',  # red
        r'$\rho_e$': '#9C27B0',  # purple
    }
    
    fig, ax = plt.subplots(1, 1, figsize=(5, 4.5))
    
    # Draw simplex boundary
    simplex_corners = np.array([[1,0,0],[0,1,0],[0,0,1]])
    sc_2d = barycentric_to_2d(simplex_corners)
    triangle = plt.Polygon(sc_2d, fill=False, edgecolor='gray', linewidth=0.5, linestyle='--')
    ax.add_patch(triangle)
    
    # Label simplex corners
    offsets = [(-0.06, -0.04), (0.02, -0.04), (0.02, 0.02)]
    labels_corner = [r'$d_1=1$', r'$d_2=1$', r'$d_3=1$']
    for i, (lbl, off) in enumerate(zip(labels_corner, offsets)):
        ax.annotate(lbl, sc_2d[i] + np.array(off), fontsize=8, color='gray')
    
    for name, lam in spectra.items():
        verts = permutohedron_vertices(lam)
        verts_2d = barycentric_to_2d(verts)
        
        if len(verts) == 1:
            ax.plot(verts_2d[0, 0], verts_2d[0, 1], 'o', color=colors[name],
                    markersize=6, label=name, zorder=5)
        elif len(verts) <= 2:
            ax.plot(verts_2d[:, 0], verts_2d[:, 1], '-o', color=colors[name],
                    linewidth=2, markersize=4, label=name, zorder=4)
        else:
            hull_pts = ordered_hull_2d(verts_2d)
            polygon = plt.Polygon(hull_pts, alpha=0.15, facecolor=colors[name],
                                  edgecolor=colors[name], linewidth=1.5, label=name, zorder=3)
            ax.add_patch(polygon)
            ax.plot(verts_2d[:, 0], verts_2d[:, 1], 'o', color=colors[name],
                    markersize=3, zorder=4)
    
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.1, 1.0)
    ax.set_aspect('equal')
    ax.legend(fontsize=9, loc='upper right', framealpha=0.9)
    ax.set_title(r'Power-distribution polytopes $\{{\bf d}\}$ (PRL $n=3$ example)', fontsize=11)
    ax.set_xlabel('Simplex coordinate', fontsize=10)
    ax.set_ylabel('Simplex coordinate', fontsize=10)
    ax.tick_params(labelsize=8)
    
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(FIGDIR, f'prl_n3_polytopes.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Saved: prl_n3_polytopes.pdf/png")

# ============================================================
# Figure 2 & 3: 4-port comparable and incomparable pairs
# ============================================================

def fig_4port_pair(lam1, lam2, name1, name2, title, filename, relation_text):
    """Visualize a 4-port pair projected to 2D."""
    v1 = permutohedron_vertices(lam1)
    v2 = permutohedron_vertices(lam2)
    
    v1_2d = simplex_3d_to_2d(project_4d_to_3d_simplex(v1))
    v2_2d = simplex_3d_to_2d(project_4d_to_3d_simplex(v2))
    
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 5))
    
    # Plot polytope 2 (outer/larger) first
    if len(v2_2d) >= 3:
        hull2 = ordered_hull_2d(v2_2d)
        poly2 = plt.Polygon(hull2, alpha=0.15, facecolor='#E91E63',
                            edgecolor='#E91E63', linewidth=1.8, label=name2, zorder=2)
        ax.add_patch(poly2)
    ax.plot(v2_2d[:, 0], v2_2d[:, 1], 'o', color='#E91E63', markersize=3.5, zorder=4)
    
    # Plot polytope 1
    if len(v1_2d) >= 3:
        hull1 = ordered_hull_2d(v1_2d)
        poly1 = plt.Polygon(hull1, alpha=0.20, facecolor='#2196F3',
                            edgecolor='#2196F3', linewidth=1.8, label=name1, zorder=3)
        ax.add_patch(poly1)
    ax.plot(v1_2d[:, 0], v1_2d[:, 1], 'o', color='#2196F3', markersize=3.5, zorder=5)
    
    # Auto-scale
    all_pts = np.vstack([v1_2d, v2_2d])
    margin = 0.05 * (all_pts.max() - all_pts.min())
    ax.set_xlim(all_pts[:, 0].min() - margin, all_pts[:, 0].max() + margin)
    ax.set_ylim(all_pts[:, 1].min() - margin, all_pts[:, 1].max() + margin)
    
    ax.set_aspect('equal')
    ax.legend(fontsize=9, loc='upper right', framealpha=0.9)
    ax.set_title(title, fontsize=11)
    
    # Add relation text
    ax.text(0.02, 0.02, relation_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', style='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
    
    ax.set_xlabel('Projected coordinate 1', fontsize=10)
    ax.set_ylabel('Projected coordinate 2', fontsize=10)
    ax.tick_params(labelsize=8)
    
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(FIGDIR, f'{filename}.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {filename}.pdf/png")

# ============================================================
# Figure 4: Combined panel — comparable + incomparable
# ============================================================

def fig_4port_combined():
    """Side-by-side panel: comparable (nested) vs incomparable (partial overlap)."""
    # Comparable
    lam_c1 = np.array([0.40, 0.30, 0.20, 0.10])
    lam_c2 = np.array([0.70, 0.20, 0.07, 0.03])
    # Incomparable (Hand-B)
    lam_i1 = np.array([0.55, 0.15, 0.15, 0.15])
    lam_i2 = np.array([0.35, 0.35, 0.25, 0.05])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    
    for ax, l1, l2, title_str, rel_str in [
        (ax1, lam_c1, lam_c2,
         r'Comparable: $\lambda_1 \prec \lambda_2$',
         r'$\{{\bf d}\}_1 \subset \{{\bf d}\}_2$ (nested)'),
        (ax2, lam_i1, lam_i2,
         r'Incomparable: $\lambda_1 \| \lambda_2$',
         r'$\{{\bf d}\}_1 \| \{{\bf d}\}_2$ (partial overlap)')
    ]:
        v1 = permutohedron_vertices(l1)
        v2 = permutohedron_vertices(l2)
        v1_2d = simplex_3d_to_2d(project_4d_to_3d_simplex(v1))
        v2_2d = simplex_3d_to_2d(project_4d_to_3d_simplex(v2))
        
        # Plot polytope 2 first (potentially larger)
        if len(v2_2d) >= 3:
            hull2 = ordered_hull_2d(v2_2d)
            poly2 = plt.Polygon(hull2, alpha=0.15, facecolor='#E91E63',
                                edgecolor='#E91E63', linewidth=1.8, zorder=2)
            ax.add_patch(poly2)
        ax.plot(v2_2d[:, 0], v2_2d[:, 1], 'o', color='#E91E63', markersize=3, zorder=4)
        
        if len(v1_2d) >= 3:
            hull1 = ordered_hull_2d(v1_2d)
            poly1 = plt.Polygon(hull1, alpha=0.20, facecolor='#2196F3',
                                edgecolor='#2196F3', linewidth=1.8, zorder=3)
            ax.add_patch(poly1)
        ax.plot(v1_2d[:, 0], v1_2d[:, 1], 'o', color='#2196F3', markersize=3, zorder=5)
        
        all_pts = np.vstack([v1_2d, v2_2d])
        margin = 0.06 * (all_pts.max() - all_pts.min())
        ax.set_xlim(all_pts[:, 0].min() - margin, all_pts[:, 0].max() + margin)
        ax.set_ylim(all_pts[:, 1].min() - margin, all_pts[:, 1].max() + margin)
        ax.set_aspect('equal')
        ax.set_title(title_str, fontsize=11)
        ax.text(0.02, 0.02, rel_str, transform=ax.transAxes, fontsize=9,
                verticalalignment='bottom', style='italic',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
        ax.set_xlabel('Projected coord. 1', fontsize=10)
        ax.set_ylabel('Projected coord. 2', fontsize=10)
        ax.tick_params(labelsize=8)
    
    # Common legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2196F3', alpha=0.3, edgecolor='#2196F3', linewidth=1.5,
              label=r'$\{{\bf d}\}_1$'),
        Patch(facecolor='#E91E63', alpha=0.3, edgecolor='#E91E63', linewidth=1.5,
              label=r'$\{{\bf d}\}_2$'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2, fontsize=10,
               bbox_to_anchor=(0.5, 1.02), framealpha=0.9)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(FIGDIR, f'4port_comparable_vs_incomparable.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Saved: 4port_comparable_vs_incomparable.pdf/png")

# ============================================================
# Figure 5: Partial-sum diagram showing crossings
# ============================================================

def fig_partial_sums():
    """Partial-sum diagrams for comparable and incomparable pairs."""
    lam_c1 = np.array([0.40, 0.30, 0.20, 0.10])
    lam_c2 = np.array([0.70, 0.20, 0.07, 0.03])
    lam_i1 = np.array([0.55, 0.15, 0.15, 0.15])
    lam_i2 = np.array([0.35, 0.35, 0.25, 0.05])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    for ax, l1, l2, title_str in [
        (ax1, lam_c1, lam_c2, r'Comparable: $\lambda_1 \prec \lambda_2$'),
        (ax2, lam_i1, lam_i2, r'Incomparable: $\lambda_1 \| \lambda_2$')
    ]:
        n = len(l1)
        ks = np.arange(1, n+1)
        ps1 = np.cumsum(np.sort(l1)[::-1])
        ps2 = np.cumsum(np.sort(l2)[::-1])
        
        ax.plot(ks, ps1, 'o-', color='#2196F3', linewidth=2, markersize=6,
                label=r'$\sum_{i=1}^k \lambda_i^{(1)\downarrow}$')
        ax.plot(ks, ps2, 's-', color='#E91E63', linewidth=2, markersize=6,
                label=r'$\sum_{i=1}^k \lambda_i^{(2)\downarrow}$')
        
        # Shade crossing regions
        for k in range(n-1):
            diff = ps1[k] - ps2[k]
            if abs(diff) > 1e-10:
                color = '#2196F3' if diff > 0 else '#E91E63'
                ax.annotate(f'{diff:+.3f}', (ks[k], (ps1[k]+ps2[k])/2),
                           fontsize=8, ha='left', va='center', color=color,
                           xytext=(5, 0), textcoords='offset points')
        
        ax.set_xlabel('$k$', fontsize=11)
        ax.set_ylabel(r'$\sum_{i=1}^k \lambda_i^\downarrow$', fontsize=11)
        ax.set_title(title_str, fontsize=11)
        ax.set_xticks(ks)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=9)
    
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(FIGDIR, f'partial_sum_crossings.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Saved: partial_sum_crossings.pdf/png")

# ============================================================
# Main
# ============================================================

def main():
    print("Generating majorization polytope figures...\n")
    
    # 1. PRL n=3
    fig_prl_n3()
    
    # 2. 4-port comparable
    lam_c1 = np.array([0.40, 0.30, 0.20, 0.10])
    lam_c2 = np.array([0.70, 0.20, 0.07, 0.03])
    fig_4port_pair(
        lam_c1, lam_c2,
        r'$\lambda_1=(0.40,0.30,0.20,0.10)$',
        r'$\lambda_2=(0.70,0.20,0.07,0.03)$',
        r'4-port comparable: $\{{\bf d}\}_1 \subset \{{\bf d}\}_2$',
        '4port_comparable_polytopes',
        r'$\lambda_1 \prec \lambda_2$ $\Rightarrow$ strict nesting'
    )
    
    # 3. 4-port incomparable (Hand-B)
    lam_i1 = np.array([0.55, 0.15, 0.15, 0.15])
    lam_i2 = np.array([0.35, 0.35, 0.25, 0.05])
    fig_4port_pair(
        lam_i1, lam_i2,
        r'$\lambda_1=(0.55,0.15,0.15,0.15)$',
        r'$\lambda_2=(0.35,0.35,0.25,0.05)$',
        r'4-port incomparable: $\{{\bf d}\}_1 \| \{{\bf d}\}_2$',
        '4port_incomparable_polytopes',
        r'$\lambda_1 \| \lambda_2$ $\Rightarrow$ partial overlap'
    )
    
    # 4. Combined panel
    fig_4port_combined()
    
    # 5. Partial-sum crossings
    fig_partial_sums()
    
    print("\nAll figures generated successfully.")
    
    # List output files
    for f in sorted(os.listdir(FIGDIR)):
        fpath = os.path.join(FIGDIR, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"  {f} ({size_kb:.1f} KB)")

if __name__ == '__main__':
    main()
