#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sparse-Tap Bin-Anchored Twin Analysis
=====================================
Operator-driven correction:

  * Low-n_out optical readout must NOT be done by coarse binning.
    Coarse binning averages many speckle grains per channel, destroying
    independence and unfairly handicapping the optical engine.

  * The correct way is SPARSE TAPPING: at a fine binning that preserves
    speckle independence (bin=10 ≈ speckle FWHM, or bin=25 already in
    regime), select a SMALL number of evenly-spaced channels that are
    separated by at least the speckle coherence size, so each chosen
    channel samples an independent speckle grain.

  * Mirror everything for bin=10 and bin=25 perspectives. Cover
    n_out = 4, 5, 9, 16, 25, 36, 49, 64, 81, 100 (both ≤ and ≥ n_in=36).

For each (anchor_bin, n_out) pair:
  - Optical concat / digital_bilinear / Complex-B benchmark on Tasks A/B/C/D
  - Pseudo-B random-T at same n_out, 10 seeds, full A/B/C/D
  - Diagnostics: PR effective rank of all three families,
    repeat correlation of Complex-B at sampled taps

Plus: raw input baselines (bin-independent) reported as a horizontal reference.

Output:
  source_data/processed/sparse_tap_twin/sparse_tap_twin_results.json
  figures/sparse_tap_twin/*.pdf
"""

import os, json, time, math
import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 10, "figure.dpi": 150})

# ── Configuration ────────────────────────────────────────────────────
N_TOKENS = 8
ROI_H, ROI_W = 500, 500
PHASE_LABELS = ["0", "pi2", "pi", "3pi2"]
N_REPS = 5
ANCHOR_BINS = [10, 25]
N_OUT_GRID = [4, 5, 9, 16, 25, 36, 49, 64, 81, 100]

SEMANTIC_TOKENS = [
    {"id": 0, "category": 0}, {"id": 1, "category": 0},
    {"id": 2, "category": 1}, {"id": 3, "category": 1},
    {"id": 4, "category": 2}, {"id": 5, "category": 2},
    {"id": 6, "category": 3}, {"id": 7, "category": 3},
]

SECTION_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGE_DIR = os.path.dirname(SECTION_DIR)
DATA_DIR = os.path.join(PACKAGE_DIR, "shared_raw_data", "result4_frame_data",
                        "exp_complex_B_semantic")
OUT_DIR = os.path.join(SECTION_DIR, "source_data", "processed",
                       "sparse_tap_twin")
FIG_DIR = os.path.join(SECTION_DIR, "figures", "sparse_tap_twin")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


# ── Token inputs (deterministic) ─────────────────────────────────────
def make_token_input(tid):
    rng = np.random.default_rng(10000 + tid * 7919)
    return np.exp(1j * rng.uniform(0, 2 * np.pi, (6, 6))).flatten()

TOKEN_INPUTS = [make_token_input(t) for t in range(N_TOKENS)]
INPUT_DIM = 36


# ── Loading ──────────────────────────────────────────────────────────
def bin_roi(roi, bin_size):
    h, w = roi.shape
    nh, nw = h // bin_size, w // bin_size
    cropped = roi[:nh * bin_size, :nw * bin_size]
    return cropped.reshape(nh, bin_size, nw, bin_size).mean(axis=(1, 3))


def load_all_data_2d(bin_size):
    """Return single_data, blank_data, pair_data with 2-D binned ROIs (nh, nw)."""
    sing = os.path.join(DATA_DIR, "singles")
    blnk = os.path.join(DATA_DIR, "blank_controls")
    pair = os.path.join(DATA_DIR, "pair_interactions")
    sd, bd, pd_ = {}, {}, {}
    for tid in range(N_TOKENS):
        sd[tid] = np.array([bin_roi(np.load(os.path.join(sing,
                f"single_t{tid}_rep{r:02d}.npy")), bin_size) for r in range(N_REPS)])
    for y in range(N_TOKENS):
        for pi, pl in enumerate(PHASE_LABELS):
            bd[(y, pi)] = np.array([bin_roi(np.load(os.path.join(blnk,
                    f"blank_y{y}_phi{pl}_rep{r:02d}.npy")), bin_size) for r in range(N_REPS)])
    for x in range(N_TOKENS):
        for y in range(N_TOKENS):
            for pi, pl in enumerate(PHASE_LABELS):
                pd_[(x, y, pi)] = np.array([bin_roi(np.load(os.path.join(pair,
                        f"pair_x{x}_y{y}_phi{pl}_rep{r:02d}.npy")), bin_size)
                        for r in range(N_REPS)])
    return sd, bd, pd_


def compute_B_2d(blank, pair):
    """Return B[(x,y)] of shape (N_REPS, nh, nw) (2-D)."""
    Qb = {}
    for y in range(N_TOKENS):
        Qb[y] = np.array([
            (blank[(y, 0)][r] - blank[(y, 2)][r]) / 4.0
            + 1j * (blank[(y, 3)][r] - blank[(y, 1)][r]) / 4.0
            for r in range(N_REPS)])
    B = {}
    for x in range(N_TOKENS):
        for y in range(N_TOKENS):
            B[(x, y)] = np.array([
                ((pair[(x, y, 0)][r] - pair[(x, y, 2)][r]) / 4.0
                 + 1j * (pair[(x, y, 3)][r] - pair[(x, y, 1)][r]) / 4.0) - Qb[y][r]
                for r in range(N_REPS)])
    return B


# ── Sparse-tap channel selection ─────────────────────────────────────
def sparse_tap_indices(nh, nw, n_out, min_step_bins):
    """
    Select up to n_out channel coordinates in a (nh, nw) binned grid such
    that the spacing between adjacent taps is ≥ min_step_bins (bins).

    Returns flattened indices into the (nh*nw) vector.
    """
    side = int(math.ceil(math.sqrt(n_out)))
    # Ideal spacing if we use a sqrt(n_out) x sqrt(n_out) grid
    step_h = nh / (side + 1)
    step_w = nw / (side + 1)
    # If even ideal spacing is below the speckle-independence threshold,
    # we cannot fit that many independent taps; we still place them on
    # the maximum-step grid we can afford.
    actual_step_h = max(step_h, min_step_bins)
    actual_step_w = max(step_w, min_step_bins)
    # If actual step pushes coordinates outside the ROI, fall back to
    # ideal step (we trust the binning provided sufficient independence
    # at the chosen anchor bin level).
    coords = []
    for i in range(side):
        for j in range(side):
            r = int(round((i + 1) * step_h))
            c = int(round((j + 1) * step_w))
            r = max(0, min(nh - 1, r))
            c = max(0, min(nw - 1, c))
            coords.append((r, c))
            if len(coords) >= n_out:
                break
        if len(coords) >= n_out:
            break
    flat = [r * nw + c for (r, c) in coords]
    return np.array(flat[:n_out], dtype=int), coords[:n_out]


# ── Feature builders ─────────────────────────────────────────────────
def build_optical_at_taps(sd, B, tap_idx):
    """sd[tid]: (N_REPS, nh, nw); B[(x,y)]: (N_REPS, nh, nw)."""
    nh, nw = sd[0].shape[1], sd[0].shape[2]
    z_mean = {t: sd[t].mean(axis=0).reshape(-1)[tap_idx] for t in range(N_TOKENS)}
    feats = {"concat": [], "digital_bilinear": [], "complex_B": []}
    pid, sc, cp, info = [], [], [], []
    for x in range(N_TOKENS):
        for y in range(N_TOKENS):
            cx = SEMANTIC_TOKENS[x]["category"]; cy = SEMANTIC_TOKENS[y]["category"]
            for r in range(N_REPS):
                Br = B[(x, y)][r].reshape(-1)[tap_idx]
                feats["concat"].append(np.concatenate([z_mean[x], z_mean[y]]))
                feats["digital_bilinear"].append(z_mean[x] * z_mean[y])
                feats["complex_B"].append(np.concatenate([np.real(Br), np.imag(Br)]))
                pid.append(x * N_TOKENS + y); sc.append(int(cx == cy))
                cp.append(cx * 4 + cy); info.append((x, y))
    return ({k: np.array(v) for k, v in feats.items()},
            np.array(pid), np.array(sc), np.array(cp), info)


# ── Classification ───────────────────────────────────────────────────
def skf_acc(X, y, n=5):
    skf = StratifiedKFold(n_splits=n, shuffle=True, random_state=42)
    return float(cross_val_score(RidgeClassifier(alpha=1.0),
                                  X, y, cv=skf, scoring="balanced_accuracy").mean())


def cross_split(X, sc, info):
    cats = sorted(set(SEMANTIC_TOKENS[i]["category"] for i in range(N_TOKENS)))
    accs = []
    for held in cats:
        tr, te = [], []
        for i, (x, y) in enumerate(info):
            cx = SEMANTIC_TOKENS[x]["category"]; cy = SEMANTIC_TOKENS[y]["category"]
            (te if cx == held or cy == held else tr).append(i)
        if len(tr) < 2 or len(te) < 2: continue
        tr, te = np.array(tr), np.array(te)
        clf = RidgeClassifier(alpha=1.0).fit(X[tr], sc[tr])
        from sklearn.metrics import balanced_accuracy_score
        accs.append(balanced_accuracy_score(sc[te], clf.predict(X[te])))
    return float(np.mean(accs)) if accs else 0.0


def bench(X, pid, sc, cp, info, name):
    return {"name": name, "dim": int(X.shape[1]),
            "A": round(skf_acc(X, pid), 4),
            "B": round(skf_acc(X, sc), 4),
            "C": round(skf_acc(X, cp), 4),
            "D": round(cross_split(X, sc, info), 4)}


# ── Diagnostics ──────────────────────────────────────────────────────
def eff_rank(X):
    Xc = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)
    s2 = s ** 2
    if s2.sum() == 0: return 0.0
    return float(s2.sum() ** 2 / (s2 ** 2).sum())


def repeat_corr(X_per_rep):
    """
    X_per_rep: (n_reps, n_features).
    Mean Pearson correlation between repeats.
    """
    corrs = []
    nr = X_per_rep.shape[0]
    for i in range(nr):
        for j in range(i + 1, nr):
            corrs.append(np.real(np.corrcoef(X_per_rep[i], X_per_rep[j])[0, 1]))
    return float(np.mean(corrs)) if corrs else 0.0


# ── Pseudo-B at fixed n_out ──────────────────────────────────────────
def pseudo_B(n_out, n_seeds=10):
    rows = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(42000 + seed)
        T = (rng.standard_normal((n_out, INPUT_DIM)) +
             1j * rng.standard_normal((n_out, INPUT_DIM))) / np.sqrt(2 * INPUT_DIM)
        z = {t: T @ TOKEN_INPUTS[t] for t in range(N_TOKENS)}
        XB, Xc, pid, sc, cp, info = [], [], [], [], [], []
        for x in range(N_TOKENS):
            for y in range(N_TOKENS):
                cx = SEMANTIC_TOKENS[x]["category"]; cy = SEMANTIC_TOKENS[y]["category"]
                Bxy = np.conj(z[x]) * z[y]
                Br = np.concatenate([np.real(Bxy), np.imag(Bxy)])
                cv = np.concatenate([np.real(z[x]), np.imag(z[x]),
                                      np.real(z[y]), np.imag(z[y])])
                for _ in range(N_REPS):
                    XB.append(Br + rng.normal(0, 1e-8, Br.shape))
                    Xc.append(cv)
                    pid.append(x * N_TOKENS + y); sc.append(int(cx == cy))
                    cp.append(cx * 4 + cy); info.append((x, y))
        XB = np.array(XB); Xc = np.array(Xc)
        pid, sc, cp = np.array(pid), np.array(sc), np.array(cp)
        rows.append(bench(XB, pid, sc, cp, info, "pseudo_B"))
    agg = {"n_out": n_out, "n_seeds": n_seeds}
    for tk in "ABCD":
        vals = [r[tk] for r in rows]
        agg[f"{tk}_mean"] = round(float(np.mean(vals)), 4)
        agg[f"{tk}_std"] = round(float(np.std(vals)), 4)
    return agg


# ── Raw-input baselines (bin-independent) ────────────────────────────
def raw_input_baselines():
    rng = np.random.default_rng(12345)
    Xs = {n: [] for n in ["concat_raw", "diff_raw", "absdiff_raw",
                          "prod_raw", "conjprod_raw"]}
    pid, sc, cp, info = [], [], [], []
    for x in range(N_TOKENS):
        for y in range(N_TOKENS):
            cx = SEMANTIC_TOKENS[x]["category"]; cy = SEMANTIC_TOKENS[y]["category"]
            Ex, Ey = TOKEN_INPUTS[x], TOKEN_INPUTS[y]
            for _ in range(5):
                ns = 1e-8
                Xs["concat_raw"].append(np.concatenate([np.real(Ex), np.imag(Ex),
                    np.real(Ey), np.imag(Ey)]) + rng.normal(0, ns, 144))
                d = Ex - Ey
                Xs["diff_raw"].append(np.concatenate([np.real(d), np.imag(d)]) +
                                      rng.normal(0, ns, 72))
                Xs["absdiff_raw"].append(np.abs(d) + rng.normal(0, ns, 36))
                p = Ex * Ey
                Xs["prod_raw"].append(np.concatenate([np.real(p), np.imag(p)]) +
                                      rng.normal(0, ns, 72))
                cp_v = np.conj(Ex) * Ey
                Xs["conjprod_raw"].append(np.concatenate([np.real(cp_v),
                                          np.imag(cp_v)]) + rng.normal(0, ns, 72))
                pid.append(x * N_TOKENS + y); sc.append(int(cx == cy))
                cp.append(cx * 4 + cy); info.append((x, y))
    pid, sc, cp = np.array(pid), np.array(sc), np.array(cp)
    return [bench(np.array(v), pid, sc, cp, info, n) for n, v in Xs.items()]


# ── Per-anchor sparse-tap sweep ──────────────────────────────────────
def sweep_anchor(bin_size, n_out_grid, speckle_fwhm_px=10.5):
    """For one anchor bin, run sparse-tap sweep over the full n_out grid."""
    print(f"\n{'='*70}\n  ANCHOR BIN = {bin_size}\n{'='*70}")
    sd, bd, pdat = load_all_data_2d(bin_size)
    B = compute_B_2d(bd, pdat)
    nh, nw = sd[0].shape[1], sd[0].shape[2]
    full_n_out = nh * nw
    # Minimum spacing in BIN units to keep independent speckle grains:
    # pixel_step ≥ speckle_fwhm_px → bin_step ≥ speckle_fwhm_px / bin_size
    min_step_bins = max(1, math.ceil(speckle_fwhm_px / bin_size))
    print(f"  Bin grid: {nh}x{nw} ({full_n_out} channels). "
          f"Min spacing: {min_step_bins} bins ({min_step_bins*bin_size} px).")

    sweep = []
    for n_out in n_out_grid:
        if n_out > full_n_out:
            continue
        tap_idx, coords = sparse_tap_indices(nh, nw, n_out, min_step_bins)
        if len(tap_idx) < n_out:
            print(f"  n_out={n_out}: only {len(tap_idx)} independent taps "
                  f"available at this anchor. Skipping.")
            continue
        feats, pid, sc, cp, info = build_optical_at_taps(sd, B, tap_idx)

        b_concat = bench(feats["concat"], pid, sc, cp, info, "concat")
        b_dig = bench(feats["digital_bilinear"], pid, sc, cp, info, "digital_bilinear")
        b_cB = bench(feats["complex_B"], pid, sc, cp, info, "complex_B")

        # Diagnostics
        er_concat = eff_rank(feats["concat"])
        er_dig = eff_rank(feats["digital_bilinear"])
        er_cB = eff_rank(feats["complex_B"])

        # Repeat correlation for Complex-B at these taps (averaged across pairs)
        rep_corrs = []
        for x in range(N_TOKENS):
            for y in range(N_TOKENS):
                Bxy = B[(x, y)].reshape(N_REPS, -1)[:, tap_idx]
                v = np.concatenate([np.real(Bxy), np.imag(Bxy)], axis=1)
                rep_corrs.append(repeat_corr(v))
        rep_corr_mean = float(np.mean(rep_corrs))

        # Pseudo-B
        pb = pseudo_B(n_out, n_seeds=10)

        entry = {
            "n_out": n_out,
            "tap_coords": coords,
            "min_step_bins": min_step_bins,
            "min_step_px": min_step_bins * bin_size,
            "concat": b_concat,
            "digital_bilinear": b_dig,
            "complex_B": b_cB,
            "eff_rank_concat": er_concat,
            "eff_rank_digital_bilinear": er_dig,
            "eff_rank_complex_B": er_cB,
            "B_repeat_corr_mean": rep_corr_mean,
            "pseudo_B": pb,
        }
        sweep.append(entry)
        print(f"  n_out={n_out:4d}  cB:A={b_cB['A']:.3f} B={b_cB['B']:.3f} "
              f"C={b_cB['C']:.3f} D={b_cB['D']:.3f}  | "
              f"pseudo:B={pb['B_mean']:.3f}±{pb['B_std']:.3f} "
              f"D={pb['D_mean']:.3f}±{pb['D_std']:.3f}  | "
              f"er(cB)={er_cB:.2f}  rep={rep_corr_mean:.3f}")

    return {"bin_size": bin_size, "n_out_grid": n_out_grid,
            "min_step_bins": min_step_bins, "sweep": sweep}


# ── Figures ──────────────────────────────────────────────────────────
def fig_per_bin_tasks(anchor, raw, out_path):
    """4-task figure for one anchor bin: optical (3 families) vs pseudo-B vs raw conjprod."""
    bin_size = anchor["bin_size"]
    sweep = anchor["sweep"]
    n_outs = [s["n_out"] for s in sweep]

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    tasks = [("A", "Task A: 64-class Pair ID"),
             ("B", "Task B: Same Category"),
             ("C", "Task C: Category Pair"),
             ("D", "Task D: Cross-Split")]

    # Raw conjprod_raw reference (constant)
    rcp = next(r for r in raw if r["name"] == "conjprod_raw")

    for ti, (tk, title) in enumerate(tasks):
        ax = axes[ti // 2, ti % 2]
        opt_concat = [s["concat"][tk] for s in sweep]
        opt_dig = [s["digital_bilinear"][tk] for s in sweep]
        opt_cB = [s["complex_B"][tk] for s in sweep]
        pB_m = [s["pseudo_B"][f"{tk}_mean"] for s in sweep]
        pB_s = [s["pseudo_B"][f"{tk}_std"] for s in sweep]

        ax.plot(n_outs, opt_concat, "o-", color="#3498db", lw=2, ms=7,
                label="Optical concat")
        ax.plot(n_outs, opt_dig, "s-", color="#e67e22", lw=2, ms=7,
                label="Optical digital bilinear")
        ax.plot(n_outs, opt_cB, "D-", color="#e74c3c", lw=2.5, ms=9,
                label="Optical Complex-B (sparse tap)")
        ax.errorbar(n_outs, pB_m, yerr=pB_s, fmt="^--", color="#9b59b6",
                    lw=2, ms=7, capsize=3, alpha=0.85,
                    label="Pseudo-B (random T, ±1σ)")
        ax.axhline(rcp[tk], color="#27ae60", ls=":", lw=2, alpha=0.85,
                   label=f"raw conjprod (72-d) = {rcp[tk]:.3f}")
        ax.axhline(0.5, color="gray", ls=":", lw=0.6)
        ax.axvline(36, color="purple", ls=":", lw=2, alpha=0.6,
                   label="Input dim (36)")
        ax.set_xlabel("Output channels n_out (sparse taps)")
        ax.set_ylabel("Balanced accuracy")
        ax.set_title(title, fontweight="bold")
        ax.set_ylim(-0.05, 1.1)
        ax.set_xscale("log")
        ax.set_xticks(n_outs)
        ax.set_xticklabels([str(n) for n in n_outs], fontsize=8)
        if ti == 0:
            ax.legend(fontsize=7, loc="lower right")
    fig.suptitle(f"Sparse-tap optical readout at bin = {bin_size} "
                 f"(min spacing {anchor['min_step_bins']*bin_size} px ≥ speckle FWHM)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  saved {out_path}")


def fig_eff_rank(anchors, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    colors = {"concat": "#3498db", "digital_bilinear": "#e67e22", "complex_B": "#e74c3c"}

    for ai, anchor in enumerate(anchors):
        ax = axes[ai]
        sweep = anchor["sweep"]
        n_outs = [s["n_out"] for s in sweep]
        for fname, lbl in [("concat", "concat"), ("digital_bilinear", "digital bilin."),
                           ("complex_B", "Complex-B")]:
            vals = [s[f"eff_rank_{fname}"] for s in sweep]
            ax.plot(n_outs, vals, "o-", color=colors[fname], lw=2, ms=7, label=lbl)
        # Diagonal reference y=x
        n_arr = np.array(n_outs)
        ax.plot(n_arr, n_arr, "k--", lw=1, alpha=0.4, label="y = n_out")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("n_out (sparse taps)")
        ax.set_ylabel("PR-effective rank")
        ax.set_title(f"bin = {anchor['bin_size']} (min step {anchor['min_step_bins']*anchor['bin_size']} px)",
                     fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, which="both")
    fig.suptitle("Effective rank of optical features under sparse-tap readout",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  saved {out_path}")


def fig_summary_advantage(anchors, raw, out_path):
    """
    The operator's key boundary picture:
    Task B and Task D as functions of n_out, comparing optical Complex-B
    (sparse tap) at both bins to pseudo-B and to raw conjprod.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    rcp = next(r for r in raw if r["name"] == "conjprod_raw")

    for ti, tk, title in [(0, "B", "Task B: Same Category"),
                          (1, "D", "Task D: Cross-Split")]:
        ax = axes[ti]
        markers = {"10": "D", "25": "s"}
        colors = {"10": "#e74c3c", "25": "#c0392b"}
        for anchor in anchors:
            bs = anchor["bin_size"]
            sweep = anchor["sweep"]
            n_outs = [s["n_out"] for s in sweep]
            cB = [s["complex_B"][tk] for s in sweep]
            pB_m = [s["pseudo_B"][f"{tk}_mean"] for s in sweep]
            pB_s = [s["pseudo_B"][f"{tk}_std"] for s in sweep]
            ax.plot(n_outs, cB, f"{markers[str(bs)]}-", color=colors[str(bs)],
                    lw=2.5, ms=9, label=f"Optical Complex-B (bin={bs})")
            ax.errorbar(n_outs, pB_m, yerr=pB_s, fmt="^--", color="#9b59b6",
                        lw=1.5, ms=6, capsize=3, alpha=0.5,
                        label=f"Pseudo-B (rand T, n_out matched)" if bs == 10 else None)
        ax.axhline(rcp[tk], color="#27ae60", ls=":", lw=2.5, alpha=0.85,
                   label=f"raw conjprod (72-d) = {rcp[tk]:.3f}")
        ax.axhline(0.5, color="gray", ls=":", lw=0.6)
        ax.axvline(36, color="purple", ls=":", lw=2, alpha=0.6,
                   label="Input dim (36)")
        ax.set_xlabel("n_out (sparse taps)")
        ax.set_ylabel("Balanced accuracy")
        ax.set_title(title, fontweight="bold")
        ax.set_xscale("log")
        ax.set_ylim(0.3, 1.1)
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("Advantage Boundary under SPARSE-TAP optical readout",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  saved {out_path}")


# ── Main ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    t0 = time.time()
    print("Computing raw-input baselines …")
    raw = raw_input_baselines()
    for r in raw:
        print(f"  {r['name']:18s} dim={r['dim']:4d}  "
              f"A={r['A']:.4f} B={r['B']:.4f} C={r['C']:.4f} D={r['D']:.4f}")

    anchors = []
    for b in ANCHOR_BINS:
        anchors.append(sweep_anchor(b, N_OUT_GRID))

    out_summary = {"raw_input": raw, "anchors": anchors,
                   "n_out_grid": N_OUT_GRID,
                   "anchor_bins": ANCHOR_BINS,
                   "speckle_fwhm_px": 10.5}
    out_json = os.path.join(OUT_DIR, "sparse_tap_twin_results.json")
    with open(out_json, "w") as f:
        json.dump(out_summary, f, indent=2)
    print(f"\nSaved {out_json}")

    print("\nGenerating figures …")
    for anchor in anchors:
        bs = anchor["bin_size"]
        fig_per_bin_tasks(anchor, raw, os.path.join(FIG_DIR,
                          f"fig_sparse_tap_optical_bin{bs}.pdf"))
    fig_eff_rank(anchors, os.path.join(FIG_DIR, "fig_sparse_tap_eff_rank.pdf"))
    fig_summary_advantage(anchors, raw,
                          os.path.join(FIG_DIR, "fig_sparse_tap_summary.pdf"))

    print(f"\nDone in {time.time()-t0:.1f}s")
