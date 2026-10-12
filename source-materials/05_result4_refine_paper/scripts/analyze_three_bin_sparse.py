#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Three-Bin Sparse-Tap Sweep at bin = {5, 10, 25}
================================================
Operator's exact request (Step 254):

  * Output binning sizes to test in parallel: 5, 10, 25 pixels.
  * Output channel count must SWEEP starting BELOW the input channel count
    (n_in = 36) and going up. That is, n_out = 4, 9, 16, 25, 36, 49, 64,
    81, 100, etc.
  * Compare against the case where the optical/digital random projection
    is REMOVED — direct linear classification on the raw input features
    [E(x), E(y)] and pair-level raw features without any T.
  * Sparse independent-speckle taps with spacing ≥ speckle FWHM (~10.5 px),
    so each tap reads one independent grain.

Per (anchor_bin, n_out):
  - Optical concat / digital_bilinear / Complex-B benchmark on Tasks A/B/C/D
  - Pseudo-B random-T at matched n_out (10 seeds, A/B/C/D mean ± std)
  - Diagnostics: PR effective rank of all three feature families,
    repeat correlation of Complex-B at sampled taps

Plus raw-input direct linear baselines (NO T at all) reported as flat
horizontal references.

Output:
  source_data/processed/three_bin_sparse/three_bin_sparse_results.json
  figures/three_bin_sparse/*.pdf
"""

import os, json, time, math
import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 10, "figure.dpi": 150})

# ── Configuration ──────────────────────────────────────────────────
N_TOKENS = 8
ROI_H, ROI_W = 500, 500
PHASE_LABELS = ["0", "pi2", "pi", "3pi2"]
N_REPS = 5
SPECKLE_FWHM_PX = 10.5

# === Operator's exact request ===
ANCHOR_BINS = [5, 10, 25]
# Cover "from BELOW input dim n_in = 36 and going up"
N_OUT_GRID = [4, 9, 16, 25, 36, 49, 64, 100, 144]

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
                       "three_bin_sparse")
FIG_DIR = os.path.join(SECTION_DIR, "figures", "three_bin_sparse")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


# ── Token inputs (deterministic) ───────────────────────────────────
def make_token_input(tid):
    rng = np.random.default_rng(10000 + tid * 7919)
    return np.exp(1j * rng.uniform(0, 2 * np.pi, (6, 6))).flatten()

TOKEN_INPUTS = [make_token_input(t) for t in range(N_TOKENS)]
INPUT_DIM = 36


# ── Loading ────────────────────────────────────────────────────────
def bin_roi(roi, bin_size):
    h, w = roi.shape
    nh, nw = h // bin_size, w // bin_size
    cropped = roi[:nh * bin_size, :nw * bin_size]
    return cropped.reshape(nh, bin_size, nw, bin_size).mean(axis=(1, 3))


def load_all_data_2d(bin_size):
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


# ── Sparse-tap selection ───────────────────────────────────────────
def sparse_tap_indices(nh, nw, n_out, min_step_bins):
    side = int(math.ceil(math.sqrt(n_out)))
    step_h = nh / (side + 1)
    step_w = nw / (side + 1)
    actual_step_h = max(step_h, min_step_bins)
    actual_step_w = max(step_w, min_step_bins)
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
    # Deduplicate (for very small ROIs at large bin)
    seen = set()
    unique_flat, unique_coords = [], []
    for f, c in zip(flat, coords):
        if f not in seen:
            seen.add(f); unique_flat.append(f); unique_coords.append(c)
    return np.array(unique_flat[:n_out], dtype=int), unique_coords[:n_out]


# ── Feature builders ──────────────────────────────────────────────
def build_optical_at_taps(sd, B, tap_idx):
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


# ── Classification ─────────────────────────────────────────────────
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


def eff_rank(X):
    Xc = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)
    s2 = s ** 2
    if s2.sum() == 0: return 0.0
    return float(s2.sum() ** 2 / (s2 ** 2).sum())


# ── Pseudo-B random-T (digital electrical surrogate) ──────────────
def pseudo_B(n_out, n_seeds=10):
    rows = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(42000 + seed)
        T = (rng.standard_normal((n_out, INPUT_DIM)) +
             1j * rng.standard_normal((n_out, INPUT_DIM))) / np.sqrt(2 * INPUT_DIM)
        z = {t: T @ TOKEN_INPUTS[t] for t in range(N_TOKENS)}
        XB, pid, sc, cp, info = [], [], [], [], []
        for x in range(N_TOKENS):
            for y in range(N_TOKENS):
                cx = SEMANTIC_TOKENS[x]["category"]; cy = SEMANTIC_TOKENS[y]["category"]
                Bxy = np.conj(z[x]) * z[y]
                Br = np.concatenate([np.real(Bxy), np.imag(Bxy)])
                for _ in range(N_REPS):
                    XB.append(Br + rng.normal(0, 1e-8, Br.shape))
                    pid.append(x * N_TOKENS + y); sc.append(int(cx == cy))
                    cp.append(cx * 4 + cy); info.append((x, y))
        XB = np.array(XB)
        pid, sc, cp = np.array(pid), np.array(sc), np.array(cp)
        rows.append(bench(XB, pid, sc, cp, info, "pseudo_B"))
    agg = {"n_out": n_out, "n_seeds": n_seeds}
    for tk in "ABCD":
        vals = [r[tk] for r in rows]
        agg[f"{tk}_mean"] = round(float(np.mean(vals)), 4)
        agg[f"{tk}_std"] = round(float(np.std(vals)), 4)
    return agg


# ── Raw-input direct-readout baselines (NO T at all) ──────────────
def raw_input_direct():
    """
    The operator-mandated control: REMOVE the optical/digital projection
    entirely. Directly classify on raw input features.
    """
    rng = np.random.default_rng(12345)
    Xs = {n: [] for n in ["concat_raw_E_xy",     # [E_x, E_y]
                          "diff_raw",            # E_x - E_y
                          "absdiff_raw",         # |E_x - E_y|
                          "prod_raw",            # E_x ⊙ E_y
                          "conjprod_raw",        # conj(E_x) ⊙ E_y
                          "single_E_x"]}         # single-token E_x only
    pid, sc, cp, info = [], [], [], []
    for x in range(N_TOKENS):
        for y in range(N_TOKENS):
            cx = SEMANTIC_TOKENS[x]["category"]; cy = SEMANTIC_TOKENS[y]["category"]
            Ex, Ey = TOKEN_INPUTS[x], TOKEN_INPUTS[y]
            for _ in range(N_REPS):
                ns = 1e-8
                Xs["concat_raw_E_xy"].append(np.concatenate([np.real(Ex), np.imag(Ex),
                    np.real(Ey), np.imag(Ey)]) + rng.normal(0, ns, 144))
                d = Ex - Ey
                Xs["diff_raw"].append(np.concatenate([np.real(d), np.imag(d)]) +
                                      rng.normal(0, ns, 72))
                Xs["absdiff_raw"].append(np.abs(d) + rng.normal(0, ns, 36))
                p = Ex * Ey
                Xs["prod_raw"].append(np.concatenate([np.real(p), np.imag(p)]) +
                                      rng.normal(0, ns, 72))
                cpv = np.conj(Ex) * Ey
                Xs["conjprod_raw"].append(np.concatenate([np.real(cpv), np.imag(cpv)]) +
                                          rng.normal(0, ns, 72))
                Xs["single_E_x"].append(np.concatenate([np.real(Ex), np.imag(Ex)]) +
                                        rng.normal(0, ns, 72))
                pid.append(x * N_TOKENS + y); sc.append(int(cx == cy))
                cp.append(cx * 4 + cy); info.append((x, y))
    pid, sc, cp = np.array(pid), np.array(sc), np.array(cp)
    return [bench(np.array(v), pid, sc, cp, info, n) for n, v in Xs.items()]


# ── Per-anchor sparse-tap sweep ────────────────────────────────────
def sweep_anchor(bin_size, n_out_grid):
    print(f"\n{'='*70}\n  ANCHOR BIN = {bin_size}\n{'='*70}")
    sd, bd, pdat = load_all_data_2d(bin_size)
    B = compute_B_2d(bd, pdat)
    nh, nw = sd[0].shape[1], sd[0].shape[2]
    full_n_out = nh * nw
    min_step_bins = max(1, math.ceil(SPECKLE_FWHM_PX / bin_size))
    print(f"  Bin grid: {nh}x{nw} ({full_n_out} channels). "
          f"Min spacing: {min_step_bins} bins ({min_step_bins*bin_size} px).")

    sweep = []
    for n_out in n_out_grid:
        if n_out > full_n_out:
            continue
        tap_idx, coords = sparse_tap_indices(nh, nw, n_out, min_step_bins)
        if len(tap_idx) < n_out:
            continue
        feats, pid, sc, cp, info = build_optical_at_taps(sd, B, tap_idx)

        b_concat = bench(feats["concat"], pid, sc, cp, info, "concat")
        b_dig = bench(feats["digital_bilinear"], pid, sc, cp, info, "digital_bilinear")
        b_cB = bench(feats["complex_B"], pid, sc, cp, info, "complex_B")
        er_concat = eff_rank(feats["concat"])
        er_dig = eff_rank(feats["digital_bilinear"])
        er_cB = eff_rank(feats["complex_B"])
        pb = pseudo_B(n_out, n_seeds=10)

        entry = {
            "n_out": n_out,
            "tap_coords_first8": coords[:8],
            "min_step_bins": min_step_bins,
            "min_step_px": min_step_bins * bin_size,
            "concat": b_concat,
            "digital_bilinear": b_dig,
            "complex_B": b_cB,
            "eff_rank_concat": er_concat,
            "eff_rank_digital_bilinear": er_dig,
            "eff_rank_complex_B": er_cB,
            "pseudo_B": pb,
        }
        sweep.append(entry)
        print(f"  n_out={n_out:4d}  cB:A={b_cB['A']:.3f} B={b_cB['B']:.3f} "
              f"C={b_cB['C']:.3f} D={b_cB['D']:.3f}  | "
              f"pseudoB:B={pb['B_mean']:.3f}±{pb['B_std']:.3f} "
              f"D={pb['D_mean']:.3f}±{pb['D_std']:.3f}  | "
              f"er(cB)={er_cB:.2f}")

    return {"bin_size": bin_size, "n_out_grid": n_out_grid,
            "min_step_bins": min_step_bins,
            "min_step_px": min_step_bins * bin_size,
            "sweep": sweep}


# ── Figures ────────────────────────────────────────────────────────
COLOR_BIN = {5: "#16a085", 10: "#e74c3c", 25: "#9b59b6"}
MARKER_BIN = {5: "o", 10: "D", 25: "s"}


def fig_three_bin_grid(anchors, raw, out_path):
    """4×3 grid: rows = Tasks A/B/C/D, cols = bin=5/10/25.
    Each panel shows optical Complex-B vs pseudo-B vs raw-input direct
    references (conjprod_raw and concat_raw_E_xy)."""
    fig, axes = plt.subplots(4, 3, figsize=(18, 18))
    tasks = [("A", "Task A: 64-class Pair ID"),
             ("B", "Task B: Same Category"),
             ("C", "Task C: Category Pair"),
             ("D", "Task D: Cross-Split")]
    raw_lookup = {r["name"]: r for r in raw}

    for ti, (tk, ttl) in enumerate(tasks):
        for bi, anchor in enumerate(anchors):
            ax = axes[ti, bi]
            sweep = anchor["sweep"]
            n_outs = [s["n_out"] for s in sweep]
            cB = [s["complex_B"][tk] for s in sweep]
            pB_m = [s["pseudo_B"][f"{tk}_mean"] for s in sweep]
            pB_s = [s["pseudo_B"][f"{tk}_std"] for s in sweep]

            ax.plot(n_outs, cB, "D-", color="#e74c3c", lw=2.5, ms=8,
                    label=f"Optical Complex-B")
            ax.errorbar(n_outs, pB_m, yerr=pB_s, fmt="^--", color="#9b59b6",
                        lw=1.5, ms=6, capsize=3, alpha=0.75,
                        label=f"Pseudo-B (rand T, ±1σ)")
            # Raw-input references — DIRECT linear classification (no T)
            ax.axhline(raw_lookup["conjprod_raw"][tk], color="#27ae60",
                       ls=":", lw=2,
                       label=f"raw conjprod (no T) = {raw_lookup['conjprod_raw'][tk]:.3f}")
            ax.axhline(raw_lookup["concat_raw_E_xy"][tk], color="#7f8c8d",
                       ls=":", lw=2,
                       label=f"raw concat[E_x,E_y] = {raw_lookup['concat_raw_E_xy'][tk]:.3f}")
            ax.axhline(0.5, color="gray", ls=":", lw=0.5)
            ax.axvline(36, color="purple", ls="--", lw=1.5, alpha=0.5,
                       label="n_in=36")
            ax.set_xscale("log")
            ax.set_xlabel("n_out (sparse independent taps)")
            ax.set_ylabel("Balanced accuracy")
            ax.set_ylim(-0.05, 1.10)
            ax.set_xticks(n_outs)
            ax.set_xticklabels([str(n) for n in n_outs], fontsize=8)
            ax.set_title(f"bin={anchor['bin_size']}px  |  {ttl}",
                         fontweight="bold", fontsize=10)
            if ti == 0 and bi == 0:
                ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("Three-Bin Sparse-Tap Sweep: Optical vs Pseudo-B vs Raw Input "
                 "(no T, direct linear readout)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  saved {out_path}")


def fig_three_bin_summary(anchors, raw, out_path):
    """Compact 2-panel summary on Tasks B and D."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    raw_lookup = {r["name"]: r for r in raw}
    for ti, tk, ttl in [(0, "B", "Task B: Same Category"),
                        (1, "D", "Task D: Cross-Split")]:
        ax = axes[ti]
        for anchor in anchors:
            bs = anchor["bin_size"]
            sweep = anchor["sweep"]
            n_outs = [s["n_out"] for s in sweep]
            cB = [s["complex_B"][tk] for s in sweep]
            ax.plot(n_outs, cB,
                    f"{MARKER_BIN[bs]}-", color=COLOR_BIN[bs],
                    lw=2.5, ms=9,
                    label=f"Optical Complex-B (bin={bs}px)")
        # pseudo-B reference (use bin=10 anchor's pseudo since pseudo only depends on n_out)
        sweep10 = next(a for a in anchors if a["bin_size"] == 10)["sweep"]
        n_outs = [s["n_out"] for s in sweep10]
        pB_m = [s["pseudo_B"][f"{tk}_mean"] for s in sweep10]
        pB_s = [s["pseudo_B"][f"{tk}_std"] for s in sweep10]
        ax.errorbar(n_outs, pB_m, yerr=pB_s, fmt="^--", color="#9b59b6",
                    lw=1.5, ms=6, capsize=3, alpha=0.7,
                    label="Pseudo-B (random T, n_out matched)")
        ax.axhline(raw_lookup["conjprod_raw"][tk], color="#27ae60",
                   ls=":", lw=2.5,
                   label=f"raw conj·prod (no T) = {raw_lookup['conjprod_raw'][tk]:.3f}")
        ax.axhline(raw_lookup["concat_raw_E_xy"][tk], color="#7f8c8d",
                   ls=":", lw=2.5,
                   label=f"raw concat[E_x,E_y] (no T) = {raw_lookup['concat_raw_E_xy'][tk]:.3f}")
        ax.axhline(0.5, color="gray", ls=":", lw=0.5)
        ax.axvline(36, color="purple", ls="--", lw=2, alpha=0.5,
                   label="Input dim n_in=36")
        ax.set_xscale("log")
        ax.set_xlabel("n_out (sparse independent taps)", fontsize=11)
        ax.set_ylabel("Balanced accuracy", fontsize=11)
        ax.set_ylim(0.3, 1.10)
        ax.set_title(ttl, fontweight="bold", fontsize=12)
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("Three-Bin Sparse-Tap Advantage Boundary",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  saved {out_path}")


def fig_three_bin_eff_rank(anchors, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for bi, anchor in enumerate(anchors):
        ax = axes[bi]
        sweep = anchor["sweep"]
        n_outs = [s["n_out"] for s in sweep]
        for fn, label, color in [
            ("eff_rank_concat", "concat", "#3498db"),
            ("eff_rank_digital_bilinear", "digital bilin.", "#e67e22"),
            ("eff_rank_complex_B", "Complex-B", "#e74c3c"),
        ]:
            vals = [s[fn] for s in sweep]
            ax.plot(n_outs, vals, "o-", color=color, lw=2, ms=7, label=label)
        n_arr = np.array(n_outs)
        ax.plot(n_arr, n_arr, "k--", lw=1, alpha=0.4, label="y = n_out")
        ax.axvline(36, color="purple", ls="--", lw=1, alpha=0.5,
                   label="n_in=36")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("n_out (sparse taps)")
        ax.set_ylabel("PR-effective rank")
        ax.set_title(f"bin = {anchor['bin_size']}px  "
                     f"(min step {anchor['min_step_px']} px)",
                     fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which="both")
    fig.suptitle("Effective Rank of Optical Features Across Three Bins",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  saved {out_path}")


def fig_raw_baselines(raw, out_path):
    """Figure showing the operator-mandated control: direct raw-input linear classification."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    tasks = [("A", "Task A: Pair ID"),
             ("B", "Task B: Same Cat."),
             ("C", "Task C: Cat. Pair"),
             ("D", "Task D: Cross-Split")]
    names = ["concat_raw_E_xy", "diff_raw", "absdiff_raw",
             "prod_raw", "conjprod_raw", "single_E_x"]
    display = ["concat\n[E_x,E_y]", "diff\nE_x-E_y", "|diff|\n|E_x-E_y|",
               "prod\nE_x⊙E_y", "conj·prod\nconj(E_x)⊙E_y", "single\nE_x"]
    colors = ["#7f8c8d", "#bdc3c7", "#95a5a6", "#e67e22", "#27ae60", "#34495e"]
    raw_lookup = {r["name"]: r for r in raw}
    for ti, (tk, ttl) in enumerate(tasks):
        ax = axes[ti]
        vals = [raw_lookup[n][tk] for n in names]
        bars = ax.bar(range(len(vals)), vals, color=colors, edgecolor="black", linewidth=0.6)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(display, fontsize=7)
        ax.set_ylabel("Balanced accuracy")
        ax.set_ylim(0, 1.10)
        ax.axhline(0.5, color="gray", ls=":", lw=0.6)
        ax.set_title(ttl, fontweight="bold")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02,
                    f"{v:.3f}", ha="center", fontsize=7, fontweight="bold")
    fig.suptitle("Raw-Input Direct Linear Classification (NO optical T, NO digital random T)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  saved {out_path}")


# ── Main ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    t0 = time.time()
    print("Computing raw-input direct-linear baselines (NO T) …")
    raw = raw_input_direct()
    for r in raw:
        print(f"  {r['name']:20s} dim={r['dim']:4d}  "
              f"A={r['A']:.4f} B={r['B']:.4f} C={r['C']:.4f} D={r['D']:.4f}")

    anchors = []
    for b in ANCHOR_BINS:
        anchors.append(sweep_anchor(b, N_OUT_GRID))

    summary = {"raw_input_direct": raw, "anchors": anchors,
               "n_out_grid": N_OUT_GRID, "anchor_bins": ANCHOR_BINS,
               "speckle_fwhm_px": SPECKLE_FWHM_PX, "input_dim": INPUT_DIM}
    out_json = os.path.join(OUT_DIR, "three_bin_sparse_results.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {out_json}")

    print("\nGenerating figures …")
    fig_three_bin_grid(anchors, raw, os.path.join(FIG_DIR, "fig_three_bin_grid.pdf"))
    fig_three_bin_summary(anchors, raw, os.path.join(FIG_DIR, "fig_three_bin_summary.pdf"))
    fig_three_bin_eff_rank(anchors, os.path.join(FIG_DIR, "fig_three_bin_eff_rank.pdf"))
    fig_raw_baselines(raw, os.path.join(FIG_DIR, "fig_three_bin_raw_baselines.pdf"))

    print(f"\nDone in {time.time()-t0:.1f}s")
