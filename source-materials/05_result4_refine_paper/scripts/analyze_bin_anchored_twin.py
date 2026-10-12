#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bin-Anchored Twin Analysis (bin=10 and bin=25)
==============================================
Mirrored control suite at each anchor bin separately.

For each anchor bin in {10, 25}, produce:
  (1) Optical benchmark table: concat / digital_bilinear / Complex-B on Tasks A/B/C/D
  (2) Pseudo-B random-T control at matched n_out (bin=10 → 2500, bin=25 → 400)
      with mean ± std over 10 seeds on Tasks A/B/C/D
  (3) Raw-input baselines (independent of bin) reported alongside
  (4) Equal-dimension fairness: at each anchor n_out, compare against random-T
      pseudo-B at the same dim, plus raw conj_prod_raw at 72 dims as reference
  (5) Bin10-vs-bin25 comparison: optical Complex-B differences at the two
      anchor bins, plus repeat-stability and effective-rank diagnostics
  (6) Diagnostics: per-class repeat correlation, total power, effective rank
"""

import os, json, time
import numpy as np
from pathlib import Path
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 10, 'figure.dpi': 150})

# ── config ─────────────────────────────────────────────────────────
N_TOKENS = 8
ROI_H, ROI_W = 500, 500
PHASE_FILE_LABELS = ["0", "pi2", "pi", "3pi2"]
N_REPS = 5
ANCHOR_BINS = [10, 25]   # the two requested anchor bins

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
                       "bin_anchored_twin")
FIG_DIR = os.path.join(SECTION_DIR, "figures", "bin_anchored_twin")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


def make_token_input(tid):
    seed = 10000 + tid * 7919
    rng = np.random.default_rng(seed)
    phases = rng.uniform(0, 2 * np.pi, (6, 6))
    return np.exp(1j * phases).flatten()

TOKEN_INPUTS = [make_token_input(tid) for tid in range(N_TOKENS)]
INPUT_DIM = 36


# ── data loading ───────────────────────────────────────────────────
def bin_roi(roi, bin_size):
    h, w = roi.shape
    nh, nw = h // bin_size, w // bin_size
    cropped = roi[:nh * bin_size, :nw * bin_size]
    return cropped.reshape(nh, bin_size, nw, bin_size).mean(axis=(1, 3)).flatten()


def load_all_data_binned(bin_size):
    singles = os.path.join(DATA_DIR, "singles")
    blanks = os.path.join(DATA_DIR, "blank_controls")
    pairs = os.path.join(DATA_DIR, "pair_interactions")
    sd, bd, pd_ = {}, {}, {}
    for tid in range(N_TOKENS):
        sd[tid] = np.array([bin_roi(np.load(os.path.join(singles,
            f"single_t{tid}_rep{r:02d}.npy")), bin_size) for r in range(N_REPS)])
    for y in range(N_TOKENS):
        for pi, pl in enumerate(PHASE_FILE_LABELS):
            bd[(y, pi)] = np.array([bin_roi(np.load(os.path.join(blanks,
                f"blank_y{y}_phi{pl}_rep{r:02d}.npy")), bin_size) for r in range(N_REPS)])
    for x in range(N_TOKENS):
        for y in range(N_TOKENS):
            for pi, pl in enumerate(PHASE_FILE_LABELS):
                pd_[(x, y, pi)] = np.array([bin_roi(np.load(os.path.join(pairs,
                    f"pair_x{x}_y{y}_phi{pl}_rep{r:02d}.npy")), bin_size)
                    for r in range(N_REPS)])
    return sd, bd, pd_


def compute_B(blank, pair):
    Qb = {}
    for y in range(N_TOKENS):
        Qb[y] = np.array([(blank[(y, 0)][r] - blank[(y, 2)][r]) / 4.0 +
                          1j * (blank[(y, 3)][r] - blank[(y, 1)][r]) / 4.0
                          for r in range(N_REPS)])
    B = {}
    for x in range(N_TOKENS):
        for y in range(N_TOKENS):
            B[(x, y)] = np.array([
                ((pair[(x, y, 0)][r] - pair[(x, y, 2)][r]) / 4.0 +
                 1j * (pair[(x, y, 3)][r] - pair[(x, y, 1)][r]) / 4.0) - Qb[y][r]
                for r in range(N_REPS)])
    return B


def build_optical_features(sd, B):
    z = {t: sd[t].mean(axis=0) for t in range(N_TOKENS)}
    feats = {"concat": [], "digital_bilinear": [], "complex_B": []}
    pid, sc, cp, info = [], [], [], []
    for x in range(N_TOKENS):
        for y in range(N_TOKENS):
            cx = SEMANTIC_TOKENS[x]["category"]; cy = SEMANTIC_TOKENS[y]["category"]
            for r in range(N_REPS):
                feats["concat"].append(np.concatenate([z[x], z[y]]))
                feats["digital_bilinear"].append(z[x] * z[y])
                Br = B[(x, y)][r]
                feats["complex_B"].append(np.concatenate([np.real(Br), np.imag(Br)]))
                pid.append(x * N_TOKENS + y); sc.append(int(cx == cy))
                cp.append(cx * 4 + cy); info.append((x, y))
    return ({k: np.array(v) for k, v in feats.items()},
            np.array(pid), np.array(sc), np.array(cp), info)


# ── classification ─────────────────────────────────────────────────
def skf_acc(X, y, n=5):
    skf = StratifiedKFold(n_splits=n, shuffle=True, random_state=42)
    return float(cross_val_score(RidgeClassifier(alpha=1.0), X, y, cv=skf,
                                  scoring='balanced_accuracy').mean())


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


# ── diagnostics ────────────────────────────────────────────────────
def repeat_correlation(B):
    """Mean across-repeat correlation for B(x,y) channel vectors."""
    corrs = []
    for k, reps in B.items():
        vec = np.concatenate([np.real(reps), np.imag(reps)], axis=1)
        for i in range(N_REPS):
            for j in range(i + 1, N_REPS):
                a, b = vec[i], vec[j]
                corrs.append(np.real(np.corrcoef(a, b)[0, 1]))
    return float(np.mean(corrs)), float(np.std(corrs))


def effective_rank(X):
    """Participation-ratio effective rank of feature matrix X."""
    Xc = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)
    s2 = s ** 2
    return float(s2.sum() ** 2 / (s2 ** 2).sum() + 1e-12)


# ── raw input baselines (bin-independent) ──────────────────────────
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
    out = []
    for n, v in Xs.items():
        X = np.array(v)
        out.append(bench(X, pid, sc, cp, info, n))
    return out


# ── pseudo-B random T at fixed n_out ───────────────────────────────
def pseudo_B_at(n_out, n_seeds=10):
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
                for _ in range(5):
                    XB.append(Br + rng.normal(0, 1e-8, Br.shape))
                    Xc.append(cv)
                    pid.append(x * N_TOKENS + y); sc.append(int(cx == cy))
                    cp.append(cx * 4 + cy); info.append((x, y))
        XB, Xc = np.array(XB), np.array(Xc)
        pid, sc, cp = np.array(pid), np.array(sc), np.array(cp)
        rows.append({"pseudo_B": bench(XB, pid, sc, cp, info, "pseudo_B"),
                     "pseudo_concat": bench(Xc, pid, sc, cp, info, "pseudo_concat")})
    agg = {"n_out": n_out, "n_seeds": n_seeds}
    for fn in ["pseudo_B", "pseudo_concat"]:
        for tk in "ABCD":
            vals = [r[fn][tk] for r in rows]
            agg[f"{fn}_{tk}_mean"] = round(float(np.mean(vals)), 4)
            agg[f"{fn}_{tk}_std"] = round(float(np.std(vals)), 4)
    return agg


# ── per-anchor full block ──────────────────────────────────────────
def run_anchor(bin_size):
    print(f"\n{'='*70}\n  ANCHOR BIN = {bin_size} (n_out = {(ROI_H//bin_size)**2})\n{'='*70}")
    sd, bd, pdat = load_all_data_binned(bin_size)
    B = compute_B(bd, pdat)
    feats, pid, sc, cp, info = build_optical_features(sd, B)

    optical_bench = {}
    for k, X in feats.items():
        r = bench(X, pid, sc, cp, info, f"optical_{k}")
        optical_bench[k] = r
        print(f"  optical_{k:18s} dim={r['dim']:5d}  A={r['A']:.4f} B={r['B']:.4f} C={r['C']:.4f} D={r['D']:.4f}")

    # diagnostics
    rep_mean, rep_std = repeat_correlation(B)
    eff_rank_B = effective_rank(feats["complex_B"])
    eff_rank_concat = effective_rank(feats["concat"])
    eff_rank_dig = effective_rank(feats["digital_bilinear"])
    self_pair_centroid = {}
    for tid in range(N_TOKENS):
        Bxx = B[(tid, tid)].mean(axis=0)
        self_pair_centroid[tid] = {"re": float(Bxx.real.mean()),
                                    "im": float(Bxx.imag.mean()),
                                    "abs_centroid": float(np.abs(Bxx.mean()))}
    print(f"  diagnostics: repeat_corr={rep_mean:.4f}±{rep_std:.4f}  "
          f"eff_rank(B)={eff_rank_B:.2f}  eff_rank(concat)={eff_rank_concat:.2f}  "
          f"eff_rank(dig)={eff_rank_dig:.2f}")

    # pseudo-B at the matched n_out
    n_out = (ROI_H // bin_size) ** 2
    print(f"  computing pseudo-B at matched n_out={n_out} ...")
    pb = pseudo_B_at(n_out, n_seeds=10)
    print(f"  pseudo_B  A={pb['pseudo_B_A_mean']:.4f}±{pb['pseudo_B_A_std']:.4f}  "
          f"B={pb['pseudo_B_B_mean']:.4f}±{pb['pseudo_B_B_std']:.4f}  "
          f"C={pb['pseudo_B_C_mean']:.4f}±{pb['pseudo_B_C_std']:.4f}  "
          f"D={pb['pseudo_B_D_mean']:.4f}±{pb['pseudo_B_D_std']:.4f}")

    return {
        "bin_size": bin_size,
        "n_out": n_out,
        "optical": optical_bench,
        "pseudo_B_matched": pb,
        "diagnostics": {
            "repeat_corr_mean": rep_mean,
            "repeat_corr_std": rep_std,
            "eff_rank_complex_B": eff_rank_B,
            "eff_rank_concat": eff_rank_concat,
            "eff_rank_digital_bilinear": eff_rank_dig,
            "self_pair_centroid": self_pair_centroid,
        },
    }


# ── figures ────────────────────────────────────────────────────────
def fig_anchor_bench(anchor_results, raw, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    tasks = [("A", "Task A: 64-class Pair ID"), ("B", "Task B: Same Category"),
             ("C", "Task C: Category Pair"), ("D", "Task D: Cross-Split")]
    bins = [r["bin_size"] for r in anchor_results]
    bin_labels = [f"bin={b}\n(n_out={r['n_out']})" for b, r in zip(bins, anchor_results)]
    width = 0.18

    for ti, (tk, title) in enumerate(tasks):
        ax = axes[ti // 2, ti % 2]
        x = np.arange(len(bins))
        # 4 optical families
        concat = [r["optical"]["concat"][tk] for r in anchor_results]
        dig = [r["optical"]["digital_bilinear"][tk] for r in anchor_results]
        cB = [r["optical"]["complex_B"][tk] for r in anchor_results]
        pB = [r["pseudo_B_matched"][f"pseudo_B_{tk}_mean"] for r in anchor_results]
        pB_err = [r["pseudo_B_matched"][f"pseudo_B_{tk}_std"] for r in anchor_results]

        ax.bar(x - 1.5*width, concat, width, label="opt concat", color="#3498db")
        ax.bar(x - 0.5*width, dig, width, label="opt digital bilin.", color="#e67e22")
        ax.bar(x + 0.5*width, cB, width, label="opt Complex-B", color="#e74c3c")
        ax.bar(x + 1.5*width, pB, width, yerr=pB_err, label="pseudo-B (rand T)",
               color="#9b59b6", capsize=3)

        # raw conjprod horizontal reference
        cp_ref = next(r["B"] if tk == "B" else r[tk] for r in raw if r["name"] == "conjprod_raw")
        ax.axhline(cp_ref, color="#27ae60", ls="--", lw=1.5, alpha=0.8,
                   label="raw conjprod (72-d)")
        ax.axhline(0.5, color="gray", ls=":", lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels(bin_labels, fontsize=9)
        ax.set_ylim(0, 1.15); ax.set_ylabel("Balanced Accuracy")
        ax.set_title(title, fontweight="bold")
        if ti == 0:
            ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("Bin-Anchored Twin Benchmark (bin=10 vs bin=25): "
                 "Optical vs Pseudo-B vs Raw Reference",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  saved {out_path}")


def fig_diagnostics(anchor_results, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    bins = [r["bin_size"] for r in anchor_results]
    xl = [f"bin={b}\nn_out={r['n_out']}" for b, r in zip(bins, anchor_results)]

    # repeat correlations
    ax = axes[0]
    means = [r["diagnostics"]["repeat_corr_mean"] for r in anchor_results]
    stds = [r["diagnostics"]["repeat_corr_std"] for r in anchor_results]
    ax.bar(range(len(bins)), means, yerr=stds, color="#16a085", capsize=5)
    ax.set_xticks(range(len(bins))); ax.set_xticklabels(xl)
    ax.set_ylim(0, 1.05); ax.set_ylabel("mean across-repeat correlation")
    ax.set_title("(a) B(x,y) Repeat Stability", fontweight="bold")
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + 0.02, f"{m:.3f}±{s:.3f}", ha="center", fontsize=9)

    # effective rank
    ax = axes[1]
    er_B = [r["diagnostics"]["eff_rank_complex_B"] for r in anchor_results]
    er_c = [r["diagnostics"]["eff_rank_concat"] for r in anchor_results]
    er_d = [r["diagnostics"]["eff_rank_digital_bilinear"] for r in anchor_results]
    x = np.arange(len(bins)); w = 0.25
    ax.bar(x - w, er_c, w, label="concat", color="#3498db")
    ax.bar(x, er_d, w, label="digital bilin.", color="#e67e22")
    ax.bar(x + w, er_B, w, label="Complex-B", color="#e74c3c")
    ax.set_xticks(x); ax.set_xticklabels(xl)
    ax.set_ylabel("participation-ratio eff. rank")
    ax.set_title("(b) Optical Feature Effective Rank", fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_yscale("log")

    # self-pair centroid magnitudes (all 8 self-pairs)
    ax = axes[2]
    for ri, r in enumerate(anchor_results):
        cents = [r["diagnostics"]["self_pair_centroid"][t]["abs_centroid"]
                 for t in range(N_TOKENS)]
        ax.plot(range(N_TOKENS), cents, "o-",
                label=f"bin={r['bin_size']}", linewidth=2)
    ax.set_xticks(range(N_TOKENS)); ax.set_xticklabels([f"t{i}" for i in range(N_TOKENS)])
    ax.set_ylabel("|centroid B(x,x)|")
    ax.set_title("(c) Self-pair Centroid Magnitude", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle("Bin-Anchored Diagnostics: Repeat Stability, Effective Rank, "
                 "Self-pair Geometry", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  saved {out_path}")


def fig_bin_delta(anchor_results, out_path):
    """Bin10 vs Bin25 detailed delta on Tasks A/B/C/D for optical Complex-B."""
    fig, ax = plt.subplots(figsize=(10, 6))
    tasks = ["A", "B", "C", "D"]
    titles = ["A: Pair ID", "B: Same Cat.", "C: Cat. Pair", "D: Cross-Split"]
    n = len(anchor_results)
    width = 0.35
    x = np.arange(len(tasks))
    colors = ["#e74c3c", "#9b59b6"]

    for i, r in enumerate(anchor_results):
        vals_cB = [r["optical"]["complex_B"][t] for t in tasks]
        ax.bar(x + (i - (n-1)/2) * width, vals_cB, width,
               label=f"Optical Complex-B (bin={r['bin_size']})",
               color=colors[i])
        for xi, v in zip(x, vals_cB):
            ax.text(xi + (i - (n-1)/2) * width, v + 0.02, f"{v:.3f}",
                    ha="center", fontsize=8, fontweight="bold")

    ax.axhline(0.5, color="gray", ls=":", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(titles)
    ax.set_ylabel("Balanced Accuracy"); ax.set_ylim(0, 1.15)
    ax.set_title("Optical Complex-B: bin=10 vs bin=25 on the Four Semantic Tasks",
                 fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  saved {out_path}")


# ── main ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    t0 = time.time()
    print("Computing raw-input baselines (bin-independent) …")
    raw = raw_input_baselines()
    for r in raw:
        print(f"  {r['name']:18s} dim={r['dim']:4d}  A={r['A']:.4f} B={r['B']:.4f} "
              f"C={r['C']:.4f} D={r['D']:.4f}")

    anchors = []
    for b in ANCHOR_BINS:
        anchors.append(run_anchor(b))

    summary = {"raw_input": raw, "anchors": anchors}
    out_json = os.path.join(OUT_DIR, "bin_anchored_twin_results.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {out_json}")

    print("\nGenerating figures …")
    fig_anchor_bench(anchors, raw, os.path.join(FIG_DIR, "fig_bin_anchor_benchmark.pdf"))
    fig_diagnostics(anchors, os.path.join(FIG_DIR, "fig_bin_anchor_diagnostics.pdf"))
    fig_bin_delta(anchors, os.path.join(FIG_DIR, "fig_bin_complex_B_delta.pdf"))

    print(f"\nDone in {time.time()-t0:.1f}s")
