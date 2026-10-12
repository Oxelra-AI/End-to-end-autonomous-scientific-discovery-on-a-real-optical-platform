#!/usr/bin/env python3
"""Generate standalone data-derived panels for the main manuscript.

Covered panels:
  Fig. 2d-g
  Fig. 3d-f
  Fig. 4g-i

The script is intentionally panel-level. It does not reconstruct the complete
Word/PDF manuscript figure layouts.
"""

from __future__ import annotations

import json
import os
from itertools import permutations
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Polygon


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "main_text_data_panels" / "generated_panels"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.7,
})


def save_panel(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)


def bin_roi(roi: np.ndarray, bin_size: int = 25) -> np.ndarray:
    h, w = roi.shape
    nh, nw = h // bin_size, w // bin_size
    cropped = roi[:nh * bin_size, :nw * bin_size]
    return cropped.reshape(nh, bin_size, nw, bin_size).mean(axis=(1, 3)).reshape(-1)


def load_records(base: Path, subdir: str, name: str) -> list[dict]:
    return json.loads((base / subdir / name).read_text())


def mean_binned_by_key(
    base: Path,
    subdir: str,
    records: list[dict],
    key_fields: tuple[str, ...],
    filter_fn,
    bin_size: int = 25,
) -> dict[tuple, np.ndarray]:
    grouped: dict[tuple, list[np.ndarray]] = {}
    for rec in records:
        if not filter_fn(rec):
            continue
        key = tuple(rec[field] for field in key_fields)
        arr = np.load(base / subdir / rec["file"])
        grouped.setdefault(key, []).append(bin_roi(arr, bin_size=bin_size))
    return {k: np.mean(v, axis=0) for k, v in grouped.items()}


def stacked_binned_by_key(
    base: Path,
    subdir: str,
    records: list[dict],
    key_fields: tuple[str, ...],
    filter_fn,
    bin_size: int,
) -> dict[tuple, np.ndarray]:
    grouped: dict[tuple, list[np.ndarray]] = {}
    for rec in records:
        if not filter_fn(rec):
            continue
        key = tuple(rec[field] for field in key_fields)
        arr = np.load(base / subdir / rec["file"])
        grouped.setdefault(key, []).append(bin_roi(arr, bin_size=bin_size))
    return {k: np.stack(v, axis=0) for k, v in grouped.items()}


def permutohedron_vertices(lam: np.ndarray) -> np.ndarray:
    vals = tuple(sorted(lam, reverse=True))
    return np.array(sorted(set(permutations(vals))))


def simplex_4d_to_2d(pts4d: np.ndarray) -> np.ndarray:
    e1 = np.array([1.0, -1.0, 0.0, 0.0]) / np.sqrt(2)
    e2 = np.array([1.0, 1.0, -2.0, 0.0]) / np.sqrt(6)
    return np.column_stack([pts4d @ e1, pts4d @ e2])


def ordered_hull(points: np.ndarray) -> np.ndarray:
    from scipy.spatial import ConvexHull
    hull = ConvexHull(points)
    return points[hull.vertices]


def generate_fig2_panels() -> None:
    # Fig. 2d: focusing operator from the first Result2 measured TM.
    k = np.load(ROOT / "01_result2_paper/source_data/TM_canonical.npy")
    row_norms = np.linalg.norm(k, axis=1, keepdims=True)
    k_norm = np.divide(k, row_norms, out=np.zeros_like(k), where=row_norms > 0)
    arr = np.abs(k_norm @ k_norm.conj().T)
    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    im = ax.imshow(arr, cmap="inferno", interpolation="nearest", aspect="equal")
    ax.set_title("Focusing operator")
    ax.set_xlabel("Output mode")
    ax.set_ylabel("Output mode")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Amp. (a.u.)")
    save_panel(fig, "main_fig2d_focusing_operator")

    # Fig. 2e: initial 16 x 16 targeted focusing output.
    focus = np.load(ROOT / "01_result2_paper/source_data/focus_target192_modes.npy").reshape(16, 16)
    fig, ax = plt.subplots(figsize=(3.0, 3.0))
    im = ax.imshow(focus, cmap="inferno", interpolation="nearest", aspect="equal")
    ax.set_title("Targeted focusing output")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Intensity (a.u.)")
    save_panel(fig, "main_fig2e_initial_focusing_output")

    # Fig. 2f: mode-number scaling.
    scaling = json.loads((ROOT / "main_text_data_panels/source_data/main_fig2_scaling_summary.json").read_text())
    modes = np.array(scaling["controlled_input_modes"])
    mean_enh = np.array(scaling["mean_enhancement"])
    max_enh = np.array(scaling["max_enhancement"])
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax.plot(modes, mean_enh, "o-", lw=1.8, ms=5, label="Mean enhancement", color="#1f77b4")
    ax.plot(modes, max_enh, "s--", lw=1.8, ms=5, label="Max enhancement", color="#d62728")
    ax.set_xscale("log", base=2)
    ax.set_xticks(modes)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("Input modes")
    ax.set_ylabel("Enhancement")
    ax.set_title("Focusing performance")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    save_panel(fig, "main_fig2f_mode_number_scaling")

    # Fig. 2g: refined 80 x 80 output-pixel focusing map.
    refined = np.load(ROOT / "02_result2_refine_paper/source_data/focusing/raw_rois/focus_single_m240_po_crop80.npy")
    fig, ax = plt.subplots(figsize=(3.0, 3.0))
    im = ax.imshow(refined, cmap="inferno", interpolation="nearest", aspect="equal")
    ax.set_title("Refined focusing output")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Intensity (a.u.)")
    save_panel(fig, "main_fig2g_refined_focusing_output")


def generate_fig3_panels() -> None:
    # Fig. 3d: predicted achievable response sets for a comparable pair.
    lam_less = np.array([0.40, 0.30, 0.20, 0.10])
    lam_more = np.array([0.70, 0.20, 0.07, 0.03])
    v_less = simplex_4d_to_2d(permutohedron_vertices(lam_less))
    v_more = simplex_4d_to_2d(permutohedron_vertices(lam_more))
    fig, ax = plt.subplots(figsize=(3.1, 2.9))
    for points, color, label, alpha in [
        (v_more, "#e91e63", r"$\lambda_2$ more coherent", 0.18),
        (v_less, "#2196f3", r"$\lambda_1$ less coherent", 0.25),
    ]:
        poly = Polygon(ordered_hull(points), closed=True, facecolor=color,
                       edgecolor=color, alpha=alpha, linewidth=1.2, label=label)
        ax.add_patch(poly)
        ax.plot(points[:, 0], points[:, 1], "o", color=color, ms=2.5)
    ax.set_aspect("equal")
    ax.set_title("Sets of achievable responses")
    ax.set_xlabel("Projection coordinate 1")
    ax.set_ylabel("Projection coordinate 2")
    ax.legend(frameon=False, loc="lower right")
    save_panel(fig, "main_fig3d_achievable_response_sets")

    # Fig. 3e: measured TM-column magnitude.
    k_obs = np.load(ROOT / "03_result3_paper/source_data/exp_transport_v3/exp_transport_v3/K_obs.npy")
    column = np.abs(k_obs[:, 2]).reshape(125, 125)
    fig, ax = plt.subplots(figsize=(3.0, 3.0))
    im = ax.imshow(column, cmap="viridis", interpolation="nearest", aspect="equal")
    ax.set_title("Measured TM-column magnitude")
    ax.set_xlabel("Output x-bin")
    ax.set_ylabel("Output y-bin")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Magnitude (a.u.)")
    save_panel(fig, "main_fig3e_measured_tm_column")

    # Fig. 3f: transport intervals across measured systems.
    meta_path = ROOT / "03_result3_paper/source_data/exp_transport_v3/exp_transport_v3/metadata.json"
    meta = json.loads(meta_path.read_text())
    systems = ["full"] + [f"radial_{i}" for i in range(8)] + [f"resp_{i}" for i in range(8)]
    systems = [s for s in systems if s in meta["system_info"]]
    labels = ["Full"] + [f"R{i}" for i in range(8)] + [f"P{i}" for i in range(8)]
    labels = labels[:len(systems)]
    x = np.arange(len(systems))
    fig, ax = plt.subplots(figsize=(5.8, 2.9))
    for idx, spec, color, label in [
        (0, "comp_less", "#2196f3", r"$\lambda_1$ less coherent"),
        (1, "comp_more", "#e91e63", r"$\lambda_2$ more coherent"),
    ]:
        centers, lows, highs = [], [], []
        for sys_name in systems:
            lo = meta["all_intervals"][sys_name][spec]["lower"]
            hi = meta["all_intervals"][sys_name][spec]["upper"]
            centers.append((lo + hi) / 2.0)
            lows.append((hi - lo) / 2.0)
            highs.append((hi - lo) / 2.0)
        ax.errorbar(x + (idx - 0.5) * 0.12, centers, yerr=[lows, highs],
                    fmt="|", ms=10, capsize=3, color=color, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Transport value")
    ax.set_title("Transport intervals across measured systems")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="upper right")
    inset = ax.inset_axes([0.42, 0.50, 0.34, 0.34])
    for idx, spec, color, _label in [
        (0, "comp_less", "#2196f3", r"$\lambda_1$ less coherent"),
        (1, "comp_more", "#e91e63", r"$\lambda_2$ more coherent"),
    ]:
        centers, lows, highs = [], [], []
        for sys_name in systems[1:]:
            lo = meta["all_intervals"][sys_name][spec]["lower"]
            hi = meta["all_intervals"][sys_name][spec]["upper"]
            centers.append((lo + hi) / 2.0)
            lows.append((hi - lo) / 2.0)
            highs.append((hi - lo) / 2.0)
        xx = np.arange(len(systems) - 1)
        inset.errorbar(xx + (idx - 0.5) * 0.10, centers, yerr=[lows, highs],
                       fmt="|", ms=7, capsize=2, color=color)
    inset.set_xticks([0, 7, 8, 15])
    inset.set_xticklabels(["R0", "R7", "P0", "P7"], fontsize=6)
    inset.tick_params(axis="y", labelsize=6)
    inset.set_title("data zoom", fontsize=7)
    inset.grid(axis="y", alpha=0.2)
    save_panel(fig, "main_fig3f_transport_intervals")


def compute_complex_b_for_pairs(selected_pairs: list[tuple[int, int]]):
    base = ROOT / "shared_raw_data/result4_frame_data/exp_bilinear_001/exp_bilinear_001"
    pair_records = load_records(base, "pair_interactions", "pair_records.json")
    blank_records = load_records(base, "blank_controls", "blank_records.json")
    selected_y = sorted({y for _, y in selected_pairs})
    selected_pair_set = set(selected_pairs)

    pair_mean = mean_binned_by_key(
        base,
        "pair_interactions",
        pair_records,
        ("x_token_id", "y_token_id", "phi_idx"),
        lambda r: (r["x_token_id"], r["y_token_id"]) in selected_pair_set,
    )
    blank_mean = mean_binned_by_key(
        base,
        "blank_controls",
        blank_records,
        ("y_token_id", "phi_idx"),
        lambda r: r["y_token_id"] in selected_y,
    )

    q_blank = {}
    for y in selected_y:
        q_blank[y] = (blank_mean[(y, 0)] - blank_mean[(y, 2)]) / 4.0 + 1j * (
            blank_mean[(y, 3)] - blank_mean[(y, 1)]) / 4.0

    b_pair = {}
    for x, y in selected_pairs:
        q_pair = (pair_mean[(x, y, 0)] - pair_mean[(x, y, 2)]) / 4.0 + 1j * (
            pair_mean[(x, y, 3)] - pair_mean[(x, y, 1)]) / 4.0
        b_pair[(x, y)] = q_pair - q_blank[y]
    return pair_mean, b_pair


def compute_refined_complex_b_for_pairs(
    selected_pairs: list[tuple[int, int]],
    bin_size: int = 10,
):
    base = ROOT / "shared_raw_data/result4_frame_data/exp_bilinear_001/exp_bilinear_001"
    pair_records = load_records(base, "pair_interactions", "pair_records.json")
    blank_records = load_records(base, "blank_controls", "blank_records.json")
    selected_y = sorted({y for _, y in selected_pairs})
    selected_pair_set = set(selected_pairs)

    pair_data = stacked_binned_by_key(
        base,
        "pair_interactions",
        pair_records,
        ("x_token_id", "y_token_id", "phi_idx"),
        lambda r: (r["x_token_id"], r["y_token_id"]) in selected_pair_set,
        bin_size=bin_size,
    )
    blank_data = stacked_binned_by_key(
        base,
        "blank_controls",
        blank_records,
        ("y_token_id", "phi_idx"),
        lambda r: r["y_token_id"] in selected_y,
        bin_size=bin_size,
    )

    q_blank_mean = {}
    q_blank_reps = {}
    for y in selected_y:
        phases = {phi: blank_data[(y, phi)] for phi in range(4)}
        q_blank_mean[y] = (phases[0].mean(axis=0) - phases[2].mean(axis=0)) / 4.0 + 1j * (
            phases[3].mean(axis=0) - phases[1].mean(axis=0)) / 4.0
        q_blank_reps[y] = (phases[0] - phases[2]) / 4.0 + 1j * (phases[3] - phases[1]) / 4.0

    b_mean = {}
    b_reps = {}
    for x, y in selected_pairs:
        phases = {phi: pair_data[(x, y, phi)] for phi in range(4)}
        q_mean = (phases[0].mean(axis=0) - phases[2].mean(axis=0)) / 4.0 + 1j * (
            phases[3].mean(axis=0) - phases[1].mean(axis=0)) / 4.0
        q_reps = (phases[0] - phases[2]) / 4.0 + 1j * (phases[3] - phases[1]) / 4.0
        b_mean[(x, y)] = q_mean - q_blank_mean[y]
        b_reps[(x, y)] = q_reps - q_blank_reps[y]
    return b_mean, b_reps


def generate_fig4_panels() -> None:
    selected_pairs = [(0, 0), (0, 1), (1, 0), (2, 3)]
    pair_mean, _b_pair = compute_complex_b_for_pairs(selected_pairs)

    # Fig. 4g: four-phase intensity modulation for representative channels.
    ref_pair = (0, 1)
    phase_values = np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2])
    phase_labels = ["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$"]
    traces = np.vstack([pair_mean[(ref_pair[0], ref_pair[1], p)] for p in range(4)])
    ranges = traces.max(axis=0) - traces.min(axis=0)
    channels = np.argsort(ranges)[-4:][::-1]
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for ch, color in zip(channels, colors):
        yvals = traces[:, ch]
        ax.plot(phase_values, yvals, "o-", lw=1.4, ms=4,
                color=color, label=f"ch{int(ch)}")
    ax.set_xticks(phase_values)
    ax.set_xticklabels(phase_labels)
    ax.set_xlabel("Phase phi (rad)")
    ax.set_ylabel("Mean intensity")
    ax.set_title("Four-phase intensity modulation")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    save_panel(fig, "main_fig4g_four_phase_modulation")

    # Fig. 4h: refined complex interaction field distributions at bin = 10 px.
    b_mean, b_reps = compute_refined_complex_b_for_pairs(selected_pairs, bin_size=10)
    all_re = np.concatenate([np.real(b_mean[p]) for p in selected_pairs])
    all_im = np.concatenate([np.imag(b_mean[p]) for p in selected_pairs])
    re_lo, re_hi = np.percentile(all_re, [1, 99])
    im_lo, im_hi = np.percentile(all_im, [1, 99])
    re_pad = 0.15 * (re_hi - re_lo)
    im_pad = 0.15 * (im_hi - im_lo)
    xlim = (re_lo - re_pad, re_hi + re_pad)
    ylim = (im_lo - im_pad, im_hi + im_pad)

    fig, axes = plt.subplots(2, 2, figsize=(5.4, 4.5), sharex=True, sharey=True)
    for ax, pair in zip(axes.ravel(), selected_pairs):
        color = "#1f77b4" if pair[0] == pair[1] else "#d62728"
        for rep in b_reps[pair]:
            ax.scatter(np.real(rep), np.imag(rep), s=1.0, alpha=0.05,
                       color=color, edgecolors="none", rasterized=True)
        z = b_mean[pair]
        ax.scatter(np.real(z), np.imag(z), s=3.5, alpha=0.35,
                   color=color, edgecolors="none", rasterized=True)
        mean_z = np.mean(z)
        ax.plot(np.real(mean_z), np.imag(mean_z), marker="*", ms=7,
                color="black", markeredgecolor="white", markeredgewidth=0.4)
        ax.axhline(0, color="0.75", lw=0.6)
        ax.axvline(0, color="0.75", lw=0.6)
        ax.set_title(f"B({pair[0]}, {pair[1]})")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal", adjustable="box")
    for ax in axes[-1, :]:
        ax.set_xlabel("Re(B)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Im(B)")
    save_panel(fig, "main_fig4h_complex_interaction_fields")

    # Fig. 4i: pair-level task performance.
    perf_path = ROOT / "04_result4_paper/source_data/processed/complex_B_semantic_results_bin10_v2/complex_B_semantic_results_bin10_v2.json"
    perf = json.loads(perf_path.read_text())["classification_results"]
    features = [
        ("zx_zy_concat", "Concat"),
        ("digital_bilinear_ew", "Digital bilinear"),
        ("complex_B", "Complex-B"),
    ]
    tasks = [
        ("task_A_pair_id", "Pair ID"),
        ("task_B_same_cat", "Same category"),
        ("task_C_cat_pair", "Category pair"),
    ]
    x = np.arange(len(features))
    width = 0.24
    fig, ax = plt.subplots(figsize=(4.0, 2.9))
    for i, (task_key, task_label) in enumerate(tasks):
        vals = [perf[feature_key][task_key] for feature_key, _ in features]
        ax.bar(x + (i - 1) * width, vals, width=width, label=task_label)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in features], rotation=12, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Balanced accuracy")
    ax.set_title("Pair-level task performance")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    save_panel(fig, "main_fig4i_pair_level_task_performance")


def main() -> None:
    generate_fig2_panels()
    generate_fig3_panels()
    generate_fig4_panels()
    print(f"Generated panels in {OUT}")


if __name__ == "__main__":
    main()
