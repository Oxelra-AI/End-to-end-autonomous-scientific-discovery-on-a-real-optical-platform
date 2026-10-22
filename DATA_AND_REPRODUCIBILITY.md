# Data and reproducibility

Run the maintained analyses from the repository root. Their outputs go to
`outputs/`; the original measurements, scripts and figures remain in
`source-materials/` as an unchanged archival snapshot.

## Setup

Use Python 3.12 and Git LFS:

```bash
git lfs pull
python -m pip install -r requirements.txt
```

`requirements.txt` pins the environment used for the committed results.
The NumPy arrays must be downloaded as data, not left as LFS pointer files;
the current loader reports a specific error if it encounters a pointer.

## Current Result 4 analyses

```bash
# Full readout, training-fold normalization, and explicit holdout units.
python -m analysis.result4 --stage main --output outputs/result4

# Sparse readouts, held-category evaluation and self-pair controls.
python -m analysis.result4 --stage refined --output outputs/refined

# Prespecified random subsets and nested feature selection.
python -m analysis.budgets --features outputs/result4 --output outputs/budgets

# Four-phase consistency, including blank-subtracted residuals.
python -m analysis.phase_readout --output outputs/phase_readout

# Rebuild the compact, version-controlled result tables after all runs finish.
python -m analysis.summarize --runs outputs --output results
```

The first two commands also accept `--deposit PATH` for an extracted copy of
the public package. Use a fresh output directory for a different dataset;
frame caches are associated with the dataset path, experiment and bin size.
No hardware access or model-service credentials are needed.

The main stage regenerates the archived full-budget scores, then evaluates
training-fold normalization and additional holdouts. The refined stage
recomputes the three-bin sparse grid and direct raw-input baselines, with and
without self-pairs. Hyperparameter `alpha=1.0`, token definitions and labels
are preserved. A compact table is in [EVALUATION.md](EVALUATION.md).

| Output | Contents |
| --- | --- |
| `environment.json` | Package versions, input-script hashes and analysis hash |
| `semantic_main.json`, `xor_main.json` | Archived, regenerated and training-normalized scores; corrected predictions |
| `*_split_comparisons.json` | Repeat-blocked, ordered-pair and unordered-pair holdouts; fixed 16-pair split for the eight-token experiment |
| `*_bin25_features.npz` | Reconstructed full-readout features for budget analyses |
| `sparse_comparisons.json`, `sparse_summary.csv` | All three bins and nine sparse-output counts, with each self-pair protocol |
| `raw_baselines.json` | The six direct raw-input feature baselines under matching category splits |
| `semantic_k50.json`, `xor_nested_k400.json` | Feature-budget results and subset choices |
| `phase_readout.json` | Phase means, repeat variation, closure and blank-subtracted closure in the released array units |

In the main-stage JSON, use `corrected.balanced_accuracy` for current LOO
scores. In category-holdout results, use `mean_fold_balanced_accuracy`.
Individual category folds overlap in their cross-category test pairs, so
folds are not independent experimental repetitions.

## Archived package

The canonical v1 deposit is [Zenodo 19890402](https://doi.org/10.5281/zenodo.19890402).
Its package is `SM_Source_Materials_CORE_5REPORTS.tar.gz`:

- MD5: `5b742fb14f82496aa6fda35ad7613c80`
- SHA-256: `5f4a9ede99ce79d3dba7f9c33b51905ebe1c32c13749d6e156239e5c075747c6`

Verify the expanded data after `git lfs pull`:

```bash
(cd source-materials && sha256sum -c provenance/checksums_sha256.txt --quiet)
```

The manifest covers 2,324 files; the checksum list is the 2,325th file.
The original scripts and generated figures remain available for historical
reproduction. Use `analysis/` for the current Result 4 evaluation. The v1
report PDFs and archived outputs retain their original values.

| Archive path | Study |
| --- | --- |
| `01_result2_paper/` | Initial transmission-matrix reproduction |
| `02_result2_refine_paper/` | Refined self-referenced transmission matrix |
| `03_result3_paper/` | Majorization-order transport |
| `04_result4_paper/` | XOR and eight-token pair benchmarks |
| `05_result4_refine_paper/` | Readout-budget and sparse-output studies |
| `shared_raw_data/result4_frame_data/` | Result 4 camera arrays |
| `provenance/` | Input, figure and checksum maps |

The existing figure commands for the first two research settings are:

```bash
(cd source-materials/01_result2_paper && python scripts/make_core_figures_v2.py)
(cd source-materials/02_result2_refine_paper && python scripts/generate_paper_figures.py)
(cd source-materials/03_result3_paper && python scripts/generate_v3_figures.py)
```

These regenerate the archived analyses for those studies; they are separate
from the current Result 4 recomputation. Source mappings are in
`source-materials/provenance/FIGURE_SOURCE_MAP.tsv` and
`source-materials/main_text_data_panels/MAIN_TEXT_DATA_PANEL_SOURCE_MAP.tsv`.

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests cover training-only normalization, held-category token separation,
reverse-pair grouping, inner feature selection and the analytical four-phase
readout. GitHub Actions runs these tests without downloading the camera data.
Full numerical runs use the commands above.
