# Data and reproducibility

## Canonical archive

The complete public archive is `SM_Source_Materials_CORE_5REPORTS.tar.gz`.
The same byte-identical file is preserved in two places:

- GitHub release: `v1.0-source-materials`
- Zenodo DOI: <https://doi.org/10.5281/zenodo.19890402>

Verify the archive against the hashes in
[`reproducibility/MANIFEST.md`](reproducibility/MANIFEST.md), then extract it:

```bash
tar -xzf SM_Source_Materials_CORE_5REPORTS.tar.gz
cd SM_Source_Materials_CORE_5REPORTS
sha256sum -c provenance/checksums_sha256.txt --quiet
```

A silent checksum command means that every archived file matches the package
manifest.

## Package structure

| Path | Role |
| --- | --- |
| `00_context/` | Qualitative visual documentation of the optical platform |
| `01_result2_paper/` | Initial transmission-matrix reproduction |
| `02_result2_refine_paper/` | Refined self-referenced transmission-matrix experiment |
| `03_result3_paper/` | Majorization-order transport experiment |
| `04_result4_paper/` | Complex-B bilinear field, XOR and semantic benchmark |
| `05_result4_refine_paper/` | Refined advantage map and sparse-output analyses |
| `main_text_data_panels/` | Audit replots for Fig. 2d-g, Fig. 3d-f and Fig. 4g-i |
| `shared_raw_data/result4_frame_data/` | Full Result 4 frame-level CMOS arrays |
| `provenance/` | File inventory, checksums, figure-source and raw-data maps |

Every report section contains `report/`, `source_data/`, `scripts/` and
`figures/` subdirectories. The audit replots are generated from the included
data; they are not intended to reproduce the final manuscript page layout.

## Environment

- Python 3.9 or newer
- NumPy
- SciPy
- Matplotlib
- scikit-learn

For headless execution:

```bash
export MPLBACKEND=Agg
```

## Main-text data panels

```bash
cd SM_Source_Materials_CORE_5REPORTS
python main_text_data_panels/scripts/generate_main_text_data_panels.py
```

## Report 1: transmission-matrix reproduction

```bash
cd SM_Source_Materials_CORE_5REPORTS/01_result2_paper
python scripts/make_core_figures.py
python scripts/make_core_figures_v2.py
```

## Report 2: refined self-referenced transmission matrix

```bash
cd SM_Source_Materials_CORE_5REPORTS/02_result2_refine_paper
python scripts/generate_paper_figures.py
```

`fix_tm_filtering.py` optionally recomputes filtered transmission-matrix
diagnostics and can overwrite filtered arrays.

## Report 3: majorization-order transport

```bash
cd SM_Source_Materials_CORE_5REPORTS/03_result3_paper
python scripts/generate_v3_figures.py
python scripts/generate_raw_data_figure.py
python scripts/analyze_majorization_spectra.py
python scripts/analyze_majorization_spectra_v2.py
```

## Report 4: Complex-B bilinear interaction field

```bash
cd SM_Source_Materials_CORE_5REPORTS/04_result4_paper
RAW=../shared_raw_data/result4_frame_data

python scripts/analyze_complex_B_xor_showcase.py \
  --data-dir $RAW/exp_bilinear_001/exp_bilinear_001 \
  --output-dir source_data/processed/complex_B_xor_showcase_results

python scripts/analyze_complex_B_semantic.py \
  --data-dir $RAW/exp_complex_B_semantic \
  --output-dir source_data/processed/complex_B_semantic_results

python scripts/generate_semantic_bin10_figures.py
python scripts/generate_semantic_bin10_figures_v2.py

python scripts/analyze_semantic_compressed_budget.py \
  --data-dir $RAW/exp_complex_B_semantic \
  --output-dir source_data/processed/semantic_compressed_budget_results
```

## Report 5: refined Complex-B advantage map

```bash
cd SM_Source_Materials_CORE_5REPORTS/05_result4_refine_paper
python scripts/analyze_refinement_phase2_fast.py
python scripts/regenerate_xor_complex_scatter_bin10.py
python scripts/analyze_three_bin_sparse.py
python scripts/analyze_advantage_boundary.py
python scripts/analyze_bin_anchored_twin.py
python scripts/analyze_sparse_tap_twin.py
```

## What is and is not reproduced

The package supports report, figure, processed-data and script inspection;
offline regeneration of the included analyses; main-text data-panel reruns;
and Result 4 raw-frame reprocessing. It does not support optical data
re-acquisition. That requires the original hardware, calibration and
instrument-control stack, which are not part of this deposit.

For the authoritative mapping from inputs and scripts to outputs, use
`provenance/FIGURE_SOURCE_MAP.tsv` and
`main_text_data_panels/MAIN_TEXT_DATA_PANEL_SOURCE_MAP.tsv` inside the archive.
