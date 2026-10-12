# Manifest

This repository archive is structured into five report-specific source-material
sections.
The purpose is to make the evidence chain readable without merging distinct
reports.

## Top-Level Guide

| File | Role |
| --- | --- |
| `SM_Reproducibility_Guide.pdf` | Lightweight source-data/code and reproducibility guide for reviewers. |
| `SM_Reproducibility_Guide.tex` | LaTeX source for the guide. |
| `SM_SCOPE_AND_RELATION_TO_MANUSCRIPT_SI.md` | Boundary between this repository archive, the submitted main manuscript, and the submitted SI. |
| `ZENODO_DEPOSIT_NOTES.md` | Notes for repository deposition, DOI reservation, and journal linking. |
| `main_text_data_panels/` | Audit-replot layer for selected main-text data panels Fig. 2d-g, Fig. 3d-f, and Fig. 4g-i. |

| Section | Source Report | Scope |
| --- | --- | --- |
| `01_result2_paper/` | `report/result2_paper.pdf` | Initial transmission-matrix reproduction. |
| `02_result2_refine_paper/` | `report/result2_refine_paper.pdf` | Refined self-referenced transmission matrix. |
| `03_result3_paper/` | `report/result3_paper.pdf` | Majorization-order transport. |
| `04_result4_paper/` | `report/result4_paper.pdf` | Earlier Complex-B bilinear interaction report: XOR showcase, semantic benchmark, and compact-budget analyses. |
| `05_result4_refine_paper/` | `report/result4_refine_paper.pdf` | Refined Complex-B advantage-map report: phase-2 refinement, three-bin sparse taps, raw-input controls, low-output tests, and sparse-tap analyses. |

## Context Files

| File | Role |
| --- | --- |
| `00_context/visual_documentation/video_qiushi.mp4` | Qualitative visual documentation of the live optical-platform interface and witness-camera view. |

The formal main manuscript and supplementary information are intentionally not
included in this repository archive. They should be submitted separately
through the journal system. For size-limited journal upload, deposit this
archive in a public repository and cite its DOI from the manuscript.

## Main-Text Data Panels

`main_text_data_panels/` maps and regenerates only audit replots for the
manuscript data panels: Fig. 2d-g, Fig. 3d-f, and Fig. 4g-i. These outputs are
not exact visual copies of the final assembled manuscript panels. See
`main_text_data_panels/MAIN_TEXT_DATA_PANEL_SOURCE_MAP.tsv` for the panel-level
data/script/output mapping and visual-status label.

## Result4 Split Rule

The two Result4 reports use related experimental acquisitions, but they are
different report products. In this package:

| Earlier Result4 | Refined Result4 |
| --- | --- |
| `04_result4_paper/` | `05_result4_refine_paper/` |
| `complex_B_xor_showcase_results` | `phase2_refinement_results` |
| `complex_B_semantic_results` | `three_bin_sparse` |
| `complex_B_semantic_results_bin10` | `advantage_boundary_results` |
| `complex_B_semantic_results_bin10_v2` | `bin_anchored_twin` |
| `semantic_compressed_budget_results` | `sparse_tap_twin` |
| original XOR/semantic analysis scripts | refinement/advantage-map/sparse-tap scripts |

Compact acquisition records are duplicated into both Result4 sections because
both reports depend on the same physical acquisitions. The full frame-level raw
data are included once, inside `shared_raw_data/result4_frame_data/`, so the
archive can be checked without duplicating the same large arrays twice.
