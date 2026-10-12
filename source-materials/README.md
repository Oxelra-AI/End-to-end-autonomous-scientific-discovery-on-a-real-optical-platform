# Source Data and Code Archive: Five-Report Layout

This folder is a source-data and code archive organized as five independent
report sections, one section per source report. For journal submission, the
recommended use is to deposit this archive in a public repository such as
Zenodo and cite the repository DOI from the manuscript Data Availability and
Code Availability statements.

Start with `SM_Reproducibility_Guide.pdf`. It is the formal supplementary
source-materials guide with manuscript title, authors, affiliations,
correspondence emails, package map, verification steps, and report-level rerun
commands.

Read `SM_SCOPE_AND_RELATION_TO_MANUSCRIPT_SI.md` before comparing this package
with the submitted main manuscript or Supplementary Information. The manuscript
and SI are authoritative for final wording, final assembled figures, captions,
and claims. This package provides source data, scripts, report-level provenance,
and audit replots.

`main_text_data_panels/` provides a narrow main-text panel layer for the
data-derived manuscript panels only: Fig. 2d-g, Fig. 3d-f, and Fig. 4g-i. The
generated outputs are audit replots, not exact visual copies of the final
assembled manuscript panels. They do not attempt to reconstruct the complete
manuscript figure layouts.

Each report section contains the same four internal folders:

| Subfolder | Meaning |
| --- | --- |
| `report/` | The exact source report PDF for that section. |
| `source_data/` | Compact data needed to audit the report figures/tables. |
| `scripts/` | Offline analysis and figure-generation scripts for that report. |
| `figures/` | Figure outputs associated with that report. |

## Five Sections

| Section | Report |
| --- | --- |
| `01_result2_paper/` | `result2_paper.pdf` |
| `02_result2_refine_paper/` | `result2_refine_paper.pdf` |
| `03_result3_paper/` | `result3_paper.pdf` |
| `04_result4_paper/` | `result4_paper.pdf` |
| `05_result4_refine_paper/` | `result4_refine_paper.pdf` |

The Result4 report and the Result4 refined report are deliberately separated.
They do not share a combined Result4 folder in this package.

## Context

`00_context/` contains qualitative visual documentation of the physical optical
platform. It does not contain the main manuscript or SI PDFs.

The formal main manuscript and supplementary information should be uploaded
separately through the journal submission system, not embedded inside this
source-materials package.

For a size-limited journal upload, submit the lightweight
`SM_Reproducibility_Guide.pdf` as the source-materials guide and use the
repository DOI for the full archive. After a DOI is reserved, replace the DOI
placeholder in `DATA_AVAILABILITY_DRAFT.md` and in the manuscript.

## Top-Level Guide

| File | Meaning |
| --- | --- |
| `SM_Reproducibility_Guide.pdf` | Lightweight reviewer-facing guide for checking and reproducing the repository archive. |
| `SM_Reproducibility_Guide.tex` | LaTeX source for the guide. |
| `SM_SCOPE_AND_RELATION_TO_MANUSCRIPT_SI.md` | Plain-language boundary between the SM package, the formal SI, and the main manuscript. |
| `ZENODO_DEPOSIT_NOTES.md` | Practical notes for depositing the full archive in Zenodo and linking it from the manuscript. |
| `00_context/visual_documentation/video_qiushi.mp4` | Visual documentation of the live Qiushi Engine optical-platform interface. |
| `main_text_data_panels/` | Audit replot layer for selected main-text data panels Fig. 2d-g, Fig. 3d-f, and Fig. 4g-i. |

## Raw Data Boundary

This package is self-contained. The Result4 frame-level CMOS `.npy`
acquisitions used by both Result4 reports are included under
`shared_raw_data/result4_frame_data/`.

## How To Read

1. Open `SM_Reproducibility_Guide.pdf` for the formal overview.
2. Open `MANIFEST.md` to see the five-report structure.
3. Open a section folder, for example `04_result4_paper/`.
4. Read that section's `README.md`.
5. Use `provenance/FIGURE_SOURCE_MAP.tsv` to map each report to its data,
   scripts, figures, and raw-data status.
6. Use `main_text_data_panels/MAIN_TEXT_DATA_PANEL_SOURCE_MAP.tsv` for
   main-text data-panel provenance and visual-status labels.
7. Use `VALIDATION_REPORT.md` to confirm file counts, PDF readability, script
   syntax checks, and checksum status.
