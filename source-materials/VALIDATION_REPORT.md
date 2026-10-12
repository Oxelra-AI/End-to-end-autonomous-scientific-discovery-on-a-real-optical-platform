# Validation Report

This report records checks performed on the five-report source-materials
package.

## Structure Check

| Check | Result |
| --- | --- |
| Package folder | `SM_Source_Materials_CORE_5REPORTS/` |
| Required report sections | 5 |
| Report PDFs found in report sections | 5 |
| Context manuscript/SI PDFs found | 0 |
| Formal reproducibility guide PDF found | 1 |
| SM/SI/manuscript relationship file found | 1 |
| Repository-deposit notes found | 1 |
| Visual documentation videos found | 1 |
| Main-text data-panel audit-replot section found | 1 |
| Main-text audit replots generated | 10 PDF + 10 PNG |
| Report PDF author metadata | `Qiushi Engine` for all five report PDFs |
| Report/guide PDF creation-modification dates visible to `pdfinfo` | 0 |
| Internal preparation files included | 0 |
| Top-level merged `result4_bilinear/` section | Not present |
| Separate `04_result4_paper/` section | Present |
| Separate `05_result4_refine_paper/` section | Present |

## Section Sizes

| Section | Size |
| --- | --- |
| `00_context/` | 8.7M |
| `01_result2_paper/` | 2.5M |
| `02_result2_refine_paper/` | 16M |
| `03_result3_paper/` | 13M |
| `04_result4_paper/` | 29M |
| `05_result4_refine_paper/` | 13M |
| `main_text_data_panels/` | 2.3M |
| `shared_raw_data/` | 1.9G |

## PDF Readability

| PDF | Pages |
| --- | --- |
| `SM_Reproducibility_Guide.pdf` | 7 |
| `01_result2_paper/report/result2_paper.pdf` | 12 |
| `02_result2_refine_paper/report/result2_refine_paper.pdf` | 18 |
| `03_result3_paper/report/result3_paper.pdf` | 8 |
| `04_result4_paper/report/result4_paper.pdf` | 22 |
| `05_result4_refine_paper/report/result4_refine_paper.pdf` | 29 |

## Main-Text Data Panels

| Scope | Result |
| --- | --- |
| Covered panels | Fig. 2d-g; Fig. 3d-f; Fig. 4g-i |
| Panel source map | `main_text_data_panels/MAIN_TEXT_DATA_PANEL_SOURCE_MAP.tsv` |
| Regeneration script | `main_text_data_panels/scripts/generate_main_text_data_panels.py` |
| Generated panel outputs | 10 PDF + 10 PNG |
| Visual status | Audit replots only; not exact final assembled manuscript panels |

## Manuscript/SI Boundary

| Item | Result |
| --- | --- |
| Formal main manuscript PDF in SM package | 0 |
| Formal SI PDF in SM package | 0 |
| Boundary document | `SM_SCOPE_AND_RELATION_TO_MANUSCRIPT_SI.md` |
| Authority statement | Manuscript and SI are authoritative for final wording, final assembled figures, captions, and claims. |
| Repository route | Full archive can be deposited in Zenodo or another suitable public repository; upload the guide PDF to the journal system if requested. |

## Visual Documentation

| File | Role |
| --- | --- |
| `00_context/visual_documentation/video_qiushi.mp4` | Qualitative recording of the live Qiushi Engine optical-platform interface and witness-camera view. |

## Script Check

| Check | Result |
| --- | --- |
| Offline Python scripts | 29 |
| Syntax failures | 0 |
| Hardware-style `exp_*.py` / `tm_*.py` scripts included | 0 |

The script check was syntax-only. It did not run optical hardware acquisition
or regenerate experimental outputs.

## Exclusion Check

| Excluded Class | Result |
| --- | --- |
| `notes/` folders | 0 |
| Internal authoring trace paths | 0 |
| Non-scientific process trace folders | 0 |
| Packaging-session path references | 0 |
| Sensitive text hits in extracted PDF text | 0 |
| Incorrect green-wavelength hits in text/PDF scan | 0 |
| Correct red-wavelength hits where wavelength is stated | Present |

## Final Inventory

After adding this report and regenerating provenance manifests:

| Item | Count/Size |
| --- | --- |
| Total files | 2325 |
| Total directories | 114 |
| Approximate package size | 1.9G |

## Provenance Files

| File | Role |
| --- | --- |
| `provenance/SOURCE_PREFIX_MAP.tsv` | Maps each top-level package section to included material. |
| `provenance/FIGURE_SOURCE_MAP.tsv` | Maps each report section to data, scripts, figures, and raw-data status. |
| `main_text_data_panels/MAIN_TEXT_DATA_PANEL_SOURCE_MAP.tsv` | Maps the ten main-text data panels to data, scripts, and generated outputs. |
| `SM_SCOPE_AND_RELATION_TO_MANUSCRIPT_SI.md` | Explains how this repository archive relates to the submitted manuscript and SI. |
| `ZENODO_DEPOSIT_NOTES.md` | Records recommended repository-deposit route and DOI-linking notes. |
| `provenance/RAW_DATA_INCLUDED.tsv` | Records raw-data locations included inside this package. |
| `provenance/FILE_TREE.txt` | Sorted file tree. |
| `provenance/file_inventory.tsv` | Sorted file-size inventory. |
| `provenance/checksums_sha256.txt` | SHA-256 checksum manifest. |

Additional review guidance is provided in `RUN_NOTES.md` and
`REPRODUCIBILITY_MATRIX.md`.

Checksum verification command:

```bash
sha256sum -c provenance/checksums_sha256.txt --quiet
```
