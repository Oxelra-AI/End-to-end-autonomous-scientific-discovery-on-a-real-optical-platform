# Submission Package Status

This file records package-structure checks. It is not a journal acceptance or
scientific-claim assessment.

## Checked Items

| Requirement | Status |
| --- | --- |
| Five report-specific sections | Present |
| Result4 and Result4 refine separated | Present |
| Source-materials guide with author, affiliation, and email metadata | Present |
| Explicit SM/SI/main-manuscript relationship file | Present |
| Main manuscript and SI excluded from archive | Verified; submit them separately through the journal system |
| Main-text data-panel audit replots for Fig. 2d-g, Fig. 3d-f, and Fig. 4g-i | Present; not exact final manuscript panels |
| Optical-platform visual documentation included | Present; qualitative only |
| Each report has its own `report/`, `source_data/`, `scripts/`, and `figures/` folders | Present |
| Notes, traces, and hardware-only scripts excluded | Checked by path/text scan |
| Shared Result4 raw frame arrays included inside the archive | Present |
| Provenance maps and checksum manifest | Present |

## Boundary

This archive includes the Result4 raw frame-level arrays under
`shared_raw_data/result4_frame_data/`.

The formal manuscript and SI are authoritative for final assembled figures,
captions, and claims. `main_text_data_panels/` is a data/code audit layer and is
not a replacement for the submitted figure files.

The full archive is too large for typical journal SI upload limits. The
recommended submission route is to deposit the full archive in a public
repository, cite the repository DOI in the manuscript, and upload only the
lightweight source-materials guide PDF through the journal system if requested.
