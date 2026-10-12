# Scope And Relation To Manuscript/SI

This folder is a source-data and code archive. It is not the formal main
manuscript and it is not the formal Supplementary Information file. For
journal submission, the recommended route is to deposit this archive in a public
repository and cite the repository DOI in the manuscript Data Availability and
Code Availability statements.

## Authoritative Documents

The formal main manuscript and the formal Supplementary Information should be
uploaded separately through the journal submission system. Those submitted PDF
files are authoritative for final wording, final figure assembly, figure panel
labels, caption text, and manuscript-level claims.

This archive supports those documents by providing data, offline scripts,
report PDFs, generated report figures, provenance maps, and checksums.

## Main-Text Data-Panel Layer

`main_text_data_panels/` is an audit-replot layer for selected data-derived
main-text panels only: Fig. 2d-g, Fig. 3d-f, and Fig. 4g-i.

The generated outputs in `main_text_data_panels/generated_panels/` are not
exact visual copies of the final assembled manuscript panels. They are
standalone data-derived replots intended to show how the included data and
scripts support the selected panels. Visual details such as crop, color scale,
font, layout, panel spacing, normalization display, and selected representative
channels or tasks can differ from the final manuscript figures.

For panel-level provenance, use
`main_text_data_panels/MAIN_TEXT_DATA_PANEL_SOURCE_MAP.tsv`. The `visual_status`
column marks these outputs as audit replots, not exact final manuscript panels.

## Report PDFs

The five PDFs under `*/report/` are linked source reports used to document the
development and provenance of the experiments. They are included as supporting
records, not as replacements for the formal manuscript or SI. If a report PDF
uses an older figure layout, older file label, or a report-specific wording, it
should be read as report-level provenance and mapped through the included data,
scripts, and provenance files.

## No Manuscript/SI Duplication

The formal main manuscript PDF and formal SI PDF are intentionally excluded from
this archive to avoid duplicate or conflicting submission records. They should
be handled in the journal's dedicated manuscript and SI upload slots. The
repository archive should be described as source data/code, not as a replacement
for formal SI.

## Sensitive-Information Boundary

This archive is intended to exclude internal session folders, exploratory notes,
task prompts, authoring traces, hardware-only acquisition scripts, and local
filesystem paths. The included video is qualitative visual documentation of the
optical platform and is not quantitative evidence for any figure.
