# Run Notes

This repository archive preserves the report-level source data, figures, and offline
scripts in a compact five-report layout. The scripts are copied as source
evidence and are not rewritten into a new package-specific workflow.

For a reviewer-facing overview, first open `SM_Reproducibility_Guide.pdf`.
It gives the formal package map, verification steps, expected environment, and
report-specific rerun commands.

Before comparing this package with the submitted main manuscript or SI, read
`SM_SCOPE_AND_RELATION_TO_MANUSCRIPT_SI.md`. The formal manuscript and SI are
authoritative for final text, final assembled figures, captions, and claims.
This archive is the data/code/provenance layer.

The formal main manuscript and supplementary information are not included in
this repository archive. Formal submission should provide them through the
journal's required upload slots. The full archive can be deposited in Zenodo or
another suitable repository and cited by DOI from the manuscript.

The video in `00_context/visual_documentation/video_qiushi.mp4` is included only
as qualitative documentation of the live optical-platform interface and witness
camera. It is not an input to any quantitative reproduction script.

The main-text data panels are regenerated with:

```bash
python main_text_data_panels/scripts/generate_main_text_data_panels.py
```

This covers only Fig. 2d-g, Fig. 3d-f, and Fig. 4g-i. The generated outputs are
audit replots, not exact visual copies of the final assembled manuscript
panels, and they do not cover the complete manuscript figure layouts.

## Important Path Note

Scripts have been adjusted to use paths inside this package. For re-execution,
run each script from its report section or use the paths already defined inside
the script.

## Result4 Raw-Frame Boundary

`04_result4_paper/` and `05_result4_refine_paper/` include processed outputs and
compact acquisition records. The shared full frame-level CMOS `.npy` arrays are
included under `shared_raw_data/result4_frame_data/`.

## Practical Review Workflow

For ordinary review:

1. Inspect the report PDF in the section's `report/` folder.
2. Inspect the corresponding processed data in `source_data/`.
3. Inspect generated figures in `figures/`.
4. Inspect the analysis or figure source in `scripts/`.
5. Use `provenance/FIGURE_SOURCE_MAP.tsv` to verify that the report, data,
   scripts, and figures belong to the same section.

For full raw reprocessing of Result4, use the scripts in the corresponding
Result4 section together with `shared_raw_data/result4_frame_data/`.
