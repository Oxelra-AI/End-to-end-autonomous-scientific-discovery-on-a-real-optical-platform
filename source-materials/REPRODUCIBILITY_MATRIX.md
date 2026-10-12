# Reproducibility Matrix

| Reproducibility Level | Supported By This Package | Notes |
| --- | --- | --- |
| Report inspection | Yes | Five report PDFs are separated into five report folders. |
| Manuscript/SI inspection | Not in SM package | The formal manuscript and SI should be uploaded separately through the journal system and are authoritative for final wording, figure assembly, captions, and claims. |
| Main-text data-panel audit replots | Yes | `main_text_data_panels/` regenerates Fig. 2d-g, Fig. 3d-f, and Fig. 4g-i as standalone audit replots, not exact final manuscript panels. |
| Physical-platform visual documentation | Yes | `00_context/visual_documentation/video_qiushi.mp4` records the live system interface and witness-camera view. |
| Figure inspection | Yes | Each report section has its own `figures/` folder. |
| Processed-data inspection | Yes | Each report section has its own `source_data/` folder. |
| Code inspection | Yes | Each report section has its own `scripts/` folder. |
| Script syntax audit | Yes | All included Python scripts parse successfully. |
| Zero-edit rerun from extracted package | Partial | Scripts use package-internal paths; some analyses may still require the expected Python environment and runtime dependencies. |
| Full Result4 raw-frame reprocessing | Yes | Full CMOS frame arrays are included under `shared_raw_data/result4_frame_data/`. |
| Hardware re-acquisition | Not supported by SM package | Requires the original optical platform and hardware-control environment. |

The package is suitable for repository deposition as a source-data/code archive,
subject to final DOI insertion and any journal-specific checks requested during
submission.
