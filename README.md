# End-to-end autonomous scientific discovery on a real optical platform

[![arXiv](https://img.shields.io/badge/arXiv-2604.27092-b31b1b.svg)](https://arxiv.org/abs/2604.27092)
[![Zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.19890402.svg)](https://doi.org/10.5281/zenodo.19890402)

This repository brings together the paper, source data, offline analysis code,
figures, provenance records and reproducibility materials for **“End-to-end
autonomous scientific discovery on a real optical platform.”**

The study introduces **Qiushi Discovery Engine**, a long-horizon research
system coupled to a real free-space optical platform. Across three studies, the
system moves from reproducing a published experiment, to translating an
abstract theory into a physical validation, and finally to proposing and
experimentally validating an optical bilinear interaction for pairwise
computation.

## Scientific result

The work reports three progressively more open research settings:

1. **Transmission-matrix reproduction.** Qiushi Engine transfers a published
   transmission-matrix experiment to a non-original optical platform, completes
   a 256 × 256 acquisition with 1,025 calibrated measurements, and improves the
   best focusing enhancement from 25.59 to 46.1.
2. **Theory-to-experiment translation.** It converts majorization order in wave
   coherence into measurable transport intervals and validates the predicted
   nesting relation on the physical platform.
3. **Open-ended discovery.** Starting from the broad theme of optical computing
   for artificial intelligence, it identifies an optical bilinear interaction:
   coherent superposition, scattering and square-law detection generate a
   measurable pair-dependent field structurally related to the bilinear
   compatibility operation in Transformer attention.

The open-ended study spans 206 Agent Steps, 1,288.1 minutes, 145.9 million
tokens, 3,242 model calls and 1,242 tool calls, producing 163 research notes and
44 scripts. The claim is grounded in physical measurements rather than a
simulation-only workflow.

## Repository map

```text
paper/
  End-to-end_autonomous_scientific_discovery.pdf
reproducibility/
  SM_Reproducibility_Guide.pdf
  MANIFEST.md
DATA_AND_REPRODUCIBILITY.md
RESEARCH_LINEAGE.md
CITATION.cff
CITATION.bib
```

The complete 1.4 GB source-material package is attached to the
[`v1.0-source-materials`](../../releases/tag/v1.0-source-materials) release as
`SM_Source_Materials_CORE_5REPORTS.tar.gz`. It is an exact copy of the Zenodo
deposit and contains all five report sections, source data, 29 offline Python
analysis scripts, generated figures, selected main-text-panel replots, raw
Result 4 CMOS arrays, provenance maps and checksums.

## Start here

- [Paper](paper/End-to-end_autonomous_scientific_discovery.pdf)
- [Source-data and code guide](reproducibility/SM_Reproducibility_Guide.pdf)
- [Data and reproduction map](DATA_AND_REPRODUCIBILITY.md)
- [Integrity manifest](reproducibility/MANIFEST.md)
- [Research lineage](RESEARCH_LINEAGE.md)
- [Zenodo record](https://doi.org/10.5281/zenodo.19890402)
- [arXiv record](https://arxiv.org/abs/2604.27092)

## Reproducibility boundary

The public package supports inspection and offline rerunning of the reported
analyses. It does not include the Qiushi Engine core system or the
instrument-control software. Re-acquiring measurements requires the original
optical platform and its calibrated control environment. The paper and formal
Supplementary Information remain the authoritative scientific description;
the source-material archive provides data, code and provenance traceability.

## Authors

Shuxing Yang, Fujia Chen, Rui Zhao, Junyao Wu, Yize Wang, Haiyao Luo, Ning Han,
Qiaolu Chen, Yuze Hu, Wenhao Li, Mingzhu Li, Hongsheng Chen, and Yihao Yang.

For correspondence, see the paper.

## Citation

Use the metadata in [`CITATION.cff`](CITATION.cff) or
[`CITATION.bib`](CITATION.bib). Cite both the arXiv paper and the Zenodo
source-data record when using the archived data or analysis code.

## Rights

No repository-wide license is asserted here. The arXiv paper and Zenodo record
retain their own terms, and third-party materials remain subject to their
respective rights. Contact the corresponding authors for uses not already
permitted by those terms.
