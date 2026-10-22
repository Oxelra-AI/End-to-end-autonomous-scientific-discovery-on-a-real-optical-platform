# End-to-end autonomous scientific discovery on a real optical platform

[![arXiv](https://img.shields.io/badge/arXiv-2604.27092-b31b1b.svg)](https://arxiv.org/abs/2604.27092)
[![Zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.19890402.svg)](https://doi.org/10.5281/zenodo.19890402)

[中文](README.zh-CN.md) · [Run the analyses](DATA_AND_REPRODUCIBILITY.md) · [Evaluation methods and results](EVALUATION.md)

**Qiushi Discovery Engine connects sustained autonomous research to a real
optical experiment.** The system develops a research plan, writes and executes
experimental code, measures optical responses, and uses the results to revise
its next experiment. This repository provides the public measurements, analysis
code and research records for three studies with progressively more open goals.

## From reproducing an experiment to developing a research direction

| Study | Scientific work | Experimental record |
| --- | --- | --- |
| Transmission matrices | Adapt a published method to a different optical platform and improve wavefront control | A 256 × 256 acquisition using 1,025 calibrated measurements; best focusing enhancement increases from 25.59 to 46.1 |
| Majorization and wave transport | Translate an abstract ordering relation into experimentally accessible transport intervals | Operators reconstructed from optical measurements, interval comparisons, and physical spot checks |
| Optical pair representations | Develop a phase-sensitive interaction readout from coherent superposition, scattering and intensity detection | Four-token XOR and eight-token category-labelled experiments, with comparisons across feature families and readout budgets |

The open-ended study spans 206 research steps and 1,288.1 minutes, producing
163 research notes and 44 scripts. The common thread is a research process
that keeps theory, implementation and physical measurement connected as the
question develops.

## Optical pair computation

Four phase-stepped measurements and a blank control produce a complex
pair-dependent readout, represented numerically by its real and imaginary
parts. This lets a linear classifier use phase-sensitive pair information
alongside the intensity-based representations tested in the study.

On the eight-token fixed-library experiment, the full complex readout achieves
100% balanced accuracy for ordered-pair identity, same-category classification
and ordered category-pair classification when holding out one repeat index
across all input pairs. These are measurements from the same acquisition
campaign. [The evaluation guide](EVALUATION.md) reports this result alongside
unseen-pair and held-category tests, defines each split, and gives the
corresponding digital baselines.

## Run it

With Python 3.12 and Git LFS installed, run these commands from the repository root:

```bash
git lfs pull
python -m pip install -r requirements.txt
python -m analysis.result4 --stage main --output outputs/result4
```

The current analysis fits normalization on training data, records held-out
predictions, and distinguishes measurements, input pairs and categories as
split units. Compressed-feature selection uses inner cross-validation when a
subset is selected. See [the reproduction guide](DATA_AND_REPRODUCIBILITY.md)
for sparse readouts, feature budgets and four-phase readout checks.

## Explore the repository

| Path | Contents |
| --- | --- |
| [`analysis/`](analysis/) | Maintained offline analysis and evaluation code |
| [`results/`](results/) | Compact results from the public measurements |
| [`source-materials/`](source-materials/) | The unchanged, expanded Zenodo v1 deposit: measurements, original scripts, figures and provenance |
| [`paper/`](paper/) | Previously released paper |
| [`reproducibility/`](reproducibility/) | Original source-data guide and integrity manifest |
| [`tests/`](tests/) | Tests for splitting, preprocessing, feature selection and phase readout |

The archive contains 2,325 files, including 2,018 NumPy arrays stored with Git
LFS. The [platform video](source-materials/00_context/visual_documentation/video_qiushi.mp4)
shows the experimental setting. New analyses write to `outputs/`, leaving the
archived measurements and results intact. Optical acquisition uses the
separately maintained hardware-control environment; this repository runs the
offline analyses.

## Authors and citation

Shuxing Yang, Fujia Chen, Rui Zhao, Junyao Wu, Yize Wang, Haiyao Luo, Ning Han,
Qiaolu Chen, Yuze Hu, Wenhao Li, Mingzhu Li, Hongsheng Chen, and Yihao Yang.

Use [`CITATION.cff`](CITATION.cff) or [`CITATION.bib`](CITATION.bib), and cite the
paper and data record when using the measurements. Related Qiushi research
projects are linked in [Research lineage](RESEARCH_LINEAGE.md).

[Organization repository](https://github.com/Oxelra-AI/End-to-end-autonomous-scientific-discovery-on-a-real-optical-platform)
· [Shuxing Yang's repository](https://github.com/Cherish0925/End-to-end-autonomous-scientific-discovery-on-a-real-optical-platform)

## Rights

The paper, Zenodo deposit and third-party materials retain their existing
terms. No repository-wide license is asserted; contact the corresponding
authors for uses not already permitted by those terms.
