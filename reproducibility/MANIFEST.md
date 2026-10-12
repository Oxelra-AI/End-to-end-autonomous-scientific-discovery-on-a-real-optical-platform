# Integrity manifest

The files below are preserved from the public arXiv and Zenodo records.

| File | Source | Integrity value |
| --- | --- | --- |
| `paper/End-to-end_autonomous_scientific_discovery.pdf` | Web-optimized complete copy of arXiv:2604.27092v1 | SHA-256 `617afa6a5f07ad5946d3e53af500f86d580db950ee2a5aa819179cb726553417` |
| Original arXiv v1 PDF in release assets | arXiv:2604.27092v1 | SHA-256 `e5bf5110d5aae245547acfb3879f394a7afafde43b7631382489c6fa892f5778` |
| `reproducibility/SM_Reproducibility_Guide.pdf` | Zenodo 19890402 | MD5 `d299f1bcba99776be6efe1100a492b7f`; SHA-256 `350c9f7a52cd5fb66ce29aa54af8d41f562c4c7761aa8a468fca0245e16b5b44` |
| `source-materials/` (2,325 separately addressable files) | Expanded byte-for-byte from Zenodo 19890402 | original package size `1,383,507,767` bytes; MD5 `5b742fb14f82496aa6fda35ad7613c80`; SHA-256 `5f4a9ede99ce79d3dba7f9c33b51905ebe1c32c13749d6e156239e5c075747c6` |

The expanded tree contains `source-materials/provenance/checksums_sha256.txt`,
which verifies the deposited files individually:

```bash
cd source-materials
sha256sum -c provenance/checksums_sha256.txt --quiet
```

The NumPy arrays are stored through Git LFS. Run `git lfs pull` before checksum
verification after a command-line clone. The original compressed package
remains available from the canonical Zenodo record, but GitHub presents the
materials in expanded form rather than as a second opaque bundle.

The Zenodo DOI is <https://doi.org/10.5281/zenodo.19890402>. The paper record is
<https://arxiv.org/abs/2604.27092>.
