# Recomputed public-data results

These files are generated from the released measurements using the maintained
`analysis/` code. See [evaluation methods](../EVALUATION.md) for the prediction
task, split unit and interpretation of each comparison.

| File | Contents |
| --- | --- |
| `main_summary.json` | Full-readout scores and additional holdout protocols for both token libraries |
| `main_comparison.csv` | The 48 archived, regenerated and training-normalized main comparisons |
| `sparse_summary.csv` | Three bin sizes × nine output counts × three representations, with all three self-pair protocols |
| `raw_input_baselines.json` | Six raw-input baselines and their category-fold scores |
| `feature_budgets.json` | Mean random-subset and nested-selection results |
| `phase_readout.json` | Four-phase means, repeat variation and closure residuals for every measured pair |
| `reproduction_summary.json` | Comparison counts and numerical agreement with the archive |
| `*_environment.json` | Runtime versions and input/analysis hashes |

Regenerate these compact files after the four runs in the
[reproduction guide](../DATA_AND_REPRODUCIBILITY.md):

```bash
python -m analysis.summarize --runs outputs --output results
```

Full per-sample predictions and feature-subset choices remain in the local
run outputs. The committed tables contain the corresponding aggregate
results. Original v1 values remain available in `source-materials/`.
