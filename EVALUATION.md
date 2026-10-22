# Optical pair evaluation

The experiments ask whether a phase-sensitive optical readout can provide a
useful representation of two inputs. The eight-token experiment tests three
labels on the same physical readout: ordered-pair identity (64 classes),
same-category relation (2 classes), and ordered category pair (16 classes).
The four-token experiment tests identity and XOR parity.

Token fields are deterministic random-phase patterns. Category labels define
the evaluation tasks; they are not encoded as semantic embeddings. Each
ordered input pair has five repeated acquisitions.

## What each split measures

| Protocol | Held out from classifier training | Question |
| --- | --- | --- |
| Measurement LOO | One measured example | Can another measurement of a known input combination be classified? |
| Repeat-blocked | The same repeat index across every input pair | Does recognition persist across the five recorded repeats? |
| Ordered-pair holdout | Every repeat of one ordered pair | Does the relation or category-pair label transfer to a new ordered combination? |
| Unordered-pair holdout | Every repeat of both `(x,y)` and `(y,x)` | Does transfer persist when the reverse pair is also unseen? |
| Fixed 16-pair holdout | One prespecified pair from each ordered category pair | How does the original held-pair comparison perform? |
| Category holdout | Every pair containing either token of one category | Does same-category classification transfer to a held-out category? |

Measurement and repeat-blocked evaluations use the same input library and
acquisition campaign. They do not measure performance on a new day or a new
optical alignment. Pair-identity classification is evaluated only where the
identity labels exist in training; holding out an entire identity class would
be a different, open-set problem.

For the main analysis, feature means and standard deviations are fitted
inside each training fold. The refined sparse analysis preserves its original
unstandardized features. Both use `RidgeClassifier(alpha=1.0)`.
The eight-token category split uses four folds. Reported category scores are
the mean of their balanced accuracies; cross-category pairs can occur in two
test folds. Five repeats of a pair are five measurements of that pair, not
five independent input combinations.

## Fixed-library recognition

Full readout, eight tokens, one repeat index held out across all 64 pairs;
training-fold normalization. Values are balanced accuracies.

| Representation | Pair identity | Same category | Category pair |
| --- | ---: | ---: | ---: |
| Concatenated single-token intensities | 100.00% | 50.00% | 100.00% |
| Elementwise product of single-token intensities | 56.25% | 100.00% | 62.50% |
| Complex pair readout | 100.00% | 100.00% | 100.00% |

This comparison supports joint decodability of the three labels for the
measured input library. The scripts also save the unseen-pair results; those
are separate evaluations and are included in `results/main_summary.json`.
For example, the full complex readout scores 57.50% for same-category
classification under ordered-pair holdout, using training-fold normalization.

The original full-budget comparisons are regenerated before applying the
preprocessing change. Across 48 scores, 47 are unchanged; the hybrid
representation's fixed 16-pair score changes from 45.83% to 50.00%.

## Category transfer and self-pairs

A self-pair `(x,x)` is always same-category. Report both the full task and the
task restricted to different inputs to identify what the classifier uses.
Removing self-pairs changes the training and test populations; these scores
are complementary tasks, not revised estimates of an identical task.

| Complex readout configuration | All pairs | Train on all pairs, test on different inputs | Train and test on different inputs |
| --- | ---: | ---: | ---: |
| 25-pixel bins, 25 spatial taps | 77.19% | 58.44% | 56.46% |
| 10-pixel bins, 100 spatial taps | 78.02% | 60.52% | 56.46% |

Under the last protocol, the direct raw-input absolute-difference and
conjugate-product baselines both score 70.83%. These are comparisons of
representations and classifiers, not matched hardware energy or latency
measurements. The complete three-bin, nine-budget grid and all six raw-input
baselines are retained in `results/`.

For the original all-pair category task, the strongest raw-input baseline is
absolute difference at 71.88%. The best optical score across bins exceeds
that baseline at seven of the nine output counts. Selecting the best bin on
these test results is a descriptive comparison, not an independently tested
configuration choice.

## Feature budgets

A spatial complex channel contributes two real coordinates. A budget of 50
real features therefore does not mean 50 complete complex spatial channels.

The eight-token budget analysis averages ten seeded random feature subsets.
At 50 real features, the complex readout reaches 99.94%, 95.79% and 99.44%
for the three fixed-library tasks under measurement LOO. The XOR budget analysis selects among five
seeded subsets using inner cross-validation and evaluates the choice in the
outer measurement-LOO fold. At 400 real features, the complex readout reaches
100% on both XOR tasks. Each preprocessing fit stays within its corresponding
training split. Subset indices and predictions are saved by `analysis.budgets`.

## Measured readout and physical model

The measured object is defined by the released four-phase processing:

$$
Q(x,y)=\frac{I_0-I_\pi}{4}
+i\frac{I_{3\pi/2}-I_{\pi/2}}{4},\qquad
B_{\mathrm{meas}}(x,y)=Q(x,y)-Q(0,y).
$$

For fixed input fields, a fixed coherent reference, correct phase steps and a
common linear intensity calibration, the ideal model gives
$B(x,y)=\overline{TE_x}\odot TE_y$. For a binned detector channel, the
corresponding kernel is the weighted sum of pixel kernels:

$$
A_k=\sum_{p\in P_k}w_p\,t_p^\dagger t_p,
\qquad B_k=E_x^\dagger A_k E_y.
$$

This averages the pixel outer products, not the fields before forming an
outer product. In complex coordinates the expression is conjugate-linear in
its first input and linear in its second.

The same ideal intensity model requires
$I_0+I_\pi=I_{\pi/2}+I_{3\pi/2}$.
`analysis.phase_readout` reports this residual, repeat variation and the
residual after blank subtraction in the released camera-array units. The
measured residuals are nonzero; connecting them to the ideal model requires
accounting for actual field encoding and intensity calibration. The routine
does not fit gains to force agreement. Its residual is not a relative error
estimate for the extracted cross-term. Repeatability, classification
performance and agreement with the ideal operator are distinct measurements.

## Result provenance

The tables above are regenerated from the public Zenodo v1 measurements.
`results/main_environment.json` and `results/refined_environment.json` record
the software versions and input hashes. The compact results can be rebuilt
with the commands in [Data and reproducibility](DATA_AND_REPRODUCIBILITY.md).
The original archive, including its historical scripts and PDFs, is preserved
under `source-materials/`.
