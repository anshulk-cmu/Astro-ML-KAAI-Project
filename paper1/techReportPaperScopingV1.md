# Technical Report: a diagnostic suite for AION-1's representation

This report is written diagnostic by diagnostic, as each one is run. It is the record for the scoping document `paperScopingV1.md`, and it replaces the track-based organisation of `technicalReport.md` (which remains the record of the Phase 1 and Track A to D work it describes).

## Reading rules

1. Every quantitative claim is tagged **[measured]** (a number an artifact contains) or **[interpreted]** (our reading of it). Summary paragraphs restate numbers that are tagged and sourced in their home sections and carry no tags of their own.
2. Every diagnostic section ends with an **Artifacts** line naming the script, the results file and the figure behind it, so any number can be traced in one step.
3. This document is a technical record of method and result. Development history, implementation decisions and the audit trail are kept in the project run log and are not reproduced here.
4. **Provenance rule.** A number enters this report only if a script in `paper1/diagnostics/` produced it and wrote it to `paper1/results/` with a provenance block recording the git revision, the SHA-256 of every input file, the seed and the environment. No value is transcribed by hand without an automated check against its artifact.
5. **Tier rule.** Every diagnostic states its evidential tier in its heading and in the master table: input-space causal, descriptive with a calibrated null, or descriptive. The tier is a column readers can see, not a detail in the methods.

## The protocol every diagnostic follows

1. **Frozen encoder.** AION-1 large, public weights, never fine-tuned. Mean pooling over encoder outputs at a 600-token budget. That budget exceeds the 256-token budget used in pretraining, so results are properties of this documented recipe.
2. **Probe fit once, on unperturbed data, evaluated held out.** Nothing is ever fit to a perturbed condition.
3. **Deterministic intervention where one exists.** Rotation, reflection, major-axis flip, blur, band masking, reddening and artificial redshifting are exact operations on the input with exactly known consequences. Where such an operation exists, the diagnostic is causal.
4. **A named null** beside every headline number.
5. **Evidential tier reported.**

## Substrate discipline

All headline probes run on **E_img**, the image-token-only embedding: nothing but pixels goes in, so anything read out of it was learned from the image. **E_full** (image plus scalar g, r, z flux plus the catalog redshift token) is used only for the leakage ablation in Diagnostic 6. Without that separation, "the model knows redshift" is ambiguous between *learned* and *was told*.

---

## Master results table

Tier and status for all nine diagnostics.

| # | Diagnostic | Pillar | Tier | Status | Headline |
|---|---|---|---|---|---|
| 1 | Angle readout characterization | I | descriptive | **run 2026-07-25** | 2.027 deg [1.946, 2.116], loop radius 0.9886, n=3,179 held out; chance 45; error scales as ellipticity^-0.97 |
| 2 | O(2) equivariance | I | **input-space causal** | not yet run here | |
| 3 | Chirality | I | **input-space causal** | not yet run here | |
| 4 | Nuisance decodability and leakage | II | descriptive | not yet run here | |
| 5 | Degradation response | II | **input-space causal** | not yet run here | |
| 6 | Decodability battery | III | descriptive | not yet run here | |
| 7 | Concept geometry under a calibrated null | III | descriptive with calibrated null | not yet run here | |
| 8 | Structured relations | III | descriptive | not yet run here | |
| 9 | Artificial redshifting | III | **input-space causal** | not yet run here | |

---

## 1. Data and the anchor sample

### 1.1 The anchor

Every diagnostic in this suite is computed on one fixed sample, the **anchor**: **48,398 galaxies**, each with a four-band image cutout, a frozen embedding, and a set of catalog labels and observing-condition covariates. All 48,398 `dr8_id` values are unique. The sample is held fixed across diagnostics so that results are comparable between them, and so that a population difference can never be mistaken for a model difference. [measured]

The imaging is from the DESI Legacy Imaging Surveys. Each cutout is **96 by 96 pixels** at **0.262 arcsec per pixel**, giving a field of view of **25.15 arcsec**, in the bands **g, r, i, z**, with fluxes in nanomaggies at zeropoint **22.5**. [measured]

### 1.2 Row contract

The row contract is enforced in `lib/data.py` and is identical for every diagnostic: **row i of the embedding corresponds to row `ok_index[i]` of the sample table**, and every catalog table is left-joined onto that order by `dr8_id`. A galaxy missing from a catalog therefore becomes a missing value rather than shifting any row, and the loader asserts that no join changes the row count.

### 1.3 Labels and their coverage

Coverage differs sharply between label sets, which is why every diagnostic reports the sample size beside each entry. Morphology vote fractions, redshift and magnitudes are complete; the stellar-population and structural quantities come from catalog crossmatches and cover under a tenth of the anchor.

| label | source | n finite | coverage |
|---|---|---|---|
| redshift | external catalog, photometric-dominated | 48,398 | 1.000 |
| spectroscopic redshift | same | 6,699 | 0.138 |
| r magnitude | Legacy Surveys catalog | 48,398 | 1.000 |
| smooth vote fraction | Galaxy Zoo DESI | 48,398 | 1.000 |
| featured vote fraction | Galaxy Zoo DESI | 48,398 | 1.000 |
| edge-on vote fraction | Galaxy Zoo DESI | 4,948 | 0.102 |
| stellar mass, elpetro | NSA crossmatch | 3,728 | 0.077 |
| specific star formation rate | NSA crossmatch | 4,760 | 0.098 |
| Sersic index | NSA crossmatch | 3,730 | 0.077 |

Redshift spans 0.0011 to 0.688 with a median of 0.174 and a first-to-ninety-ninth percentile range of 0.025 to 0.384. Sparse-label subsamples are catalog crossmatches rather than random draws and carry their own selection, which is standing caveat 4. [measured]

### 1.4 Shape catalog

48,290 anchor galaxies have a catalog shape row. Model types are SER 37,885, REX 4,111, DEV 4,051, EXP 1,736, DUP 457 and PSF 50, with 108 galaxies unmatched. Over the sources with a defined position angle, ellipticity has median 0.230 and a first-to-ninety-ninth percentile range of 0.018 to 0.730, and the propagated catalog uncertainty on position angle has median 0.249 degrees, rising to 4.51 degrees at the ninety-ninth percentile as the axis becomes ill-defined. Section 4.3 gives the parameterisation and the exclusion rule behind these figures. [measured]

### 1.5 Observing-condition covariates

Each galaxy carries the conditions under which it was observed, which Pillar II treats as the confound budget for everything in Pillar III. Over the 48,290 matched galaxies, r-band seeing has percentiles 1.06 / 1.38 / 2.17 arcsec at the 1st, 50th and 99th; r-band point-source depth in inverse nanomaggies squared has percentiles 121 / 621 / 12,433; and Galactic extinction E(B-V) has percentiles 0.0081 / 0.0307 / 0.1481. By hemisphere, **35,919 are south and 12,371 are north**, with 108 unmatched. The two hemispheres are different instruments, so this split is an instrument boundary rather than a geographic one. [measured]

**Artifacts.** `paper1/diagnostics/d0DatasetAudit.py`, `paper1/results/d0DatasetAudit.json`

## 2. The model and the embeddings

### 2.1 The frozen recipe

AION-1 large is used with its public weights and is **never fine-tuned**, in any diagnostic. Inputs pass through the model codecs to tokens, and we take the encoder output and **average over tokens**. Two properties of this recipe are ours rather than the model's, and are stated wherever results are quoted:

- **Token budget.** The encoder is called with a budget of **600 tokens**, which retains all 576 image tokens of a 96 by 96 four-band cutout. Pretraining used a smaller budget, so inference here operates outside the regime the model was trained in. Every result is a property of this documented recipe.
- **Pooling.** Mean pooling is our choice. The attentive pooling used in the model's own downstream setting is a learned layer that would require its own training target and split discipline, and would no longer be a frozen-encoder measurement.

### 2.2 The two substrates

| substrate | contents | role |
|---|---|---|
| **E_img** | image tokens only | the substrate for every headline probe |
| **E_full** | image plus scalar g, r, z flux plus the catalog redshift token | leakage ablation only |

Anything read out of E_img was learned from pixels, because nothing else went in. E_full ingests quantities the model may simply be repeating back, so it bounds leakage and is never a competing measurement. E_full is not the model's full input set: it excludes spectra, the i-band scalar flux and the infrared photometry.

### 2.3 Numerical properties

Both arrays are 48,398 by 1,024 float32 with **no NaN and no infinity**. Vector norms concentrate near a shell: E_img has mean norm 49.58 with standard deviation 2.46, E_full 48.37 with 2.52. Per-dimension standard deviations are not uniform, spanning a factor of **22.1** in E_img and **22.6** in E_full. [measured]

That spread is why every analysis begins from the **z-scored** embedding: each dimension has its mean subtracted and is divided by its standard deviation. Without it a handful of dimensions would dominate every distance and every direction. The standardisation is fit on all rows before any split, which leaks nothing because it uses no labels.

**Artifacts.** `paper1/diagnostics/d0DatasetAudit.py`, `paper1/results/d0DatasetAudit.json`

## 3. Shared methods

Machinery used by more than one diagnostic is implemented once in `paper1/lib/` and described once here. Diagnostic sections state only what they add.

### 3.1 Constants

One source of truth, `paper1/config.py`: seed **0**; test fraction **0.2**; ridge grid **10^-2 to 10^4 in seven logarithmic steps**; fixed regularisation **100** for concept directions; **1,000** bootstrap resamples; working ellipticity cut **0.3**. No diagnostic chooses its own.

### 3.2 Linear probes

A probe asks how much of a label a linear map can read out of the embedding. We fit ridge regression, minimising squared error plus a penalty proportional to the squared coefficient norm, with the penalty chosen by cross-validation on the training split alone. The score is the coefficient of determination on the held-out split,

> R2 = 1 - sum (y - y_hat)^2 / sum (y - y_bar)^2

which is 1 for a perfect readout and 0 for one no better than predicting the mean. A **fixed probe** is the same fitted object applied unchanged to other rows or to re-encoded inputs. It is what makes the causal diagnostics causal, since nothing is refit to the perturbed condition.

### 3.3 Circular readouts

A quantity that wraps cannot be regressed directly. For a quantity of period 360/k degrees we regress the pair (cos k theta, sin k theta) and recover theta_hat = atan2(s_hat, c_hat) / k. The error is the difference wrapped into the quantity's own period, so it lies in [0, 180/k] degrees, and a readout carrying no information gives a uniform error whose median is **90/k degrees**: 45 for an axial quantity, 90 for a full circle. The **loop radius** is the length of the predicted vector, whose truth counterpart is 1 by construction; it separates a confident readout from one that has collapsed toward the origin. Section 4.2 derives its relation to the component R2 values.

### 3.4 Intervals

Every reported statistic carries a 95 per cent interval from 1,000 bootstrap resamples, reported as the 2.5th and 97.5th percentiles. The resampling unit is the **held-out prediction**, so these intervals capture test-set sampling only, not refitting, resplitting or regularisation-selection variation, and they are narrower than total uncertainty. Where a conclusion could depend on the partition, the whole fit is repeated over independent splits and that spread is reported alongside, as in Section 4.10.

### 3.5 Rank statistics

Monotone trends are reported as Spearman rank correlations with bootstrap intervals rather than as Pearson coefficients, because no diagnostic has grounds to assume a linear relation. Where a covariate could confound a trend, a **partial** rank correlation is reported alongside, computed by removing the rank of the confounder from both variables by least squares and correlating the residuals.

### 3.6 Nulls

Seven null types are used across the suite: shuffled labels, matched-energy random sets, matched-norm random directions, matched-noise simulation, a calibrated generative null, a pixel-only floor, and population controls. Each is implemented once in `lib/nulls.py`. A null is reported on every axis its result is reported on, and each null is itself checked against its analytic expectation in the test suite.

### 3.7 Provenance

Every results file records the git revision and dirty flag, a UTC timestamp, wall-clock seconds, the seed, interpreter and package versions, the platform, and the SHA-256 and size of every input file read. `runAll.py --verify` re-hashes those inputs and reports any that have changed. Appendix A gives the reproduction commands.

---

# Pillar I — Observation geometry

Orientation is the ideal test case. It wraps at 180 degrees, so the faithful representation is a closed loop rather than a number line, which is a non-trivial topological requirement. The transformations acting on it are exact operations on pixels with exactly known consequences. That combination makes this pillar the strongest evidence in the suite and sets the standard the other two are measured against.

## 4. Diagnostic 1 — Angle readout characterization · *descriptive*

**Science question.** Does the model store a quantity that wraps on a closed loop rather than a line, and does the fidelity of that storage track the observable's own measurability?

### 4.1 The quantity and why it needs a circle

A galaxy's position angle is the sky direction of its long axis. It is *axial*, not directional: rotating a galaxy by 180 degrees returns the same object, so the angle lives on the interval [0, 180) with the two ends identified. The value 179 and the value 1 are two degrees apart, not 178. Any readout that treats the angle as a number on a line therefore carries a seam at the wrap point, and errors near that seam are counted as enormous when they are in fact tiny.

The standard remedy in directional statistics is to double the angle. Under theta -> 2 theta the interval [0, 180) maps onto the full circle [0, 360), the identification of the two ends becomes the ordinary closure of a circle, and standard circular machinery applies; results are halved to return to the axial scale. The technique dates to Krumbein (1939) and is the textbook treatment of axial data in Fisher (1993) and Mardia and Jupp (2000).

We therefore fit two ridge probes, one for cos 2 theta and one for sin 2 theta, and recover

> theta_hat = (1/2) atan2(s_hat, c_hat)

The circular error for one galaxy is the difference wrapped into the axial period,

> err = |arg exp(i (2 theta_hat - 2 theta))| / 2, which lies in [0, 90] degrees

A readout that knows nothing produces a uniform error distribution on [0, 90], whose median is **45 degrees**. That is the chance floor for this diagnostic, and the shuffled-label null must reproduce it.

### 4.2 Loop radius, and what it measures

The truth vector (cos 2 theta, sin 2 theta) has length exactly 1 by construction. The prediction does not: ridge regression shrinks, so the predicted point (c_hat, s_hat) can sit anywhere in the plane. Its length is the **loop radius**. Near 1 the prediction lands on the rim of the circle; near 0 it has collapsed to the centre, where the recovered angle is a direction with no magnitude behind it. Two readouts can share a median error and differ completely in radius, so the radius is reported beside every error in this section, on held-out rows only.

The radius is not the mean resultant length of directional statistics, which measures the concentration of a set of angles about their mean. It is a per-galaxy shrinkage measure of one regression prediction. Under a linear-Gaussian approximation with position angle uniform on the axial circle, Var(cos 2 theta) = Var(sin 2 theta) = 1/2 and a probe achieving held-out R-squared of R2 recovers a predicted variance of R2/2 per component, so the expected squared radius is the mean of the two component R-squared values. Section 4.11 tests that identity as a consistency check rather than assuming it.

### 4.3 Population

Position angle is defined only where the catalog fits an ellipticity. The DESI Legacy Imaging Surveys tractor catalog parameterises source shape as a complex ellipticity,

> epsilon = epsilon_1 + i epsilon_2 = ((a - b) / (a + b)) exp(2 i phi)

from which |epsilon| = hypot(e1, e2), b/a = (1 - |epsilon|) / (1 + |epsilon|) and phi = (1/2) atan2(e2, e1). The factor of two in the exponent is the same axial doubling as Section 4.1: the catalog itself stores the orientation on the doubled circle, and our angle, ellipticity, axis ratio and inclination columns are derived from these expressions. Appendix F records the numerical verification of each identity against the catalog columns.

Three tractor model types carry no ellipticity freedom: the round exponential (REX), the point source (PSF) and the duplicate-source model (DUP). For these, shape_e1 and shape_e2 are identically zero, so the derived angle is a constant rather than a measurement, and such sources cannot enter a position-angle analysis.

Of the 48,398 anchor galaxies, 48,290 have a catalog shape row. 4,618 of those are circular-by-construction models (4,111 REX, 457 DUP, 50 PSF) and are excluded, leaving **43,672 galaxies with a defined position angle**. Of these, **15,893** exceed the working ellipticity cut of 0.3, the regime in which the axis is well determined, and form the fitting population. The retained angles are uniformly distributed on the axial circle, as an isotropic sky requires; per-bin uniformity is verified in Section 4.5 and over the whole retained population in the test suite. [measured]

### 4.4 The readout

One probe pair is fit on a random 80 percent of the elongated population (12,714 galaxies, seed 0) and evaluated on the held-out 20 percent (3,179). Cross-validated regularisation selected alpha = 0.01 for both components.

| substrate | median circular error (deg) | 95% CI | loop radius | 95% CI | R2 cos / R2 sin | within 20 deg |
|---|---|---|---|---|---|---|
| E_img (image tokens only) | 2.027 | [1.946, 2.116] | 0.9886 | [0.9836, 0.9928] | 0.9703 / 0.9658 | 0.9972 [0.9953, 0.9987] |
| E_full (robustness line) | 2.146 | [2.063, 2.247] | 0.9866 | | 0.9683 / 0.9638 | |

E_full ingests scalar photometry and the catalog redshift but no shape information, and it is reported here only as a robustness line, not as a competing measurement. [measured]

### 4.5 Stress axis 1 — elongation grading

One probe, fit once as above, evaluated on every galaxy with a defined position angle that the probe never saw, binned by catalog ellipticity. Bins extend well below the working cut into the regime where the axis is genuinely ill-defined. `sigma_PA` is the catalog's own propagated uncertainty (Section 4.8); the KS p-value tests the catalog angles in that bin against a uniform distribution, guarding against a repeat of the placeholder-label problem.

| ellipticity | n | median error (deg) | 95% CI | loop radius | catalog sigma_PA (deg) | label uniformity KS p |
|---|---|---|---|---|---|---|
| 0.00 – 0.05 | 2,424 | 23.643 | [22.36, 25.15] | 0.270 | 2.097 | 0.81 |
| 0.05 – 0.10 | 4,565 | 11.993 | [11.56, 12.44] | 0.390 | 0.809 | 0.58 |
| 0.10 – 0.15 | 5,986 | 7.339 | [7.10, 7.55] | 0.543 | 0.480 | 0.92 |
| 0.15 – 0.20 | 5,784 | 5.176 | [5.02, 5.33] | 0.671 | 0.343 | 0.82 |
| 0.20 – 0.25 | 4,958 | 3.834 | [3.70, 3.96] | 0.780 | 0.262 | 0.65 |
| 0.25 – 0.30 | 4,062 | 3.046 | [2.93, 3.18] | 0.860 | 0.209 | 0.98 |
| 0.30 – 0.40 | 1,225 | 2.563 | [2.34, 2.77] | 0.945 | 0.155 | 0.03 |
| 0.40 – 0.50 | 902 | 1.965 | [1.80, 2.14] | 1.005 | 0.115 | 0.50 |
| 0.50 – 0.60 | 578 | 1.661 | [1.51, 1.81] | 1.028 | 0.091 | 0.97 |
| 0.60 – 0.70 | 334 | 1.694 | [1.40, 1.99] | 1.016 | 0.068 | 0.74 |
| 0.70 – 1.00 | 140 | 1.646 | [1.41, 2.05] | 1.003 | 0.044 | 0.60 |

The ordering is monotone across the full range: Spearman rank correlation between bin ellipticity and bin median error is **-0.991, p = 3.8e-9** over the 11 bins. Loop radius rises monotonically from 0.270 to about 1.0 over the same range, so the readout reports low confidence exactly where the axis is ill-defined rather than guessing confidently. The roundest bin sits at 23.6 degrees, **below** the 45-degree chance floor, so real angular information survives even where the catalog calls the object nearly round. One bin (0.30 to 0.40) returns a KS p-value of 0.03; across eleven tests that is not significant under any multiplicity adjustment, and no conclusion rests on it. Radii slightly above 1 in the three most elongated bins mean the probe marginally overshoots the unit circle for the highest-signal objects, which is the expected behaviour of a shrinkage estimator when shrinkage is weak. [measured]

**Error scaling (extension beyond the scoping document).** Multiplying each bin's median error by its median ellipticity gives 0.774, 0.929, 0.919, 0.903, 0.858, 0.835, 0.889, 0.875, 0.904 across the bins below ellipticity 0.6: an approximately constant product of about 0.88 degrees across a twentyfold range in elongation. Fitting log median error against log ellipticity over those bins gives a slope of **-0.974**, against **-1.109** for the catalog's own sigma_PA. A slope of -1 is the geometric expectation for any estimator of an axis whose underlying component noise does not itself depend on elongation, since the angular uncertainty of an ellipse axis scales as sigma_e / |epsilon|. The model therefore degrades with roundness in the same functional form as the catalog measurement it is scored against, with a larger effective component noise: the ratio of model error to catalog sigma_PA runs from 11.3 to 18.3 over the fitted range. Above ellipticity 0.6 the readout leaves the power law and flattens at 1.694 and 1.646 degrees, which is a floor of the readout rather than of the geometry. [measured; the geometric interpretation is interpreted]

### 4.6 Stress axis 2 — topology matching

The scoping document's prediction is that a periodic quantity should be stored as a loop and a bounded one as a line. Inclination and axis ratio are bounded companions of position angle, derived from the same catalog shape fit, and are probed as plain scalars.

| quantity | treatment | held-out R2 | 95% CI | n |
|---|---|---|---|---|
| inclination | plain scalar | 0.8445 | [0.8298, 0.8587] | 15,893 |
| axis ratio b/a | plain scalar | 0.8487 | [0.8338, 0.8627] | 15,893 |
| edge-on vote fraction | plain scalar | 0.8889 | | 4,948 |
| position angle | plain scalar on the raw angle | 0.7736 | | 15,893 |

Scored in degrees rather than R2, which puts the two treatments of each quantity on one axis: position angle recovers to 2.027 degrees under the circular treatment and 10.329 degrees [9.854, 10.875] under the plain linear treatment, a factor of 5.1. Inclination recovers to a median absolute error of 1.565 degrees [1.509, 1.635] as a plain scalar, and forcing it through the same circular machinery gives 1.529 degrees in inclination units with loop radius 0.976 — no material gain. The periodic quantity needs the loop; the bounded one does not. [measured]

The bounded arm of this contrast is **underpowered by design and must be quoted as such**: a bounded quantity has no wrap seam for a linear encoding to fail at, so the circular treatment of inclination cannot fail the way the linear treatment of position angle does. The informative half is the position-angle half.

### 4.7 Stress axis 3 — population invariance

One globally-fit probe, held fixed, evaluated separately on each stratum's held-out rows. Per-stratum refits measure within-stratum decodability and are secondary. Brightness tertiles are cut at r = 17.99 and r = 18.63.

| stratum | n | fixed-probe error (deg) | 95% CI | loop radius | refit error (deg) |
|---|---|---|---|---|---|
| bright | 1,042 | 2.053 | [1.93, 2.17] | 0.9882 | 2.032 |
| mid | 1,044 | 1.926 | [1.78, 2.11] | 0.9951 | 2.184 |
| faint | 1,093 | 2.062 | [1.93, 2.21] | 0.9817 | 2.257 |
| smooth | 1,817 | 1.970 | [1.84, 2.09] | 0.9841 | 2.104 |
| featured | 176 | 2.471 | [2.06, 3.08] | 0.9603 | 2.466 |

Every stratum's interval overlaps the all-galaxy value of 2.027, and the featured stratum, which is both the smallest (n = 176) and the highest, has an interval reaching 3.08. One coordinate system serves every stratum tested at this resolution. [measured]

### 4.8 Stress axis 4 — heteroscedasticity

Per-galaxy errors on the held-out set, against apparent magnitude and angular size. Because rounder galaxies are also on average fainter and smaller, each raw rank correlation is accompanied by a partial rank correlation with ellipticity held.

| covariate | Spearman with error | 95% CI | p | partial, ellipticity held | 95% CI |
|---|---|---|---|---|---|
| apparent magnitude r | +0.012 | [-0.023, +0.043] | 0.51 | +0.009 | [-0.026, +0.041] |
| angular size shape_r | -0.067 | [-0.103, -0.032] | 1.7e-4 | -0.025 | [-0.061, +0.010] |
| ellipticity | -0.177 | [-0.212, -0.143] | 7.6e-24 | | |

Brightness has no detectable effect on the readout across the sample's magnitude range. Angular size has a small raw effect that does not survive controlling for ellipticity, so it is attributable to the elongation dependence rather than to size itself. Elongation is the one covariate that matters, which is the same conclusion the grading reaches by a different route. [measured]

**Label-noise floor (extension beyond the scoping document).** The catalog's own position-angle uncertainty follows from the ellipticity component variances: with phi = atan2(e2, e1), Var(phi) = (e2^2 Var(e1) + e1^2 Var(e2)) / (e1^2 + e2^2)^2 and sigma_PA = sigma_phi / 2. Over the elongated population the median sigma_PA is **0.111 degrees**, interquartile range [0.060, 0.194]. The readout's 2.027 degrees is about eighteen times that, so this measurement is **not** label-noise limited: the error is the model's own, not the catalog's. [measured]

### 4.9 Nulls

| null | median error (deg) | 95% CI | loop radius | R2 cos / R2 sin |
|---|---|---|---|---|
| shuffled labels | 44.060 | [42.47, 45.76] | 0.0487 | -0.0008 / -0.0040 |
| top 2 principal components | 43.198 | [41.77, 45.26] | 0.0291 | |
| plain linear probe on raw angle | 10.329 | [9.85, 10.88] | | R2 = 0.7736 |
| theoretical chance floor | 45 | | 0 | |

The shuffled-label null returns 44.06 degrees against the theoretical 45, with its interval covering 45 and its loop radius collapsing to 0.049: the machinery finds nothing when there is nothing, and it reports that absence in the radius as well as in the error. The top-two-principal-component null sits at chance while carrying 34 percent of the embedding variance.

**Principal-component sweep (extension beyond the scoping document).** Retaining k leading components and refitting gives 43.20 degrees at k = 2 (34 percent of variance), 34.03 at k = 5 (54 percent), 17.72 at k = 10 (69 percent), 4.28 at k = 20 (83 percent), 4.22 at k = 50, 3.95 at k = 100, 3.21 at k = 200, 2.35 at k = 512 (100.0 percent to three figures), against 2.027 for the full 1,024 dimensions. The angle coordinate is essentially absent from the dominant variance directions and becomes readable only once roughly twenty components are retained. This is the concrete form of the scoping document's point that the top-PC null is load-bearing beyond this diagnostic: a structure invisible to the leading variance directions is invisible to any global statistic built on them, and targeted probes are the instrument that finds it. [measured; the implication for global statistics is interpreted]

### 4.10 Sensitivity to the train and test split

The intervals in Sections 4.4 to 4.9 resample held-out predictions and therefore capture test-set sampling only. To bound the variation the bootstrap cannot see, the entire fit and evaluation is repeated over ten independent 80/20 splits, each redrawing the partition and refitting both component probes.

| statistic | mean over 10 splits | standard deviation | minimum | maximum |
|---|---|---|---|---|
| median circular error (deg) | 1.989 | 0.050 | 1.907 | 2.040 |
| loop radius | 0.9909 | 0.0028 | 0.9857 | 0.9968 |
| R2 cos | 0.9719 | | | |
| R2 sin | 0.9673 | | | |

The across-split standard deviation of 0.050 degrees is comparable to the within-split bootstrap half-width of about 0.085 degrees, and every split lies inside a range of 0.13 degrees. The readout is a property of the representation rather than of one partition. [measured]

### 4.11 Consistency checks

- **Chance floor.** Shuffled labels give 44.060 degrees against a theoretical 45; the interval covers it.
- **Radius collapse.** The same null's loop radius is 0.0487, against 0 expected for a readout carrying no information.
- **Shrinkage identity.** The root-mean-square loop radius is 0.98461; the square root of the mean of the two component R-squared values is 0.98390. Agreement to 7e-4 supports the interpretation of loop radius as a shrinkage measure given in Section 4.2.
- **Monotonicity.** Spearman between bin ellipticity and bin median error is -0.991 (p = 3.8e-9).
- **Cross-implementation agreement.** An independent implementation of the same estimator, written against a separate data loader and sharing only the seed and split rule, returns 2.0266658 degrees for this quantity. The two agree to floating-point equality, difference **0.0** degrees.

### 4.12 What the result does and does not establish

Established, on this substrate and this population: the model carries a mod-180 angular coordinate that a linear readout recovers to about two degrees on held-out galaxies, with predictions landing on the unit circle rather than collapsing inward; the fidelity of that coordinate degrades monotonically as the axis becomes ill-defined, in the same functional form as the geometric uncertainty of the measurement itself; the coordinate is shared across brightness and morphology strata rather than refit per population; and it is essentially invisible to the leading principal components.

Not established here: that the model *uses* this coordinate for anything downstream, which no probe can show; that the coordinate transforms correctly under operations on the input, which is Diagnostic 2; and that the readout would survive on a survey with different observing conditions, which Diagnostics 4 and 5 bound. This diagnostic is descriptive in tier, and all four of its stress axes are correlational statements about a frozen representation.

**Artifacts.** `paper1/diagnostics/d1AngleReadout.py`, `paper1/diagnostics/d1AngleReadoutFigures.py`, `paper1/results/d1AngleReadout.json`, `paper1/results/d1AngleReadoutProjection.npy`, figures `d1Loop.png`, `d1Elongation.png`, `d1Nulls.png`, `d1Invariance.png`, `d1Heteroscedasticity.png`.

![Loop](figures/d1Loop.png)
![Elongation grading](figures/d1Elongation.png)
![Nulls](figures/d1Nulls.png)
![Invariance](figures/d1Invariance.png)
![Heteroscedasticity](figures/d1Heteroscedasticity.png)

## 5. Diagnostic 2 — O(2) equivariance · *input-space causal*

**Science question.** Does the internal angle coordinate transform correctly under known transformations of the input, and do the transformations compose the way they compose in the plane?

**Procedure as implemented.** *pending*

**Nulls.** *pending* — the untransformed baseline error as the noise floor; matched random directions.

**Results.** *pending*

**What it tells us.** *pending*


**Artifacts.** `paper1/diagnostics/d2Equivariance.py`, `paper1/results/d2Equivariance.json`, `paper1/figures/d2*.png`

## 6. Diagnostic 3 — Chirality · *input-space causal*

**Science question.** Does the model encode parity-odd structure, spiral arm handedness, or has it discarded a real physical observable?

**Procedure as implemented.** *pending*

**Nulls.** *pending* — the achiral elliptical population; the same procedure with a rotation instead of a flip.

**Results.** *pending*

**What it tells us.** *pending* — both outcomes are informative, which is why it is worth running. A null means the model discarded a real observable, which is a concrete demonstration that augmentation choices delete physics from representations. A positive result hands over a self-supervised handedness label as a by-product.


**Artifacts.** `paper1/diagnostics/d3Chirality.py`, `paper1/results/d3Chirality.json`, `paper1/figures/d3*.png`

---

# Pillar II — Instrumental and observational systematics

The embedding is supposed to describe the source. Every image also records the conditions it was taken under, and none of that is a property of the object. This pillar quantifies that contamination. It does double duty: these numbers set the confound budget that qualifies every physical result in Pillar III, and they are the most directly usable results in the suite for anyone doing cross-survey work.

## 7. Diagnostic 4 — Nuisance decodability and leakage · *descriptive*

**Science question.** How strongly does the embedding encode the observation rather than the object, and can it recover information that should be unavailable from the image entirely?

**Procedure as implemented.** *pending*

**Nulls.** *pending* — shuffled labels; the residualization drop compared against the drop from residualizing on random covariates of matched dimension.

**Results.** *pending*

**What it tells us.** *pending*

**Stated limitation.** The residualization drop compares R-squared on targets of different variance. It is a crude apportionment, not a mediation decomposition, and is reported as such.


**Artifacts.** `paper1/diagnostics/d4NuisanceLeakage.py`, `paper1/results/d4NuisanceLeakage.json`, `paper1/figures/d4*.png`

## 8. Diagnostic 5 — Degradation response · *input-space causal*

**Science question.** Does the representation respond to controlled degradation the way the physics requires, and where does each physical readout break?

**Procedure as implemented.** *pending*

**Nulls.** *pending* — the unperturbed baseline; perturbations applied to a matched random subset of pixels rather than physically.

**Results.** *pending*

**What it tells us.** *pending* — the severity at which each physical readout loses half its accuracy is an effective operating limit, directly usable by anyone applying the model to a survey different from the training set.


**Artifacts.** `paper1/diagnostics/d5Degradation.py`, `paper1/results/d5Degradation.json`, `paper1/figures/d5*.png`

---

# Pillar III — Physical structure

Predicting stellar mass accurately is a statement about information content. Reconstructing the Hubble tuning fork, an ordered sequence that branches exactly where bars become possible and bends rather than running straight, is a statement about structure. The first is already established by downstream benchmarks. The second is not, and it is the stronger claim about whether a model has learned physics.

## 9. Diagnostic 6 — Decodability battery · *descriptive, with two control arms*

**Science question.** Which physical quantities are linearly readable from pixels alone, particularly quantities the model never received as input?

**Procedure as implemented.** *pending*

**Nulls.** *pending* — shuffled labels; a raw-pixel-PCA floor requiring no model weights.

**Results.** *pending*

**What it tells us.** *pending* — quantities that are never model inputs in any substrate and still decode well are the cleanest evidence of learned physics rather than input recall. Any entry where the model ties the pixel floor is non-discriminating and is marked as such.


**Artifacts.** `paper1/diagnostics/d6Decodability.py`, `paper1/results/d6Decodability.json`, `paper1/figures/d6*.png`

## 10. Diagnostic 7 — Concept geometry under a calibrated null · *descriptive with calibrated null (methods contribution)*

**Science question.** Do the directions encoding distinct physical properties sit further apart than the properties' own correlations require?

**Procedure as implemented.** *pending*

**Nulls.** *pending* — the pair-specific calibrated generative null, and the sanity anchors that gate whether its output can be trusted.

**Results.** *pending*

**What it tells us.** *pending* — with per-pair p-values this becomes an inferential statement rather than a descriptive contrast, and false-discovery control across the sweep becomes possible for the first time.

**Stated assumption.** The null assumes the model's linearly-decodable structure is adequately described by a linear-Gaussian generative model in the embedding. That is an assumption, but a stated one with testable consequences, which is a strict improvement on an uncalibrated contrast.


**Artifacts.** `paper1/diagnostics/d7ConceptGeometry.py`, `paper1/results/d7ConceptGeometry.json`, `paper1/figures/d7*.png`

## 11. Diagnostic 8 — Structured relations · *descriptive*

**Science question.** Does the model organize physical relationships the way nature does, preserving ordering, regime-dependence, curvature and conditional structure, rather than merely encoding the marginal quantities?

**Procedure as implemented.** *pending*

**Nulls.** *pending* — shuffled labels for every fitted direction; a matched noise null for the curvature test; held-out evaluation for the fork; a non-branching control property.

**Instance gate.** Before any instance of 8d or 8e runs, the relation must be shown to be visible in label space with the sample actually in hand. The gate is stated in advance precisely so that dropping an instance is distinguishable from selecting the ones that worked. Gate outcomes are recorded here whether they pass or fail.

**Results.** *pending*

**What it tells us.** *pending* — this is where "encodes physics" is distinguished from "organizes physics".


**Artifacts.** `paper1/diagnostics/d8StructuredRelations.py`, `paper1/results/d8StructuredRelations.json`, `paper1/figures/d8*.png`

## 12. Diagnostic 9 — Artificial redshifting · *input-space causal*

**Science question.** When an object is transformed to look like it sits at a higher redshift, does the embedding move along the empirically derived redshift direction, by the right magnitude, and without disturbing intrinsic properties?

**Procedure as implemented.** *pending*

**Nulls.** *pending* — matched-norm random directions; the untransformed baseline; real objects at the target redshift as the reference population.

**Results.** *pending*

**What it tells us.** *pending* — this upgrades an embedding-level translation result, where one merely adds a vector and inspects neighbours, into a genuine intervention on the input. The on-manifold check is what makes it credible.


**Artifacts.** `paper1/diagnostics/d9ArtificialRedshift.py`, `paper1/results/d9ArtificialRedshift.json`, `paper1/figures/d9*.png`

---

## 13. Findings index

One row per claim, with its section, key numbers, evidential tier and artifact. Rows are added as diagnostics land. No row is marked as a headline: that selection belongs to the manuscript.

| # | Claim | Section | Key numbers | Tier | Artifact |
|---|---|---|---|---|---|
| 1 | A linear readout recovers the axial angle from the image-only embedding on held-out galaxies | 4.4 | 2.027 deg [1.946, 2.116], n = 3,179; R2 0.9703 / 0.9658; 99.72 percent within 20 deg | descriptive | d1AngleReadout.json |
| 2 | Predictions land on the unit circle rather than collapsing toward its centre | 4.4 | loop radius 0.9886 [0.9836, 0.9928] | descriptive | d1AngleReadout.json |
| 3 | The readout is not an artifact of one partition | 4.10 | 10 splits: mean 1.989, sd 0.050, range [1.907, 2.040] | descriptive | d1AngleReadout.json |
| 4 | Fidelity degrades monotonically as the axis becomes ill-defined | 4.5 | 23.643 deg at ellipticity 0.00-0.05 to 1.646 deg above 0.70; Spearman -0.991, p = 3.8e-9 | descriptive | d1AngleReadout.json |
| 5 | Error scales inversely with elongation, as a geometric axis estimator does | 4.5 | log-log slope -0.974 against -1.109 for the catalog uncertainty; error x ellipticity about 0.88 deg | descriptive | d1AngleReadout.json |
| 6 | The readout saturates above ellipticity 0.6 | 4.5 | 1.694 and 1.646 deg in the two highest bins | descriptive | d1AngleReadout.json |
| 7 | The measurement is not limited by the uncertainty of its label | 4.8 | catalog sigma_PA median 0.111 deg, IQR [0.060, 0.194], against a readout error of 2.027 deg | descriptive | d1AngleReadout.json |
| 8 | One coordinate serves every brightness and morphology stratum tested | 4.7 | fixed probe 1.926 to 2.471 deg, radii 0.9603 to 0.9951, all intervals overlapping 2.027 | descriptive | d1AngleReadout.json |
| 9 | Readout error is unrelated to apparent brightness | 4.8 | Spearman +0.012 [-0.023, +0.043], p = 0.51 | descriptive | d1AngleReadout.json |
| 10 | The apparent angular-size trend is the elongation dependence | 4.8 | raw -0.067 [-0.103, -0.032]; partial with ellipticity held -0.025 [-0.061, +0.010] | descriptive | d1AngleReadout.json |
| 11 | A periodic quantity needs the circular treatment; a bounded companion does not | 4.6 | position angle 2.027 circular against 10.329 linear; inclination 1.565 linear against 1.529 circular | descriptive | d1AngleReadout.json |
| 12 | The angle coordinate is invisible to the dominant variance directions | 4.9 | top 2 components 43.198 deg at 34 percent of variance; chance 45; about 20 components needed to reach 4.28 | descriptive | d1AngleReadout.json |
| 13 | The machinery reports nothing when there is nothing | 4.9, 4.11 | shuffled labels 44.060 deg [42.47, 45.76] against a theoretical 45, loop radius 0.0487 | descriptive | d1AngleReadout.json |

---

## Appendix A — validation and reproducibility

**Reproducing the suite.** The environment is pinned in `envInterp.txt` at the repository root (Python 3.11.14, NumPy 1.26.4, SciPy 1.17.0, scikit-learn 1.8.0). With the anchor data present under `data/`:

```
python paper1/runAll.py            status and provenance of every diagnostic
python paper1/runAll.py d1         run one diagnostic and render its figures
python paper1/runAll.py --verify   re-hash every recorded input and compare
python -m pytest paper1/tests -q   convention and null-calibration tests
```

**Provenance.** Every results file carries the git revision and dirty flag, the UTC timestamp, wall-clock seconds, the seed, interpreter and package versions, the platform, and the SHA-256 and byte count of every input file read. `runAll.py --verify` re-hashes those inputs and reports any that have changed since the run, so a result can never be silently attributed to data it was not computed from.

**Test suite.** Twenty-three tests cover the two failure modes that would invalidate results without producing an error: an incorrect convention, and an uncalibrated null.

- *Conventions* (`tests/testConventions.py`): the ellipticity identities, the axial-doubling reconstruction, inclination from axis ratio, the exclusion of circular-by-construction models, uniformity of retained position angles, and the inverse-elongation scaling of the catalog angle uncertainty, all checked against the real catalog rather than against documentation.
- *Nulls and circular machinery* (`tests/testNulls.py`): the analytic chance floors of 45 and 90 degrees for axial and full-circle quantities, wrap symmetry across the seam, collapse of both error and loop radius under shuffled labels, recovery and radius on synthetic data of known signal strength, the loop-radius shrinkage identity at three signal levels, near-orthogonality of random directions in high dimension, determinism of the split, and boundedness of the error by the period.
- *Input-space operators* (`tests/testTransforms.py`): skipped with an explicit reason until the operators exist, so a missing implementation cannot be mistaken for a pass.

**Figure integrity.** Figure scripts read only the results file and the saved projection array, never the embeddings or labels, so no plotted value can differ from the recorded one.

**Numerical audit.** Each written section is checked against its artifact by an automated comparison of every numeric and string claim before the section is considered complete.

## Appendix B — glossary

| Term | Meaning here |
|---|---|
| anchor | The fixed sample of 48,398 galaxies on which every diagnostic is computed |
| substrate | The embedding a probe reads from: E_img (image tokens only) or E_full (image plus scalar flux and catalog redshift) |
| z-score | Per-dimension standardisation: subtract the mean, divide by the standard deviation |
| probe | A ridge fit reading a label out of an embedding, scored on held-out rows |
| fixed probe | A probe fit once and applied unchanged to other rows or to re-encoded inputs |
| R2 | Fraction of label variance explained; 1 is perfect, 0 is no better than the mean |
| axial quantity | A quantity identified with itself under a half turn, so its period is 180 degrees |
| doubled angle | Regressing (cos 2 theta, sin 2 theta) so an axial quantity becomes a point on a circle |
| circular error | Angular difference wrapped into the quantity's own period |
| chance floor | Median error of a readout carrying no information: 90/k degrees, so 45 for an axial quantity |
| loop radius | Length of the predicted (cos, sin) vector, whose truth counterpart is 1 by construction |
| mean resultant length | A concentration statistic for a set of angles; not the same object as loop radius |
| position angle | Sky direction of a source's long axis, defined modulo 180 degrees |
| ellipticity | Magnitude of the complex shape parameter, hypot(e1, e2) |
| axis ratio | b/a, the ratio of minor to major axis |
| inclination | Tilt of a disk toward the line of sight, arccos(b/a) |
| sigma_PA | Catalog uncertainty on position angle, propagated from the shape component variances |
| SER, DEV, EXP | Tractor model types carrying a fitted ellipticity |
| REX, PSF, DUP | Tractor model types that are circular by construction and carry no fitted ellipticity |
| seeing, psfsize | Width of the point-spread function, in arcsec |
| point-source depth, psfdepth | Survey depth in inverse nanomaggies squared |
| E(B-V) | Galactic extinction along the line of sight |
| bootstrap interval | Percentile interval from resampling held-out predictions |
| partial rank correlation | Spearman correlation after removing the rank of a confounder from both variables |
| evidential tier | Input-space causal, descriptive with a calibrated null, or descriptive |
| provenance block | Git revision, input hashes, seed, environment and timing recorded in every results file |

## Appendix C — artifact index

| Diagnostic | Script | Results | Figures |
|---|---|---|---|
| 1 | `diagnostics/d1AngleReadout.py`, `d1AngleReadoutFigures.py` | `results/d1AngleReadout.json`, `d1AngleReadoutProjection.npy` | `d1Loop`, `d1Elongation`, `d1Nulls`, `d1Invariance`, `d1Heteroscedasticity` |
| 2 | `diagnostics/d2Equivariance.py` | `results/d2Equivariance.json` | |
| 3 | `diagnostics/d3Chirality.py` | `results/d3Chirality.json` | |
| 4 | `diagnostics/d4NuisanceLeakage.py` | `results/d4NuisanceLeakage.json` | |
| 5 | `diagnostics/d5Degradation.py` | `results/d5Degradation.json` | |
| 6 | `diagnostics/d6Decodability.py` | `results/d6Decodability.json` | |
| 7 | `diagnostics/d7ConceptGeometry.py` | `results/d7ConceptGeometry.json` | |
| 8 | `diagnostics/d8StructuredRelations.py` | `results/d8StructuredRelations.json` | |
| 9 | `diagnostics/d9ArtificialRedshift.py` | `results/d9ArtificialRedshift.json` | |

## Appendix D — standing caveats

These five apply to the whole suite and are not repeated in every section.

1. **Label noise.** Catalog labels are measurements with their own errors, not truth. Where a diagnostic scores the model on recovery of noisy labels, the achievable ceiling is unknown. The image interventions are the exception: those transformations are exactly controlled, which is why the causal diagnostics carry disproportionate weight.
2. **Bootstrap intervals** on held-out predictions capture test-set sampling only, not refitting or resplitting variation. They are narrower than total uncertainty.
3. **Photometric redshift labels** are largely colour-derived, so any image-to-redshift result partly rides through image-to-colour.
4. **Sparse-label subsamples** are catalog crossmatches, not random draws, and carry their own selection.
5. **Inference-time configuration** (token budget, pooling) differs from pretraining. Results are properties of the documented recipe and are labelled that way.

## Appendix E — environment and provenance

Analysis environment, recorded in every results file rather than described here from memory: Python 3.11.14, NumPy 1.26.4, SciPy 1.17.0, scikit-learn 1.8.0, on Windows. The full lock is `envInterp.txt` at the repository root; the embedding and re-encoding steps additionally require PyTorch 2.10.0+cu128 and polymathic-aion 0.0.2.

Randomness is seeded throughout at seed 0, and train and test splits use that seed everywhere.

Diagnostics 1 to 9 read the frozen embeddings already on disk; those that transform inputs re-encode through the frozen model on one RTX 5070 Ti laptop GPU at approximately 17 images per second, which is the unit cost for pricing any re-encoding sweep. Diagnostics that only probe existing embeddings are CPU-only.

Per-diagnostic wall-clock cost, from the provenance blocks:

| diagnostic | cost | hardware |
|---|---|---|
| d0 dataset and substrate audit | about 1 second | CPU |
| d1 angle readout characterization | about 2 minutes, including the ten-split sensitivity sweep | CPU |

`python paper1/runAll.py` prints the recorded revision, timestamp and cost for every diagnostic; `--verify` re-hashes every input file against the values stored at run time.

## Appendix F — references

Sources the suite relies on, with what each supports. Conventions are verified numerically against the data in `paper1/tests/testConventions.py` rather than taken on documentation alone.

**Diagnostic 1**

- Krumbein (1939); Fisher, *Statistical Analysis of Circular Data*, Cambridge University Press (1993); Mardia and Jupp, *Directional Statistics*, Wiley (2000) — angle doubling as the standard treatment of axial data, and the mean resultant length as a concentration statistic distinct from the loop radius of Section 4.2.
- DESI Legacy Imaging Surveys, tractor catalog documentation, <https://www.legacysurvey.org/dr10/catalogs/> — the complex-ellipticity parameterisation and its conversions. The identities are verified on all 43,672 retained sources to 1e-9, and inclination to 1e-6.
