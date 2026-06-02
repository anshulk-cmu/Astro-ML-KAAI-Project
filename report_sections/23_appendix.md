## 23. Appendix: thresholds, statistical claims, reproducibility

Every headline in this report rests on a decision rule: a threshold a number had to beat, an interval that had to exclude zero, a sign convention, or a choice of which metric to trust. Scattered through the sections, these rules are easy to lose track of. This appendix gathers them in one place so a reader can audit the logic without re-reading the whole report. Each entry gives the rule, its value, and what it decides. Every value here is measured and matches the committed results files; none is introduced for the first time.

### 23.1 Thresholds and decision rules

The table below is the full set of fixed cutoffs and decision rules the analysis used. Read each row as: this is the number, this is what passing or failing it means.

| Rule | Value | What it decides | Section |
|---|---|---|---|
| Resolution ceiling $\log_2 N$ | $\log_2(48{,}398) = 15.56$ | The largest intrinsic dimension a sample of $N$ points can resolve. Any ID estimate near or above this is suspect; our plateau at $\sim$10-12 sits safely below it. | 6, 7 |
| Astrophysical prior band | 4 to 10 effective axes | The optimistic expected range for an honest multimodal galaxy embedding, from photometry ($\sim$2-5) and spectra ($\sim$3-10). It is a band to compare against, not a prediction. Our ID sits at or just above its top. | 2, 7 |
| ID estimator agreement | Gride-large $\approx 10$, local-MLE $\approx 11.4$, PCA-PR $\approx 11.2$ (E_full) | Three mechanistically-independent estimators must agree near a plateau for the ID headline to hold. They do, which is why the small-scale TwoNN $\approx 16.5$ is read as noise-inflated rather than the answer. | 7 |
| Headline ID and CI (E_full) | TwoNN MLE $16.56$ $[16.42, 16.71]$; plateau $\sim$10-12 | The TwoNN Gamma-posterior 95% interval is reported, but the trustworthy estimate is the larger-scale plateau, not this small-scale value. E_img is similar (TwoNN $16.05$ $[15.91, 16.19]$). | 7 |
| SAE alignment null thr95 | $0.0119$ | A latent's max absolute Spearman with a label must exceed this (the 95th percentile of the label-shuffle null) to count as aligned-significant. Count passing: 717. | 13 |
| SAE alignment null thr99 | $0.0129$ | The stricter 99th-percentile cutoff for aligned-strong. Count passing: 690. The max alignment, $0.279$, is about 23 times thr95. | 13 |
| SAE seed-stability cosine | $\geq 0.6$ | A latent's decoder direction must reappear in at least half of the other four seeds at cosine $\geq 0.6$ to be called seed-stable. Aligned-and-stable count: 335. Median best cross-seed cosine across active latents: $0.460$. | 13 |
| SAE novelty cut | $> 0.7$ | Fraction of a latent's activation variance the six labels cannot linearly explain. Above $0.7$, plus seed-stable, plus not label-aligned, defines an alien candidate. Count: 59 (correlational only). | 13 |
| $\beta_1$ persistence threshold | $> 0.1$ | A diameter-normalised loop must persist beyond this to be a significant hole, not noise. Significant loops essentially never recur across 10 subsamples (mean $\leq 0.2$, max persistence $0.105$ Euclidean / $0.138$ Fermat / $0.081$ diffusion), so $\beta_1 = 0$. | 15 |
| $\beta_0$ split rule | Cut longest MST edges; piece must exceed size 1 | Cutting the longest minimum-spanning-tree edges peels off only single outliers ($[48{,}397, 1]$, then $[48{,}396, 1, 1]$), so the giant body stays whole and $\beta_0 = 1$. | 15 |
| delta-ID CI excludes zero | $\Delta\text{ID} = 1.66$ $[1.38, 1.96]$ | The passive-minus-star-forming intrinsic-dimension difference is real only if its CI excludes zero. It does, so passive galaxies sit on a slightly higher-dimensional sub-manifold (passive $17.19$ $[17.01, 17.43]$ vs star-forming $15.53$ $[15.39, 15.73]$, small-scale absolute values). | 16 |
| GMM colour cut | g-r $= 1.012$ (means $0.78$, $1.24$) | The two-component Gaussian-mixture split point on g-r that defines passive ($> 1.012$, $n = 22{,}614$) versus star-forming ($n = 25{,}782$). | 16 |
| Ollivier-Ricci sign | sign is meaningful; mean $= +0.155$ | Positive curvature means locally clustered neighbourhoods, negative means a bridge or saddle. Mean is positive with only 4.2% negative edges (p5 $+0.006$, p95 $+0.319$), read as mostly-positive curvature with weak localised branching. | 14 |
| Forman-Ricci role | rank-only (not signed) | On a kNN graph Forman is structurally negative (degree dominates), so its sign is not a clean curvature indicator; it is used only to rank candidate bridge edges. Ollivier is the trustworthy signed measure. | 14 |
| delta-hyperbolicity anchoring | AION $0.0136$ vs Gaussian anchor $0.0177$, tree $0.0079$ | Tree-likeness is read against anchors, not in absolute terms. AION falls below the matched-covariance Gaussian (mildly more tree-like than random) but well above the synthetic tree (not a tree). | 14 |
| Geometry-of-concepts significance | measured $0.072$ vs permutation null p95 $0.084$ | The featured-to-spiral hierarchy cosine must beat the 95th percentile of the permutation null to count. It does not ($0.072 < 0.084$), so the test returns an honest null. | 17 |
| Metric battery roles | Fermat primary, Euclidean control | Intrinsic metrics (Fermat, isomap, diffusion) give several times more distance contrast than Euclidean (Fermat RDR $\sim$260-281 vs Euclidean $\sim$14-15), so Fermat is the primary metric for topology and curvature, with Euclidean kept as a control and cosine as a middling control. | 5 |

A few rows deserve a plain-language reminder of why they are framed as a threshold rather than a point estimate. Three kinds of decision rule appear here. The first kind is a null comparison: a measured number is meaningful only if it beats what scrambled, structureless data would produce (the SAE alignment thresholds and the geometry-of-concepts permutation null both work this way). The second kind is an interval test: a difference is real only if its confidence interval misses zero (the delta-ID rule). The third kind is a sign convention: the number's sign carries the physics (Ollivier-Ricci positive versus negative), so we state which measure's sign we trust and which we do not (Forman is rank-only). Keeping these three logics separate is what lets the report say "measured" and "interpreted" honestly: the thresholds are where measurement stops and reading begins.

The next four paragraphs walk through the load-bearing rules in more detail, because a reader auditing a headline needs to know not just the cutoff but why that cutoff and not another.

The $\log_2 N$ ceiling and the prior band together bracket the intrinsic-dimension claim from above and from the optimistic side. The ceiling is a hard fact about resolution: with $N$ points you cannot reliably tell apart structure finer than about $\log_2 N$ independent directions, because the number of points needed to populate a $d$-dimensional neighbourhood grows exponentially in $d$. At $N = 48{,}398$ that ceiling is $15.56$. Our plateau estimate of about 10 to 12 sits below it, so the sample can in principle resolve the dimension we report; the small-scale TwoNN value of $16.56$ sits essentially at the ceiling, which is one more reason to treat it as noise-inflated rather than as the answer. The 4-to-10 prior band is a softer comparison: it is where astrophysics expected the dimension to land if the embedding were honest, derived from the rough effective dimensionality of clean photometry and galaxy spectra. Our result sits at or just above its top edge, so "low-dimensional" is fair but "as low as the best case" is not, and we say so rather than rounding down.

The SAE thresholds are null comparisons, and the two cutoffs do different jobs. thr95 ($0.0119$) is the 95th percentile of the label-shuffle null: a latent whose best label correlation exceeds it would be reached by chance fewer than one time in twenty, so it counts as aligned. thr99 ($0.0129$) is the stricter 99th-percentile version for the aligned-strong count. Both are tiny because shuffling the labels destroys almost all correlation, so even a weak real alignment stands out; the largest observed alignment, $0.279$, clears thr95 by roughly a factor of 23, which is why the strongest concepts (colour and morphology) are not in doubt. The seed-stability cosine ($0.6$) and the novelty cut ($0.7$) then layer two further filters on top of alignment: stability asks whether a direction is a property of the model or of one lucky random initialisation, and novelty asks whether the labels can explain the direction at all. The alien candidates pass stability and novelty while failing alignment, which is exactly the corner of the space where a genuinely new, model-specific axis would live, and exactly the corner where we have the least external check, so we flag those 59 as correlational only and never claim more.

The topology and curvature rules are about not mistaking noise for structure. For $\beta_1$, the persistence cut of $0.1$ exists because persistent homology always produces a swarm of short-lived "loops" that are sampling noise (thousands of them, per the mean-bars counts), and only a loop that survives well past that cut, and recurs across independent subsamples, and survives a change of metric, counts as a real hole. None did. For $\beta_0$, the MST rule is conservative in the other direction: we cut the longest connecting edges one at a time and watch what falls off, and because only single points detach while the body of tens of thousands stays whole, the cloud is one piece and not a hidden red/blue split. For curvature, the Ollivier sign is the whole point of using it: a signed local curvature tells clustered neighbourhoods (positive) from bridges (negative), and the mean of $+0.155$ with only $4.2\%$ negative edges is what licenses the "mostly positive, weakly branching" reading. Forman is deliberately demoted to ranking only, because its sign is forced negative by graph degree and would otherwise manufacture a false branching signal.

The delta-ID rule is the one interval test that carries a comparative claim, and it is framed as "excludes zero" for a reason. The absolute per-population intrinsic dimensions ($17.19$ passive, $15.53$ star-forming) are small-scale TwoNN values and share the same noise inflation as the global TwoNN, so neither absolute number is the estimate of a population's true dimension. What survives the noise is the difference, because both populations are measured the same way on the same scale, so the shared bias largely cancels. The bootstrap interval on that difference, $[1.38, 1.96]$, misses zero, so the ordering (passive slightly higher-dimensional) is the trustworthy signal even though the absolute heights are not.

### 23.2 Confidence intervals on the headline numbers

Every headline carries an uncertainty, and the report never presents a point estimate as if it had none. The intervals below are the ones a reader should quote alongside each claim. The intrinsic-dimension intervals are Gamma-posterior 95% intervals from the TwoNN MLE (small-scale, so the plateau is the real estimate); the probe intervals are 1000-times bootstrap 95% intervals; the delta-ID interval is a bootstrap interval over the stratified estimate.

| Headline quantity | Estimate | 95% interval | Note |
|---|---|---|---|
| Decodable: stellar mass (E_img) | $R^2 = 0.721$ | $[0.659, 0.772]$ | Non-input; image-only. The cleaner physics evidence. |
| Decodable: sSFR (E_img) | $R^2 = 0.760$ | $[0.692, 0.816]$ | Non-input; image-only. |
| Decodable: redshift (E_img) | $R^2 = 0.800$ | $[0.792, 0.809]$ | Mostly photo-z, so partly image-to-colour-to-redshift. |
| Decodable: g-r (E_img) | $R^2 = 0.958$ | $[0.957, 0.960]$ | Strongest single decode. |
| Decodable: spiral (E_img) | $R^2 = 0.252$ | $[0.153, 0.334]$ | Weakest; $n = 3{,}034$ only. |
| delta-ID (passive minus SF) | $1.66$ | $[1.38, 1.96]$ | Excludes zero. |
| Max SAE alignment | $0.279$ | (vs null thr95 $0.0119$) | About 23x the null. |

### 23.3 Reproducibility note

The analysis is built to be re-run from committed artefacts. Three properties make it auditable.

First, the geometry runs on the committed embeddings and labels, not on anything regenerated on the fly. The embeddings (E_full and E_img, one 1024-vector per galaxy, $N = 48{,}398$) and the label tables are fixed inputs, z-scored per dimension before every geometry arm. A second run on the same inputs reproduces the same numbers up to the documented random elements (the 80/20 probe split, the SAE seeds, the topology subsamples), each of which uses a fixed seed where one applies.

Second, every estimator was validated on synthetic data of known answer before it was trusted on AION. The intrinsic-dimension estimators were run at the matched $N = 48{,}398$ on a sphere (truth 5), a swiss roll (truth 2), and a plane (truth 5), and they recovered the known dimensions (for example TwoNN $4.96$, $1.95$, $5.08$); a two-blob control was included to show the estimator inflates on a near-zero-diameter cluster (TwoNN $\approx 212$) and must be excluded. Curvature used a matched-covariance Gaussian anchor and a synthetic tree as brackets. Topology and concept tests used shuffle and permutation nulls. The validation logic is what licenses the readings; we did not trust any tool we had not first watched recover a known answer.

Third, the figure scripts read only the committed results files. No figure recomputes a result; each plots numbers already written to disk, so a figure cannot disagree with the table that produced it.

The committed results files, and what each contains, are:

| File | Contents |
|---|---|
| `results/sanity.json` | NaN/Inf checks, embedding norms, per-dimension std range, raw PCA participation ratio and cumulative-variance counts for E_full and E_img. |
| `results/metric.json` | Distance-concentration diagnostics (RDR, NN/mean) across five metrics on a 2,000-point subsample, with connectivity checks. |
| `results/intrinsicDim.json` | Intrinsic-dimension estimates (TwoNN with CI, Gride scale curve, local Levina-Bickel MLE, PCA participation ratio) for AION E_full/E_img and for the synthetic validation manifolds. |
| `results/diffusionMap.json` | Diffusion-map eigenvalues, spectral gaps, connected-component count, and the harmonic screen for both embedding sets. |
| `results/diffcoords_E_full.npy`, `results/diffcoords_E_img.npy` | The per-galaxy diffusion coordinates (48,398 x 10) used for the coloured embeddings and label correlations. |
| `results/probes.json` | RidgeCV probe $R^2$ with bootstrap CIs (image-only and multimodal), the modality ablation, the cross-matched physical-subset probes, the disentanglement angles, and kNN purity. |
| `results/sae.json` | SAE health (FVU, alive fraction) per seed and expansion, the L0-vs-FVU frontier, the concept-scoring summary (null thresholds, counts, alien candidates), and the named-concept breakdown. |
| `results/sae_acts0.npy`, `sae_align.npy`, `sae_best_label.npy`, `sae_novelty.npy`, `sae_recurrence.npy` | Per-latent SAE arrays: activations (seed 0), alignment scores, best-matching label, novelty, and cross-seed recurrence, supporting the alignment-novelty and named-concept figures. |
| `results/curvature.json` | delta-hyperbolicity (median, mean, p95) for AION, the Gaussian anchor, and the tree validation; Ollivier-Ricci signed summary; Forman-Ricci ranking statistics. |
| `results/topology.json` | $\beta_0$ via MST edge cuts (split sizes) and $\beta_1$ via persistent homology across 10 subsamples under three metrics (mean bars, max persistence, significant-loop counts). |
| `results/stratifiedId.json` | The g-r GMM split (cut, component means), per-population TwoNN ID with CIs, and the delta-ID with its CI and the excludes-zero flag. |
| `results/geometryOfConcepts.json` | The featured-to-spiral parent-child cosine, the permutation-null 95th percentile, and the significance flag. |

What this appendix does not contain is any new measurement. Every threshold, interval, and file listed here was used to support a claim earlier in the report; collecting them in one place is meant to make the report checkable, not to add to it. A reader who wants to test whether a headline is defensible can take its row from the tables above, open the named results file, and confirm the number for themselves.
