# The Native Representational Geometry of an Astronomical Foundation Model: A Geometry-and-Concept Study of AION-1's Galaxy Embedding

## 0. Front matter

### Thesis in one paragraph

A foundation model is a large neural network trained once on a flood of unlabeled data, then reused for many tasks without retraining.

AION-1 is one such model for astronomy. It reads galaxy images and catalogue measurements and turns each galaxy into a list of 1024 numbers, a point in a 1024-dimensional space we call the embedding. Its authors said the model "organises objects along physically meaningful directions," but they never measured the shape of that space.

This report measures it. We take the model frozen, with no extra training, and ask three plain questions about the cloud of 48,398 galaxy points it produces. How many independent directions does the cloud really use (its intrinsic dimension)? What is its shape: one connected body or separate clusters, flat or curved, holed or simply connected? And does the model carry concepts of its own, beyond the human labels we already have?

The short answer is that AION-1's galaxy space is a single, smooth, low-dimensional continuum, about 10 to 12 real directions out of 1024, gently curved and not tree-like, that encodes real galaxy physics, including properties it was never told, strongly enough to read back off the image alone. That is the whole report in one sentence. The rest earns it.

### Executive summary: the four headline findings

Before the findings, the scope in one breath. We analyse $N = 48{,}398$ galaxies (drawn from Galaxy Zoo DESI, redshift $0 < z < 1$, fixed random seed; 96.8% of a 50,000 draw had usable four-band images). The model is AION-1-Large, 800 million parameters, frozen and never fine-tuned, with no morphology labels in its training. It emits one vector per image token, and we mean-pool those into a single 1024-number vector per galaxy. We study two versions of that vector: $E_{\text{full}}$, which fuses image plus photometry plus redshift, used for the geometry; and $E_{\text{img}}$, image only, the leakage-free set used for concept probes, because anything fed in as an input is trivially decodable and would make a "concept axis" circular. Every distance below is computed after z-scoring each of the 1024 dimensions (subtracting its mean, dividing by its standard deviation) so that no single high-variance dimension dominates.

We organise the work around four results. Each carries a number, a confidence interval or null where we have one, and an explicit tag: "measured" means a direct readout from the data, "interpreted" means an inference we draw from the measurement and would defend but cannot prove.

One discipline runs under all four. Every estimator was first run on synthetic data of known answer at the same sample size, so a number is only trusted on AION-1 after it recovered the truth on a control. The dimension estimators were checked on a sphere, a swiss roll, and a plane; the curvature and topology tools were checked against a matched random cloud and a synthetic tree; the concept scores were checked against label-shuffle and permutation nulls. When a tool failed its control (the small-scale TwoNN inflating on tight clusters, Forman-Ricci going structurally negative on a neighbour graph), we say so and do not lean on it. That is why some readings below are demoted to controls rather than estimates.

**Finding 1: the embedding uses only about 10 to 12 real directions, not 1024.** The intrinsic dimension is the count of independent knobs you would need to describe a galaxy's position in the cloud, which can be far below the ambient 1024. We estimated it four ways, each resting on a different piece of math, after first checking every estimator on synthetic shapes of known dimension at the same sample size.

On AION-1, three estimators that work at manifold scale agree: the Gride scale curve flattens near 10, the local Levina-Bickel maximum-likelihood estimator plateaus near 11.4, and the linear PCA participation ratio is about 11 (all measured). The smallest-scale TwoNN reading of about 16.5 is inflated by sampling noise and is not the estimate; the plateau at larger scale is (interpreted). All of these sit below the resolution ceiling of $\log_2 N = 15.56$ that a sample of 48,398 points can support.

Compressing 1024 nominal dimensions down to about 11 is a large reduction, and the closeness of the nonlinear estimate (about 10) to the linear one (about 11) tells us the manifold is only weakly curved at large scale (interpreted). Honest caveat: this dimension sits at or just above an optimistic astrophysical prior band of 4 to 10, so "low-dimensional" is true but not as low as the best case.

**Finding 2: it is one smooth, simply-connected continuum, not a set of discrete clusters.** Three independent geometric tests agree. The diffusion-map eigenvalue spectrum decays smoothly with no dominant gap (a gap would mean the cloud breaks into separate groups), and there is a single connected component (measured).

The number of separate pieces, the topological invariant $\beta_0$, is 1: cutting the longest links of the minimum spanning tree peels off only lone outliers while the roughly 48,000-point body stays whole, so this is not a two-piece red/blue split (measured). The number of independent loops or holes, $\beta_1$, is 0: significant loops essentially never recur across ten independent subsamples or survive a change of distance metric (measured, from 2,000-point subsamples).

We read the continuum as the model's encoding of the known morphological and colour sequences of galaxies, which themselves vary continuously (interpreted).

**Finding 3: physics is strongly decodable from the image alone, and the concept directions are separated.** A linear probe is a single straight-line map fit from the 1024 numbers to a physical property; its $R^2$ is the fraction of held-out variance it explains, where 1 is perfect and 0 is no better than guessing the mean.

Probing the image-only embedding, where redshift and colour are never given as inputs so decoding them is a true inference, we measure stellar mass $R^2 = 0.721$ (95% interval 0.659 to 0.772, $n = 3{,}728$) and specific star-formation rate $R^2 = 0.760$ (0.692 to 0.816, $n = 4{,}760$), both from pixels alone (measured). Colours read back at 0.91 to 0.96 and morphology vote fractions at 0.68 to 0.79 (measured). The model also separates concepts more than the labels' own correlations would force: the angle between probe directions exceeds the label-correlation null by up to about 19 degrees (measured).

We read this as a visual representation that has internalised galaxy physics it was not handed (interpreted). Honest caveat: the redshift label is mostly photometric (colour-derived), so image-to-redshift is partly image-to-colour-to-photo-z; the mass and star-formation results are the cleaner evidence because those quantities were never inputs.

**Finding 4: sparse autoencoders surface 335 seed-stable, physics-named concept directions plus 59 candidate "alien" directions.** A sparse autoencoder learns an over-complete dictionary so that each galaxy vector is rebuilt from only a few active concept units (here 32 of them), which lets the embedding reveal its own axes instead of us probing for human-named ones.

After scoring each unit against six labels with a label-shuffle null (the 95th-percentile threshold is 0.0119), we find 717 units aligned to a known property and 335 of those also stable across five random training seeds (measured). The strongest single alignment is 0.279, about 23 times the null (measured); colour and morphology are the clearest single concepts, while redshift spreads over many weak units.

Separately, 59 units are seed-stable and high-novelty (most of their activation variance is not linearly explained by any label we have) yet not aligned to any label, the "alien" candidates (measured). We flag these hard: they are correlational only, no causal or ablation test was run, so they are the most interesting and the least certain result in the report (interpreted).

### The honest one-line verdict

AION-1's galaxy embedding is a low-dimensional (about 10 to 12), single, simply-connected, mostly positively curved continuum that encodes real galaxy physics strongly, with weak localised branching and a small set of candidate concepts beyond the human labels; the dimension sits at or just above an optimistic prior, the redshift signal is partly photo-z routed through colour, and the alien concepts are correlational and untested for cause.

### How to read this report

The audience is one technical reader who may know astronomy or machine learning but probably not both, so every term is defined in plain words the first time it appears, and we explain both the math and the physics.

We separate two kinds of statement throughout. A "measured" claim is a direct readout from data and carries its uncertainty (a confidence interval, a bootstrap range, a null threshold, or an anchor to compare against). An "interpreted" claim is an inference, flagged with words like "we read this as," that we will defend but cannot prove from this data alone. Where the data cannot decide, we say so.

The sections build in order. Sections 1 to 4 set up the question, the galaxy physics a faithful embedding should reflect, the model itself, and the data and preprocessing. Section 5 explains why the choice of distance matters in high dimensions and picks the metric we use downstream. Sections 6 and 7 define and validate the intrinsic-dimension estimators and report Finding 1. Sections 8 and 9 give the diffusion-map shape and read physics off it.

Sections 10 and 11 cover decodability and concept separation (Finding 3). Sections 12 and 13 cover the sparse autoencoder and its concepts (Finding 4). Sections 14 through 17 cover curvature, topology (Finding 2), population-dependent dimension, and one honest null result. Section 18 shows real galaxy cutouts along the recovered axes. Sections 19 to 21 synthesise, list limitations, and state conclusions. Section 22 is a glossary of every term and symbol, and Section 23 is an appendix of every threshold and statistical decision rule with a reproducibility note.

If you want only the bottom line, read this front matter and Section 19. If you doubt a number, Section 23 and the glossary point you to the exact result file.

A note on what this study is and is not. The embedding is a static snapshot of present-day galaxies, so it shows the density of populations that exist now, not a movie of galaxies evolving over time. We never claim the model "learned evolution." The question of whether the model rediscovered the classic Hubble tuning-fork sequence of galaxy shapes is a control, not the thesis; the thesis is the native geometry and the concepts.

### Conventions and a few symbols you meet early

So you do not have to hunt later, here are the conventions used everywhere and the handful of symbols that appear before their home section defines them in full. Each is restated in plain words where it is first used in earnest, and all of them live in the Section 22 glossary.

| Symbol or term | Plain meaning |
|---|---|
| $N = 48{,}398$ | Number of galaxies analysed. |
| $d = 1024$ | Ambient embedding dimension, the length of each galaxy's vector. |
| $E_{\text{full}}$ | Embedding from image plus photometry plus redshift; used for geometry. |
| $E_{\text{img}}$ | Image-only embedding; the leakage-free set used for concept probes. |
| Intrinsic dimension | Count of independent directions the cloud actually uses, far below $d$. |
| $R^2$ | Fraction of held-out variance a linear probe explains (1 perfect, 0 mean-guess). |
| $\beta_0,\ \beta_1$ | Number of separate pieces, and number of independent loops or holes. |
| "measured" | A direct readout from the data, reported with its uncertainty. |
| "interpreted" | An inference we draw from a measurement and would defend but not prove. |

Two reading rules follow from the table. First, treat any number without an interval, anchor, or null as still measured but read its uncertainty in the owning section, because the headline here is a summary. Second, when a sentence says "we read this as" or carries an "(interpreted)" tag, that is the boundary between what the data shows and what we think it means; the report keeps that boundary visible on purpose.

### Figure index

Nineteen figures carry the quantitative load. Each is embedded in the section that owns it, with a full caption naming every axis, colour scale, legend, and reference line. Figures are numbered by the two-digit number in their filename. This index is the map.

| Figure | File | One-line description |
|---|---|---|
| 1 | figures/01_id_synthetic_validation.png | Intrinsic-dimension estimators recover known dimensions on synthetic manifolds (the validation check). |
| 2 | figures/02_id_aion_scale_curves.png | AION intrinsic dimension versus neighbour scale (Gride and local-MLE), showing the 10 to 12 plateau. |
| 3 | figures/03_metric_concentration.png | Distance contrast (RDR) and nearest-neighbour ratio across five candidate metrics. |
| 4 | figures/04_diffusion_spectrum.png | Diffusion-map eigenvalue spectrum and consecutive gaps (smooth decay, no dominant gap). |
| 5 | figures/05_diffusion_full_labels.png | Diffusion embedding (coords 1 by 2) coloured by six full-sample labels. |
| 6 | figures/06_diffusion_sparse_physics.png | Same embedding coloured by stellar mass, sSFR, and Sersic index (cross-matched subset). |
| 7 | figures/07_probe_decodability.png | Linear-probe $R^2$ with 95% intervals, image-only, full-sample and subset labels. |
| 8 | figures/08_modality_ablation.png | Image-only versus multimodal decodability, exposing input leakage. |
| 9 | figures/09_disentanglement_angles.png | Probe-direction angle versus the label-correlation null (the orthogonality excess). |
| 10 | figures/10_probe_pred_vs_true.png | Predicted versus true on held-out data for redshift, mass, and sSFR. |
| 11 | figures/11_sae_frontier.png | Sparse-autoencoder reconstruction error (FVU) versus sparsity level $k$. |
| 12 | figures/12_sae_named_concepts.png | Counts of aligned and seed-stable autoencoder features per named concept. |
| 13 | figures/13_sae_alignment_novelty.png | Feature alignment versus the shuffle null, and alignment versus novelty, marking the alien candidates. |
| 14 | figures/14_curvature.png | Delta-hyperbolicity against anchors plus the Ollivier-Ricci curvature summary. |
| 15 | figures/15_topology.png | $\beta_0$ via minimum-spanning-tree cuts and $\beta_1$ via loop persistence versus threshold. |
| 16 | figures/16_stratified_id.png | The g-r colour split plus passive versus star-forming intrinsic dimension and the delta-ID. |
| 17 | figures/17_pca_variance.png | Linear PCA cumulative variance, the linear baseline for the dimension result. |
| 18 | figures/18_manifold_galaxy_montage.png | Real galaxy cutouts sampled along the colour and morphology axes. |
| 19 | figures/19_persistence_diagrams.png | H1 persistence diagrams under the three metrics with a bootstrap noise band, confirming $\beta_1 = 0$. |

With the map in hand, Section 1 states the question and why, despite the model's own claims, it is still open.
