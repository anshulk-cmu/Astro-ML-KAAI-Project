# FACTS SHEET — every number is measured and committed. Do not invent or alter.

Cross-check against the JSON in `results/` named in each block. Tag values "measured".
Interpretations are flagged "[interp]" and you may expand them faithfully.

## 0. Dataset and representation
- N = 48,398 galaxies analysed (sampled 50,000 Galaxy Zoo DESI galaxies, 0 < z < 1, fixed seed;
  96.8% had usable 4-band images). Ambient embedding dimension d = 1024.
- Model: AION-1-Large, 800M parameters, frozen (no fine-tuning). One encoder transformer trained
  by multimodal masked modelling, self-supervised, no morphology labels. Emits per-token vectors;
  one 1024-vector per galaxy by MEAN POOLING over tokens.
- Two embedding sets:
  - E_full = image (g,r,i,z) + photometry (g,r,z flux) + redshift, fused as input tokens.
  - E_img = image only. This is the LEAKAGE-FREE set: redshift/colour are NOT inputs, so probing
    them is a real inference, not reading back an input.
- log2(N) = log2(48398) = 15.56. This is the rough resolution ceiling: a sample of N points can
  only resolve an intrinsic dimension up to about log2(N).

## 1. Labels (what we compare the geometry against)
- Full-N (~48,398): redshift (MOSTLY photo-z; spectroscopic z for only 6,883), magnitudes
  mag_g/mag_r/mag_z giving colours g-r and r-z, and Galaxy-Zoo vote fractions smooth, featured,
  merger. Vote fractions are predicted by a separate CNN (Zoobot/EfficientNet), accurate to ~5-10%.
- Branch-conditional (n = 3,034): spiral-arm fraction, strong-bar fraction. Defined only for
  featured, non-edge-on disks (the Galaxy-Zoo decision tree), so the sample is small by design.
- Cross-matched physical subset: stellar mass log M*/Msun (n = 3,728), specific star-formation
  rate sSFR (n = 4,760), Sersic index n (n = 3,730). These come from external SED-fit / structural
  catalogues, available for a few thousand galaxies only.
- Preprocessing for all geometry: z-score each of the 1024 dimensions (subtract mean, divide by std).
  Reason: per-dimension std spans a 23x range, so without z-scoring a few large-variance dimensions
  would dominate every Euclidean distance. Embedding vector norms are tight (~48 for E_full,
  ~49.6 for E_img, about 5% spread), so the cloud sits near a thin spherical shell.

## 2. Sanity (results/sanity.json), full 48,398
- E_full: NaN = 0, Inf = 0. Norm 48.37 +/- 2.52 (min 34.4, max 56.5). Per-dim std in [0.129, 2.93].
  PCA participation ratio (RAW, not z-scored) = 7.40. Cumulative variance: 50% in 3 PCs, 90% in 25,
  95% in 45, 99% in 143.
- E_img: norm 49.58 +/- 2.46. PCA participation ratio (raw) = 8.16. 50% in 3 PCs, 90% in 25,
  95% in 44, 99% in 137.
- NOTE: the participation ratio computed on Z-SCORED data (reported in the ID arm) is ~11, higher
  than the raw 7.4, because z-scoring spreads variance over more dimensions. Report both honestly.

## 3. Metric concentration diagnostic (results/metric.json)
2,000-point subsample, z-scored, k = 15 nearest neighbours. RDR = (Dmax - Dmin)/Dmin over all
pairwise distances (HIGHER = more contrast, LESS concentrated). NN/mean = mean nearest-neighbour
distance over mean pairwise distance (LOWER = more concentrated). All graphs connected (inf = 0).
- E_full: euclidean RDR 14.69 (NN/mean 0.433); cosine 76.80 (0.193); isomap 42.12 (0.179);
  fermat_p2 281.04 (0.164); diffusion 95.01 (0.269).
- E_img: euclidean 14.16 (0.430); cosine 79.36 (0.193); isomap 44.46 (0.178); fermat_p2 259.14
  (0.164); diffusion 99.16 (0.267).
- [interp] Raw Euclidean is the most concentrated (lowest contrast). Intrinsic metrics
  (isomap/Fermat/diffusion) give several times more contrast. This motivates using Fermat as the
  primary metric for the topology/curvature arms, with Euclidean kept as a control. Deviation from
  prior expectation: cosine did NOT concentrate like Euclidean here (its RDR ~77 is high), so cosine
  is a middling control, not a failure.

## 4. Intrinsic dimension (results/intrinsicDim.json), full 48,398, z-scored, GPU kNN to k=512
Estimators (all implemented directly): TwoNN (ratio of 2nd to 1st neighbour distance; unbiased MLE
d = (N-1)/sum log mu, with a Gamma-posterior 95% interval and a 200x bootstrap, plus the original
linear fit); Gride (generalised ratio over neighbour ranks n1, giving an ID-vs-scale curve);
local Levina-Bickel MLE (per-point ID from K neighbour distances, swept over K); PCA participation
ratio (a LINEAR baseline, (sum lambda)^2 / sum lambda^2).
- SYNTHETIC VALIDATION (matched N = 48,398, run BEFORE trusting AION):
  - sphere id5 (truth 5): TwoNN MLE 4.96 [4.91, 5.00]; Gride@n1=256 = 4.83; local-MLE@K=226 = 4.88;
    PCA-PR = 6.00 (PCA-PR reads slightly high on a curved manifold; it is a linear spread measure).
  - swiss roll id2 (truth 2): TwoNN 1.95 [1.94, 1.97]; Gride@256 = 1.93; local-MLE@226 = 1.96;
    PCA-PR = 2.98.
  - linear id5 (truth 5): TwoNN 5.08 [5.03, 5.12]; Gride@256 = 4.97; local-MLE@226 = 5.02;
    PCA-PR = 4.99.
  - twoblobs (no manifold, two tight clusters): TwoNN ~212 (a control showing the estimator inflates
    on a near-zero-diameter cluster; not a real dimension).
- AION E_full: TwoNN MLE 16.56 [16.42, 16.71]. Gride curve DECREASES with scale: 16.56 (n1=1) ->
  12.09 (n1=32) -> 11.44 (n1=64) -> 10.10 (n1=256). Local-MLE: 12.97 (K=10) -> peak ~13.1 (K=14) ->
  11.40 (K=226). PCA participation ratio (z-scored) = 11.18.
- AION E_img: TwoNN 16.05 [15.91, 16.19]. Gride 16.05 -> 9.89 (n1=256). Local-MLE 12.71 -> 11.16
  (K=226). PCA-PR (z-scored) = 11.93.
- [interp] HEADLINE: ID ~ 10-12. Three mechanistically-independent estimators agree near the plateau
  (Gride-large ~10, local-MLE ~11.4, PCA-PR ~11), all below the log2(N) ~ 15.6 ceiling. The small-scale
  TwoNN ~16.5 is noise-inflated (1st/2nd-neighbour distances pick up sampling noise); the plateau at
  larger scale is the estimate. Large-scale nonlinear ID (~10) is close to the linear PCA-PR (~11),
  which means WEAK curvature at manifold scale (a small global-minus-local gap). Compression from
  1024 to ~11 is large. ID sits AT or JUST ABOVE the optimistic astrophysical 4-10 prior band.

## 5. Diffusion map (results/diffusionMap.json + results/diffcoords_*.npy)
Full 48,398, z-scored, anisotropic alpha = 1 (Laplace-Beltrami normalisation, so the recovered
geometry does not depend on sampling density), self-tuning local-scaling bandwidth (sigma_i = distance
to the 7th neighbour), kNN graph k = 64, top 30 eigenvectors.
- E_full: 1 connected component. Eigenvalues lambda_0 = 1.000 (trivial constant mode), lambda_1 =
  0.9989, lambda_2 = 0.9772, lambda_3 = 0.9607, lambda_4 = 0.9598, ..., lambda_29 = 0.8213. All
  consecutive gaps are small (largest ~0.0217); NO dominant spectral gap.
- E_img: 1 connected component; very similar spectrum (largest gap ~0.0276).
- Harmonic screen (regress coord_k on degree-2 polynomials of coords 1-3): for E_full, coord 3 is
  90.3% explained and coord 4 is 94.0% explained by lower coords, so they are HARMONICS (repeats of
  the same geometry), not new axes.
- Diffusion coordinates (diffcoords_*.npy, 48,398 x 10), |Spearman| with labels (measured):
  - dc0: weak global mode, |rho| < 0.07 with every label (a slow mixing direction; ~94% of points
    participate, so a broad gradient, not an outlier spike).
  - dc1: smooth +0.50, featured -0.47, redshift +0.45, mass -0.27. Reads as a MORPHOLOGY/REDSHIFT axis.
  - dc2: g-r +0.57, r-z +0.47, sSFR -0.43, mass +0.33. Reads as a COLOUR/STAR-FORMATION axis.
  - dc3, dc4: harmonics (per the screen above).
- [interp] Smooth eigenvalue decay with no gap, plus a single connected component, says the cloud is
  one continuous body, not separate clusters. The physics-bearing axes are dc1 and dc2.

## 6. Probes (results/probes.json)
RidgeCV (ridge regression with cross-validated penalty, alphas 1e-2..1e4), 80/20 train/test split,
seed 0, 1000x bootstrap for the 95% CI. R^2 is the fraction of test-set variance the linear probe
explains (1 = perfect, 0 = no better than the mean).
- E_img (image-only, LEAKAGE-FREE), full N: redshift R^2 = 0.800 [0.792, 0.809]; g-r = 0.958
  [0.957, 0.960]; r-z = 0.911 [0.903, 0.918]; smooth = 0.792 [0.782, 0.802]; featured = 0.794
  [0.783, 0.804]; merger = 0.681 [0.658, 0.702]; spiral = 0.252 [0.153, 0.334] (n=3,034); bar =
  0.554 [0.476, 0.625] (n=3,034).
- Modality ablation (image-only vs multimodal, where redshift/flux ARE inputs to E_full): redshift
  E_img 0.800 vs E_full 0.976; g-r 0.958 vs 0.989; r-z 0.911 vs 0.968. The E_full lift on its own
  inputs is partly leakage, so E_img is the honest measure.
- Cross-matched physical subset, image-only E_img: stellar mass R^2 = 0.721 [0.659, 0.772] (n=3,728);
  sSFR = 0.760 [0.692, 0.816] (n=4,760); Sersic n = 0.664 [0.608, 0.713] (n=3,730). For E_full
  (has flux+z): mass 0.870, sSFR 0.848, Sersic 0.668.
- Disentanglement: for each pair of probe directions, theta = angle between the two ridge weight
  vectors; null = arccos(correlation of the two labels). excess = theta - null. excess > 0 means the
  learned directions are MORE orthogonal than the labels' own correlation forces (genuine
  separation). Values (theta / null / excess, degrees):
  redshift-smooth 93.0 / 73.7 / +19.4; redshift-r_z 69.2 / 50.7 / +18.5; g_r-r_z 40.3 / 25.2 / +15.1;
  redshift-g_r 52.2 / 39.1 / +13.1; g_r-smooth 90.5 / 78.8 / +11.7; r_z-smooth 93.3 / 84.3 / +9.0;
  r_z-merger 91.9 / 86.2 / +5.7; g_r-merger 88.1 / 85.2 / +2.9; redshift-merger 82.5 / 84.5 / -2.0;
  smooth-merger 90.4 / 95.9 / -5.6. Most excesses are positive; merger pairs are near zero.
- kNN purity (model-free, E_full, k = 20): for galaxies with smooth fraction > 0.7, 99.1% of their
  20 neighbours are also smooth (n = 34,289); for featured > 0.7, 71.4% (n = 2,224).
- [interp] Physics is strongly decodable, including NON-INPUT mass and sSFR from the IMAGE ALONE
  (0.72 and 0.76). Caveat: redshift label is mostly photo-z (colour-derived), so image->redshift is
  partly image->colour->photo-z; the image->mass/sSFR results are the cleaner evidence. Concept
  directions are more orthogonal than the label correlations require.

## 7. Sparse autoencoder, SAE (results/sae.json + sae_*.npy)
Trained on z-scored E_full, full N. TopK SAE: encoder f = TopK(W_enc (x - b_pre) + b_enc) keeps only
the k = 32 largest latent activations per galaxy (L0 = 32); decoder x_hat = f W_dec + b_pre. AuxK
term revives dead latents (top 512 dead units reconstruct the residual, weight 1/32). Expansion R in
{4, 8} (dictionary size m = R*1024). 5 random seeds. "FVU" = fraction of variance unexplained
(0 = perfect reconstruction).
- Health: R=4 (m=4,096) FVU ~0.035 (96.5% variance explained), fraction of latents alive ~0.65-0.70;
  R=8 (m=8,192) FVU ~0.034, alive ~0.33 (lower at higher expansion, expected).
- L0-vs-FVU frontier (R=4, FVU at each k): k=4 -> 0.142; 8 -> 0.092; 16 -> 0.058; 24 -> 0.043;
  32 -> 0.036; 48 -> 0.026; 64 -> 0.023; 96 -> 0.018; 128 -> 0.015. Operating point k = 32.
- Scoring (R=4, m=4,096): n_active = 2,657. Alignment of a latent = max over the 6 labels of
  |Spearman correlation| between the latent's activations and the label. Significance is set by a
  LABEL-SHUFFLE null (permute labels, recompute): thr95 = 0.0119, thr99 = 0.0129. max_align = 0.279
  (about 23x the thr95 null). Seed-stability: a latent is stable if its decoder vector reappears in
  at least half of the other 4 seeds with cosine >= 0.6. median best cross-seed cosine = 0.460.
  - aligned_sig (exceeds thr95) = 717; aligned_strong (exceeds thr99) = 690; aligned_and_stable = 335.
  - alien_candidates = 59: seed-stable AND high-novelty (novelty > 0.7, where novelty = fraction of a
    latent's activation variance NOT linearly explained by the 6 labels, a regression residual) AND
    NOT label-aligned. These are CORRELATIONAL candidates only; no causal/ablation test was run.
- Named concepts (each aligned latent named by its top-correlated label), n (n_stable, max_align):
  g-r 204 (80, 0.272); redshift 166 (87, 0.126); featured 114 (52, 0.245); r-z 89 (41, 0.210);
  smooth 81 (37, 0.279); merger 63 (38, 0.164). Clearest single concepts are colour (g-r) and
  morphology (smooth/featured); redshift is spread over many weak latents (a global property, not one
  sparse code).
- [interp] The model carries many seed-stable, physics-aligned directions (335), plus 59 candidate
  directions that are stable and not explained by any label we have. The alien candidates are the most
  interesting and the least certain; they are correlational and need future causal tests.

## 8. Curvature (results/curvature.json), full 48,398
- delta-hyperbolicity (Gromov 4-point delta, 1,000,000 random quadruples, divided by the cloud
  diameter; LOWER = more tree-like; 0 = a perfect tree). Median values: E_full Euclidean 0.01356;
  E_img Euclidean 0.01382; E_full cosine 0.02729; matched-covariance Gaussian ANCHOR 0.01774;
  synthetic TREE validation 0.00789. So AION (0.0136) sits BELOW the Gaussian anchor (mildly more
  tree-like than a matched random cloud) but well ABOVE the synthetic tree (not a tree).
- Ollivier-Ricci curvature (2,000-point subsample, k = 10, optimal-transport / earth-mover distance
  between neighbour distributions, alpha = 0.5; SIGN is meaningful: positive = locally clustered,
  negative = a bridge/saddle). mean = +0.155; fraction of edges with negative curvature = 4.2%;
  p5 = +0.006; p95 = +0.319.
- Forman-Ricci (full kNN graph k=15, augmented with triangle counts): frac_negative = 0.997, mean
  = -32.5. NOTE: on a kNN graph Forman is structurally negative (degree dominates), so its SIGN is not
  a clean curvature indicator. We use it only to RANK candidate bridge edges, not to claim negativity.
  Ollivier is the trustworthy signed measure.
- [interp] The manifold is mostly LOCALLY POSITIVELY curved (clustered neighbourhoods) with a small
  fraction (~4%) of negative-curvature bridge edges, and only a mild global tree tendency versus a
  matched random cloud. So: weak, localised branching, NOT a tree.

## 9. Topology (results/topology.json), full 48,398 + 10 x 2k subsamples
- beta0 (number of connected pieces): single-linkage on the kNN graph (k=15) gives knn_components = 1.
  Removing the longest minimum-spanning-tree edges peels off only SINGLE outliers: after 1 cut the
  sizes are [48,397, 1]; after 2 cuts [48,396, 1, 1]; after 3 cuts [48,395, 1, 1, 1]. The giant
  component stays whole. So beta0 = 1 (one connected body), NOT a red/blue beta0 = 2 split.
- beta1 (number of independent loops/holes): persistent homology on 10 independent 2,000-point
  subsamples, diameter-normalised, under Euclidean, Fermat, and diffusion metrics. Thousands of short
  "noise" bars appear, but significant loops (persistence > 0.1) essentially never recur:
  euclidean mean significant loops = 0.1 (range 0-1), max persistence 0.105; fermat 0.2 (0-1), 0.138;
  diffusion 0.0 (0-0), 0.081. So beta1 = 0 (no robust holes; rare loops are noise that does not recur
  across subsamples or survive a metric change).
- [interp] The embedding is a simply-connected continuum: one piece, no holes.

## 10. Heterogeneity / stratified ID (results/stratifiedId.json)
A two-component Gaussian mixture on g-r colour splits the sample at cut = 1.012 (component means 0.78
and 1.24). Passive (red, g-r > cut) n = 22,614; star-forming (blue) n = 25,782. TwoNN ID per
population (small-scale): passive 17.19 [17.01, 17.43]; star-forming 15.53 [15.39, 15.73]. Difference
delta-ID = 1.66 [1.38, 1.96], which EXCLUDES zero.
- [interp] Passive galaxies sit on a slightly HIGHER-dimensional sub-manifold than star-forming ones.
  The absolute IDs here are small-scale (1st/2nd-neighbour) and noise-inflated; the RELATIVE delta-ID
  is the trustworthy signal. (Prior literature, Cadiou et al. 2025, found the OPPOSITE ordering for
  raw photometry; our embedding is a different, multimodal representation, so the comparison is not
  one-to-one. State this honestly.)

## 11. Geometry of concepts (results/geometryOfConcepts.json)
Park et al. style test of a linear concept hierarchy: is the "spiral" direction equal to the "featured"
direction plus an orthogonal spiral-specific part? Measured cos(featured, spiral-residual) = 0.072 vs
a permutation null 95th percentile = 0.084. Since 0.072 < 0.084, NOT significant.
- [interp] Our data does NOT establish a clean linear featured->spiral hierarchy. This is an honest
  null result (an earlier tautological version of the test was corrected to use a permutation null).

## FIGURE INDEX (embed where the outline assigns them)
- figures/01_id_synthetic_validation.png — ID estimators recover known dims on synthetic manifolds.
- figures/02_id_aion_scale_curves.png — AION ID vs neighbour scale (Gride + local-MLE), plateau 10-12.
- figures/03_metric_concentration.png — distance contrast (RDR) and NN/mean across 5 metrics.
- figures/04_diffusion_spectrum.png — diffusion eigenvalue spectrum and gaps (smooth, no gap).
- figures/05_diffusion_full_labels.png — diffusion embedding (coords 1x2) coloured by 6 full-N labels.
- figures/06_diffusion_sparse_physics.png — same embedding coloured by mass/sSFR/Sersic (subset).
- figures/07_probe_decodability.png — probe R^2 with 95% CI, image-only, full vs subset labels.
- figures/08_modality_ablation.png — image-only vs multimodal decodability (leakage).
- figures/09_disentanglement_angles.png — probe-direction angle vs label-correlation null (excess).
- figures/10_probe_pred_vs_true.png — predicted vs true for redshift, mass, sSFR (held-out).
- figures/11_sae_frontier.png — SAE reconstruction error (FVU) vs sparsity k.
- figures/12_sae_named_concepts.png — counts of aligned and seed-stable SAE features per concept.
- figures/13_sae_alignment_novelty.png — SAE alignment vs shuffle null + alignment-vs-novelty, aliens.
- figures/14_curvature.png — delta-hyperbolicity vs anchors + Ollivier-Ricci summary.
- figures/15_topology.png — beta0 (MST splits) and beta1 (loop persistence vs threshold).
- figures/16_stratified_id.png — g-r GMM split + passive vs star-forming ID + delta-ID.
- figures/17_pca_variance.png — linear PCA cumulative variance (the linear baseline for ID).
- figures/18_manifold_galaxy_montage.png — real galaxies sampled along the colour and morphology axes.
