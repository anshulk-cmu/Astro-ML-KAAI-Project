# REPORT OUTLINE (section numbering, coverage, figures, line targets)

Report title (front matter only): "The Native Representational Geometry of an Astronomical
Foundation Model: A Geometry-and-Concept Study of AION-1's Galaxy Embedding".

Each section file starts with its own `## N. Title` heading exactly as below. Figures are
numbered by the two-digit number in their filename (figures/05_... -> "Figure 5"). Cross-
reference other sections by number and short name. Do not repeat content another section owns;
a one-line reminder of a defined term is fine.

---

### 00 — Front matter  (file 00_frontmatter.md, ~120 lines)
The big title (H1), a one-paragraph plain-language thesis, an executive summary that states the
four headline findings with their numbers and their epistemic status (measured vs interpreted),
a short "how to read this report" guide, and a compact figure index (list of the 18 figures with
one-line descriptions). State the four findings plainly: (1) intrinsic dimension ~10-12;
(2) a single smooth simply-connected continuum (beta0=1, beta1=0, no spectral gap), not discrete
clusters; (3) physics strongly decodable from the image alone (mass R^2 0.72, sSFR 0.76) and
concept directions disentangled; (4) sparse autoencoders find 335 seed-stable physics-named
concepts plus 59 correlational "alien" candidates. End with the honest one-line verdict and its
caveats. No figures embedded here except the index list.

### 01 — The question, and why it is still open  (file 01_question.md, ~150 lines)
What a foundation-model embedding is in plain words (a learned 1024-number summary of a galaxy).
The claim AION-1's authors made ("organises objects along physically meaningful directions") and
the gap: they never measured the geometry. State the two open questions we ask: (B) what is the
native geometry (dimension, curvature, topology) and (C) does the model organise by concepts beyond
the human taxonomy. Briefly place this in the ML-interpretability lineage (intrinsic dimension of
representations; geometry of concepts; the idea that big models may converge on shared structure),
in plain words, without overclaiming. Make clear the confirmatory "did it rediscover the Hubble
tuning fork" question is only a control, not the thesis. No figures.

### 02 — Background: galaxies as a measurable shape  (file 02_galaxy_background.md, ~170 lines)
The physics a faithful embedding should reflect. Explain, in simple terms with the real concepts:
the morphological continuum (ellipticals and spirals merge into each other; morphology is a
sequence); colour-mass bimodality (the red sequence of passive early-types, the blue cloud of
star-forming late-types, the sparse green valley between them); quenching and the at-least-two
distinct channels (fast disk->spheroid vs slow). Define every astro term: morphology, early/late
type, passive vs star-forming, quenching, redshift, stellar mass, sSFR, Sersic index, colour
(g-r, r-z), photometry vs spectroscopy. Then the quantitative prior: clean photometry has
intrinsic dimension ~2-5 and galaxy spectra ~3-10 effective dimensions, so an honest multimodal
embedding plausibly uses ~5-10 axes (an optimistic ceiling, not a point prediction). Stress the
load-bearing caveat: a static present-day embedding shows a DENSITY of populations, not a movie of
galaxies evolving; we must not claim the model "learned evolution". No figures.

### 03 — The model and the representation we study  (file 03_model.md, ~130 lines)
What AION-1 is at the level needed to read the geometry (do NOT cover training logistics or cloud).
One encoder-decoder transformer trained by multimodal masked modelling, self-supervised, no
morphology labels; it tokenises images, spectra, and catalogue scalars and learns to fill in masked
tokens. We freeze it, drop the decoder, take the per-token encoder outputs, and MEAN-POOL to one
1024-vector per galaxy. Define: token, encoder, embedding, mean pooling, frozen model, dimension
d=1024. Explain the two embedding sets and WHY two: E_full (image+photometry+redshift) for geometry,
E_img (image only) as the leakage-free set for concept probes, because anything fed in as input is
trivially decodable and would make a "concept axis" circular. No figures.

### 04 — Data and preprocessing  (file 04_data.md, ~150 lines)
The galaxy sample (Galaxy Zoo DESI, 0<z<1, N=48,398), the 4-band g,r,i,z images, and the labels with
their coverage and trust level: full-N morphology vote fractions and colours; redshift (mostly
photo-z, with the photo-z caveat explained); the small cross-matched physical subset (mass, sSFR,
Sersic). Define vote fraction and how Galaxy-Zoo labels are produced (a CNN predicting human vote
shares, accurate to ~5-10%), and the decision-tree conditionality (spiral/bar only for featured
disks, hence n~3,034). Then preprocessing: z-scoring per dimension and WHY (23x spread in per-dim
std would otherwise let a few dimensions dominate distances), and the observation that the vectors
sit near a thin shell (tight norms). Note honestly what we did not use (no spectra in this run; the
image-only deviation from the full multimodal plan). No figures.

### 05 — The metric problem and the concentration diagnostic  (file 05_metric.md, ~170 lines)
Why distance choice is not a detail. Explain distance concentration in high dimensions in plain words
(as dimension grows, all pairwise distances become nearly equal, so nearest-neighbour structure gets
unreliable), and define RDR = (Dmax-Dmin)/Dmin and NN/mean. Introduce the honest-metric battery:
Euclidean (control), Isomap geodesic (shortest path on a neighbour graph), Fermat / density-weighted
geodesic (paths prefer dense regions; outlier-robust; has a published convergence guarantee for
topology), and diffusion distance (averages over all paths). Embed Figure 3 and read it fully:
Euclidean is most concentrated (lowest RDR ~14-15, highest NN/mean ~0.43), intrinsic metrics give
several times more contrast (Fermat RDR ~260-281). Conclude why Fermat is the primary downstream
metric with Euclidean as control, and note the honest deviation that cosine did not concentrate like
Euclidean. FIGURE: figures/03_metric_concentration.png.

### 06 — Intrinsic dimension I: definition and estimators  (file 06_id_methods.md, ~230 lines)
Define intrinsic dimension in plain words (the number of independent knobs you need to describe a
point, which can be far below the ambient 1024). Explain WHY one number is not enough and why we use
four mechanistically-independent estimators. Then each estimator with its math and intuition, every
symbol defined: TwoNN (mu = r2/r1, the cumulative law P(mu>x)=x^-d, the unbiased MLE
d=(N-1)/sum log mu, the Gamma posterior interval, the linear-fit variant); Gride (generalised ratio
over neighbour ranks n1, giving an ID-vs-scale curve that separates true dimension from small-scale
noise inflation, no subsampling needed); local Levina-Bickel MLE (per-point ID from K neighbour
distances, swept over K, to map where dimension is higher); PCA participation ratio (the LINEAR
baseline, (sum lambda)^2 / sum lambda^2, which counts effective linear directions). Explain the
log2(N) resolution ceiling and the dimensional-collapse caveat (self-supervised encoders under-use
their nominal dimension, so a small ID is expected and not by itself proof of a clean manifold). No
figures (results in Section 07).

### 07 — Intrinsic dimension II: validation and the AION result  (file 07_id_results.md, ~270 lines)
First the validation logic: we ran every estimator on synthetic manifolds of KNOWN dimension at the
same N before trusting AION. Embed Figure 1 and read it (sphere truth 5, swiss roll truth 2, plane
truth 5; the manifold-aware estimators land on truth; PCA-PR reads slightly high on the curved
sphere because it is a linear-spread measure; the two-blob control inflates and is excluded). Then
the AION result: embed Figure 2 and read both panels fully (Gride falls from ~16.5 at the smallest
scale to ~10 plateau; local-MLE plateaus ~11.4; the PCA-PR dashed anchors ~11; the shaded 4-10 prior
band; the log2(N)=15.56 ceiling). Explain WHY the small-scale value is noise-inflated and the plateau
is the estimate. State the headline (ID ~10-12, three independent estimators agree). Then embed
Figure 17 (linear PCA cumulative variance) as the linear baseline and contrast: linear PCA needs
~45-60 components for 95% variance while the nonlinear ID is ~10-12, and the large-scale nonlinear ID
(~10) is close to PCA-PR (~11), which we read as WEAK curvature at manifold scale. Compare to the
astro prior (at/just above the 4-10 band) honestly. FIGURES: 01, 02, 17.

### 08 — The shape of the manifold: diffusion maps  (file 08_diffusion.md, ~190 lines)
The method and the spectrum. Explain diffusion maps in plain words (build a graph where near points
are strongly linked, treat it as a random walk, and the slow walk directions become coordinates).
Give the construction with symbols defined: affinity W_ij, the anisotropic alpha=1 normalisation that
recovers the Laplace-Beltrami operator so the geometry does not depend on where galaxies happen to be
dense, the Markov matrix, eigenvalues lambda and eigenvectors psi, and the diffusion coordinates
psi*lambda. Explain WHY diffusion maps over UMAP/t-SNE (deterministic, theory-backed, density-robust).
Note the practical fix we used (a self-tuning local-scaling bandwidth, after a global bandwidth gave
a degenerate fragmented graph), and that we always checked for a single connected component. Embed
Figure 4 and read it: the eigenvalue spectrum decays smoothly with no dominant gap (a gap would mean
discrete clusters), and the gaps panel confirms it; mention the harmonic screen (coords 3-4 are
repeats of lower coords). Conclude: one continuous body. FIGURE: figures/04_diffusion_spectrum.png.

### 09 — Reading physics off the manifold  (file 09_embedding_physics.md, ~190 lines)
The coloured embedding. Explain that coordinates dc1 and dc2 are the physics-bearing axes (dc0 is a
weak global mode, dc3/dc4 are harmonics), and give the measured Spearman correlations: dc1 tracks
morphology and redshift (smooth +0.50, featured -0.47, redshift +0.45), dc2 tracks colour and star
formation (g-r +0.57, sSFR -0.43, mass +0.33). Embed Figure 5 (six full-N labels over the dc1xdc2
embedding) and read every panel: axes are the two diffusion coordinates, each colour bar is one
property, the shared limits keep ~99% of points; describe the visible gradients. Then embed Figure 6
(mass, sSFR, Sersic on the cross-matched subset over a grey background) and read it, noting the
smaller n. Be careful: these are correlations and visible gradients, not a claim that the axes ARE
those properties. FIGURES: 05, 06.

### 10 — Decodability: how much physics the embedding carries  (file 10_probes.md, ~260 lines)
The probe method and results. Explain a linear probe (ridge regression: fit a single linear map from
the 1024 numbers to a property; R^2 is the fraction of held-out variance it explains), cross-validated
penalty, the 80/20 split, and the 1000x bootstrap CI. Stress the leakage-free design: probe on E_img
(image only) so redshift/colour are genuine inferences. Embed Figure 7 and read it (R^2 with CI bars,
full-N vs small-subset colour coding, n annotated). Give the numbers: colours 0.91-0.96, redshift
0.80, morphology 0.68-0.79, and the cleaner non-input evidence mass 0.72 and sSFR 0.76 from the image
alone. Embed Figure 8 (modality ablation) and explain the leakage logic: E_full scores higher on its
own inputs (redshift 0.98, colours 0.97-0.99), so E_img is the honest measure. Embed Figure 10
(predicted vs true) and read the three panels (identity line, R^2, n_test). State the photo-z caveat
clearly: image->redshift is partly image->colour->photo-z, so mass/sSFR are the stronger evidence
that the visual representation encodes physics it was not given. FIGURES: 07, 08, 10.

### 11 — Disentanglement and neighbourhood purity  (file 11_disentangle.md, ~150 lines)
Two model-light checks. First disentanglement: explain that each probe gives a direction (its weight
vector), and we measure the angle between two such directions and compare it to the angle implied by
the labels' own correlation (the null). excess = measured angle minus null; positive means the model
separates the two concepts MORE than the labels force. Embed Figure 9 and read the dumbbell plot
(grey null marker, coloured measured marker, the 90-degree line, the excess values), and report that
most pairs are positive (redshift-smooth +19, redshift-r_z +18) while merger pairs are near zero or
slightly negative, stated honestly. Then kNN purity (model-free): for smooth galaxies 99.1% of the 20
nearest neighbours are also smooth, for featured 71.4%, so like sits near like. FIGURE: 09.

### 12 — Concept discovery I: the sparse autoencoder  (file 12_sae_method.md, ~210 lines)
What an SAE is and why we use it. Plain-words idea: learn an over-complete dictionary so each galaxy's
1024-vector is rebuilt from a FEW active "concept" units, then read off what those units mean. Give the
TopK construction with symbols defined: f = TopK(W_enc(x-b_pre)+b_enc) keeps the k=32 largest latents
(L0=32), x_hat = f W_dec + b_pre; the AuxK term that revives dead latents; expansion R (dictionary size
m=R*1024). Define FVU (fraction of variance unexplained) and "alive". Embed Figure 11 (FVU vs k) and
read the reconstruction-sparsity trade-off and why k=32 is the operating point (FVU ~0.036, 96.5%
variance explained). Explain WHY this method answers question C (it lets the embedding reveal its own
axes rather than us probing for human-named ones). FIGURE: figures/11_sae_frontier.png.

### 13 — Concept discovery II: the concepts and the alien candidates  (file 13_sae_concepts.md, ~240 lines)
Scoring and results. Explain alignment (max |Spearman| between a latent's activations and the 6
labels), the LABEL-SHUFFLE null and its thresholds (thr95=0.0119, thr99=0.0129), seed-stability
(decoder vector reappears across seeds with cosine>=0.6), and novelty (activation variance NOT
linearly explained by the labels, a regression residual). Give counts: 717 aligned, 335 aligned AND
seed-stable, max alignment 0.279 (~23x the null). Embed Figure 12 (named concept counts) and read it
(g-r 204, redshift 166, featured 114, etc.; colour and morphology are the clearest single concepts;
redshift is spread over many weak latents). Embed Figure 13 and read both panels (the alignment
histogram against the null lines; the alignment-vs-novelty scatter with the 59 "alien" candidates in
the highlighted region). Be explicit and repeated: the alien candidates are CORRELATIONAL only, no
causal/ablation test was run; they are the most interesting and least certain result. FIGURES: 12, 13.

### 14 — Curvature and tree-likeness  (file 14_curvature.md, ~230 lines)
Is the manifold flat, curved, or tree-like, and where does it branch? Define curvature intuitively
(positive = surface curls like a sphere / neighbourhoods cluster; negative = saddle / a branch point),
and tree-likeness. Explain delta-hyperbolicity (Gromov 4-point delta; lower = more tree-like; 0 = a
perfect tree) and WHY we compare to anchors rather than reading the absolute number (a matched-
covariance Gaussian cloud and a synthetic tree). Explain Ollivier-Ricci curvature (optimal-transport
distance between the neighbour distributions of two linked points; SIGN is meaningful) and why it is
the trustworthy signed local measure, and why Forman-Ricci is only used to rank candidate bridges (it
is structurally negative on kNN graphs). Embed Figure 14 and read both panels (delta bars with the
Gaussian-anchor line and the tree validation; Ollivier mean +0.155 with p5-p95 whiskers and the 4.2%
negative-edge fraction, the zero line). Conclude: mostly positive (clustered) curvature, weak
localised branching, NOT a tree. Tie to the small global-minus-local ID gap from Section 7.
FIGURE: figures/14_curvature.png.

### 15 — Topology: pieces and holes  (file 15_topology.md, ~190 lines)
Define Betti numbers in plain words: beta0 = number of separate pieces, beta1 = number of independent
loops/holes. Explain persistent homology briefly (grow balls around points; track when features are
born and die; long-lived features are real, short-lived are noise) and the minimum-spanning-tree route
for beta0. Embed Figure 15 and read both panels: panel A shows that cutting the longest MST edges peels
off only single outliers while the giant ~48k component stays whole (beta0=1, not a red/blue beta0=2
split); panel B shows max loop persistence per metric against the 0.1 significance line, with the
recurring-loop counts ~0 across 10 subsamples (beta1=0). Explain the honest method choices: beta0 is
exact on the full sample, beta1 uses 2k subsamples with confidence across three metrics. Conclude: a
simply-connected continuum. Connect to the diffusion no-gap result (Section 8). FIGURE: 15.

### 16 — Heterogeneity: different types, different dimension?  (file 16_stratified.md, ~140 lines)
Do passive and star-forming galaxies live on sheets of different intrinsic dimension? Explain the
data-driven split (a two-component Gaussian mixture on g-r colour, cut at 1.012, means 0.78 and 1.24)
and why we use the RELATIVE difference, not the absolute IDs (which are small-scale and noise-inflated).
Embed Figure 16 and read both panels (the g-r histogram with the cut and component means; the passive
vs star-forming TwoNN ID with CI and the delta-ID = 1.66 [1.38, 1.96] that excludes zero). State the
result: passive sits on a slightly higher-dimensional sub-manifold. Add the honest comparison to prior
photometry work (Cadiou et al. 2025 found the opposite ordering for raw photometry; our representation
is different, so it is not a one-to-one contradiction). FIGURE: figures/16_stratified_id.png.

### 17 — The geometry of concepts: an honest null  (file 17_geom_concepts.md, ~110 lines)
The one test that did not pass. Explain the Park et al. idea (binary concepts as directions; a child
concept like "spiral" should be the parent "featured" direction plus an orthogonal spiral-specific
part). Explain the corrected test (an earlier version was tautological; we replaced it with a
permutation null). Report: cos(featured, spiral-residual) = 0.072 vs null 95th percentile = 0.084, so
NOT significant. Say plainly: our data does not establish a clean linear featured->spiral hierarchy
here, and why that is fine to report (negative results matter; the test is exploratory and the spiral
sample is small). No figure.

### 18 — The manifold made visible  (file 18_montage.md, ~110 lines)
Show the axes are physical using real cutouts. Explain the sampling (galaxies picked at evenly spaced
percentiles along each axis, representative not curated, because the absolute extremes include stars
and artefacts), and the RGB rendering (z/r/g channels with an arcsinh stretch). Embed Figure 18 and
read both rows: the colour row runs blue to red as g-r rises; the morphology row runs more-featured to
smoother along diffusion coordinate 1, titled by smooth fraction. Note honestly that the morphology
row is noisier (sampling at percentiles of one coordinate, real data). FIGURE: figures/18_manifold_galaxy_montage.png.

### 19 — Synthesis: the coherent picture  (file 19_synthesis.md, ~180 lines)
Bring the arms together. Argue that every measurement tells one consistent story: a low-dimensional
(~10-12), simply-connected, mostly-positively-curved continuum that strongly encodes galaxy physics,
with weak localised branching and a small set of candidate concepts beyond the human labels. Map each
geometric result to its physical reading (the continuum matches the known morphological continuum and
colour gradient; weak branching is consistent with, but does not prove, distinct quenching channels;
the higher-dimensional passive sheet is a real, modest, and somewhat surprising signal). Be explicit
about where physics intuition supports the result and where the data is silent. No new numbers beyond
those already reported; no figures.

### 20 — Limitations and threats to validity  (file 20_limitations.md, ~150 lines)
A frank, complete list with WHY each matters and how it bounds the claims: redshift mostly photo-z (so
image->redshift partly routes through colour); SAE "alien" features correlational only (no causal
test); ID sits at or just above the optimistic 4-10 band (so "low-dimensional" is true but not as low
as the best case); beta1 from 2k subsamples (confidence-set, not exhaustive); image-only run with no
spectra (a deviation from the full multimodal plan, so some physics axes may be missing); the sample
inherits a selection function; vote-fraction labels carry 5-10% error; the passive/SF ID is small-scale
in absolute terms. Distinguish what is solid from what is suggestive. No figures.

### 21 — Conclusions and what a full study should do next  (file 21_conclusions.md, ~110 lines)
State the defensible bottom line (the four findings and the honest verdict), then what a full study
should add: causal tests of the alien concepts (ablate/steer), the Riemannian pullback-metric flagship
for honest geodesics and where the manifold stretches, spectra and a true multimodal embedding,
cross-model checks, and per-population curvature at the branch points. End on a concrete forward step,
not a summary. No figures.

### 22 — Glossary of terms and symbols  (file 22_glossary.md, ~200 lines)
A clear table (or definition list) of every technical term and symbol used in the report, each defined
in one or two plain sentences. Cover at least: embedding, token, encoder, mean pooling, frozen model,
ambient vs intrinsic dimension, z-score, participation ratio, TwoNN, Gride, Levina-Bickel MLE, mu=r2/r1,
PCA, eigenvalue/eigenvector, diffusion map, Laplace-Beltrami, affinity, bandwidth, diffusion coordinate,
harmonic, RDR, NN/mean, Isomap, Fermat/geodesic distance, diffusion distance, ridge probe, R^2, bootstrap
CI, leakage, disentanglement, kNN purity, sparse autoencoder, TopK, L0, FVU, dictionary/dead/alive latent,
alignment, label-shuffle null, seed-stability, novelty, Spearman correlation, curvature (Gaussian/sectional),
delta-hyperbolicity, Ollivier-Ricci, Forman-Ricci, optimal transport, Betti number beta0/beta1, persistent
homology, minimum spanning tree, Gaussian mixture, redshift, photo-z, stellar mass, sSFR, Sersic index,
colour g-r/r-z, morphology, quenching, red sequence, blue cloud, green valley. Keep it faithful and short
per entry. No figures.

### 23 — Appendix: thresholds, statistical claims, reproducibility  (file 23_appendix.md, ~170 lines)
A single faithful reference table of every threshold and statistical decision rule used, with its value
and meaning: the ID resolution ceiling log2(N)=15.56; the 4-10 prior band; bootstrap/credible intervals
on every headline; the SAE null thresholds (thr95=0.0119, thr99=0.0129) and the seed-stability cosine
(0.6) and novelty (0.7) cuts; the beta1 persistence threshold (0.1); the delta-ID CI excluding zero; the
Ollivier sign convention; the metric battery roles (Fermat primary, Euclidean control). Then a short,
faithful reproducibility note: the analysis runs on the committed embeddings and labels, every estimator
was validated on synthetic known-answer data, and the figure scripts read only the committed results.
List the results files and what each contains. No figures.
