# Paper 1 Technical Report — Foundation Models Learn the Topology of Physical Parameter Spaces

**Scope.** The complete technical record behind Paper 1: every experiment on the frozen AION-1
representations of the GZ-DESI anchor — Phase 1 (the geometry of the embedding) and Tracks A–D
(the eight tests), plus the model-grid, spectrum-pathway, and depth-robustness experiments as they
complete. Each section states what was measured, how (method + math), why (what claim it supports
or kills), the results with statistics, and the honest caveats. This document is the internal
system of record; the ICLR manuscript and the ML4PS 4-pager are distilled from it.

**Predecessor.** This file replaces the Phase 1 geometry report (24 sections, preserved in git
history at commit `1ad5d72` and earlier). Phase 1 content is being rewritten into Part II rather
than pasted, so the whole document carries one voice and one set of conventions.

**Pre-registration anchor.** The per-paradigm prediction table (§22) was frozen by repo commit
`3aa2233ecb09928f0dde55ca17e2ce0f979125d6` (2026-07-07), before any non-AION model was embedded.

---

## Status ledger

| Part | Section | Status |
|---|---|---|
| I | 1–4 (setup, data, methods) | [ ] to write |
| II | 5–15 (Phase 1 geometry) | [ ] to write (results final) |
| III | 16 Track A | [ ] to write (all results final incl. 16.11 flip, 2026-07-07) |
| III | 17 Track B | [ ] to write (results final) |
| III | 18 Track C | [ ] to write (results final) |
| III | 19 Track D | [ ] to write (results final at image scope) |
| IV | 20–26 (planned experiments) | [~] placeholders; fill as each run completes |
| V | 27–30 (synthesis) | [ ] write last |
| App | A–D | [ ] running upkeep |

Legend: `[ ]` not yet written · `[~]` skeleton/awaiting results · `[x]` written and number-checked
against its results JSON.

**Conventions (binding for every section).**
- Every quantitative statement is tagged **[measured]** or **[interpreted]**.
- Every headline number carries its 95% bootstrap CI; angle quantities report BOTH percentile and
  basic (pivotal) intervals, and the text says which one is quoted and why.
- Nulls and controls are reported next to the result they discipline, never in a separate section.
- Corrections and retractions are kept in the record (what was first believed, what test overturned
  it), not silently overwritten.
- Every section ends with an **Artifacts** line: the exact scripts and results files behind it.

---

# Table of contents

## Part I — Setup

**1. Overview and thesis.**
The one-sentence claim; the three-way structure (ground truth, intervention, generality); what
"topology of a label space" means operationally (RP¹ vs S¹ vs interval); the map from the four
exploratory tracks to the eight locked tests T1–T8 and the contributions C1–C7; relationship to
Paper 2 (instrument calibration only, no claim overlap).

**2. The model under study: AION-1.**
Architecture and training summary (multimodal masked modeling over Legacy Survey imaging +
catalog + spectra tokens); the three public sizes (base 768-d / large 1024-d / xlarge); tokenizer
modality list (image, flux, shape e1/e2/r, Ra/Dec, Z) and why that matters for leakage; embedding
recipe used everywhere: CodecManager encode → `model.encode(..., num_encoder_tokens)` → mean pool;
E_img (image-only, leakage-free probe space) vs E_full (image+flux+z); measured throughput and
hardware.

**3. Data: the GZ-DESI anchor and its extensions.**
- 3.1 Anchor construction: GZ-DESI catalogs (8.69M), selection chain (r<19 extended, μ>18,
  non-PSF — Walmsley 2023 verified), the seeded 50k draw, cutout fetch (96×96 griz, DR10), final
  N=48,398 (96.8% fetch yield). Label inventory: full-coverage vs sparse (SDSS/NSA) labels.
- 3.2 Embeddings on disk: E_img / E_full (48,398×1024), E_img_base (×768); xlarge pass planned (§22).
- 3.3 Tractor shapes: e1/e2/r → paDeg (mod 180), ellipticity, b/a, inclination; ellipSnr; 48,290
  matched; the ellip>0.3 elongation population (15,893) and why PA is undefined below it.
- 3.4 Survey covariates + footprint: psfsize/psfdepth (griz), EBV, release → south 35,919
  (DECam, training domain) / north 12,371 (BASS/MzLS); the i-blank structure (north 100%,
  south 24%).
- 3.5 WISE W1/W2 → Stern-wedge AGN proxy (357 candidates; approximate, flagged).
- 3.6 Stellar populations: Firefly DR16 (5,512 joint age+metal, the only joint source);
  FastSpecFit iron v3.0 (13,044 anchor + 7,061 faint ages; ZZSUN unpopulated — no metallicity at
  scale, verified).
- 3.7 Spectra: SPARCL fetch, 20,986 spectra / 17,643 galaxies (DESI-DR1 13,915, SDSS 4,724,
  BOSS 1,501, EDR 846; ~3.3k dual-instrument).
- 3.8 Faint extension: 150k catalog+shapes (r 19–21, random_id draw), 136,407 good cutouts
  (90.9% yield), fetch infrastructure note.
- 3.9 The 8-representation model shelf: aion-base/large/xlarge, AstroCLIP, Stein MoCo-v2
  (authenticated; 3-band), Zoobot convnext-nano, DINOv2-base, pixel-PCA; provenance checks.

**4. Shared methods and statistics.**
- 4.1 Preprocessing: per-dimension z-score; the measured 23× per-dim scale spread that forces it.
- 4.2 Probes: RidgeCV (α ∈ 10^{-2..4}), 80/20 held-out R², bootstrap-resampled residuals for CIs.
- 4.3 Concept directions (Ridge α=100, unit-normalized) and the angle machinery: embedding angle
  vs the label-correlation null arccos(ρ); the EXCESS as the disentanglement statistic; each
  direction fit on its own natural population (the mask bug and its fix, §17.3).
- 4.4 Bootstrap intervals: percentile AND basic (pivotal); why bootstrap-refit directions bias
  angles toward 90° and when each interval is quotable.
- 4.5 Circular statistics: the doubled angle for mod-180 quantities, circular error, the 45°
  chance floor, wrap handling at antipodes.
- 4.6 Nulls-and-controls policy (shuffle nulls, matched random directions/feature sets, label
  nulls) and the multiplicity policy (effect size + CI per cell; exploratory batteries never
  promoted post hoc; Holm on the pre-registered family only).
- 4.7 Compute: local RTX 5070 Ti (12 GB), measured 17 img/s at batch 32; environment inventory;
  bf16 policy for oversized models.

## Part II — Phase 1: the geometry of the embedding (the substrate for everything else)

**5. Sanity and linear structure.**
NaN/inf audit; norm concentration (‖x‖ ≈ 48–50 ± 2.5, near-shell); per-dim std spread 23×;
PCA participation ratio (raw 7.4–8.2 vs z-scored ~11) as the LINEAR-dimension null; variance
curve (50% in 3 PCs / 90% in 25). [results/sanity.json]

**6. Metric concentration diagnostic.**
Five metrics (Euclidean, cosine, Isomap, diffusion, Fermat p=2) on a fixed 2k subsample; RDR and
NN/mean statistics; Euclidean most concentrated (RDR 14.7), Fermat highest contrast (281);
the measured deviation from the spec's cosine expectation; the resulting metric assignment
(Fermat primary for topology/curvature, diffusion for coordinates, Euclidean control) — and the
forward pointer to §17.7 on why this does NOT imply geodesic neighbors are semantically better.
[results/metric.json]

**7. Intrinsic dimension.**
Four estimators implemented from first principles (TwoNN with Γ-posterior; Gride scale curve
n1=1..256; Levina–Bickel K-sweep; PCA-PR), each with math; synthetic certification at matched N
(sphere-5 → 4.96, swissroll-2 → 1.95, linear-5 → 5.08, two-blobs control); AION measured:
small-scale ~16 inflated by noise, Gride decays to ~10.1, local-MLE plateau ~11.4, PCA-PR ~11.2
→ headline ID ≈ 10–12; the global–local gap (~+1) as the first weak-curvature signal; scale
sensitivity sweeps. [results/intrinsicDim.json]

**8. Diffusion map.**
α=1 Laplace–Beltrami normalization; the global-BGH bandwidth failure (fragmented graph, λ≈1
degeneracy) and the self-tuning local-scaling fix (σ_i = 7th-NN distance) with the
connected-component check; smooth spectrum (no cluster gap); harmonic screen (coords 4–5 are
harmonics of 0–3); the λ₁≈0.999 verification (participation ratio 94% → genuine global gradient,
not an outlier mode); the dc1×dc2 physics plane used in figures. [results/diffusionMap.json]

**9. Decodability and disentanglement (Phase 1 probes).**
Leakage-free E_img decodability (redshift 0.800, g−r 0.958, r−z 0.911, morphology 0.68–0.79) and
non-input physics (mass 0.721, sSFR 0.760, Sérsic 0.664); the E_full modality ablation
(redshift 0.98 — input leakage demonstrated); disentanglement excess angles (+9° to +19°) with
the NaN-poisoning bug and fix; kNN purity; the photo-z caveat and why image→mass/sSFR is the
cleaner evidence. [results/probes.json]

**10. Curvature.**
δ-hyperbolicity (10⁶ quads, diameter-normalized) against matched-covariance Gaussian AND
synthetic-tree anchors — mildly more tree-like than random, far from a tree; Ollivier–Ricci
(2k, EMD, α=0.5): mean +0.155, 4.2% negative bridge edges; Forman–Ricci structural negativity on
kNN graphs (rank-only use); coherence with the ID gap. Verdict: mostly positively curved
continuum, weak localized branching. Medians vs means stated explicitly. [results/curvature.json]

**11. SAE concept discovery.**
TopK+AuxK architecture and training math (k=32, R=4, aux on dead latents, α=1/32); L0-vs-FVU
frontier (10 points); health (FVU 0.035, alive 0.70); the fixed-threshold scoring bug and the
label-shuffle-null fix (zero-inflated activations); 717 significantly aligned / 335 seed-stable /
~59 alien candidates; concept naming by top label (g−r 204, redshift 166, featured 114...);
cross-seed stability statistics (median best decoder-cosine 0.46) — the number §19.7's cross-scale
result must be read against. [results/sae.json]

**12. Topology.**
β₀ exact on all 48,398 (kNN-MST single-linkage; largest split [48397,1] → outliers, not
bimodality → β₀=1); β₁ via 10×2k subsamples under Euclidean/Fermat/diffusion (~0 significant
loops); the follow-up: full persistence diagrams + Fasy/Chazal bootstrap bands
(Fermat's 0.138 loop under its 0.197 band → noise); the honest 2k-resolution limit; the forward
pointer to §16 (thin feature-subspace rings are invisible to global H1 — and were then found by
probing). [results/topology.json, results/persistence.json]

**13. Heterogeneity: stratified intrinsic dimension.**
GMM(2) split on g−r (passive 22,614 / SF 25,782); TwoNN per population; ΔID = 1.66,
CI [1.38, 1.96] excludes 0 → different populations occupy different-dimensional submanifolds;
small-scale inflation caveat (relative signal only). [results/stratifiedId.json]

**14. Geometry of concepts (exploratory).**
Park et al. hierarchy test on featured→spiral; the tautological first construction and the
permutation-null correction; result: not significant (0.072 vs null p95 0.084) — an honest
exploratory null. [results/geometryOfConcepts.json]

**15. Phase 1 synthesis.**
The four Strong-GO criteria scored; the coherent picture (low-dim, simply connected, mostly
positively curved continuum that strongly encodes physics); what Phase 1 could NOT see (thin
subspace structure) — the hinge into Part III.

## Part III — The four exploratory tracks (the eight tests on AION-large)

*Opens with the track→test map: A→T1/T2/T3/T4(half)/T7 · B→T5 (+C7 negatives) · C→T6(image
side)+translation · D→T8(image side). All Part III numbers are final and audited; every section
cites its JSON.*

**16. Track A — angular quantities live on closed loops.**
- 16.1 Setup: PA from Tractor e1/e2; mod-180 physics → the doubled-angle encoding; the elongation
  condition.
- 16.2 Headline loop: (cos 2θ, sin 2θ) probe → median 2.03° [1.95, 2.10], 99.7% within 20°,
  R² 0.97/0.97 (n=15,893); E_full replication 2.15°.
- 16.3 Controls: shuffle null 44.1° (≡ 45° chance — calibrates the metric), top-2-PC 43.2° (the
  loop is invisible to dominant variance), linear-scalar 0.77 (the wrap costs a line ~0.2 of R²),
  vacuous ellipSnr cut documented as vacuous.
- 16.4 Conditioning and invariance: error vs ellipticity (4.9° → 1.4°, with the honest top-bin
  uptick to 1.8°); loop radius 0.986–0.993 across brightness tertiles and morphology strata (T3).
- 16.5 Inclination as an arc, not a loop: b/a inclination R² 0.84, edge-on vote 0.89 (T1's
  interval case).
- 16.6 RA leakage in E_img (negative control turned side-finding): R² 0.41/0.17; conditions
  explain only ~42/52%; residual 0.24/0.08 → sky-correlated population structure; observing
  conditions strongly encoded (psfsize 0.69 / depth 0.44 / EBV 0.30) — the instrument-confound
  quantification later used by §19.5 and Paper 2.
- 16.7 RA/Dec through AION's native coordinate codecs: RA a true k=1 loop (2.9°, R² 0.99/0.98,
  wraps at 0/360); Dec a scalar arc (0.99) with the underpowered circular control honestly
  labeled.
- 16.8 Cross-modal consistency (T4, image↔catalog): E_shape from the shape codec; img↔shape 3.6°
  [3.41, 3.72] — same loop, two modalities.
- 16.9 Unsupervised discoverability (T7): best PCA pair (of 1225) ring-R² 0.41; Phase 1 SAE
  overlap (best single |corr| 0.50; 15/4096 > 0.3) → findable but only partially, supervised
  probe needed for the clean view.
- 16.10 Causal test I — rotations: fixed probe + physically rotated inputs; the three-link sign
  convention (verified independently); slope −0.999, intercept −0.01, max residual 0.05°
  (90° antipode excluded, per-galaxy circular error there 2.4°); the 180° fixed point 1.9° as an
  interventional proof of the mod-180 symmetry.
- 16.11 Causal test II — mirror flip + composition (completes O(2)). **[results final 2026-07-07]**
  Geometry: flip ⇒ θ→−θ (mod 180), fixed points {0°, 90°}, displacement −2θ; interpolation-free
  intervention; frame-offset conjugation (−2(c−90)) as a measurable. Measured: flip readout vs
  −truth 1.99° (= the 2.03° baseline — reflection costs nothing) / vs −readout 2.38° [2.34, 2.43];
  displacement slope 0.983 (wrap fold-back explains the attenuation); fixed-point/antinode medians
  6.0°/83.9° — both on the |2θ| law including the ±5° window; frame offset −0.09° (convention is
  90° to sub-0.1°); composition flip∘rot30 → −θ−30 at 2.59° [2.54, 2.63]. O(2) closed by
  input-space intervention. [results/trackA_flip.json]

**17. Track B — the Hubble tuning fork.**
- 17.1 Decodability: featured 0.794, bar 0.554 / spiral 0.252 (branch n=3,034), Sérsic 0.664.
- 17.2 The handle: sequence axis validated against independent labels (+0.89/−0.87/−0.32);
  monotonicity by Spearman (+0.87/−0.81/−0.30).
- 17.3 The angle battery and the mask bug: the asymmetric-population fit error that silently
  moved the headline (+11.9° → +0.7°), its diagnostic proof, and the fix (per-direction natural
  populations); final: seq–bar excess +11.9° [9.6, 15.1] percentile / [8.7, 14.3] basic;
  seq–spiral +27.5 (with the bootstrap-bias note — this pair's raw CI is not quotable);
  bar–spiral −5.2 (slightly MORE aligned than labels predict).
- 17.4 The fork opens: bar-direction spread 0.48 → 0.82 (ratio 1.70); out-of-sample defense:
  shuffled-vote refits 0.70–0.93 (never a fan), split-half held-out 1.53–1.79 (mean 1.65);
  spiral does NOT fan (0.76) — the fan is bar-specific.
- 17.5 The handle bends: path/chord 1.248 (noise-corrected 1.221) vs the matched straight-line
  null 1.038 ± 0.001; the mis-calibrated first null and why it was replaced (an
  estimator-regime error that under-sold a real finding).
- 17.6 Fork dimensionality and the preferred axis: disk-residual PC1+PC2 = 48.6% (low-dim
  branch) but PC1 ⊥ bar (|cos|=0.00) — the dominant disk-specific axis is unidentified (open);
  independent Sérsic direction at 91.1° from the sequence (correlated labels, orthogonal
  directions).
- 17.7 Geodesic vs Euclidean (honest negative, both targets 43%/37% < 50%): the 10-dim
  diffusion-proxy construction; reconciliation with §6 — intrinsic metrics win GLOBALLY
  (contrast/topology) while weak curvature (§10) predicts no local-neighbor advantage; the
  planned Fermat-distance robustness variant.
- 17.8 Redshift contamination: seq–redshift excess −20.9 [−22.7, −19.1] — the sequence axis is
  more redshift-entangled than labels force; carried as a caveat on "pure Hubble type".

**18. Track C — redshift translation, property angles, and the age–metallicity degeneracy.**
- 18.1 Firefly data and the log-transform discipline (age already log; metallicity spans 200× —
  the linear treatment that produced the retracted −14° conflation claim; the correction story
  kept in full).
- 18.2 Decodability with CIs: the collapse pattern — redshift 0.800 / colours 0.91–0.96 /
  mass 0.72 / sSFR 0.76 vs age 0.072 [0.034, 0.100] / metal 0.251 [0.196, 0.297].
- 18.3 The 28-pair angle battery: strong structural pairs (redshift–colour aligned +13 excess;
  mass/sSFR/morphology disentangled +18 to +37) stable under both CI constructions; the 5 pairs
  involving weak age/metal probes that flip between constructions (quoted only with both).
- 18.4 The degeneracy result (T6 image side): embed 93.7° vs truth 91.5° → excess +2.3°
  [−4.2, +5.8] / [−1.3, +8.8], consistent with zero across mass-weighted and z≥0.15 variants →
  the degeneracy is inherited as MISSING INFORMATION, not as extra entanglement.
- 18.5 Manifold translation (embedding-level intervention, labeled as such): +0.1 z-step lands
  among real higher-z neighbors at slope 0.318 [0.308, 0.331] vs random 0.011; morphology
  preserved at the kNN floor (0.079 vs 0.073/0.073); the decoder-calibration bug and fix.
- 18.6 The DERIVED linear steer block (algebraically the angle battery; kept for intuition,
  never causal evidence).
- 18.7 Selection vs evolution: z decodes within magnitude tertiles (0.78/0.74/0.75); partial-out
  leaves residual-z R² 0.352 [0.335, 0.369] beyond mag+size+colours; downsizing null at fixed
  mass (n=1,392/663; the third bin under the n>100 gate — recorded).

**19. Track D — SAE dictionaries and the Engels reducibility test.**
- 19.1 SAE retrain with saved decoders: Phase 1-identical protocol; health across 5 seeds ×
  2 embeddings (FVU 0.033–0.036, alive 0.65–0.71); why retraining was necessary.
- 19.2 The Engels pipeline: Mistral kNN-graph clustering (τ sweep reproduces the paper's three
  regimes; 74 clusters at τ=0.5, 72 tested); cluster-restricted reconstruction → PCA → 2D;
  separability S (min rotated binned MI, bits) and ε-mixture M with the documented deterministic
  deviation from the paper's GD; the paper's calibration values.
- 19.3 What the candidates are: top scores S 1.1–1.7 bits but M 0.37–0.53 (no calendar-grade
  circle); #18 partially colour (0.35), #67 partially morphology (0.42), most unnamed; mixed
  seed stability; the two self-caught bugs (rank-biserial for binary labels; nats→bits).
- 19.4 The loop is in superposition (headline): no cluster plane near the known loop plane
  (chance-level angles — null to be re-persisted, see gaps ledger); best-cluster ablation does
  nothing (2.35→2.41 vs controls 2.35; best cluster = #39 at 9.6°, per JSON); top-K
  loop-correlated ablation degrades monotonically (K=15/50/200 → 5.7/9.3/16.7°, controls flat) —
  replicated on E_full (6.5/9.5/17.2) and at base scale (6.3/9.8/17.5). The known-ground-truth
  irreducible feature is fractured across hundreds of dictionary words.
- 19.5 Instrument-identity census (Matt's diagnostic): 452 physics-first vs 442 instrument-first
  (i-blank 321 — the largest systematics family), strong-core counts; the honest
  median-at-null caveat; hand-off to Paper 2's exclusion list.
- 19.6 AGN proxy: clean null (best cluster R² 0.064) with physical reasons.
- 19.7 Cross-scale: geometry conserved (base raw loop 2.5°; same fracture pattern) but
  vocabulary not (22% matched at ≥0.5; read against §11's cross-seed 0.46).
- 19.8 Deferred (data-gated): spectrum-side dictionary, line systems, decoder-level causal
  ablation, activation montages. Gaps ledger: re-persist the random-plane null; persist base
  SAE weights; efull WISE rerun.

## Part IV — Completing the grid (planned; fill as each run lands)

**20. North replication (T4).** [~] Per-instrument refit/evaluation of the headline tests on the
12,371 BASS/MzLS galaxies (already inside E_img); success criterion: loop/fork/degeneracy numbers
reproduce within CIs on DECam-south vs north.

**21. RA/Dec confound audit (T1 gate).** [~] (a) Input-token audit: E_img is position-blind by
construction — verify tokenizer inputs; (b) mediation: regress the loop coordinate on
EBV/stellar-density/depth/PSF maps, report partial R² of RA/Dec beyond them; both pre-registered
readings written before running.

**22. The model grid and emergence (C6, F3).** [~] Embed the anchor through all 8 representations;
run the full test grid on each; score the FROZEN predictions (commit `3aa2233`): reconstruction
objectives keep the loop; rotation-augmented contrastive (AstroCLIP, Stein) lose it by design;
label-supervised (Zoobot) loses it; DINOv2 hflip-only → folded arc; pixel-PCA wrong topology;
scale trend base→large→xlarge (completes the 2-point trend, incl. the degeneracy-angle-vs-scale
question). Holm-corrected family summary.

**23. Pooling and depth robustness.** [~] Headline numbers under attentive pooling and at two
intermediate encoder depths (AION-large only); headline-only-if-stable rule.

**24. The spectrum pathway (T4/T6 completion).** [~] Embed 20,986 spectra; image↔spectrum
consistency on shared galaxies; the degeneracy test on the spectrum side (prediction: spectra
break what images cannot know); dual-instrument subsample as a free cross-instrument check.

**25. Depth robustness to the noise floor (T3 completion, F6).** [~] Embed 136,407 faint cutouts
(r 19–21); loop error vs magnitude; FastSpecFit-faint ages as the extended T6 age side.

**26. T8 completion.** [~] Spectrum-side SAE dictionary + line systems; image-vs-spectrum
dictionary comparison; optional decoder-level ablation and montages.

## Part V — Synthesis

**27. The contributions, scored.** C1–C7 against final numbers; the pre-registered prediction
scorecard; what emerged only from combining tracks (e.g., Phase 1 β₁=0 + Track A loop = global
topology is blind to thin feature subspaces — and that is itself a finding).

**28. Honest negatives, corrections, and limitations.** The kept negatives (geodesic, PC1⊥bar,
spiral weakness, downsizing null, AGN null, Dec control underpowered); the corrections ledger
(Track B mask bug, Track B curvature null, Track C metallicity transform, Track A causal-JSON
frame fields); standing limitations (photo-z labels, single-survey anchor, Firefly subset size,
correlational SAE claims, north-up assumption).

**29. Reproducibility and release.** Artifact map (every JSON/figure/script per section); seeds
and environments; the galprobe packaging plan; the pre-registration commit chain; what is public
data vs what needs recomputation.

**30. Related work and positioning.** The five threads (probing/world models; feature geometry;
SAEs on scientific models; astro FMs + convergence; invariance/equivariance); nearest neighbors
and the open gap; the Platonic-refinement framing (convergence-on-what, answered with ground
truth).

## Appendices

**A. Notation and glossary.** Every symbol and term defined once (ID, RDR, FVU, RP¹, excess
angle, basic CI, ...).
**B. Environments and compute log.** Exact package versions per phase; GPU throughput
measurements; total compute budget.
**C. Artifact index.** Section ↔ script ↔ results-file ↔ figure crosswalk, with checksums for
frozen numbers.
**D. Changelog.** Dated record of every section fill/revision of this report.

---
*Fill order: §4 (methods, anchors everything) → §16 (Track A, freshest + flip lands today) →
§17–19 → Part II (Phase 1 rewrite) → §1–3 → Part IV as runs complete → Part V last.*
