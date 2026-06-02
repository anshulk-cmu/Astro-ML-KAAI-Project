# The native representational geometry of an astronomical foundation model — and the physical concepts it invents

### A geometry-and-concept-discovery study of AION-1's galaxy embedding manifold

> **One-paragraph thesis.**
> AION-1 — Polymathic AI's billion-parameter, omni-modal astronomy foundation model — compresses every galaxy into a 1024-number vector, with no morphology labels and no notion of galaxy evolution.
> The model's authors *assert* that this embedding "organizes objects along physically meaningful directions," but they never measure its geometry.
> We do.
> We download AION-1-Large, generate multimodal embeddings for ~10⁴ real galaxies that carry independent physical labels, and characterize the **shape** of the resulting manifold:
> its intrinsic dimension (global, local, Bayesian, spectral), its curvature and tree-likeness (a battery, not one fragile number), its Riemannian structure (geodesics and curvature under a learned pullback metric), its topology (under a metric with a convergence guarantee), and the **concepts it organizes by** — including a sparse-autoencoder search for structure that lives *beyond* the human Hubble/Galaxy-Zoo taxonomy.
> The deliverable is the first geometric characterization of an astronomical foundation model's latent manifold, framed as an astro instantiation of the "intrinsic dimension of representations," "geometry of concepts," and "Platonic representation" programs in machine-learning interpretability.

This document is three things at once:

- an **exploratory go/no-go specification** — is the signal there, stable, and worth a full study?
- a **pre-registration** — what is the claim, what is the null, what falsifies it?
- a **publishability blueprint** — what is novel, for which audience, at which venue?

It contains **no code** and **no solution** — only data, models, mathematics, formulas, libraries, statistical bounds, expected results, and a toy-scale interpretation guide.
Every external claim is sourced; the consolidated reference list is in Section 23.
The companion paper grounding all model facts is OCR'd in-repo at [aion1.md](papers/aion1/aion1.md).

**Provenance & corrections.**
This is the fourth revision.

- v1 contained precise-but-wrong numbers and one mathematically *unsound* diagnostic (the "JL-gap ⇒ non-Euclidean" inference).
- v2 corrected those against primary sources and made the methods rigorous.
- v3 reframed for novelty and folded in four advanced method families — diffusion maps + geodesic/Fermat metrics, a Riemannian curvature battery, Bayesian/GPLVM manifold learning, and sparse-autoencoder concept discovery — plus the comparative-topology and Platonic cross-checks.
- v4 (this version) folds in two locked scope decisions — **multimodal embeddings** as the core representation, and the **Riemannian pullback-metric flagship** — and is rendered at full granularity.

All corrections remain flagged **[CORRECTION]**; method families added after v2 are flagged **[v3]**/**[v4]**.
The revision history below is retained for provenance; earlier standalone version files have been removed.

---

## 0. How to read this document

This is a long, deliberately complete specification. Three reader paths:

- **Project lead deciding go/no-go.** Read §1 (the question), §4 (scope + the toy-scale contract), §16 (the pipeline), §18 (decision rules), §20 (abort/pivot conditions), and §24 (open decisions). Skim Appendix D (the validation harness) — it is what makes any reported number trustworthy.
- **ML-interpretability reviewer.** Read §1 (framing + the Ansuini / geometry-of-concepts / Platonic lineage), §§7–15 (the methods and their guarantees), §21.5 (anticipated objections), and Appendix D. The novelty argument is in §3.5–§3.6 and §21.
- **Astronomer.** Read §2 (galaxy background), §3 (what AION-1 is), §§11–12 (concept axes and discovery), §19 (what to believe at toy scale), and Appendix C (what each result *means* for galaxy science).

Two conventions used throughout:

- **Tags.** `[CORRECTION]` marks a fix to a v1 error; `[v3]`/`[v4]` mark method families added after the original methods-driven draft.
- **Embedding sets.** `E_full` = the multimodal embedding (image + photometry + redshift + spectra), used for the *geometry* arms; `E_held(target)` = a target-withheld embedding (image + photometry only), used for the *concept* arms so a "redshift/mass axis" is a genuine inference rather than a read-back of an input (Section 5.1).

**If you read only one thing:** §19 (interpretation — what each number is *allowed* to mean) together with §20 (what would falsify each claim). At toy scale, the honesty of the interpretation, not the cleverness of the method, is what separates a suggestive plot from a defensible measurement.

---

## Table of contents

- **Part I — Framing**
  - 1. The reframed question (geometry + concept discovery)
  - 2. Background: galaxy evolution as a measurable shape
  - 3. Background: AION-1, the gap, and the ML lineage we plug into
  - 4. Scope, the toy-scale contract, and venue targets
- **Part II — Ingredients**
  - 5. The model specification (we build multimodal embeddings)
  - 6. The data specification
  - 7. The metric problem and the honest-metric battery `[v3]`
- **Part III — The measurements**
  - 8. Method A — Intrinsic dimension: four estimators triangulated `[v3 upgrade]`
  - 9. Method B — Manifold reconstruction: diffusion maps + Bayesian GPLVM `[v3]`
  - 10. Method C — Curvature, the Riemannian pullback flagship, and tree-likeness `[v3/v4]`
  - 11. Method D — Concept axes and the geometry of concepts `[v3 upgrade]`
  - 12. Method E — Unsupervised concept discovery (sparse autoencoders) `[v3, novel payload]`
  - 13. Method F — Topological skeleton (Mapper)
  - 14. Method G — Persistent homology with a guaranteed intrinsic metric `[v3 upgrade]`
  - 15. Method H — Cross-domain (CAMELS) & cross-model (Platonic) checks `[v3 upgrade]`
- **Part IV — Reading the result**
  - 16. Execution pipeline (dependency-ordered)
  - 17. Statistical bounds, nulls, and the reproducibility contract
  - 18. Expected results and pre-registered decision rules
  - 19. Interpretation at toy scale: what to believe and what not
  - 20. Falsifiability and threats to validity
  - 21. Deliverables, novelty, and venues
  - 21.5 Anticipated reviewer objections (dual-venue FAQ) `[v4]`
- **Part V — Reference material**
  - 22. Glossary of symbols
  - 23. References
  - 24. Open questions and locked decisions
  - Appendix A — Worked quantitative companion `[v3]`
  - Appendix B — Library & function quick-reference `[v3]`
  - Appendix C — What each result would mean for galaxy science `[v3]`
  - Appendix D — Synthetic validation harness (known-answer tests) `[v4]`

---

# Part I — Framing

## 1. The reframed question (geometry + concept discovery)

The original framing — *"did AION-1 rediscover the Hubble tuning fork?"* — is **too weak to carry a study.**
The verification made this concrete:

- A decade of unsupervised-morphology work already establishes that galaxy morphology is a *continuum*, not discrete classes.
- It also shows machine methods go "Beyond the Hubble Sequence," recovering continuous morphology manifolds and machine-defined classes (Cheng et al. 2021, *MNRAS* 503, 4446).
- AstroCLIP already showed self-supervised galaxy embeddings retain "near-perfect" redshift and mass.

So a reviewer would say: *we already know SSL galaxy embeddings encode morphology, redshift, and mass.*
The confirmatory question is a likely desk-reject; it survives here only as **one control figure**, not the thesis.

**[v3] The reframed thesis is a hybrid of two genuinely open questions.**

> **(B) What is the native representational *geometry* of an astronomical foundation model?**
> How many real dimensions does AION-1's 1024-d manifold use, globally *and* locally?
> Is it flat, curved, or tree-like, and *where* does it branch?
> What is its topology under a metric that is actually trustworthy in high dimensions?

> **(C) Does AION-1 invent physical *concepts* beyond the human taxonomy?**
> If we let the embedding reveal its own organizing axes (sparse-autoencoder dictionary learning) rather than probing for human-named ones, which discovered concepts align with known physics (mass, sSFR, redshift, morphology) and which are *physically coherent but unnamed* — structure the model uses that astronomers never wrote down?

These plug into three active ML-interpretability programs, which is what makes the work legible to the ML audience as well as the astronomy one:

- **Intrinsic dimension of representations** — Ansuini et al. 2019 (NeurIPS): representation ID is orders of magnitude below ambient, invisible to PCA, and *predicts* accuracy; recent work shows *local* ID predicts alignment and generalization.
- **The geometry of concepts** — Park et al. 2023/2024 (ICML): binary concepts are directions, categorical concepts are simplices/polytopes, and hierarchy appears as *orthogonality under a whitened (causal) inner product*.
- **The Platonic Representation Hypothesis** — Huh et al. 2024 (ICML): large models converge toward a shared representation of reality; recent critiques (2026) show the robust signal is *local* neighborhood geometry, not global alignment.

The five original outputs survive and are *upgraded*, not replaced:

1. **Intrinsic dimension** — now global + local + Bayesian + spectral (four independent estimators).
2. **Curvature / tree-likeness** — now a falsifiable battery, plus a Riemannian pullback metric, that *localizes* where branching happens.
3. **Physical concept axes** — now tested against the geometry-of-concepts laws (simplex / orthogonality).
4. **Topological skeleton** — Mapper, plus a convergence-guaranteed intrinsic-metric persistent homology and comparative topology (RTD).
5. **Cross-domain alignment** — CAMELS sim-vs-real, *plus* a cross-model Platonic check against AstroCLIP.

And one genuinely new, headline output is added:

6. **[v3] Unsupervised concept discovery** — a sparse autoencoder over the embedding, scored against physical catalogs, to surface aligned and "alien" concepts.

## 2. Background: galaxy evolution as a measurable shape

### 2.1 The continuum, the bimodality, and the two pathways

Three empirical facts frame what a *faithful* embedding should look like.

**The morphological continuum.**

- Hubble's 1926 text already says the elliptical and spiral sections "merge into each other."
- Morphology is a sequence along which physical properties vary monotonically (Roberts & Haynes 1994, *ARA&A* 32, 115).
- So a faithful embedding might look like a *curved low-dimensional spine*, not separated blobs.

**Color–mass bimodality.**

- Galaxies split into a **red sequence** (passive, early-type) and a **blue cloud** (star-forming, late-type).
- These are separated by a sparse **green valley** (Schawinski et al. 2014).
- This predicts a possible *two-component* (β₀ = 2) topological signature.

**Multiple evolutionary pathways.**

- Schawinski et al. 2014 ("Green Valley is a Red Herring", arXiv:1402.4814) show the green valley is **not** a single transitional state.
- Early-type galaxies quench fast, with a disk→spheroid transformation.
- Late-type galaxies quench slowly while keeping their morphology.
- So there are at least **two distinct quenching channels.**

**[CORRECTION — interpretive, load-bearing for Section 19.]**

- "Young blue spirals slowly transform into old red ellipticals" is a *statistical tendency over a population with multiple channels*, not a literal time-trajectory each galaxy follows.
- A "tree" or "continuum" in a *static* embedding of *present-day* galaxies is a **density of states / branching of populations**, not a movie of galaxies evolving.
- We must not claim AION-1 "learned galaxy evolution"; at most it learned a present-day manifold whose branches echo distinct evolutionary endpoints.
- This is exactly why **local curvature that localizes branch points to physical transitions** (Section 10) is more defensible than any single global "tree-likeness" claim.

### 2.2 What "low-dimensional and physically aligned" means quantitatively

**Spectra.**

- Connolly et al. 1995 (*AJ* 110, 1071) found galaxy SEDs are described by the **first two eigenspectra** / a one-parameter family.
- **[CORRECTION]** v1 said "~3"; that is wrong — the "~3" / "7–10" figures are from Yip et al. 2004 (*AJ* 128, 585), ~170k SDSS spectra.
- Yip: >99% of galaxies on a 2-D locus in the first three eigencoefficient ratios; ~92% variance in 10 components; ≤8 for extreme emission-line objects.

**Photometry.**

- Cadiou, Laigle & Agertz 2025 (*MNRAS* 537, 1869; arXiv:2404.02962): nonlinear **ID ≈ 4.3 ± 0.5 (star-forming), ≈ 2.9 ± 0.2 (passive)**.
- This is a *direct precedent* for our heterogeneous-ID hypothesis (Section 8).
- "Galaxy Manifold" (arXiv:2210.05862): 2 parameters → 93.2% variance, linear early/late split at 85%.

**Net prior.**

- Clean photometry ID ≈ 2–5; spectra effective dim ≈ 3–10.
- Multimodal fusion can add physical axes (dust, AGN, metallicity, environment, inclination, redshift).
- **"~5–10 intrinsic dimensions" is an optimistic upper ceiling, not a point prediction.**
- A result of d̂ ≈ 3–6 is *consistent with the literature*, not disappointing.

### 2.3 The Hubble taxonomy (the control, not the thesis)

- Hubble 1926 (*ApJ* 64, 321) and 1936 (*The Realm of the Nebulae*): the tuning fork — ellipticals on the handle, S0 at the fork, two spiral prongs, irregulars aside.
- We use this only to **anchor interpretation and as a control**: a linear-probe / silhouette check that the human taxonomy is decodable.
- The novelty is the *geometry* and the *concepts beyond* the taxonomy.

## 3. Background: AION-1, the gap, and the ML lineage we plug into

All model numbers are from the AION-1 paper (arXiv:2510.17960), its HTML, and the HF card `polymathic-ai/aion-base`; cross-references appear as [aion1.md:line](papers/aion1/aion1.md).

### 3.1 What AION-1 is

- **Identity.** "AstronomIcal Omni-modal Network," Polymathic AI (Flatiron/Simons + NYU, Cambridge, Princeton, LBNL).
- **Architecture.** One encoder–decoder transformer trained by **multimodal masked modeling** (4M-style; Mizrahi et al. 2023) over **39 modalities** — multiband images, optical spectra, catalog scalars.
- **Tokenizers.** FSQ image codebook 2¹², LFQ spectrum 1024, scalar 1024, scalar-field FSQ 1000.
- **Self-supervised**, no labels ([aion1.md:9](papers/aion1/aion1.md#L9), [aion1.md:46](papers/aion1/aion1.md#L46)).

### 3.2 Training data

**[CORRECTION]** Not "200M galaxies."

- **>200 million *observations* of stars, galaxies, AND quasars** across five surveys — Legacy Survey, HSC, SDSS, DESI, Gaia — from the Multimodal Universe (arXiv:2412.02527).
- ~4M cross-matched multimodal objects actually trained on (HF card).
- Per-survey: Legacy ~122M, HSC ~2.5M, SDSS ~4M, DESI ~1M, Gaia ~220M ([aion1.md:98-126](papers/aion1/aion1.md#L98-L126)).

### 3.3 Variants — and our choice

**[CORRECTION — variant-dependent dimension]** ([aion1.md:301-308](papers/aion1/aion1.md#L301-L308)):

| Variant | Blocks (enc/dec) | Hidden d | MLP | Heads | Params |
|---|---|---|---|---|---|
| AION-1-Base | 12 / 12 | **768** | 3072 | 12 | 300M |
| AION-1-Large | 24 / 24 | **1024** | 4096 | 16 | 800M |
| AION-1-XL | 24 / 24 | **2048** | 16384 | 32 | ~3B |

- So v1's "768" and "1024" were the Base and Large variants.
- **Decision (given "the 1B model in fp16"):** there is no exact-1B checkpoint; the ~1B variant is **AION-1-Large (800M), d = 1024**, ~1.6 GB in fp16.
- Every dimension-dependent statement assumes **d = 1024.**

### 3.4 Embeddings — no CLS token; we build them ourselves

**[CORRECTION]**

- Freeze the encoder, discard the decoder.
- The encoder emits a *per-token* sequence Z = {z₁,…,z_T}, z_t ∈ ℝ^d (HF API: up to `num_encoder_tokens` ≈ 600 tokens × d).
- One vector per galaxy via **mean pooling** e = (1/T) Σ_t z_t (Eq. 7) or **attentive pooling** (Eq. 8).
- Loaded via `from aion import AION; AION.from_pretrained('polymathic-ai/aion-large')` + `CodecManager`.
- **[v3 confirmed]** the project will download AION-1-Large and generate embeddings — so generating ~10⁴ embeddings (cheap, single GPU, minutes–hours) and training light heads (SAE, GPLVM, the pullback decoder) is in scope.

### 3.5 What the paper already established — and the gap

AION-1 (frozen + light head) already shows:

- PROVABGS property regression (R² up to ~0.96), GZ10 morphology (AION-L 87.2%), GZ-DECaLS retrieval (spirals nDCG@10 ≈ 0.64) ([aion1.md:405-604](papers/aion1/aion1.md#L405-L604)).
- So physics is **decodable** and similar objects are **close.**

**The gap (triple-confirmed):**

- AION-1 ran only linear probing + cosine retrieval — **no** intrinsic dimension, curvature, manifold, concept, PCA, t-SNE, or UMAP analysis of its latent.
- AstroCLIP (arXiv:2310.03024) noted physical info is near-perfect but did **no** manifold geometry.
- The closest concept-discovery prior — the Euclid galaxy-morphology SAE (arXiv:2510.23749, Oct 2025) — found features "beyond the Galaxy Zoo tree" but did **no** intrinsic-dimension, curvature, or topology analysis.
- **Geometry-of-the-embedding-manifold for an astro foundation model is genuinely unoccupied.**

## 4. Scope, the toy-scale contract, and venue targets

- **Prototype, inference + light heads.** N ≈ 10⁴ galaxies; 2k subsample for the O(N³)-ish methods.
- We freeze AION-1; we train only tiny heads (SAE, GPLVM, the pullback decoder).
- The aim is to show the signal is real and stable enough for a full study.
- **Scope deliberately increased (per request)** to the six outputs of Section 1 plus the four advanced method families, the multimodal embedding, and the Riemannian flagship.

**The toy-scale contract (binding on interpretation).**

- At N = 2k in 1024-d we are deep in the curse of dimensionality.
- Pairwise distances concentrate; ID estimators are biased; persistent-homology features beyond H₀/H₁ are noisy; a single subsample is a random draw; "any direction correlates with something."
- Therefore, non-negotiably:
  - every headline number ships with a **bootstrap / credible interval**;
  - every geometric claim is **triangulated across ≥ 2 independent estimators or metrics**;
  - every topological feature is checked across **independent subsamples** with confidence bands;
  - every discovered concept survives a **permutation null + multiple-testing correction + seed-stability** check;
  - ID **> ~8–10** is treated with skepticism (a 2k sample resolves ID only up to ≈ log₂ n ≈ 11).

**Venue targets (dual, per request).**

- **Fast / exploratory:** NeurIPS **ML4PS** (Machine Learning and the Physical Sciences) workshop — 4-page, non-archival, astronomy + ML-interpretability in scope.
- **ML audience:** a NeurIPS/ICML **mechanistic-interpretability** workshop — foreground the geometry-of-concepts / Platonic framing.
- **Archival:** **MNRAS / ApJ / A&A** (ML section) if the concept-discovery results are strong and physically validated.

---

# Part II — Ingredients

## 5. The model specification (we build multimodal embeddings)

| Item | Specification | Source |
|---|---|---|
| Checkpoint | `polymathic-ai/aion-large`, fp16 | HF card; [aion1.md:301-310](papers/aion1/aion1.md#L301-L310) |
| Embedding dim d | 1024 | Fig. 6, [aion1.md:305](papers/aion1/aion1.md#L305) |
| Encoder output | per-token Z ∈ ℝ^{T×d}, T ≲ 600 | HF API; Eq. 7–8 |
| Pooling | **mean** (primary); attentive (sensitivity) | [aion1.md:381-395](papers/aion1/aion1.md#L381-L395) |
| State | **frozen**; decoder discarded (a *separate* light decoder is trained for the pullback metric, §10.2) | [aion1.md:377-379](papers/aion1/aion1.md#L377-L379) |
| Tokenization | `CodecManager` per modality | §4 |
| Image preproc AION saw | 160×160 → center-crop 96×96; arcsinh; inverse-variance | [aion1.md:98](papers/aion1/aion1.md#L98), [aion1.md:142](papers/aion1/aion1.md#L142) |

### 5.1 [v4 — locked] Multimodal embeddings as the core representation

**Decision:** the core embedding fuses **image + photometry + redshift** (and, where available, spectra), not images alone.

- AION-1 supports this natively: concatenate the tokens from any subset of modalities and pass the union through the frozen encoder; no fusion module is needed ([aion1.md:397](papers/aion1/aion1.md#L397)).
- Per-galaxy modalities fused: Legacy `{g,r,i,z}` image (96×96), `{g,r,i,z}` + WISE fluxes, E(B−V), ellipticity, R_eff (scalars), redshift (scalar), and the DESI/SDSS spectrum when cross-matched.
- This yields a *richer* manifold whose geometry reflects multi-instrument physics — the more interesting object to characterize.

**[v4 — CRITICAL circularity safeguard.]**

- Any quantity fed to the encoder as an *input* is trivially decodable from the embedding, so it cannot serve as a non-trivial concept-probe target.
- Concretely: if redshift is an input token, the "redshift concept axis" (Section 11) and the "regress out redshift" step (Section 6.5) become circular.
- **Rule:** maintain **two embedding sets** —
  1. **E_full** (image + photometry + redshift + spectra): the primary object for the *geometry* arms (ID, curvature, topology, manifold reconstruction, SAE discovery).
  2. **E_held(target)** (image + photometry only, withholding the probe target): used for the *concept-disentanglement and nuisance-removal* arms, so a "redshift/mass axis" is a genuine inference, not a read-back of an input.
- Report which embedding set every number comes from. This safeguard is what keeps the concept results honest under multimodal fusion.

### 5.2 Two operational warnings (silent corruptors)

1. **Pool before you analyze.**
   - Forgetting to pool yields a ~6M × 1024 *token* cloud, not a 10⁴ × 1024 *galaxy* cloud.
   - Every galaxy-level statistic is then meaningless and homology will not finish.
2. **Reproduce the 160→96 crop + arcsinh on calibrated fluxes.**
   - Feeding raw cutouts, or JPEG viewer images, changes the input distribution and the embedding.
   - Use **FITS** (linear nanomaggies); tokenize through `CodecManager`.

Pooling has geometric consequences (mean vs attentive change δ, homology lifetimes, SAE features); we fix mean pooling and report attentive as a robustness check.

## 6. The data specification

### 6.1 Morphology labels — Galaxy Zoo DESI

- Walmsley et al. 2023 (*MNRAS* 526, 4768; arXiv:2309.11425).
- **[CORRECTION]** **8.67M** galaxies (rounds to "8.7M"), **r < 19.0**, **DESI-LS DR8**.
- Produced by **a variant of EfficientNetB0** (98-unit sigmoid head); *Zoobot* is the *framework*, not an architecture.
- Outputs are **predicted vote fractions** (smooth/featured, bar, spiral, merger), accurate to **5–10%.**
- Hosted on **Zenodo** (DOI 10.5281/zenodo.8360385; pin record + checksum).
- **[CRITICAL] decision-tree conditionality:** bar/spiral fractions are only defined for featured, non-edge-on disks — filter on branch relevance or you fabricate structure.
- **[v4 — locked]** label source = **GZ-DECaLS-SGC** (~171k, AION's retrieval/training footprint), the cleaner in-distribution control (Section 24).

### 6.2 Physical parameters (concept targets)

For the concept-axis and SAE arms we need *continuous physical scalars*:

- **redshift** — GZ DESI `external_catalog.parquet` `redshift` = **SDSS spec-z where available, else photo-z** (hybrid; spec fraction unverified, likely a minority — propagate photo-z error, flag spec vs photo). **[CAVEAT]** verify the cross-match join key.
- **stellar mass, sSFR, age, metallicity** — cross-match to **PROVABGS** (Hahn et al. 2023), the SED-fit catalog AION-1 itself used ([aion1.md:401](papers/aion1/aion1.md#L401)); ~120k DESI BGS galaxies with z, M⋆, t_age, Z_met, SFR.
- **structural** — Sérsic index n, half-light radius R_eff, asymmetry, concentration (Legacy `tractor` / GZ) as additional "candidate concept" targets.

### 6.3 Images and the cutout service

- Endpoints: `https://www.legacysurvey.org/viewer/cutout.fits?ra=..&dec=..&layer=ls-dr10&pixscale=0.262&bands=griz&size=160` (FITS, not JPEG). Max size 512.
- **[CAVEAT]** rate limiting → HTTP 429, no published quota; use a bounded pool (4–8), exponential backoff, write-`.tmp`-then-rename.
- **[CAVEAT]** north/south (BASS/MzLS vs DECaLS) PSF/noise heterogeneity can leak as a spurious axis — match AION's SGC/DR10 footprint; record declination.

### 6.4 Simulations — CAMELS (cross-domain arm)

- CAMELS (Villaescusa-Navarro et al. 2021; arXiv:2010.00619), (25 h⁻¹ Mpc)³, parameters **[Ω_m, σ₈, A_SN1, A_AGN1, A_SN2, A_AGN2]** (on-disk order; reading by wrong position is a silent bug), suites TNG/SIMBA/Astrid.
- **Not in AION training** → genuine OOD.
- **[CAVEAT]** must be rendered into mock g,r,i,z matching survey PSF/noise; realism mismatch is a confound; prior work (arXiv:2410.10606) shows all hydro models are detectably misspecified vs real SDSS — *a gap is the expected null.*
- **[v4 — locked]** optional/stretch (Section 24, Q4): delivered after Methods A–G + AstroCLIP.

### 6.5 Sample design and the inherited selection function

- ~10⁴ galaxies (morphology + physical params); fixed-seed 2k subsample for homology/curvature/δ.
- **Selection function** (AION §8.1, [aion1.md:655-657](papers/aion1/aion1.md#L655-L657)): magnitude/quality cuts, SGC footprint, 1″ cross-match, plus GZ r<19.
- Morphology is entangled with apparent size and redshift in ground-based imaging.
- **Regress out redshift before calling an axis "physical"** — using **E_held(redshift)** (Section 5.1) to avoid circularity.
- De-duplicate and remove r₁=r₂ ties before any NN-based estimator.

### 6.6 Build vs reuse — and the AION-Search warning

- **[v3/v4 resolved]** We **build** embeddings from FITS cutouts + scalar/spectral tokens through AION-1-Large.
- **[CRITICAL]** do *not* substitute AION-Search embeddings (`astronolan/*-embeddings`, arXiv:2512.11982) — they are contrastively re-aligned to a *text* space, so their geometry/topology differ from raw frozen encoder outputs; using them would invalidate every geometric conclusion.
- AstroCLIP weights *are* public and *are* used — but only for the explicit cross-model Platonic check (Section 15.2), not as a stand-in.

## 7. The metric problem and the honest-metric battery `[v3]`

Everything downstream is a function of distances on **E ∈ ℝ^{N×d}** (N ≈ 10⁴, d = 1024).
The metric is not a detail — three original arms (TwoNN's r^d derivation, δ-hyperbolicity, Vietoris–Rips) inherit their guarantees from it, and **in 1024-d the raw metric is broken.**

### 7.1 Distance concentration — quantify it, then fix it

- As d grows, the **relative distance ratio** RDR = (D_max − D_min)/D_min → 0 (Minkowski/concentration theorem; arXiv:2401.00422): every pair becomes nearly equidistant.
- So nearest-neighbor structure — the input to *all* metric-based arms — is unreliable.
- A NeurIPS 2024 theorem (Damrich et al., arXiv:2311.03087) shows Euclidean Vietoris–Rips *fails* in high dimensions ("all distances become too similar, hiding the true loops"), while diffusion/spectral distances recover the correct topology.

**[v3] Diagnostic (report this first):**

- Compute RDR and the NN/mean-distance ratio under each candidate metric on the 2k subsample.
- A near-zero RDR under Euclidean/cosine and a substantially larger RDR under the honest metrics is the *measured* justification — turning "why not just use cosine?" into a cited figure.

### 7.2 The honest-metric battery (precomputed metric everywhere downstream)

We replace the raw metric with **three independent manifold-respecting metrics**, and require conclusions to be stable across them (or to differ in an interpretable, density-driven way):

1. **Isomap geodesic** (Tenenbaum et al. 2000).
   - Shortest path on a kNN graph with Euclidean edge weights: d_G(x,y) = min over paths Σ |x_i − x_{i+1}|.
   - The baseline "honest ruler." `sklearn.manifold.Isomap.dist_matrix_`.
2. **Fermat / density-weighted geodesic** (Groisman, Jonckheere & Sapienza; arXiv:1810.09398).
   - D_{Q,α}(x,y) = inf over paths Σ |q_{j+1} − q_j|^α, α > 1.
   - Continuum limit: the density-conformal d_{f,β} = inf_γ ∫ f^{−β} ds, β = (α−1)/d.
   - α = 1 recovers Euclidean (control); α > 1 pulls paths through high-density regions and is outlier-robust.
   - `fermat` package (or sklearn-kNN + `scipy.sparse.csgraph.shortest_path` with weights |·|^α).
3. **Diffusion distance** (Coifman & Lafon 2006).
   - D_t(x_i,x_j)² = Σ_l λ_l^{2t} (ψ_l(x_i) − ψ_l(x_j))², equal to Euclidean distance in the diffusion-map coordinates (Section 9).
   - Averages over *all* paths, so it down-weights spurious kNN edges. `datafold`.

**[v3 — locked keystone, a publishability anchor.]**

- Fernandez, Borghini, Mindlin & Groisman (JMLR 2023, "Intrinsic Persistent Homology via Density-based Metric Learning") **prove** Gromov–Hausdorff convergence of (sample, Fermat distance) to (manifold, intrinsic Fermat distance).
- Via bottleneck stability, the Vietoris–Rips/Čech persistence diagrams converge.
- This upgrades the homology arm (Section 14) from "we ran ripser on Euclidean distances" to "we computed an **intrinsic, density-aware, outlier-robust, embedding-invariant** persistent homology *with a published convergence guarantee*."
- **[v4 — locked]** Fermat is the **primary** metric for topology/curvature; diffusion supplies the coordinate system; Euclidean is the control.

**Caveats (binding).**

- Each metric has a critical hyperparameter — k (Isomap/Fermat graph), ε (diffusion bandwidth), α (Fermat exponent).
- Too small disconnects the graph; too large short-circuits the manifold.
- Sweep them, verify a single connected component, report sensitivity.
- The convergence theorems assume i.i.d. sampling from a smooth compact manifold — treat them as asymptotic motivation on finite, selection-biased, possibly multi-component data, and **always include the Euclidean baseline as a control.**
- Do *not* re-embed the geodesic/diffusion distances before feeding δ-hyperbolicity — pass them as a *precomputed metric* (re-embedding reintroduces distortion).

---

# Part III — The measurements

## 8. Method A — Intrinsic dimension: four estimators triangulated `[v3 upgrade]`

**Goal.**
Estimate how many real degrees of freedom AION-1's manifold uses — globally, *locally*, with *uncertainty*, and *spectrally* — and compare to ambient d = 1024 and the astrophysical prior of ~3–10.
**A single number is not defensible; we report four mechanistically-independent estimators and show agreement.**

### 8.1 TwoNN (the base estimator) and its derivation

- TwoNN (Facco et al. 2017, *Sci. Rep.* 7:12140) uses μ_i = r₂(x_i)/r₁(x_i).
- Under local uniform density, successive enclosed volumes v₁ = ω_d r₁^d and (v₂ − v₁) are i.i.d. exponential.
- With w = (v₂−v₁)/v₁ (ratio of i.i.d. exponentials, P(w>t)=1/(1+t)) and μ^d = v₂/v₁ = 1+w, one gets P(μ>x) = x^{−d}, so:
  - pdf f(μ) = d·μ^{−d−1}
  - CDF F(μ) = 1 − μ^{−d}
  - identity −log(1−F(μ))/log μ = d
- **Linear fit (original paper):** regress −log(1−F_emp(μ_i)) on log μ_i through the origin, discarding the 10% largest μ; slope = d̂. (`skdim.id.TwoNN`.)
- **MLE:** ℓ(d) = N log d − (d+1)Σ log μ_i ⟹ d̂ = N/Σ log μ_i.
- **[CORRECTION]** that plain MLE is *biased*; the unbiased version is **d̂ = (N−1)/Σ log μ_i** (Denti et al. 2022, arXiv:2104.13832), with exact law d̂/d ~ InverseGamma(N, N−1).

### 8.2 [v3] Local intrinsic dimension (Levina–Bickel MLE)

A single global number averages away heterogeneity.
The **local** ID at a point z, from its K nearest-neighbor distances T_j(z), is:

  **m̂_K(z) = [ (1/(K−1)) Σ_{j=1}^{K−1} log( T_K(z) / T_j(z) ) ]^{−1}.**

- Sweeping K and mapping m̂_K(z) over the manifold tells us *where* dimensionality is higher (e.g., is the merger/rare-object region higher-dimensional than the red sequence?).
- This connects directly to the ML claim that *local* ID predicts alignment/generalization. `skdim.id.MLE`.

### 8.3 [v3] Bayesian intrinsic dimension (Gride + heterogeneous Hidalgo)

**Bayesian TwoNN / Gride** (Denti et al. 2022; `dadapy.compute_id_2NN_wprior`, `return_id_scaling_gride`).

- A conjugate Gamma(a,b) prior gives a closed-form posterior **d | data ~ Gamma(a + N, b + Σ log μ_i)** → a *credible interval* on ID.
- So "11.3" becomes "11.3 [9.8, 12.6] (95% CrI)."
- **Gride** generalizes the ratio to the (n₂)-th over (n₁)-th neighbor, giving an **ID-vs-scale curve** that separates true dimension from noise-inflated small-scale ID, without subsampling.
- Density: f_μ(μ) = d (μ^d − 1)^{n₂−n₁−1} / [ μ^{d(n₂−1)+1} B(n₂−n₁, n₁) ].

**[v3 novel] Hidalgo — heterogeneous ID** (Allegra et al. 2020, *Sci. Rep.* 10:72222; `intRinsic::Hidalgo` via rpy2).

- A Bayesian mixture in which each galaxy belongs to one of K submanifolds of *different* dimension d_k: p(μ_i) = Σ_k p_k d_k μ_i^{−(d_k+1)}.
- A **local-homogeneity prior** ∝ ζ^{N_i(ζ)}(1−ζ)^{q−N_i(ζ)} favors configurations where a point's neighbors share its label.
- This is the *only* tool in the stack that can test the hypothesis (motivated by Cadiou et al. 2025) that **star-forming vs passive galaxies occupy submanifolds of different intrinsic dimension.**
- A result like "star-forming: d = 5.1 [4,6]; passive: d = 3.0 [2.5,3.5], P(distinct) > 0.99" is a genuinely novel, physically-interpretable claim that cross-validates the Mapper/SAE clusters.

### 8.4 [v3] Spectral and ARD intrinsic dimension (independent mechanisms)

- **Spectral-gap ID** from the diffusion operator (Section 9): d_spec = argmax_k (λ_k − λ_{k+1}). Mechanistically independent of nearest-neighbor ratios.
- **BGH log-log slope** (Section 9.2): the max slope of log T(ε) vs log ε estimates ≈ d_int/2 — a third independent ID.
- **ARD latent dimension** from the Bayesian GPLVM and Bayesian PCA (Section 9.3): the number of un-pruned latent dimensions, a model-based marginal-likelihood criterion.

### 8.5 The triangulation claim and the linear null

- We report **TwoNN (linear + unbiased MLE, IG CI), local Levina–Bickel (vs K), Gride scale curve, Hidalgo per-population, spectral gap, BGH slope, and GPLVM/BPCA ARD** — and headline only where ≥ 3 independent estimators agree.
- The **linear null** is Bayesian PCA's automatically-determined dimension (Section 9.3): if the nonlinear IDs are *much lower* than the linear PCA dimension, we have quantitative evidence of nonlinear curvature.
- Libraries: `scikit-dimension`, `dadapy`, `intRinsic` (rpy2), `datafold`.

> **Expected at toy scale.** d̂ ≈ 4–8 with CI ±1–2, lower for passive than star-forming (the Hidalgo headline). **Dimensional-collapse caveat:** SSL encoders under-use their nominal dimension, so d̂ ≪ 1024 is expected and not by itself evidence of a clean physical manifold.

## 9. Method B — Manifold reconstruction: diffusion maps + Bayesian GPLVM `[v3]`

**Goal.**
Build a *geometry-faithful*, *reproducible*, and *probabilistic* low-dimensional coordinate system — replacing the stochastic UMAP picture — to (a) color by physics, (b) supply the diffusion metric to the topology/curvature arms, and (c) give a generative latent with uncertainty.

### 9.1 Diffusion maps (Coifman & Lafon 2006)

Construction:

- Build a Gaussian affinity L_{ij} = exp(−‖x_i − x_j‖²/ε).
- **Anisotropic α-renormalization** L^{(α)} = D^{−α} L D^{−α} (D_{ii} = Σ_j L_{ij}).
- Row-normalize to a Markov matrix M = (D^{(α)})^{−1} L^{(α)}.
- Take eigenpairs M ψ_l = λ_l ψ_l and embed Ψ_t(x) = (λ₁^t ψ₁(x), …, λ_k^t ψ_k(x)).

The α knob (use α = 1):

- As ε → 0: α = 0 → density-biased graph Laplacian; α = ½ → Fokker–Planck; **α = 1 → the Laplace–Beltrami operator**, recovering Riemannian geometry **independent of sampling density.**
- This matters because our 10k galaxies are *not* uniformly sampled (selection effects), and only α = 1 guarantees the recovered geometry reflects the manifold, not where galaxies happen to be dense.
- `sklearn.SpectralEmbedding` is the α = 0 special case — **do not use it** (silently density-biased).

Why over UMAP/t-SNE:

- Deterministic, theory-backed (provable Laplace–Beltrami convergence), and astronomy-validated (diffusion maps beat PCA on SDSS spectra; Richards et al. 2009).
- A strictly stronger foundation for the "embedding manifold" claim than a UMAP cartoon.

### 9.2 [v3] Self-tuning bandwidth (BGH + variable bandwidth)

- **BGH auto-ε** (Berry–Giannakis–Harlim): T(ε) = (1/N²) Σ_{i,j} exp(−‖x_i−x_j‖²/2ε); pick ε* = argmax d log T / d log ε; the max slope estimates ≈ d_int/2 — a free, *independent* ID estimate (Section 8.4). `pydiffmap` `epsilon='bgh'`.
- **Variable bandwidth** (Berry & Harlim 2016): K_ε(x,y) = exp(−‖x−y‖²/(4ε ρ(x)ρ(y))) with ρ(x) ∝ q(x)^β (β < 0), giving uniform error control under wildly non-uniform galaxy density — important when rare quasars/lenses coexist with common galaxies.

### 9.3 [v3] Bayesian GPLVM + Bayesian PCA

Unlike diffusion maps (which only embed), a **Bayesian GPLVM** (Titsias & Lawrence 2010) infers a *generative*, *probabilistic* nonlinear latent with calibrated uncertainty and automatic dimension selection.

- Generative model: y_{:,d} ~ GP(0, K_{XX}) across the D = 1024 channels.
- ARD-SE kernel: k(x_i,x_j) = σ_f² exp(−½ Σ_q α_q (x_{i,q} − x_{j,q})²).
- Maximizing the variational ELBO drives irrelevant α_q → 0, **automatically pruning latent dimensions** — a model-based ID that cross-checks TwoNN.
- It also gives:
  - per-point latent uncertainty (error ellipses vs UMAP's overconfident dots);
  - a forward map to *sample synthetic galaxy embeddings* for counterfactuals and for testing the concept probes;
  - a marginal-likelihood comparison across latent dimensions.
- `GPyTorch` (`BayesianGPLVM` + `VariationalLatentVariable`); feasible at 2k points with 50–100 inducing points, minutes on a GPU.

Linear null:

- Bayesian PCA / PPCA (Tipping & Bishop 1999; Bishop 1999): t = Wx + μ + ε with an ARD prior on the columns of W (precision α_i per column drives redundant columns to zero), giving the automatically-determined *linear* dimension.
- If the GPLVM/TwoNN ID ≪ Bayesian-PCA dimension, the manifold is genuinely nonlinear. `sklearn.FactorAnalysis`.

**Answer to the project's question — "can only Bayesian approaches find manifolds?"**

- **No.** Diffusion maps, Isomap, LLE, and the frequentist TwoNN all recover manifold structure without any prior.
- What Bayesian methods *uniquely* add — and why they earn their place at toy scale — is **calibrated uncertainty, heterogeneous/local ID, and a generative latent**, jointly.
- Framing the paper around this question and answering it explicitly is itself a clean narrative spine.

**Caveats.**

- Diffusion maps: dense kernel + eigensolve is O(N²) memory (kNN-sparsify; `megaman` for >50k); higher eigenvectors are noisy at 2k (trust the first handful, bootstrap eigenvalue error bars); watch for *repeated/harmonic* eigenvectors; report across diffusion time t ∈ [4,64].
- GPLVM: non-convex ELBO — re-run from PCA init across seeds, watch for ARD pruning the wrong dimensions; report the lengthscale spectrum, not a single count.
- **Validate the whole pipeline on a synthetic manifold of known topology embedded in 1024-d with added noise before trusting AION-1.**

## 10. Method C — Curvature, the Riemannian pullback flagship, and tree-likeness `[v3/v4]`

**Goal.**
Replace the single fragile global δ-hyperbolicity number with (i) a learned **Riemannian metric** giving honest geodesics and curvature, and (ii) a **convergent, falsifiable battery** that asks "is it negatively curved / tree-like?" several independent ways *and localizes where it branches.*

### 10.1 [CORRECTION] Why the v1 JL-gap "curvature" diagnostic is dropped

- v1 proposed JL_dim = ⌈4 ε⁻² log N⌉, Gap = JL_dim − d̂, "large gap ⇒ non-Euclidean." **Unsound.**
- The formula is wrong: the canonical bound is k ≥ 4 ln N/(ε²/2 − ε³/3) ≈ 8 ln N/ε² (sklearn `johnson_lindenstrauss_min_dim`), so v1 underestimated by ~2–3×.
- The inference is a non sequitur: JL_dim depends **only on N and ε** (data-blind), so the gap is large for *every* dataset regardless of geometry.
- We keep JL **only** as a distance-preservation budget for the random projection in Section 14, and replace the curvature diagnostic with §§10.2–10.5.

### 10.2 [v4 — locked flagship] The Riemannian pullback metric (decoder → G = JᵀJ → geodesics → curvature)

This is the most mathematically prestigious arm and the one almost nobody has done on a galaxy foundation model.
It makes the metric *explicit* and lets us measure honest distances and curvature on the embedding manifold.

**The construction.**

- Train a light, smooth decoder g: ℝ^{1024} → ℝ^{D_out} from the frozen embedding to a galaxy observable space — e.g. reconstruct the `{g,r,i,z}` image, or predict the physical-parameter vector (M⋆, sSFR, n, …).
- The decoder Jacobian J_g(z) = ∂g/∂z induces a **pullback Riemannian metric** on the embedding space:
  - **G(z) = J_g(z)ᵀ J_g(z).**
- Curve length under g: L[g(γ)] = ∫₀¹ ‖J_γ γ̇‖ dt.
- **Magnification factor / volume element:** dV = √(det G(z)) dz — a scalar field showing where the manifold locally stretches/compresses (flagging "transition" regions like the green valley).
- Stochastic decoder g(z)=μ(z)+σ(z)⊙ε: expected metric M̄_z = J_z(μ)ᵀJ_z(μ) + J_z(σ)ᵀJ_z(σ).

**Geodesics (honest distances on the curved surface).**

- Geodesics minimize energy: E[γ] = ∫₀¹ ⟨γ̇, M_γ γ̇⟩ dt, with fixed endpoints.
- Euler–Lagrange gives the geodesic ODE: γ̈^k + Γ^k_{ij} γ̇^i γ̇^j = 0.
- Christoffel symbols: Γ^k_{ij} = ½ g^{kl}(∂_i g_{jl} + ∂_j g_{il} − ∂_l g_{ij}).
- The **geodesic-vs-Euclidean-chord ratio** is itself a localized curvature/distortion statistic, far richer than one δ.

**Curvature (the tensor, not a scalar summary).**

- Sectional curvature: K(u,v) = ⟨R(u,v)v,u⟩ / (⟨u,u⟩⟨v,v⟩ − ⟨u,v⟩²), with Riemann tensor R.
- A concrete, falsifiable hypothesis to test (Shao et al. 2018 found learned image manifolds are surprisingly *near-zero* curvature, so linear latent paths approximate geodesics): does AION-1's galaxy manifold share this near-flatness, or is it genuinely curved where morphology transitions?

**What it adds and why it's the flagship.**

- Replaces the heuristic "Euclidean distance is fine" with an exact statement (√det G field) of *where* it fails.
- Provides geodesic-corrected neighborhoods that make the TwoNN/Mapper/persistent-homology inputs more honest.
- Tests a concrete, falsifiable claim on a *science* foundation model.

**Libraries & cost.**

- `geomstats` (`PullbackMetric` / `PullbackDiffeoMetric` build G = JᵀJ from an immersion; `RiemannianMetric.exp/log/geodesic/dist`; `Connection.curvature`, `ricci_tensor`, `sectional_curvature`).
- `torch.func.jacrev` (or autograd) for J of the trained decoder; `scipy.integrate.solve_bvp` for the geodesic BVP.
- Building G and its inverse for the geodesic ODE is the heaviest step but fine at 2k–10k points if g is light; reference solvers in `geometric_ml` (Arvanitidis).

### 10.3 Global-vs-local ID gap (curvature from dimensions)

- A curved d-manifold needs more than d linear dimensions to embed, so **(global PCA-variance ID) − (local TwoNN/Levina–Bickel ID) > 0 signals curvature**; ≈ 0 signals a locally-flat manifold.
- Curvature-aware ID (**CA-PCA**, arXiv:2309.13478) hardens this: it corrects the eigenvalue thresholds for curvature via λ_i^{(d)} = λ̂_i + c(d) Σ_{j>d} λ̂_j ≈ 1/(d+2), c(d) = (5d²+6d+8)/(d(d+2)(d+4)).

### 10.4 [v3] Local curvature on the point cloud (second fundamental form)

Generator-free signed curvature *per galaxy*:

- At each point, local PCA gives the tangent/normal split.
- A least-squares quadratic (second fundamental form) in the normal directions yields principal curvatures k₁,k₂,….
- Hence **Gaussian K = k₁k₂** and **mean H = (k₁+k₂)/2.**
- Negative K = saddle/hyperbolic (branching/transition — e.g. the green-valley saddle); positive K = elliptic/cluster.
- Color this field on the diffusion/Mapper layout and correlate with mass/sSFR.
- (Demonstrated to reveal saddle geometry at biological *branching points* in single-cell data — a direct analogy; arXiv:2502.03750, arXiv:2308.02615.)
- `sklearn` (local PCA) + `numpy` lstsq.

### 10.5 [v3] Graph Ricci curvature (Ollivier / Forman) — localizing the branches

Build a kNN graph and assign each **edge** a discrete Ricci curvature.

- **Ollivier–Ricci:** κ(x,y) = 1 − W₁(m_x, m_y)/d(x,y), with the α-lazy walk measure m_x^α(z) = α if z=x, (1−α)/deg(x) if z ~ x.
  - κ < 0 on **bridges/branch points** (poorly-overlapping neighborhoods, expensive transport); κ > 0 inside clusters.
  - The Lin–Lu–Yau limit κ_LLY = lim_{α→1} κ^α/(1−α) is less α-sensitive.
  - `GraphRicciCurvature.OllivierRicci` (POT + NetworKit; Sinkhorn for speed).
- **Forman–Ricci:** F(e) = 4 − deg(v) − deg(w) (unweighted) — an O(1)-per-edge proxy to screen the *full 10k* graph, then drill into negative-curvature regions with Ollivier on the 2k subsample.
  - Agreement between the two is a robustness check.

**This is the qualitative leap:**

- Instead of one global yes/no, a per-edge signed-curvature *field* that says **where** the manifold branches.
- We cross-tabulate it against morphology (do negative-curvature ridges coincide with spiral↔elliptical / star-forming↔quiescent transitions?), the concept axes (Section 11), and the SAE features (Section 12).

### 10.6 [v3] Tree-likeness battery (is it actually a tree?)

Negative curvature is necessary but not sufficient for "tree" — so we add direct tree tests.

- **Hyperbolic-embedding distortion** (Sala et al. 2018; Nickel & Kiela 2017).
  - Fit the 2k distance matrix into a Poincaré ball B^k (d_H(x,y) = acosh(1 + 2‖x−y‖²/((1−‖x‖²)(1−‖y‖²)))) with Riemannian Adam.
  - Compare average distortion D(f) = (1/C(N,2)) Σ_{u≠v} |d_V − d_U|/d_U against a matched-dimension Euclidean MDS baseline, vs k.
  - Trees embed in hyperbolic space with arbitrarily low distortion but never in Euclidean space (Bourgain) — so *much lower hyperbolic distortion at small k* is strong tree evidence.
  - `geoopt` (PoincareBall/Lorentz, RiemannianAdam); float64, minimize squared distance (not raw acosh).
- **Additive-tree / four-point distribution.**
  - Fit a neighbor-joining tree and report (a) the *full distribution* of the per-quadruple Gromov δ (mean and 95th percentile, normalized by diameter) over ~10⁶ sampled quadruples — **not** the single worst-case max — and (b) the tree distortion ‖d_X − d_T‖.
  - A 2025 differentiable Gromov-hyperbolicity relaxation (arXiv:2505.21073) gives a principled "nearest tree metric" and residual.
  - This is the proper, robust replacement for v1's single δ.
- **Cophenetic correlation.**
  - Build a dendrogram (average/Ward) and measure how faithfully it reproduces pairwise distances (`scipy.cluster.hierarchy.cophenet`) — a cheap "is it a hierarchy?" score.

**The convergence claim.**

- If hyperbolic distortion is low *and* the additive-tree/cophenetic fit is good *and* negative Ricci curvature concentrates on bridges that map to morphological transitions, the tuning-fork claim is *triangulated across geometric, combinatorial, and agglomerative viewpoints.*
- If they disagree, the nuanced finding ("hierarchical in some respects, not a clean tree") is itself publishable.

**Caveats.**

- All distance-based tests inherit the metric (Section 7) — report under ≥ 2 metrics.
- Curvature sign depends on graph construction (k, mutual-kNN, pruning) — sweep k, verify stable negative-curvature ridges.
- δ-distribution needs sampling (exhaustive is O(N⁴)).
- Hyperbolic optimization is non-convex with NaN risk near the ball boundary (Lorentz model, h-MDS init).
- Validate every curvature/ID number on synthetic manifolds of known curvature (sphere K>0, saddle K<0, torus mixed) first.

## 11. Method D — Concept axes and the geometry of concepts `[v3 upgrade]`

**Goal.**
For physical labels (redshift, smooth/bar/spiral fractions, mass, sSFR, Sérsic n), find the decoding directions and test whether their *geometry* obeys the laws that hold for LLM representations.
**Use E_held(target) (Section 5.1) for any target that is also an input modality, to avoid circularity.**

### 11.1 Ridge probes with the label-correlation null

- On standardized Z, fit ridge probes ŵ_y = argmin ‖y − Zw‖² + λ‖w‖² (CV-λ).
- The disentanglement matrix is θ_ij = arccos(ŵ_i·ŵ_j/(‖ŵ_i‖‖ŵ_j‖)).
- **The guard that makes the angle meaningful:** under a perfectly linear, whitened, disentangled representation with correlated labels, cos θ_ij = ρ_ij (the label correlation), so the *null* angle is **arccos(ρ_ij)**.
- Non-orthogonal probes are *expected from label correlation alone* (TCAV §3.6; Träuble et al. 2021).
- Report ρ_ij beside θ_ij; interpret only deviations from arccos(ρ_ij); gate on held-out R²; bootstrap CIs; report λ-sensitivity (ridge shrinkage rotates directions by d_j²/(d_j²+λ)).
- **Confounded labels ≠ confounded representation.**
- High-dim near-orthogonality means "orthogonal" is the uninformative default (only small angles are diagnostic).

### 11.2 [v3] The geometry of categorical & hierarchical concepts (Park et al. 2024)

The LLM-interpretability "geometry of concepts" makes *testable* predictions we port to galaxies:

- a **binary** concept (e.g. barred vs not) = a single direction v_c = mean(z|c=1) − mean(z|c=0);
- a **categorical** concept with m values (e.g. Hubble stage) = a **simplex/polytope** of m vertices;
- a **hierarchy** (smooth ⊃ {E, S0}; featured ⊃ {spiral, barred}) appears as **child vectors orthogonal to parent vectors under the whitened/causal inner product** ⟨a,b⟩_C = aᵀ Σ⁻¹ b.

Test:

- Estimate concept vectors, whiten by Σ^{−1/2}, and check (a) predicted parent–child orthogonality and (b) simplex structure for sub-types.
- **This is novel:** it asks whether a *frozen astronomy* foundation model's geometry obeys the same concept laws as language models.
- *Violations* are themselves findings (the geometry-of-concepts assumptions were derived for LLM unembeddings; their transfer to a multimodal encoder is a hypothesis, not a given).
- `scikit-learn` (ridge, Cholesky/PCA whitening) + `numpy/scipy`.

> **Expected.** High R² for smooth/spiral/redshift/mass; the interesting result is whether morphology axes are separated from redshift *beyond* their label correlation (did AION factor out the nuisance?), and whether the smooth/featured hierarchy is near-orthogonal under the causal inner product.

## 12. Method E — Unsupervised concept discovery (sparse autoencoders) `[v3, novel payload]`

**Goal.**
Stop *choosing* the concepts.
Let the embedding **reveal its own** organizing axes, then test which align with physics and which are physically-meaningful-but-unnamed ("alien") concepts.

This is the single highest-novelty arm, and the gap is real:

- The only published astro-SAE work (Euclid morphology, arXiv:2510.23749; scientific-text, arXiv:2408.00657) is very recent and does **not** target a multimodal astronomy *foundation* model or analyze manifold geometry.

### 12.1 The sparse autoencoder

A sparse autoencoder (SAE) decomposes each dense 1024-d embedding into a sparse combination of an overcomplete dictionary (width m = R·1024, expansion R ∈ {2,4,8}).

- **L1 SAE** (Bricken et al. 2023; Cunningham et al. 2023): x̂ = W_dec f + b, f = ReLU(W_enc(x − b) + b_enc), loss ‖x − x̂‖² + λ‖f‖₁; decoder columns (dictionary atoms = candidate monosemantic concept directions) unit-normalized.
- **[v3 preferred at toy scale] TopK + AuxK** (Gao et al. 2024): h = TopK(W_enc(x − b_pre)) sets L0 = k directly (no L1 shrinkage bias); the AuxK auxiliary loss ‖e − ê‖² (dead latents reconstruct the live-feature residual e = x − x̂) **resurrects dead atoms** — critical in the small-data regime.
- **BatchTopK** (Bussmann et al. 2024) selects top n·k across a batch, letting rare morphologies (mergers, lenses) use more active concepts — well-matched to the rare-structure-discovery angle.
- `dictionary_learning` (saprmarks) or `SAELens`.

### 12.2 [v3] Scoring discovered features against physics

For each dictionary feature f_j:

- **alignment** s_j = max_p |Spearman(f_j, y_p)| over physical labels y_p (mass, sSFR, redshift, morphology, Sérsic n) — high s_j = AION re-discovering known physics;
- **novelty** = 1 − R²(f_j | all labels) — features with strong, coherent activation but *low* explainable variance from all physical labels are candidate **"alien" concepts** (the headline novel result), ranked for follow-up via top-activating-galaxy montages and ablation.

This mirrors the Euclid-SAE protocol (max|Spearman| + R² vs Galaxy Zoo) but adds the geometry:

- we then locate the discovered features in the manifold (do "alien" features live on negative-curvature ridges? in higher-local-ID regions?), tying the SAE arm to Sections 8–10.

### 12.3 [v3] Guards (without which this is circular)

In 1024-d *any* direction correlates with *something*. Therefore:

- **seed/width stability** — train at R ∈ {2,4,8} across ≥ 3 seeds, headline only features that recur (SAEs do not find canonical units);
- **permutation null + multiple-testing correction** on the Spearman/R² scores;
- **causal validation** — ablate/steer a feature and confirm the predicted change in reconstructed embedding / downstream probe (a correlation is not "the model uses it");
- report variance-explained, L0, %-alive.
- Compute is cheap (a 1024-in / few-thousand-atom SAE on 10k vectors trains in minutes on one GPU); the bottleneck is *analysis rigor*, not FLOPs.

> **Expected.** A subset of features cleanly tracking mass / sSFR / redshift / smoothness (AION re-discovering physics), and a smaller, seed-stable subset of high-coherence / low-R² features (candidate alien concepts) — e.g. the Euclid-SAE-style "dust lanes in edge-on disks" or "ellipticals with bluer companions." Even a *handful* of validated alien features is a strong, dual-venue result.

## 13. Method F — Topological skeleton (Mapper)

**Goal.**
A graph summary: does the manifold branch like a tuning fork, flow like a continuum, or show no stable structure?

- Mapper (Singh, Mémoli & Carlsson 2007): filter f → overlapping cover → cluster within preimages → nerve graph; for a scalar filter it approximates the Reeb graph of f.
- `kepler-mapper` (`Cover(n_cubes, perc_overlap)`; default clusterer is DBSCAN **not** KMeans; DBSCAN eps=0.5 is meaningless in 1024-d — cluster in the diffusion coordinates).

**[CRITICAL] The filter manufactures the structure.**

- A redshift lens *forces* the spine to track redshift; "branches" found that way cannot be a discovery.
- We therefore use **non-redshift filters** — diffusion coordinates (Section 9), local ID, eccentricity/centrality, or local curvature (Section 10) — and report *which* physical variable each branch correlates with.
- Branches must be **stable across the (n_cubes, perc_overlap, clusterer) grid**, survive a **label-shuffle null**, and ideally come with Carrière–Michel–Oudot confidence regions (JMLR 2018).
- Treat any single graph as one sample.

> **Expected.** A robust 2–3-branch tuning fork would be: branches stable under a *non-redshift* (e.g. diffusion-coordinate) filter, correlating with smooth/featured/barred, surviving the null. A single stable spine = continuum. Unstable branches = null.

## 14. Method G — Persistent homology with a guaranteed intrinsic metric `[v3 upgrade]`

**Goal.**
Count, with confidence, the connected components (β₀) and loops (β₁) of the manifold — under a metric that is *trustworthy* in high dimensions and a guarantee that the diagram converges.

### 14.1 The construction and the corrected complexity

- Vietoris–Rips across a growing threshold → persistence diagram (birth vs death; far from diagonal = signal).
- **[CORRECTION]** "O(N³)" is imprecise: the cubic bound is in the *number of simplices* (Otter et al. 2017); the VR complex has 2^O(N) simplices uncapped, O(N^{k+1}) capped at dim k.
- For β₀,β₁, ~C(2000,3) ≈ 1.3×10⁹ simplices at 2k — we **must** set maxdim = 1 and an edge threshold.
- `ripser.py` (Bauer's Ripser) / `GUDHI`.

### 14.2 [v3] The intrinsic-metric upgrade (keystone)

- Compute VR persistence on the **Fermat distance** (Section 7), invoking Fernandez et al. (JMLR 2023): the sample diagram converges (Gromov–Hausdorff + bottleneck stability) to the diagram of the manifold under the *intrinsic* Fermat distance.
- So the Betti numbers become:
  - **embedding-invariant** (re-rotating/re-scaling AION's output leaves them unchanged in the limit);
  - **density-aware**;
  - **outlier-robust**;
  - correct over a *longer* filtration interval (convexity radius ≫ Euclidean reach).
- Headline figure: Euclidean-vs-Fermat-vs-diffusion barcode comparison (a metric-robustness argument referees in both fields respect).
- The NeurIPS-2024 result (Section 7.1) is the independent justification that the Euclidean diagram is *unreliable* in 1024-d.

### 14.3 [v3] Comparative topology (RTD)

- A within-set diagram is hard to interpret.
- **Representation Topology Divergence** (Barannikov et al., ICML 2022; `RTD`) compares the multi-scale topology of two *matched* point clouds, yielding a single dissimilarity.
- Use it to quantify how AION-1's manifold topology differs from (a) **AstroCLIP**'s on the same galaxies (the Platonic cross-check, Section 15) and (b) the **physical-parameter manifold** (mass/sSFR/redshift space).
- This serves both Framing B (geometry) and the Platonic narrative.

### 14.4 Confidence (corrected)

- Many independent 2k subsamples → report the spread of β₀,β₁.
- **Fasy et al. 2014** confidence band: **[CORRECTION]** half-width **√2·c_n**, not 2c_n; estimate c_n by bootstrap.
- **[CORRECTION]** PCA-to-50 has no (1±ε) distance guarantee (so the homology-stability theorem doesn't cover it); a **JL random projection** is the safe reduction (the one legitimate use of the JL bound).
- Use the Section-7 metric; handle β₀'s essential (infinite) class explicitly in any bottleneck/Wasserstein comparison.

> **Expected.** β₀ ≈ 1–2 with confidence (a stable β₀ = 2 would match the red/blue bimodality); β₁ mostly noise at 2k — do not over-read.

## 15. Method H — Cross-domain (CAMELS) & cross-model (Platonic) checks `[v3 upgrade]`

### 15.1 CAMELS sim-vs-real (optional/stretch)

- Mock-observe CAMELS per suite (matched PSF/noise), embed, and measure graded overlap (kNN-density/OOD scoring, MMD/OT, and the bottleneck/RTD distance between persistence diagrams).
- Report **per suite** (TNG/SIMBA/Astrid).
- The informative outputs are *graded* overlap, *which physical directions* (Sections 11–12) drive divergence, and *physics vs mock-realism* — not a binary "they don't overlap" (the expected null, arXiv:2410.10606).
- Optional because of the rendering effort (Section 24, Q4).

### 15.2 [v3] AION-1 vs AstroCLIP — an astrophysical Platonic test

Embed the *same* galaxies with AION-1 and with public AstroCLIP, and compare:

- **local neighborhood alignment** (mutual-kNN: fraction of shared neighbors) vs **global** (CKA), privileging *local* structure per the 2026 critique that global alignment degrades at scale while local neighborhood geometry is the robust signal;
- **RTD** between the two manifolds, and between each and the physical-parameter space.

This turns "galaxy embedding geometry" into "an astrophysical test of representational convergence" — broadening appeal to the core ML-interpretability audience and de-risking the astro-only positioning.
`repsim`/`torchmetrics` (CKA), `RTD`.

---

# Part IV — Reading the result

## 16. Execution pipeline (dependency-ordered)

No hour budget — order follows data dependencies; each stage gates the next.

1. **Assemble the labeled sample.** Cross-match GZ-DECaLS-SGC morphology with PROVABGS physical params + redshift; apply r<19 and branch-relevance filters; record footprint/declination. → ~10⁴ galaxies with {morphology, z, M⋆, sSFR, n, …}.
2. **Obtain images faithfully.** FITS cutouts (size 160, ls-dr10, g,r,i,z) with backoff + `.tmp` rename; replicate the 160→96 crop + arcsinh.
3. **Embed (multimodal).** Tokenize image + photometry + redshift (+ spectra) via `CodecManager`; run frozen AION-1-Large (fp16); **mean-pool** → **E_full** ∈ ℝ^{10⁴×1024}. Also build **E_held(redshift)** and **E_held(mass)** (image+photometry only) for the concept arms. De-duplicate. Embed the same galaxies with AstroCLIP for §15.2.
4. **Metric battery (§7).** RDR concentration diagnostic; build Isomap-geodesic, Fermat(α), and diffusion distance matrices; verify single connected component; pick t/k/α/ε with sensitivity sweeps.
5. **Manifold reconstruction (§9).** Diffusion map (α=1, BGH ε); spectral-gap + BGH-slope ID; Bayesian GPLVM (ARD) + Bayesian PCA (linear null) → coordinates to color and to feed Mapper/curvature.
6. **Intrinsic dimension (§8).** TwoNN (linear + unbiased MLE, IG CI), local Levina–Bickel (vs K), Gride scale curve, Hidalgo per-population; triangulate.
7. **Riemannian flagship (§10.2).** Train the light decoder g; build G = JᵀJ; compute magnification √det G, geodesics, sectional curvature; compare geodesic vs Euclidean neighborhoods.
8. **Curvature & tree battery (§§10.3–10.6).** Global−local/CA-PCA gap; second-fundamental-form curvature field; Ollivier/Forman-Ricci on kNN; hyperbolic-distortion-vs-Euclidean; NJ additive-tree + δ-distribution; cophenetic. Cross-tabulate negative-curvature ridges with morphology.
9. **Concept axes (§11).** Ridge probes on E_held (CV-λ, R², ρ-null, bootstrap); geometry-of-concepts simplex/orthogonality under the causal inner product.
10. **Concept discovery (§12).** Train TopK + BatchTopK SAEs (+AuxK) on E_full at R∈{2,4,8}×{≥3 seeds}; score features (max|Spearman|, residual-R²); permutation null + correction; causal ablation; locate features in the manifold.
11. **Topology (§§13–14).** Mapper (≥2 filters incl. non-redshift, stability + null); Fermat-metric VR persistence (maxdim=1) over many 2k subsamples with Fasy bands; Euclidean-vs-Fermat-vs-diffusion barcodes; RTD vs AstroCLIP and vs physical space.
12. **(Optional) CAMELS (§15.1).**
13. **Synthesize** into one of the decision outcomes (§18); every number with its interval, metric, embedding set, and null.

## 17. Statistical bounds, nulls, and the reproducibility contract

| Quantity | Estimators | Error bar | Null / baseline |
|---|---|---|---|
| Intrinsic dim | TwoNN(lin+MLE), local Levina–Bickel, Gride, Hidalgo, spectral-gap, BGH-slope, GPLVM/BPCA-ARD | bootstrap; IG CI; Gamma CrI; ARD spectrum | ambient 1024; shuffled-feature; Bayesian-PCA linear dim; prior 3–10 |
| Pullback geometry | √det G field; geodesic-vs-chord ratio; sectional curvature | bootstrap over galaxies; decoder-seed variance | near-zero-curvature hypothesis (Shao); Euclidean chord |
| Curvature/tree | global−local & CA-PCA gap; 2nd-fund-form K; Ollivier/Forman; hyperbolic distortion; NJ δ-distribution; cophenetic | resample variance; bootstrap; distortion-vs-k curves | synthetic tree (≈0); matched random cloud; Euclidean-MDS distortion; δ-distribution (not max) |
| Concept axes | ridge θ_ij (on E_held); geometry-of-concepts simplex/orthogonality | held-out R²; bootstrap θ; λ-sensitivity | arccos(ρ_ij) label-corr null; high-d random ~90° |
| Concept discovery (SAE) | TopK/BatchTopK features; Spearman/R² scoring | seed × width stability; var-explained/L0/%-alive | permutation null + multiple-testing; causal ablation |
| Mapper | nerve graph, ≥2 filters | parameter-grid stability; CMO confidence regions | label-shuffle; multi-seed |
| Persistent homology | Fermat-metric VR, maxdim=1 | many 2k subsamples; Fasy band (√2·c_n) | Euclidean-vs-Fermat-vs-diffusion bottleneck; RTD |
| Cross-model | mutual-kNN, CKA, RTD | bootstrap over galaxies | shuffled correspondence; local vs global |

**Reproducibility contract.** Pin:

- AION-1-Large checkpoint + `aion`/`CodecManager` versions; which modalities are in E_full vs each E_held;
- Zenodo record + checksums; cutout `layer` + crop/normalization; metric choice + k/α/ε/t;
- all RNG seeds (subsample, diffusion, GPLVM, decoder, SAE, UMAP, bootstrap);
- library versions (`scikit-dimension`, `dadapy`, `intRinsic`, `datafold`/`pydiffmap`, `GraphRicciCurvature`, `geoopt`, `geomstats`, `dictionary_learning`/`SAELens`, `ripser.py`/`GUDHI`, `RTD`, `scikit-learn`, `GPyTorch`).

Point estimate **and** interval for every number; **multiple-comparisons discipline** — any single significant branch/loop/feature is provisional until it survives its null *and* a robustness scan.

## 18. Expected results and pre-registered decision rules

Re-calibrated against the verified priors; rules fixed *before* running.

### Outcome 1 — "Structured, physically-aligned, mildly-hierarchical manifold with discovered concepts" (best case → dual-venue paper)
- ID: estimators agree at ≈ 4–8; **Hidalgo finds distinct star-forming vs passive dimensions** (P > 0.95).
- Riemannian: √det G peaks at morphological transitions; geodesic ≠ Euclidean where curvature is non-zero.
- Curvature: global−local gap > 0; **negative Ollivier-Ricci ridges localize to morphological transitions**; hyperbolic distortion ≪ Euclidean at small k; NJ δ-distribution low; cophenetic high.
- Concepts: morphology axes separated beyond ρ (on E_held); smooth/featured hierarchy near-orthogonal under the causal inner product; **SAE yields seed-stable physics-aligned features AND ≥ a few validated "alien" features.**
- Topology: Mapper non-redshift branches stable + survive null; Fermat β₀ = 2 persistent; RTD shows interpretable AION-vs-AstroCLIP / AION-vs-physical differences.
- **Reading.** AION-1's native geometry is low-dimensional, physically-aligned, locally hierarchical, and contains concepts beyond the human taxonomy. **Strong dual-venue paper.**

### Outcome 2 — "Physically-aligned continuum, concept-rich but not tree-like" (middle → still publishable)
- ID ≈ 4–8; curvature mild but **no stable tree** (hyperbolic ≈ Euclidean distortion; δ-distribution moderate; single Mapper spine); concept axes well-decoded; SAE still yields aligned + some alien features.
- **Reading.** AION organizes galaxies on a *continuous, physically-aligned, concept-structured* manifold rather than a discrete hierarchy — a substantive claim about scientific FMs, consistent with the continuum astrophysics.

### Outcome 3 — "Generic / tangled geometry" (null → still publishable as a negative result)
- Concept axes not separated beyond ρ; geometry-of-concepts laws violated; no stable Mapper branches; curvature ≈ matched-random; ID collapsed or implausibly high; SAE features unstable across seeds.
- **Reading.** The embedding is geometrically generic and concept-tangled; the "physically meaningful directions" claim is unsupported *at the level of geometry* (decodability can still hold — §19). A calibrated negative that distinguishes *decodability* from *geometric structure* and tests whether the geometry-of-concepts laws transfer to a scientific encoder.

### How the arms combine (decision matrix)

The outcome is read from the *joint* pattern of the arms, not any single one:

| Signal | Outcome 1 (hierarchy) | Outcome 2 (continuum) | Outcome 3 (null) |
|---|---|---|---|
| ID triangulation (≥3 agree) | yes, ≈ 4–8 | yes, ≈ 4–8 | disagree / collapsed / implausibly high |
| Hidalgo heterogeneity | distinct passive vs SF | maybe | no |
| Tree battery (hyperbolic + NJ + cophenetic) | converges "tree" | converges "not a tree" | incoherent |
| Ricci ridges ↔ morphology | aligned | n/a (no ridges) | not aligned |
| Concept axes vs ρ-null (E_held) | separated beyond ρ | separated beyond ρ | at the ρ-null |
| Geometry-of-concepts laws | hold | partially | violated |
| SAE features | aligned + validated alien | aligned + some alien | seed-unstable |
| Mapper (non-redshift) | stable branches | stable single spine | unstable |
| Fermat β₀ | 2 (bimodality) | 1–2 | unstable |

- **Borderline cases are expected and informative.** A common realistic pattern is "ID and concepts clean, tree battery says continuum-not-tree" → Outcome 2; or "geometry clean but SAE unstable" → Outcome 2 with concept discovery demoted to the single-validated-example fallback (§20).
- **No single arm can promote to Outcome 1 alone.** The hierarchy claim requires the tree battery *and* the Ricci-ridge↔morphology mapping *and* a non-redshift Mapper branch — by design, to defeat confirmation bias.

> **All three are publishable.** Every outcome is informative — that asymmetry of value is what makes this a good exploratory bet at increased scope.

## 19. Interpretation at toy scale: what to believe and what not

Each number, the tempting wrong reading, and the defensible one.

**Intrinsic dimension.**
- *Wrong:* "d̂ = 6 ⇒ galaxies have six degrees of freedom."
- *Defensible:* "Under the Fermat metric, AION-1-Large's multimodal embedding of this r<19 SGC sample uses ≈ 6 effective local dof (95% CI/CrI [a,b]) by ≥ 3 independent estimators, ≪ 1024 and consistent with the ~3–10 ID literature; Hidalgo separates passive (≈3) from star-forming (≈5)."
- Remember **dimensional collapse**: d̂ ≪ 1024 is expected for SSL and not by itself evidence of clean physics.

**Pullback geometry.**
- *Wrong:* "√det G is large here, so this is where galaxies evolve."
- *Defensible:* "Under the decoder g, the manifold's volume element peaks near the smooth↔featured boundary, and geodesics there deviate most from Euclidean chords — a static geometric transition, not a temporal one." The metric depends on the decoder target; report the target and a decoder-seed sensitivity.

**Curvature.**
- *Wrong:* "δ small ⇒ it's the Hubble tree."
- *Defensible:* "Hyperbolic distortion is X× lower than Euclidean at k=5, the NJ δ-distribution median is low, and negative Ollivier-Ricci edges concentrate on the smooth↔featured boundary — three independent signals of localized hierarchy." Believe "tree-like" only on convergence of the battery, not one number.

**Concept angles / geometry-of-concepts.**
- *Wrong:* "redshift and spiral axes 40° apart ⇒ confounded."
- *Defensible:* "On E_held(redshift) (so redshift was not an input), 40° vs the arccos(ρ=0.5)=60° null ⇒ *more* aligned than labels predict; both probes high-R²; λ-sensitive; exploratory." **Confounded labels ≠ confounded representation**; the orthogonality test is a *hypothesis* for a multimodal encoder, not an assumption.

**SAE features.**
- *Wrong:* "feature 217 correlates with mass ⇒ AION uses mass."
- *Defensible:* "feature 217 is seed-stable across R∈{2,4,8}, max|Spearman| with mass = 0.7 (permutation p < 0.001, corrected), and ablating it shifts the mass-probe prediction as predicted." For "alien" features: "high activation coherence, low R² from all labels, seed-stable, with an interpretable montage — a candidate concept beyond the catalog, pending physical follow-up." A single-seed, un-ablated correlation means nothing.

**Mapper branches.** Only branches that are stable, non-redshift-filtered, morphology-correlated, and null-surviving count.

**Persistent homology.**
- *Wrong:* "a β₁ loop ⇒ cyclic evolution."
- *Defensible:* "β₀ = 2 persists across 50 Fermat-metric 2k subsamples outside the Fasy band — robust two components consistent with bimodality; β₁ inside the band, not interpreted."

**The "learned evolution" claim.**
- *Defensible:* "AION-1's *static* embedding of *present-day* galaxies has a geometry whose branches/axes echo the population structure of the morphological sequence — a density of states, not a time-trajectory; with ≥ two quenching channels (Schawinski 2014), a branch mixes evolutionary endpoints."

**Decodability vs geometry (subtle, central).**
- AION-1 *already* decodes physics at high accuracy.
- A *null* geometric result does **not** contradict that — information can be linearly readable while the manifold is curved, tangled across concepts, or topologically featureless.
- Frame a negative as "decodable but not geometrically clean/disentangled," never "AION-1 doesn't encode physics."

**Selection function & projection artifacts.**
- Everything is conditional on the r<19/SGC/cross-matched slice.
- Morphology is entangled with apparent size/redshift (regress out z first, on E_held).
- Bimodality can be a coordinate distortion of a continuum (check under physical vs observed coordinates).

**Metric stability.**
- Any geometric claim must hold under ≥ 2 of {Euclidean, geodesic, Fermat, diffusion} or be reported as metric-dependent.

## 20. Falsifiability and threats to validity

**Pre-registered falsifiers.**
- "Hierarchical manifold" (Outcome 1) is **falsified** if hyperbolic distortion ≈ Euclidean, OR the NJ δ-distribution is high, OR negative-curvature ridges don't map to morphology, OR Mapper branches fail the non-redshift/stability/null tests.
- "Disentangled physics / geometry-of-concepts holds" is **falsified** if concept angles (on E_held) match the arccos(ρ) null, OR the simplex/orthogonality tests fail, OR probe R² is low.
- "Concepts beyond taxonomy" is **falsified** if no SAE feature is seed-stable, or all features are fully explained by trivial nuisances, or none survive ablation.
- "Low-dimensional manifold" is **falsified** if the ID estimators disagree beyond their CIs or collapse under the ties/un-pooled checks.
- "Heterogeneous ID (passive ≠ star-forming)" is **falsified** if Hidalgo's K = 2 fit does not beat K = 1 (posterior P(distinct) < 0.9) or the segmentation is unstable across ζ.
- "Curvature is real, not a metric artifact" is **falsified** if the curvature/tree verdict flips between the Euclidean, geodesic, Fermat, and diffusion metrics.
- "Platonic convergence" is **falsified** if the AION–AstroCLIP local (mutual-kNN) alignment is no better than the shuffled-correspondence null.
- "The pullback metric is informative" is **falsified** if √det G is flat everywhere (no transition structure) or its peaks move arbitrarily with the decoder seed/target.

**Threats — symptom and mitigation (a risk register).**

| # | Threat | How it shows up | Mitigation |
|---|---|---|---|
| 1 | Pipeline correctness | shapes wrong; homology never finishes; an "axis" is a read-back of an input | pool to one vector/galaxy; FITS + 160→96 crop; never AION-Search; **E_held** for any input-modality probe; the Appendix-D harness |
| 2 | Metric inconsistency | δ/Betti flip between metrics; anchors computed under a different metric | one metric set; recompute every anchor under it; report ≥ 2 metrics per claim |
| 3 | Toy-scale under-resolution | ID > 10 unstable; β₂ noisy; one subsample's loop vanishes in the next | maxdim = 1; many subsamples + Fasy bands; skepticism above ID ~8–10; CIs everywhere |
| 4 | Confirmation bias toward the fork | a plausible 2-branch Mapper that is actually a lens artifact | non-redshift filters; parameter-grid stability; label-shuffle null; rules fixed before running |
| 5 | SAE circularity / cherry-picking | a "concept" that is one seed's artifact, or any-direction-correlates-with-something | seed × width stability; permutation null + correction; **causal ablation**; report L0/%-alive |
| 6 | Geometry-of-concepts non-transfer | orthogonality/simplex laws derived for LLMs may not hold for a vision/multimodal encoder | treat as a *hypothesis to test*, not an assumption; a violation is a reported finding |
| 7 | Pullback-metric dependence | √det G / curvature changes with the decoder target or architecture | report decoder-target and decoder-seed sensitivity; use the physical-parameter decoder as primary, image-reconstruction as a check |
| 8 | Cross-model correspondence | AION-vs-AstroCLIP alignment inflated by mismatched galaxies | exact same galaxies; shuffled-correspondence null; privilege *local* (mutual-kNN) over global CKA |
| 9 | Selection function / projection | morphology entangled with apparent size/redshift; bimodality a coordinate artifact | condition every claim on the slice; regress out z on E_held; check physical vs observed coordinates |
| 10 | CAMELS mock-realism | a "physics gap" that is really a rendering mismatch | match PSF/noise + selection; report per-suite, graded; keep optional/stretch |

**Abort / pivot conditions (when to stop or change course).**

- **Abort the geometry arms** if the Appendix-D harness fails broadly — e.g. even on a clean synthetic sphere in ℝ^1024 the ID estimators disagree wildly and the homology is unstable at N = 2k. That would mean the toy scale is fundamentally under-resolved for *any* dataset, and the honest move is to report the harness result and either raise N or narrow the claims, not to publish unstable AION-1 numbers.
- **Pivot to images-only** if the multimodal E_full is dominated by a single modality (e.g. the embedding geometry is essentially the redshift token's), detectable as a near-1-D manifold whose principal direction is redshift — in which case the *appearance* geometry (images-only) is the more interesting object.
- **Drop the tree claim, keep the continuum claim** if the tree-likeness battery disagrees with itself (hyperbolic distortion low but cophenetic poor, or vice versa) — Outcome 2 is still a paper.
- **Demote concept discovery to a single validated example** if SAE features are broadly seed-unstable; even one rigorously-validated "alien" feature is worth reporting, but a catalog of unstable features is not.

**Minimal publishable unit.**

- The smallest result worth writing up is: **a validated intrinsic-dimension triangulation (≥ 3 estimators agreeing, with CIs) + one robust curvature/tree statement + one seed-stable, ablation-validated SAE concept**, all on a metric that passed the concentration diagnostic, with the Appendix-D harness in the supplement.
- That alone is the first geometric characterization of an astro FM and clears the ML4PS bar; everything beyond it (Hidalgo heterogeneity, the Riemannian flagship, the AstroCLIP Platonic test, CAMELS) strengthens toward the archival venue.

## 21. Deliverables, novelty, and venues

**Artifacts (figures/tables).**
1. ID triangulation: TwoNN/local/Gride/Hidalgo/spectral/BGH/ARD with CIs vs ambient + literature; Hidalgo per-population.
2. Metric-concentration (RDR) table + the honest-metric choice.
3. Diffusion-map coordinates colored by physics; GPLVM latent with uncertainty; Bayesian-PCA linear null.
4. Pullback-metric panel: √det G field, geodesic vs Euclidean interpolations, sectional-curvature map.
5. Curvature field (second-fundamental-form) + Ollivier/Forman-Ricci graph colored by curvature with morphology overlay; hyperbolic-vs-Euclidean distortion-vs-dimension; NJ δ-distribution; cophenetic.
6. Disentanglement matrix θ_ij vs ρ_ij + R² (on E_held); geometry-of-concepts simplex/orthogonality test.
7. **SAE feature catalog**: aligned features (with physics) and seed-stable "alien" features (with montages + ablations).
8. Mapper (≥2 filters) + Fermat-metric persistence with Fasy bands + Euclidean-vs-Fermat barcodes + RTD vs AstroCLIP / physical space.
9. (Optional) CAMELS per-suite overlap.

**Novelty (triple-confirmed gap).**
- AION-1, AstroCLIP, and the Euclid-SAE paper all stop short of any manifold-geometry analysis.
- This study is the first to (a) characterize an astro FM's embedding geometry (ID/curvature/Riemannian/topology) with a robust, multi-estimator, intrinsic-metric methodology, (b) test the geometry-of-concepts laws on a scientific encoder, and (c) discover physically-meaningful concepts *beyond* the human taxonomy and locate them in the manifold — under the Platonic umbrella with a concrete cross-model test.

**Venues.**
- ML4PS (fast, exploratory, dual-audience) → mech-interp workshop (ML framing) → MNRAS/ApJ/A&A (archival, if concept discovery validates).
- The "is Bayesian the only way to find a manifold?" question is a clean narrative spine that unifies the methods into a study rather than a grab-bag.

## 21.5 Anticipated reviewer objections (dual-venue FAQ) `[v4]`

Writing the rebuttals now is part of pre-registration: each is a design constraint, not an afterthought.

**From a machine-learning / interpretability reviewer.**

- *"Intrinsic-dimension estimators are notoriously unreliable — why trust one number?"*
  - We never report one. We triangulate ≥ 3 mechanistically-independent estimators (TwoNN, local Levina–Bickel, Gride, spectral gap, BGH slope, GPLVM-ARD), report CIs, and show each passing a known-answer test on a matched-N, matched-d synthetic sphere (Appendix D).
- *"Persistent homology in 1024-d is just noise."*
  - Agreed for the *Euclidean* metric — that is exactly the Damrich et al. (NeurIPS 2024) result we cite. We therefore use the **Fermat metric**, which carries the Fernandez et al. (JMLR 2023) Gromov–Hausdorff + bottleneck convergence guarantee, restrict to β₀/β₁, and report Fasy confidence bands over many subsamples.
- *"SAE features are not canonical and are easy to cherry-pick."*
  - We require seed × width stability (R ∈ {2,4,8}, ≥ 3 seeds), a permutation null with multiple-testing correction, and **causal ablation**; we report L0/%-alive/variance-explained, and we report unstable features as unstable.
- *"Geometry-of-concepts was derived for LLM unembeddings, not a multimodal encoder."*
  - Correct — so we treat the orthogonality/simplex laws as a *hypothesis to test*, and we plant known concepts in the synthetic harness to confirm the test recovers them. A violation on AION-1 is a reported finding, not a failure.
- *"Why diffusion maps and not UMAP?"*
  - Determinism and a convergence theorem (Laplace–Beltrami at α = 1), plus the measured distance-concentration diagnostic that shows the raw metric is broken. UMAP is stochastic and distorts global geometry.
- *"N = 2k is tiny."*
  - It is a prototype, and the Appendix-D harness states exactly what is resolvable at 2k (e.g. β₂ is not). We do not claim anything the harness says we cannot measure.

**From an astronomy reviewer.**

- *"We already know SSL galaxy embeddings encode morphology, redshift, and mass (AstroCLIP)."*
  - Yes — that is *decodability*, and it is settled. We measure *geometry* (dimension, curvature, topology, concept structure), which is unmeasured for any astro foundation model; a null geometric result would *not* contradict the known decodability.
- *"Morphology is a continuum, not a tuning fork — this is old news."*
  - Exactly why the confirmatory question is only a control. We test continuum-vs-hierarchy *quantitatively* (the tree battery) and search for organizing structure *beyond* the Hubble/Galaxy-Zoo taxonomy (the SAE arm).
- *"Photo-z contamination and selection effects will drive your 'physical axes'."*
  - We use E_held (withholding the target modality), regress out redshift, propagate photo-z error, condition every claim on the r<19/SGC slice, and check physical-vs-observed coordinates.
- *"Isn't this numerology on a black box?"*
  - Every number carries a matched known-answer synthetic validation, a physical cross-tabulation (curvature ridges vs morphology; features vs PROVABGS), and a pre-registered falsifier.
- *"What is the astrophysical payoff?"*
  - Appendix C: heterogeneous ID is a testable physical prediction (passive simpler than star-forming); candidate "alien" SAE concepts are follow-up targets for discovery; the sim-vs-real arm localizes where simulations diverge from reality.

---

# Part V — Reference material

## 22. Glossary of symbols

| Symbol | Meaning |
|---|---|
| d | ambient embedding dim (= 1024, AION-1-Large) |
| e, E_full, E_held | mean-pooled embedding; full multimodal set; target-withheld set (image+photometry) |
| N | #galaxies (~10⁴; 2k subsample for topology/curvature/δ) |
| r₁,r₂,μ | 1st/2nd NN distances and ratio (TwoNN) |
| d̂, m̂_K(z) | global ID; local (Levina–Bickel) ID at z from K neighbors |
| F(μ) | Pareto CDF 1 − μ^{−d} |
| ψ_l, λ_l, Ψ_t, D_t | diffusion eigenvectors/values; diffusion-map embedding; diffusion distance |
| α | diffusion anisotropy (=1, Laplace–Beltrami) / Fermat exponent (context-dependent) |
| ε, k, t | kernel bandwidth / kNN graph degree / diffusion time |
| g, J, G = JᵀJ | decoder; its Jacobian; pullback Riemannian metric |
| √det G, Γ^k_{ij}, K(u,v) | magnification factor; Christoffel symbols; sectional curvature |
| K_Gauss, H | Gaussian / mean curvature (second fundamental form) |
| κ(x,y), F(e) | Ollivier-/Forman-Ricci edge curvature |
| d_H, D(f) | Poincaré distance; hyperbolic-embedding distortion |
| δ-distribution | Gromov four-point hyperbolicity (report distribution, not max) |
| ŵ_y, θ_ij, ρ_ij | ridge probe direction; probe angle; label correlation (null = arccos ρ_ij) |
| ⟨a,b⟩_C = aᵀΣ⁻¹b | causal/whitened inner product (geometry of concepts) |
| f, W_enc, W_dec, s_j | SAE feature activations / encoder / decoder dictionary / alignment score |
| β₀,β₁,β₂, c_n | Betti numbers; Fasy radius (band half-width √2·c_n) |
| W_∞, W_q, RTD | bottleneck / Wasserstein-q / Representation Topology Divergence |
| RDR | relative distance ratio (D_max−D_min)/D_min |

**Abbreviations.**

| Acronym | Expansion |
|---|---|
| FM / SSL | foundation model / self-supervised learning |
| ID / LID | intrinsic dimension / local intrinsic dimension |
| TwoNN / Gride | two-nearest-neighbor ID estimator / generalized-ratio ID estimator |
| GPLVM / ARD | Gaussian-process latent-variable model / automatic relevance determination |
| PPCA / BPCA | probabilistic PCA / Bayesian PCA |
| SAE / AuxK | sparse autoencoder / auxiliary-k dead-latent loss (TopK SAEs) |
| L0 / R | active-feature count per sample / SAE dictionary expansion factor |
| TDA / PH / VR | topological data analysis / persistent homology / Vietoris–Rips |
| RTD / CKA / MMD / OT | representation topology divergence / centered kernel alignment / maximum mean discrepancy / optimal transport |
| BGH | Berry–Giannakis–Harlim auto-bandwidth selection |
| CI / CrI | (frequentist) confidence interval / (Bayesian) credible interval |
| SGC / DR8 / DR10 | Southern Galactic Cap / Legacy Survey Data Releases 8, 10 |
| FSQ / LFQ | finite-scalar / look-up-free quantization (AION tokenizers) |
| BGS / SED / sSFR | Bright Galaxy Survey / spectral energy distribution / specific star-formation rate |
| OOD / nDCG | out-of-distribution / normalized discounted cumulative gain |
| PROVABGS | PRObabilistic Value-Added Bright Galaxy Survey (SED-fit catalog) |
| Reeb graph | quotient of a space by connected components of level sets of a filter (Mapper target) |

## 23. References

**AION-1 & ecosystem**
- AION-1, arXiv:2510.17960 — https://arxiv.org/abs/2510.17960 ; HTML https://arxiv.org/html/2510.17960v1 ; OCR [aion1.md](papers/aion1/aion1.md)
- HF card `polymathic-ai/aion-base` — https://huggingface.co/polymathic-ai/aion-base ; GitHub https://github.com/PolymathicAI/AION ; blog https://polymathic-ai.org/blog/aion-1/
- Multimodal Universe, arXiv:2412.02527 — https://arxiv.org/abs/2412.02527
- AstroCLIP, *MNRAS* 531, 4990; arXiv:2310.03024 — https://arxiv.org/abs/2310.03024
- AION-Search, arXiv:2512.11982 — https://arxiv.org/abs/2512.11982 (embeddings NOT to be substituted)

**Data**
- Galaxy Zoo DESI (Walmsley et al. 2023), *MNRAS* 526, 4768; arXiv:2309.11425 — https://arxiv.org/abs/2309.11425 ; Zenodo 10.5281/zenodo.8360385 — https://zenodo.org/records/8360385 ; Zoobot https://github.com/mwalmsley/zoobot
- PROVABGS (Hahn et al. 2023), *ApJ* 945, 16 — referenced [aion1.md:401](papers/aion1/aion1.md#L401)
- DESI Legacy Surveys (Dey et al. 2019); cutout svc https://www.legacysurvey.org/dr10/description/
- CAMELS (Villaescusa-Navarro et al. 2021), arXiv:2010.00619 — https://arxiv.org/abs/2010.00619 ; sim-vs-real OOD arXiv:2410.10606

**Intrinsic dimension**
- Facco et al. 2017 (TwoNN), *Sci. Rep.* 7:12140 — https://www.nature.com/articles/s41598-017-11873-y
- Denti et al. 2022 (unbiased MLE, Gride), arXiv:2104.13832 ; Beyond the noise arXiv:2405.15132
- Allegra et al. 2020 (Hidalgo), *Sci. Rep.* 10:72222 — https://www.nature.com/articles/s41598-020-72222-0
- DADApy, *Patterns* 2022 — https://dadapy.readthedocs.io/ ; intRinsic, arXiv:2102.11425
- Ansuini et al. 2019 (ID of representations), NeurIPS; arXiv:1905.12784 ; local-ID-predicts-generalization (2026) arXiv:2601.22722
- `scikit-dimension` — https://scikit-dimension.readthedocs.io/
- Cadiou et al. 2025, *MNRAS* 537, 1869; arXiv:2404.02962 ; Galaxy Manifold arXiv:2210.05862

**Diffusion maps / metrics**
- Coifman & Lafon 2006, *ACHA* 21:5 — https://math.pku.edu.cn/teachers/yaoy/Fall2011/Lafon06.pdf
- Berry & Harlim 2016 (variable bandwidth), arXiv:1406.5064 ; `datafold` https://datafold-dev.gitlab.io/datafold/ ; `pydiffmap`
- Richards et al. 2009 (diffusion maps on SDSS spectra), arXiv:0807.2900
- Damrich et al. 2024 (spectral PH in high-d), NeurIPS; arXiv:2311.03087
- Isomap (Tenenbaum et al. 2000); Fermat distance, arXiv:1810.09398 ; `fermat` https://pypi.org/project/fermat/
- Distance concentration, arXiv:2401.00422

**Riemannian / curvature**
- Arvanitidis et al. 2018 (Latent Space Oddity), arXiv:1710.11379 ; Shao et al. 2018, arXiv:1711.08014 ; `geomstats` arXiv:2004.04667
- Second-fundamental-form curvature: arXiv:2502.03750, arXiv:2308.02615 ; CA-PCA arXiv:2309.13478 ; curvature-profile ID arXiv:2509.13385
- Ollivier-Ricci/Forman: `GraphRicciCurvature` https://github.com/saibalmars/GraphRicciCurvature ; Forman arXiv:1603.00386
- Hyperbolic embeddings: Sala et al. 2018, arXiv:1804.03329 ; Nickel & Kiela 2017, arXiv:1705.08039 ; `geoopt` arXiv:2005.02819 ; differentiable Gromov hyperbolicity arXiv:2505.21073

**Bayesian manifold learning**
- Titsias & Lawrence 2010 (Bayesian GPLVM), AISTATS — https://proceedings.mlr.press/v9/titsias10a/titsias10a.pdf ; `GPyTorch` GPLVM tutorial
- Tipping & Bishop 1999 (PPCA); Bishop 1999 (Bayesian PCA), NeurIPS

**Concept probes / SAEs / geometry of concepts**
- Alain & Bengio 2016 (linear probes), arXiv:1610.01644 ; Kim et al. 2018 (TCAV), arXiv:1711.11279
- Park et al. 2023 (causal inner product), arXiv:2311.03658 ; Park et al. 2024 (geometry of categorical/hierarchical concepts), arXiv:2406.01506
- Bricken et al. 2023 (Towards Monosemanticity) — https://transformer-circuits.pub/2023/monosemantic-features ; Cunningham et al. 2023, arXiv:2309.08600
- Gao et al. 2024 (TopK SAE) ; Bussmann et al. 2024 (BatchTopK), arXiv:2412.06410 ; `dictionary_learning` https://github.com/saprmarks/dictionary_learning
- Euclid galaxy-morphology SAE, arXiv:2510.23749 ; scientific-concept SAEs, arXiv:2408.00657 ; open-ended visual discovery SAEs, arXiv:2511.17735

**Topology / Platonic**
- Singh, Mémoli & Carlsson 2007 (Mapper) — https://research.math.osu.edu/tgda/mapperPBG.pdf ; `kepler-mapper` ; Carrière–Michel–Oudot 2018, arXiv:1706.00204
- ripser.py (Tralie et al. 2018, JOSS) ; Bauer 2021 (Ripser), arXiv:1908.02518 ; GUDHI https://gudhi.inria.fr/
- Otter et al. 2017 (PH complexity) ; Fasy et al. 2014 (confidence sets), arXiv:1303.7117
- Fernandez et al. 2023 (intrinsic PH via Fermat metric), JMLR 24 — https://www.jmlr.org/papers/volume24/21-1044/21-1044.pdf
- Barannikov et al. 2022 (RTD), ICML; arXiv:2201.00058
- Huh et al. 2024 (Platonic Representation Hypothesis), arXiv:2405.07987 ; Back into Plato's Cave (2026), arXiv:2604.18572

**Astrophysics priors**
- Hubble 1926 (*ApJ* 64, 321), 1936 ; Roberts & Haynes 1994, *ARA&A* 32, 115
- Schawinski et al. 2014, arXiv:1402.4814 ; Beyond the Hubble Sequence (Cheng et al. 2021), *MNRAS* 503, 4446
- Connolly et al. 1995, arXiv:astro-ph/9411044 ; Yip et al. 2004, arXiv:astro-ph/0407061

## 24. Open questions and locked decisions

**Locked (this revision):**
- **Embeddings:** built by us; **multimodal** (image + photometry + redshift + spectra) as E_full, with **E_held(target)** for concept arms (Section 5.1).
- **Riemannian pullback-metric flagship:** in-scope now — train a light decoder (Section 10.2).
- **Label source:** GZ-DECaLS-SGC (~171k, AION footprint).
- **Primary metric:** Fermat for topology/curvature; diffusion for coordinates; Euclidean control.
- **Cross-model:** AstroCLIP Platonic check (mutual-kNN + RTD) included.

**Still open (defaults in force unless you say otherwise):**
1. **Decoder target for the pullback metric.** g → reconstruct the `{g,r,i,z}` image (geometry of *appearance*) vs g → predict the physical-parameter vector (geometry of *physics*)? *Default:* physical-parameter decoder (cleaner interpretation), with image-reconstruction as a sensitivity check.
2. **CAMELS arm.** Optional/stretch — promote to in-scope? *Default:* deliver after Methods A–G + AstroCLIP.
3. **Spectra in E_full.** Include DESI/SDSS spectra for the (smaller) cross-matched spectroscopic subset, accepting a reduced N, or keep E_full to image+photometry+redshift for the full ~10⁴? *Default:* image+photometry+redshift for the full sample; a spectroscopic-subset variant as a robustness check.
4. **Compute ceiling for GPLVM/decoder.** Single-GPU minutes–hours is assumed; confirm if a multi-GPU/longer budget is available for larger inducing-point sets or image-reconstruction decoders.

If you have preferences on any of these, I will fold them in; otherwise I proceed with the stated defaults.

---

## Appendix A — Worked quantitative companion (toy-scale numbers for every method) `[v3]`

This appendix makes the math concrete at our exact scale — N ≈ 10⁴ galaxies, a 2k subsample for the O(N³)-class methods, ambient d = 1024.
Two kinds of number appear, kept rigorously distinct:

- **Formula-derived (exact):** follows deterministically from a formula and N/d/ε — correct now, before any data.
- **Illustrative (synthetic):** a small hand example showing *what a method outputs and how to read it* — **not** a result from AION-1, and labeled "illustrative." No AION-1 number is invented anywhere in this document.

### A.1 Distance concentration (Section 7)

- RDR = (D_max − D_min)/D_min → 0 as d → ∞.
- For i.i.d. isotropic data the coefficient of variation of pairwise distance scales like O(1/√d): in d = 1024 the spread is ~3% of the mean, so nearest and farthest neighbors differ by a few percent and NN identity is unstable.
- **Illustrative target:** Euclidean/cosine RDR ~0.1–0.5 on the 2k subsample; Fermat/geodesic/diffusion several-fold larger. **Report the actual table; do not assume these numbers.**

### A.2 Johnson–Lindenstrauss budget (Section 14; NOT a curvature test) — formula-derived

| ε | k at N = 2,000 | k at N = 10,000 |
|---|---|---|
| 0.05 | 25,162 | 30,489 |
| 0.10 | 6,515 | 7,894 |
| 0.20 | 1,754 | 2,125 |
| 0.50 | 364 | 442 |

- At ε = 0.1 the budget (7,894) exceeds the ambient d = 1024 → JL guarantees no reduction here (why v1's "JL gap" was dropped).
- At ε = 0.5 a 442-dim random projection provably preserves squared distances within ±50% — an acceptable *safe* reduction for homology. Use JL only as this budget.

### A.3 TwoNN credible interval (Section 8.1) — formula-derived

| N used | bias factor N/(N−1) | relative std of d̂ ≈ 1/√(N−2) |
|---|---|---|
| 2,000 | 1.0005 | 2.24% |
| 10,000 | 1.0001 | 1.00% |

- At 10⁴ points a TwoNN d̂ ≈ 6 has *statistical* uncertainty only ~±0.06.
- So the **dominant error is model/assumption error**, which is why we triangulate across four estimators rather than trusting one tight CI.

### A.4 Gride scale curve and Hidalgo heterogeneity (Section 8.3) — illustrative

- **Gride.** Plot d̂ vs neighbor order n₂ ∈ {2,4,…,64}. A fall from ~9 (n₂=2) to a plateau ~5 (n₂≥16) signals small-scale noise inflating ID; the plateau is the manifold dimension. Report the curve, not one point.
- **Hidalgo.** A two-component fit returning "passive: d = 3.0 [2.5,3.5]; star-forming: d = 5.1 [4.2,6.0], P(distinct)=0.99" would quantitatively confirm the Cadiou et al. 2025 photometry result inside a foundation-model embedding — a new claim. Sweep ζ ∈ {0.5,0.8,0.95}, K ∈ {1,2,3}; check MCMC mixing/label-switching.

### A.5 Diffusion spectral gap and BGH slope (Sections 8.4, 9) — illustrative

- Eigenvalues like (1, 0.97, 0.94, 0.90, 0.86, 0.55, 0.51, …) have their largest drop between λ₄ and λ₅ → d_spec = 5, independent of TwoNN.
- BGH: a peak log-log slope of ~2.5 implies d_int ≈ 5. Agreement of TwoNN, spectral gap, BGH slope, and GPLVM-ARD at ≈ 5 is far stronger than any single value.

### A.6 Curvature signs and a worked saddle (Section 10.4) — illustrative

- Unit sphere: k₁ = k₂ = 1, Gaussian K = +1 (cluster).
- Saddle z = x² − y²: k₁ = +2, k₂ = −2, K = −4 (branch/transition).
- Flat plane: K = 0.
- So a patch returning K < 0 sits on a saddle (a branch point such as the green valley); K > 0 is a compact cluster. Sweep neighborhood size k; report a curvature-vs-k curve.

### A.7 Ollivier-Ricci on a two-cluster graph (Section 10.5) — illustrative

- Two tight clusters joined by one bridge edge a→b, α = ½ lazy walk.
- Intra-cluster edge: heavily overlapping neighborhoods → W₁ small relative to d=1 → κ = 1 − W₁/d > 0 (community).
- Bridge edge: neighborhoods in different clusters → mass transported far → W₁ > d → κ < 0 (branch/bottleneck).
- The publishable test: do negative-κ edges coincide with morphological transitions? Screen the full 10⁴ graph with Forman F(e)=4−deg(v)−deg(w); confirm with exact Ollivier on 2k.

### A.8 Hyperbolic vs Euclidean distortion of a known tree (Section 10.6) — illustrative

- A balanced binary tree of 2,000 nodes typically embeds into H² with average distortion D(f) ≈ 1.0–1.1; the best Euclidean MDS at 2-D gives ≈ 1.3–1.5 and worsens with depth.
- A 4-cycle embeds *better* in Euclidean than hyperbolic.
- Test for AION-1: fit both at k ∈ {2,5,10,20}; a hyperbolic curve well below Euclidean at small k is strong tree evidence. Use float64, Lorentz model, h-MDS init.

### A.9 Pullback metric — a worked stretch (Section 10.2) — illustrative

- If the decoder g maps a unit latent step to a 2× longer step in observable space along one direction and 1× along another, the eigenvalues of G = JᵀJ are (4, 1) and √det G = 2 — the manifold stretches 2× there.
- A √det G field peaking at a morphological boundary means a small embedding step corresponds to a large change in galaxy appearance/physics — the geometric signature of a transition region.

### A.10 SAE dictionary arithmetic (Section 12) — formula-derived

- d = 1024; expansion R ∈ {2,4,8} → width m ∈ {2048,4096,8192}.
- TopK k = 32 at R = 4 → L0/m = 32/4096 ≈ **0.78% active** per galaxy.
- Parameters ≈ 2·d·m = 2·1024·4096 ≈ **8.4M** — a tiny head, minutes on one GPU on 10⁴ vectors.
- "Alien" candidate (illustrative): activation coherence s_j ~ 0.6 yet residual novelty > 0.8, seed-stable across R, permutation-null-surviving, and ablation-confirmed.

### A.11 Persistent-homology simplex budget (Section 14.1) — formula-derived

| N | vertices | edges C(N,2) | triangles C(N,3) |
|---|---|---|---|
| 2,000 | 2,000 | 1,999,000 | 1.33×10⁹ |
| 10,000 | 10,000 | 4.9995×10⁷ | 1.67×10¹¹ |

- 10⁴ points dim-2 → 1.67×10¹¹ triangles (infeasible); 2k → 1.33×10⁹, tractable only with maxdim=1 + edge threshold.
- β₂ (C(N,4) ≈ 6.6×10¹¹ tetrahedra at 2k) is out of reach and under-resolved — restrict to β₀, β₁.

### A.12 Geometry-of-concepts worked check (Section 11.2) — illustrative

- Binary "barred" → one direction v_bar.
- Categorical m = 3 (E, S0, irregular) → a 3-vertex simplex in a 2-D affine subspace; test near-equal whitened edges.
- Hierarchy "smooth ⊃ {E,S0}": under ⟨a,b⟩_C = aᵀΣ⁻¹b, the child direction (E vs S0) should be near-orthogonal to the parent (smooth vs featured); a whitened cosine ≈ 0 supports the laws, large is a finding that they do not transfer.

### A.13 Geodesic-vs-chord distortion (Section 10.2) — illustrative

- On a flat manifold the geodesic distance equals the Euclidean chord, so the ratio d_geo/d_chord ≈ 1 everywhere.
- On a curved manifold the ratio exceeds 1, and *by how much* localizes curvature: a ratio of, say, 1.4 between a red-sequence galaxy and a blue-cloud galaxy means the honest path between them along the manifold is 40% longer than the straight line — they are "geometrically farther apart" than the embedding's raw distance suggests.
- Report the *distribution* of d_geo/d_chord and where its tail concentrates; a tail localized at the smooth↔featured boundary corroborates the Ricci-curvature ridge.

### A.14 RTD and CKA — reading a comparative-topology number (Sections 14.3, 15.2) — illustrative

- RTD = 0 means two matched point clouds have identical multi-scale topology; larger RTD means more topological disagreement.
- A finding like "RTD(AION, physical-params) < RTD(AstroCLIP, physical-params)" would say AION-1's manifold topology is *closer* to the true physical-parameter manifold than AstroCLIP's — a concrete, quantitative Platonic statement.
- CKA (global) vs mutual-kNN (local) can disagree: per the 2026 critique, trust the *local* mutual-kNN agreement when they conflict, and report both.

### A.15 Compute budget — what "the prototype ran" looks like

- Embedding 10⁴ galaxies through frozen AION-1-Large (fp16): single GPU, minutes–hours.
- Metric battery, diffusion map, ID estimators, curvature field, Mapper, ridge probes, SAE: minutes each.
- Exact Ollivier-Ricci on 2k, Fermat-metric persistence, GPLVM, pullback decoder + geodesics: minutes to tens of minutes.
- CAMELS arm (optional): dominated by mock-observation rendering, not the geometry.
- **Rigor (intervals, nulls, seed stability, metric robustness), not FLOPs, is the binding constraint.**

---

## Appendix B — Library & function quick-reference `[v3]`

One row per method; exact packages and entry points. Versions pinned per Section 17.

| Arm | Method | Library · entry point | Key parameters |
|---|---|---|---|
| Embed | frozen AION-1-Large | `aion` · `AION.from_pretrained('polymathic-ai/aion-large')`, `CodecManager` | fp16; mean-pool ≤600 tokens; E_full / E_held |
| Metric | geodesic | `scikit-learn` · `manifold.Isomap.dist_matrix_` | `n_neighbors=k` |
| Metric | Fermat | `fermat` · `Fermat(alpha, path_method='D')` | `alpha ∈ {2,3,4}`, `k≈log N` |
| Metric | diffusion | `datafold` · `DiffusionMaps(GaussianKernel(ε), alpha=1, time_exponent=t)` | `alpha=1`, BGH ε |
| Metric | concentration | `scipy.spatial.distance.pdist` + numpy | per-metric RDR |
| ID | TwoNN | `scikit-dimension` · `id.TwoNN`; `dadapy` | `discard_fraction=0.1` |
| ID | local Levina–Bickel | `scikit-dimension` · `id.MLE` | sweep `K` |
| ID | Bayesian / Gride | `dadapy` · `compute_id_2NN_wprior`, `return_id_scaling_gride` | Gamma prior; `range_max=64` |
| ID | Hidalgo | `intRinsic` (R, `rpy2`) · `Hidalgo` | `K`, `ζ` |
| Manifold | diffusion map | `datafold`/`pydiffmap` · `DiffusionMaps` | `alpha=1`, `epsilon='bgh'` |
| Manifold | Bayesian GPLVM | `GPyTorch` · `BayesianGPLVM` | `ard_num_dims`, inducing 50–100 |
| Manifold | Bayesian PCA | `scikit-learn` · `decomposition.FactorAnalysis` | linear null |
| Riemannian | pullback G=JᵀJ | `geomstats` · `PullbackMetric`; `torch.func.jacrev` | decoder g; `solve_bvp` geodesics |
| Curvature | global−local / CA-PCA | `scikit-dimension` + CA-PCA code | — |
| Curvature | 2nd fundamental form | `scikit-learn` PCA + `numpy.lstsq` | per-point kNN, sweep k |
| Curvature | Ollivier / Forman | `GraphRicciCurvature` | `alpha=0.5`, `method='Sinkhorn'` |
| Tree | hyperbolic distortion | `geoopt` · `PoincareBall`/`Lorentz` | float64, h-MDS init |
| Tree | additive tree / δ-dist | `scipy.cluster.hierarchy` + NJ | ~10⁶ quadruples; distribution |
| Tree | cophenetic | `scipy.cluster.hierarchy.cophenet` | average/Ward |
| Concept | ridge probes | `scikit-learn` · `RidgeCV` (on E_held) | CV-λ; R²; ρ-null; bootstrap |
| Concept | geometry of concepts | numpy/scipy + whitening | Σ⁻¹ causal inner product |
| Discovery | TopK/BatchTopK SAE | `dictionary_learning` / `SAELens` | R∈{2,4,8}, k, AuxK, ≥3 seeds |
| Discovery | feature scoring | `scipy.stats.spearmanr`, `scikit-learn` R² | permutation null + correction |
| Topology | Mapper | `kepler-mapper` | ≥2 filters; pass KMeans explicitly |
| Topology | persistence | `ripser.py` / `GUDHI` (`metric='precomputed'`) | maxdim=1, threshold, Fermat |
| Topology | comparative | `RTD` | matched points |
| Cross-model | CKA / mutual-kNN | `repsim` / `torchmetrics` | AstroCLIP on same galaxies |

---

## Appendix C — What each result would mean for galaxy science `[v3]`

A bridge for the astronomy reader: each geometric measurement translated into an astrophysical statement (and its limits).

- **A low intrinsic dimension (d̂ ≈ 3–6)** says AION-1 compresses a galaxy's appearance to a handful of effective parameters — consistent with galaxy SEDs/photometry being described by ~2–5 physical knobs. It does **not** say those are *the* physical parameters; it bounds *how many* independent directions the model uses.
- **Heterogeneous dimension (passive < star-forming)** would echo the intuition that quiescent galaxies form a tighter, simpler family while star-forming galaxies vary along more axes (gas, spirals, bars, dust). A novel, testable prediction the Hidalgo arm can confirm or refute.
- **Localized negative curvature at a morphological boundary** is the geometric fingerprint of a *transition region* between populations — the manifold analog of the green valley — without claiming any galaxy traverses it in time.
- **A √det G magnification peak at the smooth↔featured boundary** would say a small embedding step there corresponds to a large change in galaxy appearance/physics — the model's "resolution" is concentrated at the transition.
- **A clean low-distortion hyperbolic / tree fit** would support a hierarchical (machine "tuning fork") organization; its *failure* supports the modern continuum view. Either is meaningful.
- **Concept axes obeying the geometry-of-concepts laws** would suggest the model encodes morphological *categories* the way LLMs encode word categories — a surprising cross-domain regularity; a *violation* says scientific encoders organize concepts differently, also worth reporting.
- **An "alien" SAE feature** that is physically coherent but unmatched by any catalog label is a candidate for follow-up: a montage of its top-activating galaxies, inspected by astronomers, could reveal a real-but-unnamed structural regularity. This is where an interpretability prototype feeds back into discovery science.
- **A sim-vs-real gap (CAMELS)**, if it survives the mock-realism controls, localizes *where* a simulation's galaxy populations diverge from reality in a learned feature space — potentially pointing at which subgrid physics is mis-tuned — but only graded, per-suite, after matching selection functions.

The throughline: at toy scale every one of these is an *exploratory* signal to be triangulated and validated, never a definitive astrophysical claim. The prototype's job is to show the signal is real and stable enough to justify the full study.

---

## Appendix D — Synthetic validation harness (known-answer tests) `[v4]`

Every estimator in this study has a known bias in high ambient dimension or small sample size.
The document repeatedly says "validate on a synthetic manifold of known X first" — this appendix specifies exactly that harness.

**The binding rule.**

- **No AION-1 number is reported for a method until that method returns the correct answer on a synthetic dataset at matched (N, d = 1024, noise level).**
- This converts every method from "trust the library" to "trust the library *because it passed a known-answer test on data shaped like ours*."
- It is also the single cheapest way to catch a pipeline bug (un-pooled tokens, wrong metric, mis-set `maxdim`) before it contaminates a science claim.

**Construction (shared).**

- Generate each synthetic manifold in its native low dimension k.
- Embed it isometrically into ℝ^1024 by a random orthonormal rotation (so the *intrinsic* geometry is known but the *ambient* coordinates look like AION's).
- Add isotropic Gaussian noise at a few SNRs to mimic encoder/tokenizer jitter.
- Sample N = 2,000 and N = 10,000 points to match both regimes.
- Run the *entire* downstream pipeline (metric battery → estimator) exactly as for AION-1.

**The known-answer suite.**

| Synthetic dataset | Native k | What it certifies | Expected output (pass criterion) |
|---|---|---|---|
| k-sphere S² , S³, S⁵ | 2,3,5 | intrinsic-dimension estimators (TwoNN, local, Gride, spectral, ARD) | d̂ ≈ k within CI; *not* tree-like (δ-distribution high; hyperbolic ≈ Euclidean distortion) |
| Swiss roll | 2 | ID + curvature: a curved 2-manifold | d̂ ≈ 2; global−local gap > 0 (curved); β₀ = 1 |
| Noisy circle | 1 | persistent homology loop detection | β₀ = 1, **β₁ = 1** persists outside the Fasy band across subsamples |
| Torus T² | 2 | homology of a richer space; mixed curvature | d̂ ≈ 2; β₀ = 1, β₁ = 2, (β₂ = 1 only at large N); local K both signs |
| Two Gaussian blobs | ~k | bimodality / β₀ | **β₀ = 2** persistent; Mapper two components; matches the red/blue-cloud test |
| Linear subspace + noise | k (linear) | the *linear null* | Bayesian-PCA dim = k; TwoNN ≈ k; curvature ≈ 0; geodesic ≈ Euclidean (G ≈ const) |
| Union of a 3- and a 5-manifold | {3,5} | **Hidalgo heterogeneous ID** | recovers two components with d ≈ {3,5}, P(distinct) high |
| Synthetic metric tree / dendrogram | tree | the tree-likeness battery | δ_rel ≈ 0; hyperbolic distortion ≪ Euclidean at small k; high cophenetic; negative Ricci on branch edges |
| Planted concept directions (orthogonal & correlated) | — | geometry-of-concepts + ridge-probe ρ-null | recovers orthogonal planted axes as ⟨·,·⟩_C-orthogonal; recovers correlated axes at exactly arccos(ρ) |
| Planted monosemantic feature | — | the **SAE** | a dictionary atom recovers the planted feature; the alignment score flags it; ablation removes it |

**Reading the harness.**

- A method that *fails* its known-answer test is reported as **unreliable at our scale** and either dropped or down-weighted — this itself is a finding worth stating (e.g., "β₂ is not resolvable at N = 2k even on a torus, so we do not report it for AION-1").
- A method that *passes* earns the right to a headline AION-1 number, with the synthetic test shown in a supplementary figure (referees in both fields ask for exactly this).
- The harness also calibrates the **matched-random anchor** used throughout: the "no special structure" baseline for δ_rel, curvature, and RDR is the *isotropic Gaussian blob in ℝ^1024 at matched N* — computed here, labeled "this work," never imported from another paper.

**Cost.**

- The synthetic generators are a few lines of NumPy; running the pipeline on each is the same minutes-scale cost as one AION-1 pass.
- Total harness cost ≈ one extra pipeline run per synthetic dataset — negligible against the value of every reported number carrying a passed known-answer test.

This harness is what turns the prototype from "we ran a stack of geometry tools on a foundation model" into "we ran a *validated* stack whose every number is backed by a matched known-answer test" — the difference between a suggestive plot and a defensible measurement.

---

## Document changelog

- **v1** (original, 49 lines): hour-by-hour sprint plan; contained the unsound JL-gap diagnostic and several precise factual errors.
- **v2** (methods-driven): corrected every error against primary sources; replaced the JL-gap diagnostic; added error bars, nulls, and a standalone framing.
- **v3** (novelty): reframed to *geometry + concept discovery*; added diffusion maps + geodesic/Fermat metrics, the Riemannian curvature battery, Bayesian/GPLVM manifold learning, sparse-autoencoder concept discovery, RTD comparative topology, and the AstroCLIP Platonic cross-check; added the worked-math appendices.
- **v4** (this version): locked decisions folded in — multimodal embeddings with the `E_held` circularity safeguard; the Riemannian pullback-metric flagship (decoder → G = JᵀJ → geodesics/curvature); GZ-DECaLS-SGC labels; Fermat metric primary; the synthetic validation harness (Appendix D); the reviewer-objection FAQ; the decision matrix; and a full granular re-render.

**Verification provenance.**

- All factual and mathematical claims were checked by two adversarial multi-agent research passes (12 + 7 agents) against primary sources — arXiv papers, journal articles, Hugging Face model cards, and library documentation.
- The AION-1 paper was OCR'd locally to ground every model fact ([aion1.md](papers/aion1/aion1.md)).
- Inline citations and §23 give the sources; `[CORRECTION]` tags mark every place a v1 claim was found wrong and fixed.

**Status.** Specification complete and internally consistent. Open decisions are listed in §24; unless overridden, the stated defaults are in force. The next step is execution against the §16 pipeline, beginning with the §5 embedding build and the Appendix-D validation harness.
