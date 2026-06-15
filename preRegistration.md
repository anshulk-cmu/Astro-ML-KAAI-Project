# Pre-registration — ASTRID x AION-1 sim-vs-real test (Phase 2, stage S0)

This document freezes the predictions, thresholds, decision rules, and data conventions of the
Phase 2 experiment BEFORE any simulated galaxy is rendered or embedded. It is the binding
reference for what counts as confirmatory; everything not listed here as confirmatory is
exploratory. Deviations during execution are permitted only with a logged rationale in
`runLog.md`. Demotions of confirmatory items to exploratory are permitted only before stage S6
(the frozen-encoder audit) begins, never after results are seen.

**Status declaration (2026-06-10).** As of this commit, zero simulated galaxies have been
rendered or embedded. Stage S1 (data census on the ASTRID archive at PSC Vera) involved
catalog-level and star-particle-level photometry and SFR statistics only (no images rendered,
nothing embedded); no ASTRID data has passed through AION-1.
The real anchor (48,398 Galaxy Zoo DESI galaxies, embedded and measured in Phase 1) is frozen:
no re-fetching, no re-embedding, no re-selection.

The experiment: forward-model ~50,000 ASTRID galaxies into Legacy-Survey-like mock
observations, embed them with the frozen AION-1-Large encoder, and attribute every component
of the sim-vs-real embedding gap to instrument, dust, demographics, or residual physics, with
per-galaxy false-discovery-rate control. Fine-tuning comes last, is calibration-only, and can
never revise the frozen-encoder result.

---

## 1. The gap decomposition (frozen identity and guards)

Domain-classifier AUC at each forward-model configuration; rungs R0-R3 are the realism ladder
(idealized; PSF + depth-matched noise; + correlated noise; RealSim-style injection into real
DR10-South cutouts); `fid` and `best` are dust configurations. The terms telescope exactly:

```
GAP_total        = AUC(R1, fid)        - 0.5
GAP_instrument   = AUC(R1, fid)        - AUC(R3, fid)
GAP_dust         = AUC(R3, fid)        - AUC(R3, best)
GAP_demographic  = AUC(R3, best)       - AUC_cond(R3, best)
GAP_conditional  = AUC_cond(R3, best)  - 0.5
```

Guards, all binding:

- `AUC_cond` (primary estimator) is the held-out AUC of ONE pooled, full-n classifier trained
  with importance weights `w(stratum) = n_real(stratum) / n_sim(stratum)` equalizing the
  (observed-z, M*) stratum mix. Per-stratum classifiers are secondary only, each read against a
  real-vs-real null floor at matched per-stratum n (power-collapse guard).
- `best` dust theta is selected by minimizing AUC_cond on a galaxy-level selection split; every
  GAP term is reported on the disjoint estimation split (winner's-curse guard). The 17-config
  dust span (max minus min over defensible configs, kappa = 0 excluded as an ablation endpoint,
  joint bootstrap over galaxies recomputing all 17 AUCs per resample) is reported at rung R2 as
  the dust sensitivity span, with an R2-to-R3 rank-stability check on (fiducial, best, one
  extreme).
- The chain is reported on a second scale alongside AUC (cross-fitted log-loss and MMD^2);
  conclusions must be sign-consistent across scales.
- AUC(R3) is read against the injected-reals floor: ~5k real anchor galaxies passed through the
  identical injection operator and tested against untouched reals. All terms downstream of R3
  are interpreted net of this floor.
- Every gap is a (statistic, classifier capacity, space) triple, never a bare number. Two
  spaces, both pre-registered: full 1024-d z-scored embedding, and 32-d PCA fit on the real
  anchor only. Z-scoring uses real-sample statistics only (pooled scaling is run once as a
  robustness check). The logistic AUC is the headline scale; `Phi_img` (image-token-only
  embedding) is primary, `Phi_full` secondary.
- All GAP terms carry bootstrap CIs over galaxies AND classifier seeds, and each term is
  separately localized (which galaxies, which directions, which SAE concepts).

**Physics sufficiency conditions (pre-registered; ALL required for a "wrong galaxies" claim):**

(a) GAP_conditional > 0, sign-stable across scales and splits;
(b) survival of the dust sensitivity span and the SKIRT screen-vs-radiative-transfer systematic;
(c) the domain direction decodes onto physics-bearing directions, not noise/PSF-like
    directions, and does not resemble the real-only negative-control split directions;
(d) stability across `Phi_img`/`Phi_full` and classifier families;
(e) a non-trivial FDR-controlled per-galaxy discovery rate (the rate-vs-structure
    discriminator: a pure mix shift flags few or no individual galaxies; a structure problem
    flags many at controlled FDR).

The sufficiency verdict is an intersection-union test (all five sub-conditions at level alpha).

---

## 2. Pre-registered falsifiable predictions (P1-P3)

Imported from the published ASTRID z=0 paper (arXiv:2510.13976), independent of this project.
The pipeline must rediscover these blind; a pipeline that fails to recover known deficits has
no authority on unknown ones.

- **P1 (quiescent deficit).** Published: ASTRID's quiescent fraction at M* ~ 5x10^10 Msun is
  ~10% vs >40% observed. Prediction: in volume-weighted Sample B, the passive region of the
  real manifold (real galaxies below the Phase 1 passive/star-forming GMM cut on the sSFR
  probe; geometrically the high-g-r arm) is underpopulated by sims by a factor f_P1 in
  [2.5, 6] within the M* in [10^10.4, 10^11] stratum. Measurement: occupancy ratio
  `rho_P1 = (fraction of Sample B in passive region) / (fraction of real in passive region)`
  in that stratum, weighted bootstrap CI. **Falsifier: the rho_P1 CI contains 1.** The
  published contrast motivates direction and rough size but is NOT the acceptance band (it is
  z=0, volume-complete, sSFR-defined; this measurement is z=0.1-0.4, flux-limited,
  embedding-region-defined). Gate G-S2.5: the P1 stratum must reach effective sample size
  ESS = (sum w)^2 / sum w^2 >= 1,000, else the CI is widened accordingly.
- **P2 (red-sequence onset).** Published (sim-relative): ASTRID's colour transition sits at
  M* ~ 10^11 vs TNG ~10^10, MillenniumTNG ~10^10.5. Prediction: the mass at which
  embedding-space bimodality emerges (GMM/dip-test separation along the colour direction per
  mass bin, Phase 1 machinery) is >= 0.3 dex higher in Sample B than in the real sample.
  **Falsifier: onset masses equal within CIs.**
- **P3 (demographics at the knee).** Published: ASTRID's galaxy stellar mass function is
  ~0.5 dex low at the M* ~ 10^11 knee. Prediction: Sample A's matching exhausts ASTRID's
  high-mass supply at M* > 10^10.8 (per-bin starvation `s_bin = 1 - N_drawn/N_requested` with
  binomial CIs), quantitatively consistent with the published GSMF deficit. This doubles as a
  pipeline consistency check.

## 3. Methodological hypotheses

- **H-inst:** a substantial fraction of GAP_total is instrument (AUC drops materially R1 to
  R3). Failure suggests the realism ladder is broken, not that the instrument term is zero.
- **H-demo:** conditioning on (observed-z, M*) materially reduces the gap, consistent with
  P1-P3 being label shift.
- **H-phys:** GAP_conditional > 0 with all sufficiency conditions met, localized
  preferentially in compact/quenched/spheroidal galaxies (measured, not assumed).
- **H-null-geom:** the simulated cloud (full realism, best dust) reproduces the Phase 1
  geometry within tolerance: ID plateau within +/- 1.5 of the real anchor's ~10-12; beta0 = 1;
  beta1 = 0; positive mean Ollivier-Ricci curvature. Any violation is a finding.

## 4. Outcome decision matrix (all four publishable; wording pre-committed)

| Outcome | Operational signature | Pre-committed interpretation |
|---|---|---|
| O-A instrument-dominated | AUC(R3, best) - 0.5 < 0.05 marginal | Frozen FM cannot distinguish well-forward-modeled ASTRID from reality at this fidelity; the test bounds the sim-real gap from above; the realism ladder quantifies the instrument terms. |
| O-B demographic-dominated | marginal AUC large; conditional AUC - 0.5 < 0.05 | ASTRID builds individually realistic galaxies in the wrong proportions; P1-P3 quantified in embedding units; per-stratum occupancy maps delivered. |
| O-C intrinsic-physics | conditional AUC - 0.5 >= 0.05, survives dust grid, decodes on structure/physics directions | Localized physics mismatch; per-galaxy FDR-controlled discovery list + concept attribution; the strongest claim, requires ALL sufficiency conditions. |
| O-D mixed | none dominates | The full decomposition with CIs is itself the result; percentages of GAP_total per term. |

The 0.05 AUC margins are working thresholds for labeling outcomes, not test criteria; all
underlying quantities are reported continuously with CIs. They are frozen here to prevent
outcome shopping.

## 5. Forbidden claims

- That ASTRID (or any simulation) "is realistic" from alignment success: alignment is
  calibration.
- That a residual gap is physics without surviving the dust grid and decoding checks.
- That embedding-displacement directions are causal physics statements; they are measured
  correlates with named uncertainty.
- Any claim about galaxy evolution dynamics from static snapshot comparisons.
- Any extrapolation beyond "as observed through the GZ DESI selection applied identically to
  both domains."

---

## 6. Tier 0 calibration acceptance thresholds (frozen; nothing is interpreted until passed)

Null suites: real-vs-real splits (random x20 seeds; sky-region; survey-condition quartiles;
north-vs-south, which is expected to reject and calibrates instrument sensitivity) and
sim-vs-sim splits (random halves x20; snapshot-vs-snapshot in overlapping mass bins). Every
Tier 1-2 statistic records its null over these splits; a sim-vs-real value is only quoted as a
gap relative to the random-split floor.

Injection-recovery battery (planted in held-out halves of the real sample): I1 mean shift
+delta ALONG THE PHASE 1 sSFR PROBE DIRECTION (delta in {0.1, 0.25, 0.5} of the real cloud's
per-direction std), plus an off-subspace variant: the same delta along a random direction
orthogonal to the real top-32 PCA subspace; I2 demographic deletion: REMOVE a fraction phi in
{0.25, 0.5, 0.75} of passive galaxies (emulates P1); I3 localized cluster: displace a random
2% of galaxies by a FIXED vector with pre-registered in-/off-subspace decomposition (100%
in-subspace, 50/50, and 100% off-subspace variants); I4 nuisance-only shift: add R1-style
extra noise to one half's IMAGES and re-embed a 5k subset; I5 scale alternative: inflate
variance x1.2 along a random in-subspace direction, plus one covariance-rotation variant.

**Pre-registered acceptance (binding):**
- I2 at phi = 0.5 must be recovered per its restated signature: a marginal-vs-conditional AUC
  drop of the size implied by (observed-z, M*)-only conditioning (computed analytically from
  stratum mixes), NOT full absorption, AND per-galaxy flag rates near the null rate (a mix
  shift flags no individual galaxy).
- I3 at 2% must be recovered with recall >= 0.5 at FDR 0.1, INCLUDING the off-subspace
  variant.
- Otherwise the battery is revised before any sim data are scored, and the revision is logged.

## 7. Per-galaxy discovery lists (Tier 2): validity-first construction (frozen)

- **Primary conformal route:** consensus rank-average of real-only-trained score families ONLY
  (kNN distance to the real cloud, k = 20; relative Mahalanobis in 32-d; normalizing-flow
  log-likelihood with GMM fallback), ranks computed jointly over calibration-union-test points
  per fold; conformal p-value `p(y) = (1 + #{cal >= s(y)}) / (n_cal + 1)`; Benjamini-Hochberg.
  Preprocessing fit on the real training split only, excluding the calibration set.
- **Classifier-native route:** AdaDetect (the sanctioned way to use classifier scores in a
  guaranteed discovery list).
- The classifier density ratio and the MMD witness are descriptive localization scores only,
  NEVER conformal inputs (they violate score exchangeability by construction).
- Targets: FDR q in {0.05, 0.1, 0.2}; **the primary list is real-only-trained consensus +
  conformal-BH at q = 0.1**; AdaDetect reported alongside with overlap statistics.
- Score families enter the consensus only if they pass the Tier 0 injection thresholds.
- Symmetric run: real galaxies scored against the sim cloud ("real galaxies ASTRID cannot
  make").

## 8. Geometry and concept tolerances (Tiers G and C)

- G1 intrinsic dimension: |ID_sim - ID_real| <= 1.5 at the plateau (H-null-geom tolerance);
  identical estimator grids as Phase 1.
- G2 topology: pooled beta0/MST-cut statement ("do sims fuse into the real continuum?");
  beta1 via the Phase 1 subsample-with-confidence protocol (10 x 2k, three metrics,
  Fasy/Chazal bands) — one of the two sanctioned subsampling exceptions.
- G3 curvature: delta-hyperbolicity (1e6 quadruples) + Ollivier-Ricci (2k subsample, the other
  sanctioned exception); question: do sim-real bridge edges concentrate negative curvature?
- G4 stratified ID: Phase 1 passive/SF GMM split applied to sims; compared against the real
  Delta-ID = 1.66, CI [1.38, 1.96].
- G5 diffusion map: pooled diffusion embedding; do sims interleave along dc1/dc2 (the
  morphology/redshift and colour/sSFR axes) or occupy a shifted ridge? Visual +
  per-coordinate two-sample sliced tests.
- All G results (G1-G5) re-run on best-dust (C4) and 3 extreme dust configs at R2 (dust-span
  error bars on every geometric claim).
- C-1 concept activation parity: per-concept AUC + KS for the 335 named seed-stable Phase 1
  concepts, BH-corrected as one family.
- C-2 alien-concept firing (59 candidates): three pre-registered outcomes (fires everywhere /
  fires on no sim config / fires only under specific dust-AGN configs), each informative.
- C-3 SAE re-fit control: fresh 5-seed SAE on pooled real+sim; dictionary overlap at decoder
  cosine >= 0.6.
- Baseline discipline: every concept-level claim must beat PCA directions AND random unit
  directions.

## 9. Multiplicity management (frozen families)

- **Headline family (8 confirmatory items):** the five GAP terms (Phi_img, logistic,
  R3/best-theta, estimation split), rho_P1, the P2 onset-mass difference, and the physics
  sufficiency verdict. Simultaneous (Bonferroni-adjusted) CIs within the family; the
  sufficiency verdict is an intersection-union test; Holm for point hypotheses.
- **Q-family (confirmatory falsifiers, BH at q = 0.1):** Q1 flag-rate contrast, Q2
  channel-direction AUC, Q4 span-test residual, Q5/P2 onset comparison, H-null-geom checks
  (G1-G5 across configs). Any member not executable as specced is demoted to exploratory
  before S6, never after.
- **Within-family BH at q = 0.1:** concept family (335 + 59); probe-axis family; strata
  family; discovery lists are FDR-controlled by construction.
- Effect-size-first: every quoted effect carries its Tier 0 floor and a CI; p-values appear
  only alongside effect sizes.
- Replication discipline: headline numbers recomputed on rank-matched vs absolute-matched
  Sample A variants and both embedding variants; quoted only if sign-stable across all four.
- Resolution sensitivity (pre-registered re-run): all headline numbers are recomputed
  excluding the lowest-z bin (z < 0.15) entirely; softening makes the z ~ 0.1 bin only
  marginally safe, and it is the most populated bin of a flux-limited sample.

## 10. Alignment phase (Stages A/B): principles and integrity thresholds (frozen)

Principles: (1) the frozen-encoder measurement is the science result; nothing in alignment
revises it. (2) Never optimize global domain-AUC toward 0.5 (label-shift theorem: marginal
alignment provably maps sim galaxies onto wrong real counterparts). All objectives are
conditional or instrument-pair-restricted. (3) Alignment success is measured on held-out
criteria that protect the physics: the known demographic deficits must survive; a procedure
that erases them is rejected regardless of its AUC. (4) Cheapest-first staging: embedding-space
maps (Stage A) before any encoder adaptation (Stage B); B runs only if A demonstrably cannot
reach the calibration goal.

Calibration goal: conditional AUC -> ~0.5 within (z, M*, sSFR, size)-matched strata for
instrument-attributable directions, while occupancy/demographic statistics remain unchanged.

Stage B trains LoRA adapters (r in {4, 8, 16}) on instrument pairs only (rung pairs + the
injected-reals pairs); never sees real-vs-sim labels; LP-FT ordering; WiSE-FT alpha sweep
reported in full; >= 3 seeds.

**Representation-integrity battery (run before/after every adaptation; failing ANY criterion
rejects the adapted model for calibration use, though the result is still reported):**

| Criterion | Measurement (real 48,398, paired pre/post) | Threshold |
|---|---|---|
| Probe retention | mass/sSFR/colour/morphology probe R^2 deltas | each delta > -0.02 |
| Geometry retention | ID plateau shift; beta0; beta1; Ollivier mean sign | abs(dID) <= 1.0; beta0 = 1; beta1 = 0; sign unchanged |
| Concept survival | refit-SAE dictionary match (decoder cosine >= 0.6, Hungarian) vs Phase 1 dictionary | >= 90% of the 335 matched |
| Representational drift | layerwise CKA pre/post on the same real galaxies | reported (diagnostic, no threshold) |
| Demographic protection | rho_P1 re-measured post-alignment against the FROZEN-space passive region | TOST equivalence: post/pre ratio within [0.8, 1.25] at alpha = 0.05 |

The unrestricted within-stratum OT map is run once as a diagnostic of the erase-physics
failure mode; it is expected to FAIL demographic protection. If it passes, the battery is too
weak and is revised (logged).

---

## 11. Frozen data conventions (established by measurement during S1)

Every item below is tagged **[measured]** (verified on the data, scripts and logs in
`vera/`) or **[adopted]** (a convention chosen and frozen here). The ASTRID PHOTOMETRIC
products carry no embedded metadata (every block attr on the photometric files is empty,
[measured]; recipe/band provenance exists nowhere on disk except directory names; subfind
headers carry only box-level metadata, no product conventions), so these conventions were
established by measurement, not documentation.

### 11.1 ASTRID product map

- **`photometric/PIG_*_photometric` is the INTRINSIC (dust-free) tier** [measured]: the z=0
  g-band luminosity function computed from `PIG_817_photometric` overshoots the observed SDSS
  Schechter (Blanton 2003) by ~7x at M_g = -21.9 and ~1,500x at M_g = -22.9, the canonical
  dust-free signature. Interpretation tag: the Schechter reference parameters were quoted
  from memory (h-conversion arithmetic verified; full source verification pending with the
  dust-tier LF check); the orders-of-magnitude bright-end excess is robust to any plausible
  parameter error; faint-end ~1.5x agreement partly reflects real ASTRID offsets.
- **Attenuation lives in the `Fatemeh/LSST/dust/` tier (kappa = 2.9)** [measured]: per-star
  dust-attenuated LSST apparent magnitudes on a 21-bin observed-redshift grid (z0.0-z2.0).
  Row alignment with the intrinsic tier and the subfind star block is PROVEN: over a 2M-star
  seeded sample, A_g = attenuated - intrinsic has min exactly 0.0 and zero negative values at
  any tolerance. A_g median 0.038, p95 1.08, max 6.56 mag; 33% of stars effectively
  unattenuated; rises monotonically with stellar metallicity through the 9th decile (slight
  turnover in the top decile).
- **Dust fiducial** [adopted, team-recommended]: kappa_ISM = 2.9, gamma = -0.7 (the
  Hafezianzadeh LSST-paper calibration), the centre of the 17-config ablation grid
  (kappa in {0, 1.45, 2.9, 5.8} x gamma in {-0.7, -1.0} x nebular {off, on} + one
  Charlot-Fall two-component variant; kappa = 0 is an ablation endpoint, not a defensible
  config).
- **SFR convention: `newSFa`, pinned project-wide** [adopted, measured impact]: the main
  `SubhaloSFR` column IS newSFa (byte-identical to the NewgalSFR block for all 44,366,700
  subhalos). The conventions differ by ~3x for the majority of 10^9-10^11 Msun galaxies
  (newSFa lower in ~97% of differing cases in that range; 89-99% across all mass bins); the
  quiescent fraction (sSFR < 1e-11/yr) at
  10^11.5-12 Msun is 0.59 (newSFa) vs 0.31 (oriSFa). The recipe also moves photometry
  non-negligibly (u median +0.071 mag; 38% of galaxies shift g-r by > 0.02 mag, with a
  mass-dependent sign flip in g) [measured]. Which recipe the released photometric products
  assume is an open team question (Section 14); if the answer is oriSFa or unknown, the
  recipe becomes one more ablation axis rather than a silent error.
- **Snapshot map (724 is excluded everywhere)** [measured]: snapshot 724 (z=0.3) is
  flagged-bad; the team's own z~0.3 product re-projects snapshot 743 (the
  "Fatemeh/LSST/PIG_724_photometric" directory contains the snapshot-743 catalog re-projected
  to z~0.31; row counts match PIG_743_subfind, not PIG_724_subfind). Frozen working map for
  observed-z bins:

  | observed-z bin | snapshot |
  |---|---|
  | [0.075, 0.15) | 771 (z=0.1) |
  | [0.15, 0.25) | 743 (z=0.2) |
  | [0.25, 0.35) | 743 re-projected (z=0.3; NOT 724) |
  | [0.35, 0.50) | 692 (z=0.4) |
  | >= 0.50 tail | 660 (z=0.49677, measured; header scale factor a=0.668106) |

- **DES bands derive from LSST via per-snapshot colour terms** [measured]: per-star DES
  photometry exists only at snapshots 544 and 817; at the working snapshots DES is obtained
  from LSST magnitudes via a 10-bin (lsst g-z) colour term, residual std 0.0006-0.019 mag
  across all checks (z band weakest). The colour term is calibrated per snapshot from
  per-subhalo SubGroups (all 16 bands exist there at every working snapshot) and is
  RE-CALIBRATED on attenuated colours for the dust tier. For the [0.25, 0.35) bin (snapshot
  743 re-projected to z~0.31, LSST-only product), the colour term is taken from snapshot
  743's native-frame SubGroups with the z=0.2 -> z~0.31 frame difference quoted as a
  systematic (expected ~0.01-0.02 mag from the measured redshift dependence of the offsets);
  if that systematic exceeds 0.02 mag in any band, the term is recomputed at the assigned-z
  frame from `fsps_grids/`.
- **Dedup key: `SubhaloIDMostbound`** [adopted, team-recommended]: no ready cross-snapshot
  descendant links exist for the low-z snapshots (`SubIdx`/`subfind-idx` are same-snapshot
  indices). Particle IDs are conserved, so subhalos in different snapshots sharing the same
  most-bound particle ID are the same object. Where the direct match fails (most-bound swap
  during mergers), the fallback is top-N bound-member checking with a position/mass
  plausibility bound (v_pec x delta_t; adjacent working snapshots are ~1-2 Gyr apart, up to
  ~1.9 Gyr for the 743-692 step after the 724 exclusion, i.e. ~300-600 pkpc drift at
  300 km/s, so pure position tolerance is forbidden: it would selectively miss satellites
  and cluster members).
- **Known product anomalies** [measured]: `PIG_817_photometric_newSFa` has 44,366,711
  SubGroups rows, 11 more than PIG_817_subfind, aligning with no subfind catalog; it is never
  index-aligned blindly. `PIG_724_photometric_with_issue` and `dust/z0.3_issue` are
  flagged-bad, never used. Membership/cross-referencing goes through
  `SubhaloOffsetType`/`SubhaloLenType` only (offset slicing verified exact; `SubhaloGroupNr`
  appears chunk-local and star-block `SubgroupID` rank-in-group, verified on one test
  subhalo — neither is ever used as a global index) [measured].

### 11.2 Unit and convention contract

- [adopted] ASTRID h = 0.6774; positions comoving kpc/h; masses in 10^10 Msun/h (MP-Gadget).
  All real-side masses are converted to the same h convention before matching:
  `log10 M*[Msun] = log10(M_code) + 10 - log10(h)` for code masses, and any literature mass
  quoted at h = 0.7 is shifted by `2 log10(0.7/0.6774)` where applicable; the conversion is
  applied once, here, and never re-derived downstream.
- **M* definition: stellar mass within 2x the stellar half-mass radius** (the z=0 paper's
  convention) for matching and for P1-P3 [adopted]. The extracted catalogs currently carry
  total bound stellar mass (SubhaloMassType[4]); aperture masses are computed EXACTLY for the
  FULL candidate pool (every subhalo above the selection floor at every working snapshot) by
  star-particle slicing on Vera before the final Sample A draw — no proxy, no conditional
  stand-in (an earlier draft's rank-agreement substitution rule was removed the same day on
  the user's no-shortcuts directive; amendment logged in `runLog.md`). The measured
  aperture-vs-total mapping is still reported, as a QC diagnostic only. All pre-registered
  absolute mass thresholds (10^9.3, 10^10.4, 10^10.8, 10^11) refer to aperture masses.
- [adopted] Magnitudes AB; AION-1 flux convention nanomaggies at zeropoint 22.5
  (`flux = 10^((22.5 - m)/2.5)`); mock images delivered as raw nanomaggie pixel values, never
  pre-normalized (the tokenizer applies arcsinh internally).
- [adopted] Pixel scale 0.262 arcsec/px; 96 px = 25.15 arcsec field of view; angular-diameter
  distances under Planck 2020 (the AION-1 convention); each mock stores its (z, FOV_kpc) pair.
- Magnitude-frame map of the archive [measured]: all z > 0 main products are APPARENT; only
  the two snapshot-817 products are ABSOLUTE; `restframe/` is absolute;
  `Aperture/2_Stellar_half_mass/` is apparent.

### 11.3 The real anchor (frozen)

48,398 Galaxy Zoo DESI galaxies with DR10 griz cutouts, embeddings (`E_full`, `E_img`,
1024-d), catalog scalars, and Phase 1 measurements. Reference values the sim cloud is tested
against: intrinsic dimension ~10-12 (Gride plateau ~10.1, local MLE ~11.4, PCA-PR ~11.2);
beta0 = 1, beta1 = 0 under three metrics; Ollivier-Ricci mean +0.155 with 4.2% negative
edges; delta-hyperbolicity 0.0136 (Gaussian anchor 0.0177, tree 0.0079); image-only probe
R^2: mass 0.721, sSFR 0.760, redshift 0.800, colours 0.91-0.96, morphology 0.68-0.79; 335
seed-stable named SAE concepts + 59 alien candidates; passive/SF Delta-ID 1.66 [1.38, 1.96].

Anchor caveats carried verbatim into all Phase 2 interpretation: redshifts mostly photometric
(~6.7k of 48,398 spectroscopic); morphology labels are CNN vote-fraction predictions with
5-10% error; mass/sSFR/Sersic cover only ~3.7-4.8k galaxies; selection function uncorrected.

Footprint rule [adopted]: AION-1's image tokenizer was trained on DR10-South; all headline
sim-vs-real comparisons run on the SOUTHERN subset of the anchor (MEASURED: 35,919 galaxies,
release != 9011; northern subset = 12,371, release 9011 BASS/MzLS, reserved as the
instrument-shift negative control; 48,290/48,398 covariate-matched, 108 unmatched). Real
anchor galaxies at z < 0.07 are excluded from matched-pair analyses only (resolution rule);
no mocks below z = 0.1.

I-band missingness [measured, plan §6.3 conditional now RESOLVED]: 21,029/48,398 anchor images
(43.5%) have a blank i-band channel (g-band blank = 3, so it is i-band coverage not artifact).
North is 100% i-blank (BASS/MzLS have no i); the SOUTHERN headline subset is 24.0% i-blank
(8,615 grz-only) vs 76% griz (27,304). This is MEANINGFUL -> the i-band-absent mock variant IS
produced, and i-band missingness is mirrored into mocks matched to footprint/region (it is a
footprint-correlated instrument confound, §17.3).

### 11.4 Anchor selection chain and self-pass gate

The mock-side observational selection reproduces the anchor's catalog-level chain, applied on
truth photometry + destination-brick extinction (mimicking low-noise Tractor model
magnitudes), NOT on realized-frame aperture photometry:

1. Tractor model magnitude r < 19.0 (GZ DESI: selects extended sources to an approximate
   limit beyond which galaxies rarely have meaningful resolved morphology; ~0.01 mag noise at
   the cut);
2. surface brightness mu_r > 18 mag/arcsec^2 and non-PSF morphological type (provenance
   VERIFIED against Walmsley et al. 2023, MNRAS 526, 4768: r < 19.0 + mu > 18 mag/arcsec^2 +
   non-PSF gave 8,956,477 sources; the SB cut removes stellar contamination);
3. GZ DESI image-quality cut: discard images with > 20% missing flux in any band (8,956,477
   -> 8,689,370, the catalog row count Phase 1 ingested; this cut acts on real images and is
   mirrored on mocks by the missing-band mask convention, not re-applied);
4. Phase 1 chain: 0 < z < 1, valid coordinates; AION input requirement mag_z < 21.

**Anchor self-pass gate (binding):** the mock-side selection operator applied to the real
anchor's own catalog values must pass ~100% within measurement error; failures mean the
operator is not the anchor's selection. Selection is applied per mock AFTER realism; per-bin
pass rates are logged; failing mocks are retained on disk but excluded from all embedding
comparisons.

### 11.5 Full-anchor mass-proxy route (frozen before S2, per plan 5.3)

The anchor's elpetro masses cover only ~3.7-4.8k of 48,398 (NSA cross-match, low-z-biased),
insufficient to define the (observed-z, M*) matching target. Frozen route:

- **Primary:** acquire full-anchor photometric stellar masses from a Legacy Surveys
  photometric value-added catalog (Zou et al.-style), cross-matched by position. Validation
  against the elpetro overlap, per z-bin: Spearman rank correlation, median offset, and
  scatter all logged. **Acceptance: Spearman rho >= 0.8 within z-bins on the overlap** (rank
  matching needs rank fidelity, not absolute calibration). For absolute mass strata (P1-P3),
  the proxy is recalibrated to the elpetro scale by the measured median offset and the
  scatter is carried as an error band on stratum boundaries.
- **Fallback (if no proxy meets acceptance):** match on fully-covered observables
  (observed-z, absolute M_r, g-r) and demote M* to a derived covariate; P1/P2 mass strata
  then use the proxy with its systematic as an explicit error band.
- The acceptance threshold rho >= 0.8 is set here, before any candidate catalog has been
  examined.

---

## 12. Frozen sample-construction rules (stage S2)

- Two samples, ~50k each, drawn from the union of snapshots per the Section 11.1 map. Sample
  A = demographics-matched (the "are the galaxies it makes realistic?" sample, primary for
  Tiers 1-3); Sample B = volume-weighted-as-observed via importance sampling (the "does it
  make the right mix?" sample, primary for P1-P3 occupancy).
- Selection plate: SubFind subhalos with M* >= 10^9.3 Msun (feasibility floor, finalized at
  the G-S2.1 pilot and logged; not a science cut); >= 300 star particles; central/satellite
  recorded, not selected on; dedup per Section 11.1 with at most one snapshot instance per
  linked chain per sample, STRICT (one instance ever) at M* > 10^10.8.
- **Anti-comb rule (binding):** mocks are NOT placed at snapshot redshifts. Each drawn
  subhalo receives a continuous target redshift sampled from the real anchor's z distribution
  within its bin; distance modulus, angular scale, dimming, and band-shifting are evaluated
  at the assigned z using the nearest snapshot's stellar populations. A 4-tooth redshift comb
  in a space where redshift decodes at R^2 = 0.8 would hand every classifier a fake feature.
- **Observed-z convention (both sides):** each mock also receives an observed redshift =
  assigned z + photo-z noise drawn from the anchor's stratum-matched photo-z error model
  (fit on the anchor's spec-z subset). ALL stratification, matching, and conditioning use
  observed-z on both sides.
- Sample A: stratified on observed-z bins of width 0.05 x log10 M* bins of 0.1 dex over
  [9.3, 11.8]; within-bin draws WITHOUT replacement, seed = 0, to the real bin count scaled
  by 50,000/N_real_south; primary matching is RANK matching within z-bins (1-D optimal
  transport on M* ranks; invariant to monotone mass-calibration offsets); absolute-value
  matching is the robustness variant. Apparent magnitude, size, colour, sSFR are deliberately
  NOT matched (they are outputs under test). The draw-render-select loop iterates until
  POST-selection counts match the target; starved bins feed P3.
- **One-entry rule (set here; closes a hazard opened by the 724 exclusion):** each physical
  galaxy (each dedup chain, and in particular each snapshot-743 subhalo) enters a given
  sample AT MOST ONCE across ALL observed-z bins, at all masses — the [0.15, 0.25) and
  [0.25, 0.35) bins both draw from snapshot 743, so without-replacement bookkeeping is
  shared across those two bins. The G-S2.3 audit is extended accordingly: zero same-subhalo
  duplicates between the two 743-fed bins at ALL masses (not only M* > 10^10.8).
- Sample B: importance-sampled (per-cell weights proportional to V_shell(z) x n_ASTRID(cell),
  oversampled inversely to the pre-estimated selection probability p_sel(cell)); every
  surviving mock carries its importance weight; all B statistics are weight-aware; ESS
  deliverable per stratum (gate G-S2.5: P1 stratum ESS >= 1,000).
- Exit gates: G-S2.1 pilot 500 mocks end-to-end (median surface brightness within
  1 mag/arcsec^2 of matched reals); G-S2.2 per-bin standardized (z, M*) differences < 0.05
  post-selection except enumerated starved bins; G-S2.3 dedup audit (zero duplicates at
  M* > 10^10.8 within any sample); G-S2.4 provenance manifest (every mock traceable to
  snapshot, subhalo ID, seed, view, theta_dust, rung).
- One random viewing direction per galaxy (isotropic, fixed seed); the 4-view orientation
  control runs on a fixed 3,000-galaxy subset analyzed separately (views never enter
  two-sample tests as extra samples).

---

## 13. Exploratory scope (declared, NOT confirmatory)

The following are in paper scope but carry no pre-registered thresholds; nothing in them can
be promoted to confirmatory after S6 begins:

- Interpretability tracks adopted 2026-06-10: (A) angular quantities as loops in feature
  subspaces; (B) Hubble-fork branch-point recovery; (C) redshift-axis steering /
  causal-translation tests; (D) irreducibly multidimensional SAE features (Engels
  reducibility test on the Phase 1 dictionary).
- Decoder-based cross-modal generation (mock image -> generated spectrum vs mock/real
  spectra), with the AION-1 authors' caveat that generations are plausibility samples, not
  calibrated posteriors.
- Wave 2 (DESI-like spectra) is gated by G-W2 stack-level validation; the FSPS-vs-BPASS
  library difference is treated as an explicit modeling systematic when it runs.
- Q1-Q5 designs are specified in the plan; their confirmatory falsifiers are the Q-family
  members listed in Section 9; everything else within them is exploratory.

## 14. Open items registered as non-blocking (with mitigations)

| Item | Status | Mitigation if unresolved |
|---|---|---|
| Which SFR convention the released photometric/dust products assume | Open with ASTRID team (Patrick Lachance is the confirmer) | Recipe difference measured (Section 11.1); becomes an ablation axis, not a silent error |
| The 11-extra-rows `_newSFa` anomaly | Open with team | Product never blind-aligned; not on any critical path |
| Quenching-channel labels (compaction vs inside-out) at working snapshots | Open with team; gates Q2 ONLY | Pre-registered descope ladder: team labels -> rebuild partial histories via most-bound-ID chains over the 33 available subfind epochs -> Q2 reported as a null/descoped |
| Dust-tier g-band LF check against the published calibration target | Queued on Vera before S3 photometry use | If it fails, the dust tier is recomputed from `fsps_grids/` with documented parameters |
| Value-distribution diff of the re-projected z~0.31 product vs the neighbouring 743/771 products (binding pre-use check; identity evidence so far: row counts, frame convention, byte-level cmp) | Queued on Vera before S3 use of that product | If distributions are inconsistent, the [0.25, 0.35) bin is re-projected from snapshot 743 ourselves via `fsps_grids/` |
| Southern-subset / i-band-missingness / DESI-spectra counts | Measured during the covariate cross-match (S1 carryover, in progress) | Counts are descriptive; rules fixed in Section 11.3 |

## 15. Declaration

Frozen by commit to the repository on 2026-06-10, before stage S2 (sample construction),
before stage S3 (forward modeling), and before stage S4 (any sim embedding). The five
catalog parquets extracted from the ASTRID archive (4.85M subhalos with M* > 10^9 Msun, 29
columns, both SFR conventions, star-block offsets) are staging data for S2 and contain no
embeddings. Amendments to this document require a logged rationale in `runLog.md`;
confirmatory-to-exploratory demotions are permitted only before S6 begins.
