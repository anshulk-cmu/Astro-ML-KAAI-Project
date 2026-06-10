# AION-1 Galaxy Embedding — Representational-Geometry Prototype

A bounded, evidence-driven study of the **geometry** of AION-1's galaxy embedding, run end-to-end on **48,398 real galaxies**, to decide — with statistics, not vibes — whether a full-scale study is worth committing to.

## The question
AION-1 (Polymathic AI) compresses any galaxy into a **1024-number vector**. Its authors assert the embedding "organizes objects along physically meaningful directions" but never measured its geometry. We do: intrinsic dimension, curvature, topology, how well physics decodes from it, and what concepts it invents.

## What we did
1. **Data.** Sampled 50,000 Galaxy Zoo DESI galaxies (seed = 0, 0 < z < 1) and fetched 4-band (g, r, i, z) Legacy-Survey DR10 cutouts. The cutout service caps one IP at ~1.5 img/s, so we parallelised across a **12-machine AWS fleet** with continuous checkpointing → **48,398 / 50,000 (96.8 %)** images. (An earlier 24-IP run was rate-limited and stranded its data; we redesigned it — full story in `runLog.md`.)
2. **Embeddings.** Ran frozen **AION-1-Large** on an L40S GPU → `E_full` (image + g,r,z flux + redshift) and `E_img` (image-only, leakage-free for probes), mean-pooled to 1024-d.
3. **Geometry — 10 arms, full data, no subsampling** except where it is combinatorially impossible (β1, exact δ): metric diagnostic, intrinsic dimension (triangulated + synthetic-validated), diffusion map, decodability probes + disentanglement, curvature (Ollivier-Ricci, δ-hyperbolicity), sparse-autoencoder concept discovery, persistent homology, population-stratified ID, geometry-of-concepts.

## Key findings (measured)
- **Intrinsic dimension ≈ 10–12** — ≥3 independent estimators agree, certified on synthetic spheres/rolls. A huge compression from 1024.
- **A single, smooth, simply-connected continuum** (β0 = 1, β1 = 0; the diffusion spectrum has no gap) — *not* discrete red/blue clusters; the populations form a connected gradient.
- **Physics is strongly encoded.** From the **image alone**, ridge probes recover stellar **mass R² = 0.72**, **sSFR 0.76**, redshift 0.80, colours 0.91–0.96, morphology 0.68–0.79; the concept directions are disentangled beyond label correlations.
- **Sparse autoencoders find 335 seed-stable, physics-named concepts** (colour, redshift, morphology, merger) plus **~59 "alien" candidates** not explained by any catalogue label.
- **Mostly positive curvature, weak branching** (not tree-like). Passive galaxies occupy a **significantly higher-dimensional** submanifold than star-forming (ΔID = 1.66, CI excludes 0).
- **Verdict: STRONG-GO** — low-dimensional, physically meaningful, concept-rich, structured. *Honest caveats:* ID sits at/just above the optimistic 4–10 band; SAE "alien" features are correlational only (causal tests deferred); redshift labels are mostly photo-z; β1 uses subsample-with-confidence.

## Repository
- `code/` — `buildSample`, `embed`, the `fleet/` download pipeline, and `analysis/` (one script per arm + `common`, `synthetic`).
- `results/` — one JSON per arm (+ small arrays).
- `runLog.md` — every step, decision, and **corrected mistake** (e.g. a degenerate diffusion bandwidth, an SAE zero-inflation scoring bug, a tautological hierarchy test — all caught and fixed).
- `problem.md` (full-study methods spec), `analysisPlan.md` (locked prototype plan), `prototypePlan.md`, `papers/` (reference SAE papers).
- Raw images/embeddings live in `data/` (git-ignored; reproducible).

## Reproduce
`setupAion.sh` (model + AION) → `stageInspect.py` (catalogs) → `buildSample.py` → `fleet/launchFleet.py` + `gather.py` (images) → `embed.py` → `analysis/*` in the order listed in `analysisPlan.md`. Every measurement is pulled back to local; the AWS box was compute-only and was **terminated after verification** (no billable resources remain).

## Phase 2 (in progress): is the ASTRID simulation in-distribution for AION-1?

The frozen encoder becomes the measuring instrument: forward-model **~50k ASTRID galaxies** (CMU McWilliams cosmological simulation, run to z = 0) into Legacy-Survey-like mock observations, embed them with the same frozen model, and attribute every component of the sim-vs-real gap — instrument, dust, demographics, residual physics — with per-galaxy false-discovery-rate control and pre-registered falsifiable predictions (ASTRID's published quiescent-fraction and luminosity-function deficits must be rediscovered blind).

**Groundwork complete (measured, on PSC Vera where the data live):**
- **Data census.** 49 SubFind catalogs (44.4M subhalos at z = 0), 80+ photometric products mapped; working snapshots locked (z 0.1→771, 0.2→743, 0.3→743, 0.4→692; snapshot 724 is flagged-bad and excluded). Per-galaxy star access via catalog offsets, verified exact.
- **Conventions established by measurement, not documentation** (the products carry no metadata): the main photometry tier is **intrinsic** (its z=0 g-band luminosity function overshoots the observed bright end ~1,500×, the classic dust-free signature); attenuation lives in a separate κ = 2.9 tier, proven row-aligned star-for-star; the two star-formation conventions differ by ~3× (quiescent fraction nearly doubles at high mass) → pinned to `newSFa`; DECam bands derive from LSST bands via a measured colour term (residual ≤ 0.019 mag).
- **Catalogs extracted** (5 snapshots, 4.85M galaxies above 10⁹ M☉) — sample construction runs locally.

`vera/` holds the Phase 2 scripts, logs, and results; the execution plan is pre-registered before any simulated galaxy is embedded. Science scope now also includes four interpretability tracks on the real embeddings (angular loops, Hubble-fork branch point, redshift-axis steering, multidimensional SAE features) and cross-modal decoder generation.

**Principles:** local is the system of record; every number is tagged *measured* vs *interpreted*; full data, no shortcuts, honest limits.
