# Paper 1 — committed end-to-end plan (for review with Matt)

**Working title:** *Foundation models learn the topology of physical parameter spaces*
(chosen 2026-07-05; the earlier subtitle "a causal probe battery for astronomical
representations" is dropped)

**Venue plan (locked):** ICLR 2027 main track (abstract ~mid-Sept, full ~late Sept 2026 —
confirm exact dates). A 4-page distillation goes to ML4PS @ NeurIPS 2026 (deadline
~late Aug/Sept) as the non-archival flag-plant. Journal follow-up (HSC extension) later.

---

## 1. Thesis (one sentence)

A frozen multimodal foundation model trained on sky surveys does not merely predict
physical labels — it reconstructs the *shape* of each label's space (circles for periodic
angles, arcs for bounded ones, a fork for morphology), respects the symmetries physics
demands (verified by intervention, not correlation), and correctly represents what each
observing modality can and cannot know (images inherit the age–metallicity degeneracy;
spectra break it).

## 2. Why this clears the main-track bar

1. **Ground truth.** Unlike LLM interpretability, every probed concept has an
   independently measured physical value (catalog angles, spectroscopic ages). The
   applied image transformations (rotation/flip) are exact interventions; the catalog
   labels carry their own measurement/model error (photo-z, Tractor shape fits, CNN vote
   fractions, SED/spectral fits) and are treated as measured references, not exact truth.
2. **Causal, not correlational.** Symmetries are verified by physically transforming real
   inputs and re-encoding (rotation done: readout tracks rotation with |slope|=0.999,
   median-fit residuals ≤0.05°, per-galaxy errors ~2-3°; mirror flip done 2026-07-07:
   reflection 2.38°/1.99°, composition 2.59° — O(2) complete).
3. **Generality by design.** One fixed battery, eight representations (scale, training
   paradigm, domain, floor), four data regimes (two instruments, two modalities, two
   depths) — with *pre-registered per-paradigm predictions*, so the comparison is
   hypothesis testing, not a leaderboard.
4. **Reusable artifact.** The battery ships as code + public-data recipe ("galprobe"):
   any scientific representation can be scored on topology, symmetry, invariance,
   cross-modal consistency, and degeneracy inheritance.
5. **[added 2026-07-04] A claim with reach beyond astronomy.** T6 co-headlines with the
   loop: the same frozen model represents what each observing modality *cannot* know
   (images barely encode age, R²=0.07, carrying the age–metallicity degeneracy as
   missing information — with no extra entanglement beyond the true correlation; the
   spectrum pathway breaks the degeneracy). A model internalizing the epistemic limits
   of its own sensors is the result an ML reader takes away.
6. **[added 2026-07-04] A sharper answer to representational convergence.** The grid
   refines the Platonic Representation Hypothesis in astronomy (Platonic Universe,
   arXiv:2509.19453; recalibration arXiv:2602.14486): models converge on *decodability*
   but diverge in *geometry* according to training objective — convergence-on-what,
   answered with ground truth.

## 3. Contributions (the claims list, as it will appear — items tagged [prediction] are
## pre-registered but UNRUN as of 2026-07-09 and must not be read as current results)

C1. A physically grounded probe battery (8 tests, with nulls + bootstrap CIs as the
    design standard) for scientific representations; released with code and public data.
C2. Topology matching: the model builds the correct label-space topology in three
    distinct cases — position angle → mod-180 circle (RP¹) learned from images,
    RA → circle (S¹) via the native catalog coordinate codecs (an information-preserving
    codec-geometry result, weaker than the image-learned loop), Dec/inclination →
    bounded arcs — and the structure is invisible to PCA/global topology (must be
    probed for).
C3. Interventional symmetry verification: the complete O(2) group (rotations + mirror)
    acts on the internal angle coordinate exactly as physics requires, including the
    180°-rotation fixed point.
C4. Cross-modal consistency: image-derived and catalog-derived angle readouts agree
    (3.6°; two separately target-trained probes — consistency, not a shared-subspace
    proof). [prediction] Cross-instrument: results replicate on BASS/MzLS north.
C5. Modality-resolved degeneracy inheritance: the image pathway barely encodes stellar
    age (R²=0.07) and only weakly metallicity (0.25) — the classic degeneracy expressed
    as missing information — with no consistent extra entanglement beyond the label
    correlation (age–metal angle 93.7° vs 91.5° truth; excess consistent with zero
    under both CI constructions for the headline and mass-weighted variants; the
    z≥0.15 variant is marginally positive under the basic interval only, so the honest
    summary is "no consistent excess", not "all variants zero"). [prediction] The
    spectrum pathway (same model, same galaxies) breaks the degeneracy.
C6. [prediction] Emergence across models: the battery run on 8 representations tests
    whether the training objective — not the data — determines which physics survives
    (frozen per-paradigm predictions below).
C7. Honest negatives, kept: geodesic distance does not beat Euclidean for morphological
    similarity; the fork's dominant residual axis is not the bar; unsupervised search
    recovers only part of the loop (R² 0.41 vs 0.97 supervised).

## 4. The battery (test names locked)

| # | Test | Status |
|---|------|--------|
| T1 | Topology matching (RP¹ / S¹ / interval) | DONE (PA 2.0°, RA 2.9°, Dec/incl arcs) |
| T2 | Causal symmetry: rotation + mirror flip (O(2)) | DONE (rotation slope −0.999; flip 2.38°/1.99°; composition 2.59° — 2026-07-07) |
| T3 | Invariance (brightness, morphology, SNR/quality) | DONE (loop radius ~0.99 everywhere) |
| T4 | Cross-modal + cross-instrument consistency | image/shape DONE (3.6°); **north = new, free** |
| T5 | Morphology hierarchy (fork, branch, bar independence) | DONE (+11.9° CI[9.6,15.1]; fan 1.70×) |
| T6 | Degeneracy inheritance, modality-resolved | image side DONE; **spectrum side = new** |
| T7 | Unsupervised discoverability (PCA/SAE) | DONE (0.41 vs 0.97) |
| T8 | Feature reducibility (Engels test, retrained dictionary w/ saved decoders) | image side DONE (2026-07-02/04); **spectrum side = new**. Note: the old 335+59 counts are retired — corrected tie-aware/FDR scoring gives 380 aligned-and-stable + 3 stable-unexplained (2026-07-09) |

### 4.1 Controls & robustness (added 2026-07-04, pre-run — closes the reviewer-facing design gaps)

- **RA/Dec confound audit (gates T1's sky-coordinate claim; ~1 day, W1).**
  (a) Input-token check: the probed embedding must be position-blind by construction —
  probe on E_img and audit the tokenizer input list to confirm no coordinate/scalar token
  carries sky position. (b) Mediation check: observables that vary smoothly with sky
  position (Galactic E(B−V), stellar density, imaging depth/PSF from the survey CCD maps)
  are regressed against the loop coordinate; report the partial R² of RA/Dec beyond them.
  Pre-registered reading, fixed now: unmediated loop = the model tracks where it looked;
  fully mediated = position-correlated observables imprinted on images (still a finding —
  a survey-systematics detector — but a different claim, and the paper says which one).
- **Pooling + depth robustness (~half day, W2, AION-large only).** Headline battery
  numbers re-run under attentive pooling (vs mean) and at two intermediate encoder
  depths. Report stability; headline only pooling-stable results. (Standard reviewer ask;
  cheap to preempt.)
- **Multiplicity policy (whole battery, stated once).** Every cell of the 8-test × 7-model
  grid reports effect size + bootstrap CI, never a bare p. The pre-registered prediction
  family is scored ✓/✗ with a Holm-corrected family summary; exploratory cells are labeled
  exploratory and cannot be promoted after results are seen (same discipline as
  preRegistration.md §9).
- **Verifiable pre-registration.** The §5 prediction table is frozen by repo commit
  BEFORE any non-AION model is embedded; the commit hash goes in runLog.md and is cited
  in the paper (and the ML4PS 4-pager). "Pre-registered" must be checkable, not asserted.

## 5. The model grid (locked, 8 representations — all weights local in data/models/)

| Representation | Axis it tests | Pre-registered prediction |
|---|---|---|
| AION-large (ours) | reference | full battery passes (measured) |
| AION-base ✓ | scale (point 1 of 3) | same structure, degraded precision; degeneracy angle narrower |
| **AION-xlarge ✓ (12.5GB; DECIDED: embed in bf16 on the 12GB card; AWS GPU = sanctioned fallback if it still doesn't fit — fleet playbook ready)** | scale (point 3 — the trend line Matt asked for) | structure sharpens monotonically with scale |
| AstroCLIP ✓ (official polymathic-ai) | contrastive multimodal, same domain | **no PA loop by design** — recipe verified (DINOv2-style pretraining, random rotations U(0,π) + flips, arXiv:2310.03024v2); fork present |
| Stein MoCo-v2 ✓ (authenticity fingerprint-verified; 3-band grz input — feed cutouts minus i) | contrastive, rotation-augmented | **no PA loop by design** (augmentation invariance); fork present |
| Zoobot convnext-nano ✓ | supervised morphology | no PA loop (labels orientation-free); strong fork + inclination |
| DINOv2-base ✓ (RGB composites) | generic vision domain | hflip-only augmentation ⇒ **folded arc** (RP¹/reflection, fixed points 0°/90°), not a closed loop; no degeneracy structure |
| Raw-pixel PCA (no weights) | floor | orientation trivially present in pixels but **wrong topology** in top components |

(Correction vs the first draft: "Stein SimCLR" is precisely **MoCo-v2** — momentum encoder,
65,536-negative queue, ResNet-50, emb 128 — same contrastive-augmentation family, cite accurately.)

The per-paradigm predictions are written down before running — the comparison is a set of
falsifiable hypotheses about *why* structure emerges (reconstruction objectives must keep
orientation; augmentation-invariant and label-supervised objectives must discard it).

## 6. The data grid (locked; acquisition executed 2026-07-04)

| Dataset | Size | Role | Status |
|---|---|---|---|
| GZ-DESI anchor, south (DECam) | 35,919 | headline | ✓ on disk |
| GZ-DESI anchor, north (BASS/MzLS) | 12,371 | cross-instrument replication | ✓ on disk |
| Faint Legacy extension (r 19–21) | **150,000** catalog+shapes; **136,407** good cutouts (90.9%) | depth/SNR robustness (loop vs magnitude to the noise floor) | ✓ complete on disk (fleet done + torn down 2026-07-04) |
| DESI/SDSS/BOSS spectra for anchor | **20,986 spectra / 17,643 galaxies** (3× plan; ~3.3k dual-instrument) | spectrum-pathway tests (T4/T6) + free cross-instrument spectra sample | ✓ on disk (100% of manifest) |
| DESI age expansion (FastSpecFit iron v3.0) | **13,044 anchor + 7,061 faint** ages (+mass/SFR/Dn4000, all with ivars) | tightens T6's age side | ✓ on disk. **Caveat: metallicity-at-scale does not exist in any public DESI product** (ZZSUN unpopulated) — the joint age–metal test keeps Firefly's n=5,512 |
| WISE W1/W2 + AGN proxy | 47,566 | Track D AGN label | ✓ on disk |

HSC (different telescope end-to-end) is explicitly OUT of this paper — journal version.

## 7. Figure plan (7 main figures)

F1 (hero): the loop — cos2θ/sin2θ colored by true PA, with controls (shuffle, PCA, linear).
F2: causal symmetry — recovered vs applied rotation (slope line) + mirror-flip panel.
F3: the grid — battery × 8 models heatmap with pre-registered predictions marked ✓/✗.
F4: the fork — handle/prongs scatter + branch fan + bar-independence CIs.
F5: modality-resolved degeneracy — image vs spectrum pathway, decodability + angle
    (co-headline with F1 in abstract/intro — the reach-beyond-astronomy result, §2.5).
F6: depth robustness — loop error vs magnitude, bright anchor → r=21 noise floor.
F7: discoverability + reducibility — supervised vs unsupervised recovery; irreducible-2D features.

## 8. Paper skeleton

1. Introduction (thesis; why ground truth + intervention beats LLM-style probing)
2. Setup: models, data, the battery (nulls, CIs, pre-registered predictions)
3. Topology (T1) + invariance (T3) + discoverability (T7)
4. Causal symmetry (T2) + consistency (T4)
5. Hierarchy (T5) + degeneracy across modalities (T6) + reducibility (T8)
6. Emergence across models (the grid, F3)
7. Negatives, limitations, and what they mean
8. Related work — five threads, full map + drafted prose in [relatedWork.md](relatedWork.md): (i) probing & world models (Othello line, Vafa, Belinkov, El Banani battery); (ii) feature geometry (Engels, Kantamneni, Modell, Gurnee, Park; grokking circles; grid cells); (iii) SAEs on scientific models (Euclid SAE, InterPLM, single-cell); (iv) astro FMs + convergence (AION, AstroCLIP, AstroPT, Zoobot; Platonic Universe); (v) invariance/equivariance (Xiao, Dangovski)
9. Release: galprobe battery + reproduction recipe

## 9. Timeline (11 weeks from Jul 6; ICLR full ~late Sept)

- W1: Matt sign-off on this plan; mirror-flip test; north replication; Track C brought to audit grade (CIs, log-metallicity, steering relabel); **RA/Dec confound audit (§4.1); freeze the §5 prediction table by repo commit before any grid embedding**.
- W2–3: model grid — embed anchor through aion-base, AstroCLIP, SimCLR, Zoobot, DINOv2, pixel-PCA; run battery on all; **pooling/depth robustness ablation on AION-large (§4.1)**.
- W3–4: spectra fetch + spectrum-pathway T4/T6.
- W4–6: faint extension (fetch 100k on Vera, embed, depth curve); DESI VAC age/metal expansion.
- W6–7: Track D reducibility (T8); freeze all numbers.
- W7–8: figures F1–F7; galprobe packaging.
- W8–10: writing (plain-language discipline), internal adversarial review, group pass.
- W11: buffer + submit. ML4PS 4-pager distilled in parallel (early Sept).

Compute: everything runs on the local GPU (measured 17 img/s ⇒ ~100 min per 100k per
model); aion-xlarge runs bf16 locally (AWS GPU is the sanctioned fallback if needed).
CORRECTION (2026-07-09): "no cloud spend" was wrong — the faint-extension cutouts were
fetched with the resurrected AWS fleet (~$3 total, torn down 2026-07-04), not Vera.

## 10. Asks for Matt (next week)

1. Confirm venue order (ICLR main + ML4PS flag-plant) and co-author expectations.
2. AstroCLIP / Zoobot / SimCLR checkpoints or precomputed Legacy embeddings, if the group has them (saves W2).
3. Whether any newer public astro foundation model should join the grid — sweep done 2026-07-04; concrete candidates: **Wu & Walmsley's released Euclid MAE (arXiv:2510.23749) or AstroPT (arXiv:2405.14930)** as a second reconstruction-objective member (~2 h compute; doubles the reconstruction class so all three paradigm classes have ≥2 members). Additive; skippable without touching claims.
4. ~~AstroCLIP's augmentation recipe (does it rotate?)~~ **RESOLVED 2026-07-04**: it rotates (U(0,π) + flips, verified against arXiv:2310.03024v2) — prediction locked in §5. Checkpoint access still wanted.
5. Framing check: is "topology of parameter spaces" the pitch he'd co-sign, or does he want the AI-for-science framing led differently?

## 11. Relationship to Paper 2

Paper 2 (ASTRID sim-vs-real, pre-registered, ApJ/MNRAS) cites this paper as the
instrument calibration. No claim overlap: Paper 1 never touches simulation data; Paper 2
never claims interpretability results. The two share code and the frozen anchor only.
