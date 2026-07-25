# Handoff — AstroML Paper 1 diagnostic suite

Written 2026-07-25 at the end of the Diagnostic 1 session. Read this top to bottom before touching anything. Everything here is verified state, not recollection.

---

## 1. Where the project stands

Two papers come out of this repository.

- **Paper 1 (interpretability).** A diagnostic suite measuring whether AION-1's frozen embedding represents the world the way physics says it should. This is the active work.
- **Paper 2 (ASTRID sim-vs-real).** Pre-registered, samples built, 99,585 mocks rendered on the Vera cluster. **Paused since 2026-06-17** on two decisions you never gave: the empty-centre criterion for injection, and the R3 gain. No simulated galaxy has been embedded, so the pre-registration gate is intact. Do not touch it without an explicit instruction.

Paper 1 was reorganised on 2026-07-21 when Matt returned `paperScopingV1.md` (synced from HackMD, local-only, gitignored). It replaces the old Track A–D framing with **nine diagnostics in three pillars**, each carrying a declared evidential tier. That document is the specification. Read it in full before implementing any diagnostic; do not work from this summary of it.

The older work lives in `code/analysis/` and is described by `technicalReport.md`. **Neither is deleted or edited.** `paper1/` is a clean rebuild against the new structure.

---

## 2. Repository state, exactly

Branch `main`. Six commits from this session, **none pushed**:

```
2c0f892 Document cache-free invocation
9b9b194 Suppress bytecode and pytest cache generation
4fa481f Handle label sentinels at source and declare label provenance
1a0758e Fill report appendices and findings index; regenerate D1 artifacts
60db172 Scope provenance dirty flag to analysis code
daa5a98 Add paper1 diagnostic suite with dataset audit and Diagnostic 1
```

Also staged earlier in the session: the rename `Technical-Report.md → technicalReport.md`. Everything at the repository root is now camelCase (`paperScopingV1.md`, `paperSuggestions.md`, `reportSections/`, `recordings/`).

Untracked and intentionally so: `vera/` outputs, `code/sim/*.py`. Gitignored and local-only: `runLog.md`, `paperScopingV1.md`, `paperSuggestions.md`, `relatedWork.md`, `problem.md`, `slides/`, `data/`, `papers/`.

`runLog.md` is 1,347 lines and is the project's history. **It is the only place development history goes.**

---

## 3. Binding rules

From `paper1/standard.md` (read it — it is binding on all nine diagnostics, with D1 as the reference implementation) and from the user's standing instructions:

1. **The technical report is a professional academic record of method and result.** No bugs, no failures, no reruns, no "prior work" narrative. All of that goes in `runLog.md`.
2. **Log to `runLog.md` as you go**, immediately after each step, not batched at session end.
3. **Report everything measured; select no headline.** That choice belongs to the manuscript later.
4. **No number enters the report unless a `paper1/` script produced it** with a provenance block, and an automated audit compares the written value against the artifact.
5. **Nulls must themselves be calibrated** against their analytic expectation, and reported on every axis the result is reported on.
6. **Conventions are verified numerically, not quoted**, and pinned by a test that fails loudly *before* expensive computation.
7. **No shortcuts or scope cuts without explicit approval.** Present the tradeoff instead.
8. Code: camelCase files and folders, minimal comments, no Jupyter notebooks, clean and simple.
9. Commits: short imperative subject, no AI attribution ever, **do not push** unless asked.
10. Check every result through three lenses before reporting: intuition, mathematics, physics.

---

## 4. Environment and exact commands

`conda` is **not** on the Git-Bash PATH. Call the interpreter directly:

```
/c/Users/worka/anaconda3/envs/interp/python.exe
```

Python 3.11.14, NumPy 1.26.4, SciPy 1.17.0, scikit-learn 1.8.0, PyTorch 2.10.0+cu128, polymathic-aion 0.0.2. GPU is an RTX 5070 Ti laptop, 12 GB, and re-encoding runs at about **17 images/s** — the unit cost for pricing any sweep.

Always pass `-B`, which keeps the tree free of `__pycache__`:

```
python -B paper1/runAll.py              # status and provenance of all diagnostics
python -B paper1/runAll.py d1           # run one, then render its figures
python -B paper1/runAll.py --figures d1 # figures only
python -B paper1/runAll.py --verify     # re-hash every recorded input
python -B -m pytest paper1/tests -q     # 26 passed, 1 skipped
```

---

## 5. What is finished

**Diagnostic 0 — dataset and substrate audit** (`d0DatasetAudit.py`). Reference artifact backing report sections 1 and 2: anchor counts, row contract, recipe constants, substrate statistics, label coverage with sentinel handling, shape-catalog composition, covariates.

**Diagnostic 1 — angle readout characterization**, complete under the standard. All four stress axes and three nulls. Key values live in `paper1/results/d1AngleReadout.json`; do not re-type them, read them. Summary: readout 2.027° [1.946, 2.116] on 3,179 held-out of 15,893, loop radius 0.9886; nulls shuffled 44.060, top-2 PCs 43.198, plain linear 10.329, against a 45° chance floor; grading monotone 23.643° → 1.646° with Spearman −0.991; error ∝ ellipticity^−0.974 against −1.109 for the catalog's own uncertainty; label floor σ_PA = 0.111°; ten-split sensitivity 1.989 ± 0.050.

**Report** `paper1/techReportPaperScopingV1.md`, 525+ lines: reading rules, protocol, substrate discipline, master table, sections 1–3, section 4 (D1, subsections 4.1–4.12), sections 5–12 as stubs for D2–D9, section 13 findings index (13 rows), appendices A–F.

---

## 6. Gotchas that cost time this session — do not rediscover them

- **The Grep tool respects `.gitignore`**, which here hides `runLog.md`, `slides/`, `data/`, `papers/`. Any negative "does not exist" claim made with Grep alone is unreliable. Use raw `grep`/`find` for negative claims.
- **Bash heredocs fail intermittently** in this shell ("unexpected EOF"). Write a Python script to the scratchpad with the Write tool and execute it instead.
- **WebFetch returned no output on every URL** this session, three separate attempts. WebSearch worked. Subagents are network-denied in this project, so network calls belong in the main session.
- **Pytest's default collection pattern does not match camelCase filenames.** `paper1/pytest.ini` sets `python_files = test*.py`. Without it, pytest collects zero tests and reports success.
- **argparse rejects an empty default when `choices` is set with `nargs="*"`.** Validate manually.
- **Provenance `code_dirty` is scoped to `paper1/{lib,diagnostics,config.py,runAll.py,tests}`**, not the whole repo, so untracked Phase-2 files do not make it permanently true.
- **Two catalog traps, both already fixed in code**: 4,618 sources (REX, PSF, DUP) carry `shape_e1 = shape_e2 = 0` exactly and have placeholder position angles — use `data.pa_defined()`. And `total_ssfr_median` encodes "not measured" as −99.0 — use `data.valid()`, never bare `isfinite`.

---

## 7. The exact next step: Diagnostic 2, O(2) equivariance

**Step 1 — the gate. Do this first, before touching any embedding.**

Implement `rotate` and `mirror` in `paper1/lib/transforms.py` (contracts are already written in the file), then make `paper1/tests/testTransforms.py` real. It currently skips itself via `importorskip` plus a `hasattr(transforms, "rotate")` guard, so it will activate automatically once the operators exist. The tests must pin:

- the rotation sign on a synthetic bright bar of known angle: whether `scipy.ndimage.rotate(+φ)` shifts the array-frame angle by `+φ` or `−φ`;
- the array-frame versus catalog-frame handedness;
- that the mirror is an exact pixel permutation (pixel multiset unchanged, no interpolation);
- error bounded by the period.

D2 lives or dies on this bookkeeping. Previously it was verified by hand and survived only as prose. Tests must pass before step 2.

**Step 2 — the diagnostic.** `paper1/diagnostics/d2Equivariance.py`, reusing `lib/circular.py` unchanged.

Cached encodes already exist and need **no GPU**: `results/trackA_causal_ckpt/rot_{30,60,90,120,150,180}.npy` and `results/trackA_flip_ckpt/{flip,flip_rot30}.npy` — eight files, each `(15893, 1024)` float32, 65,097,856 bytes, about 520 MB total. Rows are the elongated population in **ascending `ok_index` order**, independently re-verified this session; that population is exactly the 15,893 the current `pa_defined` mask selects, so they align with the new harness without adjustment. Implement `lib/encode.py`'s `adopt_legacy` to register them after checking shape, dtype and row order — never adopt blind.

Measure: rotation slope across the grid; the 180° self-map; mirror reflection with its fixed points at 0°/90° and antinodes at 45°/135°; composition (flip then rotate 30°); all-population **and** held-out summaries; and exclude 90° from the slope fit while still reporting its per-galaxy circular error, since ±90° is one point on a mod-180 loop.

**The piece never run anywhere: the invariance complement.** Read colour, morphology and stellar mass out of the *transformed* embeddings with fixed probes and confirm they do **not** move. Colour and morphology votes have full coverage; stellar mass covers only 3,728 anchor-wide, so on the elongated subset it will be small — measure the number and report it with its n rather than dropping it.

Nulls: the untransformed baseline as the noise floor (2.027°), plus matched-norm random directions, which the old code never had.

**Step 3.** Figures via a separate `d2EquivarianceFigures.py` reading only the artifact. Report section 5. Automated number audit. `runLog.md` entry. Commit.

Expected cost: CPU only, minutes.

---

## 8. State of Diagnostics 3–9

From a verified audit this session. Percentages are the fraction already existing under the *old* code, not under `paper1/`.

- **D3 chirality — 0%.** Needs a major-axis flip operator (rotate axis horizontal, flip vertically, rotate back). The existing mirror is the wrong operator, orientation-dominated. **No handedness labels are held locally**: GZ DESI asks arm tightness and count, both mirror-symmetric; clockwise/anticlockwise belongs to GZ1/GZ2 and carries a documented S-wise reporting bias.
- **D4 nuisance leakage — ~40%.** Seeing 0.685, depth 0.441, extinction 0.303 exist for r band only. Dec has never been probed from the image substrate (the 0.988 figure in older records is a codec readback, a different substrate). Hemisphere and band-blankness probes do not exist. Both nulls missing.
- **D5 degradation — 0%.** All seven perturbations. The Vera realism code (`realismR1.py`, `injectR3.py`) is **not on this machine** — verified absent from the tree, all four tarballs and git history.
- **D6 decodability — ~65%.** Main table exists. Missing: leakage ablation for morphology votes, selection-vs-physics for anything but redshift, shuffled-label nulls, and the **pixel-PCA floor**, which exists nowhere.
- **D7 concept geometry — ~35%.** The statistic exists on 44 pairs. The calibrated generative null (7c), sanity anchors, per-pair p-values with FDR, concept arithmetic and the whitened robustness column are all new. This is the paper's declared methods contribution.
- **D8 structured relations — ~60%.** Tuning fork (8a–8c) essentially complete. 8d, 8e and the instance gate do not exist.
- **D9 artificial redshifting — 0%.** FERENGI is Barden, Jahnke & Häußler 2008 (arXiv:0812.1022), code at github.com/MegaMorph/ferengi. Five of six steps are simple; the k-correction is the hard one, and our 20,986 spectra over 17,643 anchor galaxies can supply real SEDs instead of templates.

---

## 9. Open decisions awaiting the user

1. Whether to push. Six commits sit local.
2. Whether `paperScopingV1.md` stays gitignored (currently yes, matching `paperSuggestions.md`).
3. Which model subset gets the expensive causal diagnostics (3, 5, 9).
4. Whether Pillar II splits into a companion paper.
5. Subsample sizes for D5 and D9.
6. Paper 2 remains blocked on the injection criterion and R3 gain.

---

## 10. First actions in the next session

1. Read `paperScopingV1.md` Diagnostic 2 in full, and `paper1/standard.md`.
2. `python -B paper1/runAll.py` and `python -B -m pytest paper1/tests -q` to confirm the state described here.
3. Implement `lib/transforms.py` `rotate` and `mirror`; write the real `tests/testTransforms.py`; get them passing.
4. Then, and only then, write `d2Equivariance.py` against the cached encodes.
5. Log every step to `runLog.md` as you go.
