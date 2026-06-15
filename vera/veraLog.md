# Vera Run Log — Phase 2 (ASTRID forward-modeling, AION-1 ID test)

## 2026-06-10 02:19 — Task 1: workspace + environment

- Baseline home usage: 1.3G / 5.0G (du -sh ~; df -h /hildafs home mount).
- Deleted scratch file ./test.py (was empty, 0 bytes).
- Created subdirs: code/ logs/ envs/ results/ data/
  - `mkdir -p code logs envs results data`

## 2026-06-10 02:20–02:30 — structure recon (login node, light reads only)

- SLURM partitions (sinfo): RM* (default, 30 nodes, 28 cpu, 128G+, 7d), TWIG-GPU, HENON-GPU,
  HENON, RITA-GPU, SIRIUS, MIKO.
- PIG_817_subfind top-level blocks: 0 (gas), 1 (DM), 4 (stars), 5 (BH), FOFGroups, SubGroups,
  NewgalSFR, Header. Header: BoxSize=250000 ckpc/h, h=0.6774, Time=1 (z=0),
  NumFOFGroupsTotal=157,097,137, NumPartInGroupTotal=[6.589e10 gas, 9.240e10 DM, 0, 0,
  6.901e10 stars, 1.203e7 BH], MassTable[4]=3.167e-5 (1e10 Msun/h).
- SubGroups has 39 columns incl. SubhaloIDMostbound, SubhaloMassType, SubhaloPos,
  SubhaloHalfmassRadType, SubhaloSFR + SubhaloSFR_old (+ SfrIn{Half,Max,}Rad and _old variants),
  SubhaloOffsetType/SubhaloLenType (slicing!), SubhaloVmax, SubhaloVelDisp, spins, inertia tensors.
- Separate NewgalSFR block: SubhaloSFR, SubhaloSfrInHalfRad/InMaxRad/InRad (new-SF-recipe SFRs).
- Star block (4/) fields: BirthDensity, Generation, GroupID, ID, LastEnrichmentMyr, Mass,
  Metallicity, Metals, NewIndex, Position, SmoothingLength, StarFormationTime, SubgroupID,
  TotalMassReturned, Velocity.
- Exact row counts (awk sum over plain-text BigFile headers, no data read):
  - subfind SubGroups: 44,366,700 subhalos. subfind 4/: 69,007,912,666 star rows (matches Header).
  - photometric/PIG_817_photometric: 4/ = 69,007,912,666 rows, SubGroups/ = 44,366,700 rows
    -> EXACTLY row-aligned with PIG_817_subfind. Bands: des{g,i,r,y,z}, lsst{u,g,r,i,z,y},
    sdss{u,g,r,i,z} = 16 columns, each in per-star (4/) and per-subhalo (SubGroups/) form, f4.
  - Fatemeh/LSST/PIG_724_photometric: lsst{u,g,r,i,z,y} only; 4/ = 59,728,055,907 rows,
    SubGroups/ = 46,359,683 rows; SubGroups dtype is f8 (vs f4 in PIG_817 product).
- attr-v2 on photometric blocks is EMPTY -> no embedded band/recipe metadata; newSFa/oriSFa
  markers appear only in sibling directory names (PIG_817_photometric_newSFa,
  ..._oriSFa_z01, ..._newSFa_z01, removemass variants) and as NewgalSFR/_old fields in subfind.
- NOTE: full per-star column = 69e9 rows (~276 GB / f4 col). Never read whole star columns;
  slice via SubGroups/SubhaloOffsetType + SubhaloLenType.
- Wrote env.sh (activates aion env; exports CONDA_PKGS_DIRS, PIP_CACHE_DIR, HF_HOME,
  MPLCONFIGDIR under workspace), code/peekSubfind.py, code/peekPhotometric.py.
- Miniforge installer downloading to envs/ (github release, slow link ~85 kB/s).

## 2026-06-10 02:25–02:32 — Task 1 cont.: bootstrap switched to micromamba

- GitHub-served downloads throttled (~85-110 kB/s): miniforge installer and micro.mamba.pm both
  slow. conda.anaconda.org (~0.8-43 MB/s) and PyPI (~2 MB/s) fast.
- Killed miniforge curl; fetched micromamba 2.8.1 binary directly from conda-forge channel
  (conda.anaconda.org/conda-forge/linux-64/micromamba-2.8.1-0.tar.bz2) -> envs/bin/micromamba.
  Task spec allows "miniforge (or micromamba)".
- Creating env: micromamba create -n aion -c conda-forge python=3.11 numpy scipy pandas astropy
  matplotlib pyarrow h5py, with MAMBA_ROOT_PREFIX=envs/mamba, CONDA_PKGS_DIRS=envs/condaPkgs.
- env.sh updated to micromamba shell-hook activation; exports MAMBA_ROOT_PREFIX, CONDA_PKGS_DIRS,
  PIP_CACHE_DIR, HF_HOME, MPLCONFIGDIR all under workspace.
- SLURM account assoc confirmed: account=phy200026p, no partition restriction (srun on RM ok).

## 2026-06-10 02:35–02:40 — Task 1 complete

- micromamba create -n aion finished. pip install bigfile unyt OK (bigfile 0.1.52, unyt 3.1.0,
  pulls cython/sympy/mpmath).
- Import check via `source env.sh`: python 3.11 at envs/mamba/envs/aion/bin/python;
  numpy 2.4.6, scipy 1.17.1, pandas 3.0.3, astropy 7.2.0, matplotlib 3.10.9, pyarrow 24.0.0,
  h5py 3.16.0, bigfile 0.1.52, unyt 3.1.0 -> ALL IMPORTS OK.
- Quota: home 1.3G before and after (unchanged). micromamba had written a 14M repodata-shard
  cache to ~/.cache/conda (XDG path) -> deleted; added XDG_CACHE_HOME=$WS/envs/xdgCache to
  env.sh. ~/.conda/environments.txt (4K tracking file) left in place. envs/ = 2.7G in workspace.

## 2026-06-10 02:40–02:50 — Task 2: peekSubfind.py (srun job 962613 on RM, exit 0)

- Ran: srun -p RM -A phy200026p -N1 -n1 -c8 --mem=32G python code/peekSubfind.py
  -> logs/peekSubfind.out (+ .err with srun queue msgs). Ran within minutes of submission.
- Total subhalos: 44,366,700.
- Mstar > 10^9.3 Msun (h=0.6774 applied, code mass 1e10 Msun/h): 716,068.
- Mstar > 10^10 Msun: 255,435. Max Mstar = 9.74e12 Msun.
- Full field list with dtypes in logs/peekSubfind.out (FOFGroups 21 cols, SubGroups 39 cols,
  4/ 15 cols, NewgalSFR 4 cols). 4/Position is float64; most else float32.
- Example subhalo idx 15610655 (Mstar=1.50e10 Msun): SubhaloPos=[52091.9, 63128.8, 14992.5] ckpc/h,
  HalfmassRadType=[27.3 gas, 36.0 DM, -, -, 4.13 stars, -] ckpc/h, SubhaloSFR=1.85 (Msun/yr).
  Stars: offset=56,876,812,810, count=71,173. Slice via SubhaloOffsetType/LenType verified:
  star Mass sum = 1.500e10 Msun == SubhaloMassType[4]; star GroupID unique=[147496],
  SubgroupID unique=[413]. Star masses 1.75e5–8.43e5 Msun; StarFormationTime (scale factor a)
  0.071–1.000 (median 0.644); Metallicity (mass frac) 8.4e-8–5.4e-2 (median 1.1e-2).
- CAVEAT: SubhaloGroupNr=79 does NOT match star GroupID=147496 -> SubhaloGroupNr appears
  chunk-local, not a global FOF index; star-block SubgroupID (413) is rank-in-group, not global
  subhalo index. Use SubhaloOffsetType/SubhaloLenType for membership, not these IDs.

## 2026-06-10 02:42 — Task 3: peekPhotometric.py (login node, structure + 5-elem samples)

- logs/peekPhotometric.out. Both products: top-level blocks 4/ (per-star) + SubGroups/
  (per-subhalo); one column per band; all block attrs EMPTY (no embedded metadata).
- photometric/PIG_817_photometric: 16 bands (des g,i,r,y,z; lsst u,g,r,i,z,y; sdss u,g,r,i,z),
  all f4. Rows: 4/=69,007,912,666, SubGroups/=44,366,700 -> exactly row-aligned with
  PIG_817_subfind (same offsets/slicing apply). Samples NEGATIVE (per-star ~ -6 to -7.5,
  per-subhalo -21.8 to -26.6) -> ABSOLUTE (rest-frame) magnitudes.
- Fatemeh/LSST/PIG_724_photometric: lsst u,g,r,i,z,y only. 4/=59,728,055,907 rows f4;
  SubGroups/=46,359,683 rows f8 (float64, dtype differs from 817 product). Samples POSITIVE
  (per-star ~33-36, per-subhalo ~15-19) -> APPARENT magnitudes. Different snapshot (z>0),
  counts self-consistent but not aligned to PIG_817.
- Sibling dirs note: photometric/ also has PIG_817_photometric_newSFa, _newSFa_z01, _oriSFa_z01,
  newSFa_removemass_z0{1,2,3}, restframe/, fsps_grids/; Fatemeh/LSST also has
  PIG_724_photometric_with_issue (avoid) and Aperture/.

## Problems & unexpected findings (consolidated, 2026-06-10)

1. MAGNITUDE CONVENTION MISMATCH between the two photometric products:
   photometric/PIG_817_photometric values are NEGATIVE (per-star ~ -6 to -7.5, per-subhalo
   -21.8 to -26.6) => absolute / rest-frame magnitudes; Fatemeh/LSST/PIG_724_photometric values
   are POSITIVE (per-star ~ +33 to +36, per-subhalo ~ +15 to +19) => apparent magnitudes.
   Not interchangeable without distance modulus / K-correction. Confirm which convention the
   forward-modeling pipeline expects before using either.
2. No metadata inside the photometric files: every block attr (attr-v2) is empty — no band
   names, no newSFa/oriSFa recipe markers, no units. Recipe provenance exists ONLY in directory
   names (PIG_817_photometric vs _newSFa vs _oriSFa_z01 vs newSFa_removemass_z0{1,2,3}) and in
   subfind's NewgalSFR block / SubhaloSFR*_old fields. Which recipe the plain
   "PIG_817_photometric" dir corresponds to is NOT recorded in the data — must ask the group.
3. ID-field gotcha in PIG_817_subfind: example subhalo has SubhaloGroupNr=79 while its star
   particles have GroupID=147496; star SubgroupID=413 is rank-in-group, not a global subhalo
   index. SubhaloGroupNr looks chunk-local. Membership/cross-referencing must go through
   SubhaloOffsetType/SubhaloLenType (verified exact: sliced star Mass sum == SubhaloMassType[4]).
4. Scale hazard: star block (4/) has 69,007,912,666 rows => any full per-star column is
   ~276 GB (f4). Whole-column star reads are impossible/forbidden; always slice. Per-subhalo
   tables (44.4M rows) are fine (~0.2-1 GB per column).
5. dtype inconsistencies: PIG_724 per-subhalo magnitudes are float64 vs float32 in the PIG_817
   product; in subfind, 4/Position is float64 while most other fields are float32.
6. File-count quirk (harmless): subfind 4/ blocks are split over 258 physical files, the
   PIG_817 photometric 4/ blocks over 256 — identical total rows; absolute-offset slicing
   unaffected.
7. Fatemeh/LSST contains PIG_724_photometric_with_issue — explicitly flagged-bad product, avoid.
   Aperture/ subdir also present (unexplored).
8. PIG_724 catalog (46,359,683 subhalos, 59.7e9 stars) is a different snapshot, NOT row-aligned
   with PIG_817; only the PIG_817 photometric product is row-aligned with PIG_817_subfind.
9. Infrastructure: GitHub-served downloads throttled on Vera (~85-110 kB/s; miniforge installer,
   micro.mamba.pm), while conda.anaconda.org (up to ~43 MB/s) and PyPI (~2 MB/s) are fast =>
   used micromamba from the conda-forge channel instead of miniforge (spec allowed either).
   Any future GitHub fetches (model weights, repos) may be similarly slow.
10. Home-quota leak averted: micromamba wrote a 14 MB repodata cache to ~/.cache/conda despite
    CONDA_PKGS_DIRS — XDG path, now redirected via XDG_CACHE_HOME in env.sh; stray cache
    deleted; home back to baseline 1.3G. ~/.conda/environments.txt (4K) remains (harmless).
11. pip side-effect: installing bigfile/unyt also pulled cython, sympy, mpmath (build/runtime
    deps; pip downgraded mpmath 1.4.1 -> 1.3.0 for sympy compatibility). No other extras.
12. Context for selection: 716,068 subhalos above Mstar=10^9.3 Msun and 255,435 above 10^10
    (max 9.74e12) — ample for the planned ~50k-galaxy sample.
13. Sibling data inventory (unexplored, may be useful later): PIG2/ holds 33 subfind catalogs
    for snapshots 023-817 — including PIG_724_subfind, the matching catalog for Fatemeh's
    PIG_724 photometry — plus ~40 FOF-only PIG_* dirs, one PIG_047_subfind_old, and BadGasFlag/
    and NewSFR/ dirs; photometric/ holds products for snapshots 544-817 plus restframe/ and
    fsps_grids/ (FSPS SED grids?); Fatemeh/LSST holds LSST products for snapshots 348-724.

## Status: Tasks 1-3 complete. Stopped per instructions.

## 2026-06-10 03:05–03:25 — Task 4: censusPhotometric.py (login node, header+1024-sample reads)

- Wrote code/censusPhotometric.py; ran on login (light). Outputs: results/photometricCensus.json,
  logs/censusPhotometric.out (+ manual addenda for dust/ and the 724/743 check).
- Census coverage: photometric/ (9 main products 544-817, 4 newSFa/oriSFa variants, 3 removemass
  dirs, 34 restframe dirs), Fatemeh/LSST/ (18 main products 348-817, 20 Aperture products,
  dust/), vs 35 PIG2 + 14 PIG3 subfind catalogs (reference counts in JSON).
- FOCUS ANSWERS: PIG_660/692/743 photometric/ products HAVE all 16 bands (des+lsst+sdss) but
  SubGroups-only (no per-star 4/ block); PIG_771 has 16 bands WITH per-star 4/. All four are
  APPARENT-frame. In photometric/, only PIG_544, PIG_771, PIG_817 retain per-star 4/ blocks.
- CONVENTION MAP (revises earlier item 1): ALL main products at z>0 (photometric/ 544-771 and
  Fatemeh/LSST 348-771) are APPARENT mags; only the two snapshot-817 (z=0) products are
  ABSOLUTE. restframe/ products are ABSOLUTE everywhere (GALEX fuv+nuv per-subhalo only;
  per-star 4/ only for 214/348/505/523/544; restframe per-star sign-classification unreliable
  since old-SSP UV absolute mags can be positive). Aperture (2x stellar-half-mass-radius LSST,
  SubGroups-only, snapshots 348-771, no 817) is APPARENT.
- PROBLEM — PIG_817_photometric_newSFa has 44,366,711 SubGroups rows = 11 MORE than
  PIG_817_subfind (44,366,700); aligns with NO subfind catalog; sdss bands only (5), absolute,
  per-star 4/ row count matches 817 exactly. The 11 extra rows are unexplained -> ask group;
  do not index-align this product blindly.
- PIG_817_photometric_newSFa_z01 and _oriSFa_z01: sdss-only, APPARENT, exactly 44,366,700 rows
  -> snapshot 817 artificially placed at z=0.1; these two are the matched pair for the
  newSFa-vs-oriSFa photometric comparison.
- PIG_817_newSFa_removemass_z0{1,2,3}: EMPTY directories (no bigfile blocks). restframe/
  PIG_23/31/47 also empty.
- MISLABELED PRODUCT — Fatemeh/LSST/PIG_724_photometric contains the snapshot-743 catalog
  (46,359,683 subhalos / 59,728,055,907 stars = PIG_743_subfind counts, NOT PIG_724_subfind's
  46,940,435 / 55,329,029,829), with data values different from PIG_743_photometric (checked:
  cmp differs) -> consistent with snapshot-743 galaxies re-projected to z~0.31 because real
  snapshot 724 is flagged bad ("_with_issue" twin, dust z0.3_issue). The REAL snapshot-724
  catalog appears only in Aperture/2_Stellar_half_mass/PIG_724_photometric (46,940,435 rows)
  and dust/4/z0.3_issue. Never align "PIG_724_photometric" to PIG_724_subfind.
- NEW (undocumented in task): Fatemeh/LSST/dust/ = dust-attenuated (kappa_2_9) per-star LSST
  apparent mags on a 21-bin observed-redshift grid z0.0-z2.0 (snapshot mapping logged in
  censusPhotometric.out addendum; z0.2 AND z0.3 both use snapshot 743). Per-subhalo dust mags
  exist ONLY for z0.5; dust/allbands_kappa2.9/z0.3 is empty. For Legacy-Survey forward modeling
  the dust per-star mags may be the most relevant product present.
- Subfind reference notes: PIG3 duplicates PIG2 counts where both exist; PIG3 *_subfind_reassign
  (609-692, +658 which has no star block) have identical nsub to PIG2 -> "reassign" changes
  membership, not counts. PIG2/PIG_047_subfind_old (44,080,448) != PIG_047_subfind (38,844,175).
- dtypes: photometric/ 544-743 SubGroups are f4; Fatemeh SubGroups f8 (348-817); per-star always
  f4 (full dtype map in JSON).

## 2026-06-10 03:30–03:40 — Task 5: checkLumFn.py (srun on RM, exit 0)

- Wrote code/checkLumFn.py; ran via srun -p RM -c8 --mem=48G. Outputs: results/lumFn817.json,
  results/lumFn817.png, logs/checkLumFn.out.
- Inputs: photometric/PIG_817_photometric/SubGroups/{sdss_g,des_g} + subfind SubhaloMassType.
  Volume (250/0.6774)^3 = 5.027e7 Mpc^3, bins 0.25 mag over [-24,-14], cuts: all / Mstar>1e8.
- QC: 28,160,471 of 44,366,700 subhalos have non-finite (NaN) mags in BOTH bands ==
  EXACTLY the number with Mstar==0 (44,366,700-16,206,229) -> NaN = no stars, clean sentinel.
  No zero or positive mags; min mag -26.62 (sdss_g) / -26.65 (des_g). Mstar>1e8: 2,541,960.
- LF shape Schechter-like, monotonic over the bin range: phi(-23.9)=1.02e-5 ->
  phi(-14.1)=1.30e-2 [Mpc^-3 mag^-1]. sdss_g vs des_g nearly identical (des slightly brighter,
  counts ~1-5% higher per bin). Mstar>1e8 cut only matters faintward of Mg ~ -16.5.
  No fitting/interpretation done (verification against published calibration happens locally).

## 2026-06-10 03:35–03:50 — Task 6: sfrConventionDiff.py (srun on RM, exit 0; re-run with
   direction stat added)

- Wrote code/sfrConventionDiff.py; ran via srun. Outputs: results/sfrConventionDiff.json,
  logs/sfrConventionDiff.out.
- IDENTITY: SubGroups/SubhaloSFR == NewgalSFR/SubhaloSFR for ALL 44,366,700 subhalos (max abs
  diff 0.0) -> the main SubhaloSFR column IS the new convention; NewgalSFR block is a redundant
  copy; SubhaloSFR_old is the original recipe. No negative SFRs anywhere.
- THE CONVENTIONS DIFFER STRONGLY in the science mass range (new=NewgalSFR, old=_old;
  relDiff=(max-min)/max; first run numbers):
  - 10^9-10^9.5 Msun: 87.3% differ >5%, 84.1% differ >2x, median relDiff 0.69, p90 0.79.
  - 10^10-10^10.5: 94.6% differ >5%, 92.7% >2x, median 0.74.
  - Trend monotonic 10^8 -> 10^12; even at 10^8-10^8.5: 72% differ >5%, 48% >2x.
  - Overall across all 44.4M (dominated by no-star/zero-SFR subhalos, both-zero counted equal):
    only 10.3% differ >5%, median 0.
- Quiescent fraction (sSFR<1e-11/yr) moves with convention in EVERY bin, new > old:
  e.g. 9.0-9.5 dex: 0.166 vs 0.154; 11.0-11.5: 0.255 vs 0.226; 11.5-12.0: 0.593 vs 0.315
  (nearly 2x at the high-mass end!).
- VERDICT for our science: newSFa-vs-oriSFa is NOT negligible — typical SFRs differ by ~3x for
  the majority of 10^9-10^11 Msun galaxies and the quiescent classification shifts, drastically
  above 10^11.5. The recipe choice must be pinned project-wide (and matched to whichever recipe
  the photometric products assume).
- DIRECTION (re-run with fracNewLowerOfDiffering added): among subhalos whose two SFRs differ
  at all, the NEW value is the LOWER one in 89-99% of cases in every mass bin (0.89 at
  10^8-10^8.5, 0.97 at 10^9-10^11, 0.99 at 10^11.5-12) -> newSFa systematically suppresses
  SFR (typically ~3x) relative to oriSFa; that is why quiescent fractions rise under newSFa.
  Re-run reproduced all first-run numbers exactly.

## 2026-06-10 03:55 — sync package

- Packed phase2sync.tar.gz (81 KB) at workspace root: veraLog.md + code/ (5 scripts) + logs/
  (8 files) + results/ (photometricCensus.json, lumFn817.json, lumFn817.png,
  sfrConventionDiff.json). Home quota still 1.3G.

## Status: Tasks 4-6 complete. Stopped per instructions.

## 2026-06-10 04:10 — Tasks 7-10 received (local verified 4-6; photometric/ = intrinsic,
   dust/ = kappa2.9-attenuated tier)

- Plan: 4 independent scripts (bandOffsets, recipePhotomDiff, dustAttenCheck, extractCatalogs),
  all heavy reads via srun on RM, run in parallel. Random subsamples = 200 seeded chunks x 10k
  contiguous stars (seed 42), identical row indices across products where compared.

## 2026-06-10 04:20–04:35 — Task 9: dustAttenCheck.py (srun job 962619, exit 0)

- Row counts identical across all three products (dust z0.1 / Fatemeh 771 intrinsic / subfind
  771 stars: 64,219,305,883). 2M-star seeded chunk sample, identical indices.
- A_g = attenuated - intrinsic: min 0.0 EXACTLY, fracNegativeAny = 0.0 -> ROW-ALIGNED confirmed;
  no negative tail at any tolerance.
- A_g distribution: median 0.038, p25 0.003, p75 0.238, p95 1.080, p99 2.925, max 6.56 mag;
  32.7% of stars effectively unattenuated (A_g<0.01).
- A_g rises monotonically with stellar metallicity (median 0.009 in lowest decile -> 0.105 in
  decile 9; slight turnover in decile 10). results/dustAttenCheck.json.

## 2026-06-10 04:20–04:40 — Task 8: recipePhotomDiff.py (srun, exit 0)

- newSFa_z01 vs oriSFa_z01 (snapshot 817 @ z=0.1, SDSS SubGroups, both 44,366,700 rows),
  Mstar>1e9: 1,013,097 subhalos (1 dropped non-finite).
- Delta(newSFa-oriSFa) per band: u median +0.071 (std 0.075; 35% of galaxies |d|>0.1!);
  g median +0.008 (48% |d|>0.02); r -0.002; i -0.007; z -0.013 (20-27% |d|>0.02 each).
- Mass dependence with SIGN FLIP in g: +0.016 at 10^9-10^10, ~0 at 10^10-10^10.5,
  -0.022..-0.036 above 10^10.5.
- g-r color shift: median +0.015, 38% of galaxies |d(g-r)|>0.02.
- VERDICT: recipe choice moves photometry well above the 0.02 mag threshold for a large
  fraction of the sample (worst in u/g and in colors; mass-dependent sign) -> NOT negligible;
  recipe must be pinned and matched between catalog SFRs and photometry.
  results/recipePhotomDiff.json.

## 2026-06-10 04:20 — Task 7 DATA GAP + adaptation (correction to Task 4 summary)

- bandOffsets.py v1 failed: photometric/PIG_771_photometric/4/ contains ONLY sdss_{u,g,r,i,z}
  per-star — NO des/lsst per-star columns. Census table's "bands" column reflected SubGroups;
  per-group truth (in photometricCensus.json): per-star des+lsst exist ONLY for PIG_544 and
  PIG_817. CORRECTION to earlier statement "771 has 16 bands WITH per-star 4/": 771's per-star
  block is SDSS-only.
- Consequence: per-star DES photometry does NOT exist at snapshots 660/692/743/771 anywhere.
- Adapted bandOffsets.py: per-star offsets at 817 (absolute, z=0) and 544 (apparent, z~1)
  bracket our z range; per-subhalo offsets (all 16 bands available) at 771 and 817.

## 2026-06-10 04:40–04:55 — Task 7: bandOffsets.py adapted re-run (srun, exit 0)

- Per-star 2M-star seeded samples at 817 (absolute, z=0) and 544 (apparent, z~1);
  per-subhalo full finite sets at 771 (16.9M) and 817 (16.2M). results/bandOffsets.json.
- des_X - lsst_X at 817 per-star: medians g -0.011, r -0.036, i -0.053, z -0.071;
  scatter sigma 0.003-0.025. After 10-bin color term (lsst g-z): residual std
  g 0.0006, r 0.0018, i 0.0038, z 0.0135 mag.
- At 544 (z~1): offsets larger (r -0.120, i -0.204, z -0.162) and strongly color-dependent,
  but residual std after color term still small: g 0.0025, r 0.0067, i 0.0074, z 0.0188.
- Per-subhalo 771 (the frame relevant for snaps 660-771): medians g -0.009, r -0.030,
  i -0.028, z -0.021; residual std after color term 0.001-0.008 mag.
- VERDICT: DES mags ARE derivable from LSST via a color term to <~0.01-0.02 mag residual in
  griz; offsets are redshift-dependent, so the term must be calibrated per snapshot — doable
  from the per-subhalo SubGroups data which exist with all 16 bands at every target snapshot.
  No need to run FSPS ourselves for snaps 743/692/660 unless <0.01 mag absolute accuracy in z
  is required (z band is the weakest: resid 0.008-0.019).
- Caveat: calibration is intrinsic-vs-intrinsic; for the dust tier, recalibrate the color term
  on attenuated colors (same procedure, color bins from attenuated lsst g-z).

## 2026-06-10 04:40–05:00 — Task 10: extractCatalogs.py (srun, exit 0)

- All 5 snapshots read from PIG2 subfind (no PIG3 fallback needed), chunked 8M-row reads,
  Mstar>1e9 filter, photometric SubGroups row-alignment asserted per snapshot.
- data/catalogs/: catalog817.parquet 1,013,097 rows (667,208 centrals) 126.0 MB;
  catalog771 1,002,773 (666,894) 124.9 MB; catalog743 983,697 (659,921) 122.9 MB;
  catalog692 939,605 (639,049) 118.1 MB; catalog660 913,248 (624,910) 115.2 MB.
  TOTAL 607.1 MB (<2 GB -> included in phase2sync.tar.gz).
- 29 columns: index, mStar/mGas/mDM [Msun], posX/Y/Z [ckpc/h], velX/Y/Z, idMostbound,
  rHalfStar, sfrNew/sfrOld [Msun/yr], vmax, velDisp, nStar, starOffset, rankInGr, isCentral,
  desG/R/I/Z, sdssG, lsstG/R/I/Z (mags as stored: absolute 817, apparent intrinsic z>0).
  Units embedded in parquet schema metadata. Sanity: catalog817 row count == Task 8 Mstar>1e9
  count (1,013,097); row 0 = known most-massive subhalo (9.74e12 Msun).
- isCentral = SubhaloRankInGr==0 (66% centrals at Mstar>1e9).

## 2026-06-10 05:00 — sync package rebuilt

- phase2sync.tar.gz rebuilt at workspace root: 510 MB, 42 entries (veraLog.md, code/ 9 scripts,
  logs/ 16 files, results/ 8 files, data/catalogs/ 5 parquet). Home quota still 1.3G.

## Status: Tasks 7-10 complete. Stopped per instructions.

## 2026-06-14 — Phase2sync2 brief received (temporal grid + exact aperture masses + per-galaxy physics)

- Re-oriented on existing workspace (Tasks 1-10 complete): 5 catalogs in data/catalogs/
  carry index, starOffset, nStar, rHalfStar, posX/Y/Z, idMostbound, mStar -> Task B can JOIN
  on `index` and slice stars without re-reading SubGroups offset arrays.
- Confirmed via Snapshots.txt (measured, file lines = snap+1):
  snap 660 a=0.668703, 692 a=0.714286, 743 a=0.833333, 771 a=0.909091, 817 a=1.
  -> z (interpreted, 1/a-1): 660 z=0.49551, 692 z=0.40000, 743 z=0.20000, 771 z=0.10000, 817 z=0.
- env.sh aion: astropy 7.2.0, numpy 2.4.6, bigfile 0.1.52 import OK.

### Task A: snapshotGrid.py (login node, header-only reads = light)

- Ran `python code/snapshotGrid.py` on login node (header attrs are metadata only). exit 0.
  logs/snapshotGrid.out, results/snapshotGrid.json.
- COSMOLOGY (measured, from snap-817 Header): FlatLambdaCDM, H0=67.74 km/s/Mpc (h=0.6774),
  Om0=0.3089, OmegaLambda=0.6911 (flatness residual exactly 0), OmegaBaryon=0.0486,
  CMBTemperature=2.7255. Identical Omega0/OmegaLambda/HubbleParam across all 5 headers.
- Temporal grid (a,z measured; age/lookback/D_A/kpc-per-arcsec interpreted via astropy):
  snap 817 a=1.000000 z=0.00000 age=13.8027 Gyr lookback=0.0000 D_A=0      kpc/asec=0      (z=0 degenerate)
  snap 771 a=0.909091 z=0.10000 age=12.4589 lookback=1.3438 D_A=392.84 Mpc kpc/asec=1.90456
  snap 743 a=0.833333 z=0.20000 age=11.2928 lookback=2.5099 D_A=702.39     kpc/asec=3.40527
  snap 692 a=0.714286 z=0.40000 age= 9.3885 lookback=4.4142 D_A=1142.34    kpc/asec=5.53821
  snap 660 a=0.668106 z=0.49677 age= 8.6325 lookback=5.1703 D_A=1292.64    kpc/asec=6.26689
- Adjacent cosmic-time gaps (measured-derived; feed dedup v_pec*dt bound):
  817->771 dt=1.3438 Gyr; 771->743 dt=1.1661; 743->692 dt=1.9043; 692->660 dt=0.7560.
- SANITY PASS: 817 -> a=1.0, z=0 exactly. age(z=0)=13.80 Gyr consistent with Planck-ish cosmo.
- FLAG (measured): snap 660 Header Time=0.6681056 DISAGREES with Snapshots.txt[660]=0.668703
  (|da|=5.97e-4, dz~1.3e-3); other 4 agree to <1e-6. Header Time is the TRUE scale factor of
  the written snapshot (MP-Gadget writes at the sync point >= requested grid time); Snapshots.txt
  is the *requested* grid. catalog660.parquet was built from THIS file, so its galaxies live at
  a=0.668106 (z=0.49677). -> Table uses header Time (authoritative). Effect on D_A/scale ~0.1-0.2%.
  Recorded headerVsTxtAgree/AbsDiff per snap in JSON.

### Task B PRE-FLIGHT (read-only, login node; from LOCAL catalogs only -- no dataset read)
- Computed star-span vs useful-star counts to size the heavy job BEFORE any srun:
  per masked subhalo we must read ALL its stars (no-shortcut aperture mass), Position f8x3 +
  Mass/StarFormationTime/Metallicity f4 = 36 B/star.
- RESULT (measured): masked subhalos (Mstar>1e9) contain ~95% of ALL stars in each snapshot.
  useful stars per snap: 817 65.67e9, 771 60.84e9, 743 56.30e9, 692 47.64e9, 660 43.93e9.
  TOTAL useful = 274.4e9 stars ~= 9.9 TB of reads across the 5 snapshots.
  Contiguous-span (one block/snap) = 289.3e9 stars -> contiguous-read efficiency 94.8%
  (only ~5% wasted on intervening unmasked subhalos) -> the brief's chunk-contiguous strategy
  is near-optimal; NO gap-skipping needed.
- Largest single subhalo: 48.6M stars (817) = 1.75 GB by itself -> chunk by a STAR-COUNT budget
  (e.g. 300-500M stars/chunk ~= 11-18 GB) not by subhalo count; split per subhalo in memory.
- IMPLICATION: this is a ~10 TB read / 274e9-star compute job (output is still only tens of MB).
  Far heavier than "tens of MB" suggests. Age per star needs an interpolation table
  a_form -> t(a_form) [Gyr] built once per snapshot (cannot call astropy 2.7e11 times).
  -> reported to brain; awaiting go-ahead before launching the ~1000-subhalo/snap PILOT.

### Task B PILOT submitted (srun on RM, background)
- code/apertureMass.py written: shared computeGalaxy() (periodic min-image, aperture 2x/1x rHalf,
  mass-wt age via per-snap a_form->t interp table 5000 pts, massWtMetalAper, sfr100/sfr10 from
  current star mass within 2x rHalf, naive-vs-minimage radii for wrap check). Modes pilot|full;
  full() raises (gated on STOP-2). Cosmology pinned from Task A.
- Pilot: ~1000 subhalos/snap STRATIFIED in log M* (declining allocation 140/75/25/12/6 per
  0.25-dex bin from 1e9 floor up), all 5 snaps, fresh subfind metadata (off/len/pos/rHalf) read
  per sampled idx -> join-validated vs catalog; + one contiguous 100M-star probe at 817 for the
  true full-run inner-loop rate. Writes results/apertureQcPilot.json.
- Submitted: srun -p RM -A phy200026p -N1 -n1 -c8 --mem=48G --time=2:00:00 python
  code/apertureMass.py pilot  (NOT on login node). Awaiting completion.

### Task B PILOT v1 (job 973998, exit 0) -- BUG FOUND, results INVALID, re-running
- BUG (measured): used catalog ROW POSITION as the subfind subhalo index. Catalog holds only
  Mstar>1e9 subhalos (~1.01M of 44.4M; `index` col scattered to ~44M), so row `i` != subhalo
  index idx[i]. -> sliced wrong subhalos. Symptoms: JOIN starOffset mismatch 955/956,
  max|dPos|~2.45e5 (~box), SLICE relDev up to 6.2, nStarNonPos=563 (sampled massive-group
  satellites). Gate still ~0.48 ONLY because each wrong subhalo was self-consistent (own
  center+rHalf+stars) -> the aperture MATH is fine, the SAMPLE was wrong.
- VALID from v1 (bug-free path): contiguous probe used catalog starOffset/pos/rHalf -> 0.34 GB/s
  read, compute 4.98e6 st/s, COMBINED 3.27e6 st/s. Full-run extrapolation (serial read+compute):
  817 5.83h, 771 5.42h, 743 5.04h, 692 4.31h, 660 3.99h; TOTAL 24.58h, no-817 18.75h. This
  estimate is independent of the bug (probe path was correct).
- FIX: read fresh subfind at gi=idx[row]; added idMostbound identity check (bulletproof join);
  replaced flawed wrap heuristic with deterministic wrapTest() scanning catalog for centers
  within 1xrHalf of a box face. Re-running pilot.

### Task B PILOT v2 CORRECTED (job 974000, exit 0) -- ALL GATES PASS
- Index fix confirmed. JOIN bulletproof: offMism=0, lenMism=0, idMostboundMism=0, max|dPos|=0,
  max|dRHalf|=0 across all 5 snaps (~4767 sampled subhalos total). (measured)
- HALF-MASS GATE mStarAper1/mStarTotalStars: median=0.5000 (IQR within [0.4999,0.5001]) for ALL
  5 snaps; holds in EVERY nStar bin incl. lowest [2000,5000): 817 .4999, 771 .5002, 743 .5000,
  692 .4996, 660 .4998. True pool star floor min nStar 2440-2845 (NOT 300). PASS. (measured)
- SLICE: max relDev(totalStars vs catalog mStar) = 1.24-1.32e-7 (f4 precision), all snaps. (measured)
- aper2/total DIAG: median 817 .910, 771 .911, 743 .910, 692 .903, 660 .895; IQR ~[.83,.94].
  Mild trend (higher z -> slightly smaller fraction). (measured)
- nStarAper: min ~3600-4300, median ~26k-32k, max ~20-32M. PATHOLOGIES rHalfNonPos=0,
  emptyAperture=0, tinyNstar=0 for ALL snaps (v1's 243 tinyNstar was the bug). (measured)
- WRAP TEST (deterministic, centers within 1xrHalf of a face): real straddlers found, e.g.
  817 idx=2800087 center=(186653,186757,249999) rHalf=4.2: naiveMaxR=249999 vs minImageMaxR=47.4,
  18613 stars wrapped, mAper2 naive=5.30e9 vs minImage=8.37e9 (naive 37% LOW). Many more at
  817 & 660. Proves periodic min-image is necessary AND correct. (measured)
- THROUGHPUT: per-subhalo-seek 1.26-1.56M st/s; contiguous probe (full-run path) read 0.37 GB/s
  (10.4M st/s), per-subhalo-loop compute 7.8M st/s, combined 4.46M st/s. Pilot serial-process
  extrapolation: no-817 13.7h. (measured)
- Full-run code (full(), apFull.sbatch) prepared: vectorized bincount segmented reductions +
  1-thread prefetch -> read-bound; 4-snap array skipping 817; per-chunk checkpoint. Verifying
  processChunk == scalar computeGalaxy before any heavy launch.

### Task B HEAVY-RUN code VERIFIED & sized (still gated; NOT launched)
- VERIFY mode (job, 817, 100M-star block): vectorized processChunk == scalar computeGalaxy to
  relDiff ~1e-12 (nStarAper exact) -> bincount path numerically correct. BUT vectorized compute
  3.5M st/s vs scalar 7.8M st/s vs read 10.3M st/s -> vectorized is COMPUTE-BOUND (full-array
  fancy-index gathers are memory-bandwidth bound). DECISION (measured): heavy run uses the
  scalar computeGalaxy loop, parallelized across 8 cores. (measured)
- full() rewritten: multiprocessing.Pool(8), each worker reads one contiguous 50M-star chunk
  (CHUNK_STARS=50M -> ~1.8GB/worker, x8 ~15GB, fits --mem=64G) + scalar loop; per-chunk parquet
  checkpoint (resumable); concat+sort-by-index at end; pathology rHalf<=0 -> NaN+flag (never 0).
- SMOKE test (job 974002; snap 660, first 16 chunks, 8 workers): 735M stars in 34s =
  21.5M st/s AGGREGATE (2.1x single-stream read 10.3M st/s -> READ-BOUND, good). Correctness
  max relDev(totalStars vs catalog mStar)=1.42e-7. (measured)
- WALL-CLOCK (measured 21.5M st/s, span/rate): 771 0.82h, 743 0.77h, 692 0.65h, 660 0.61h.
  4-task sbatch array (one node x8 workers each) -> WALL ~0.82h (~49 min, snap 771).
  Conservative w/ cross-task Lustre contention ~1-1.5h. << 24h. (interpreted from measured rate)
- DELIVERABLES READY (gated): code/apFull.sbatch (array 0-3 -> 771/743/692/660, 817 skipped,
  -c8 --mem=64G -t6h); full() writes data/apertureExtra{s}.parquet (12 cols per contract) +
  results/apertureQc_{s}.json; `python code/apertureMass.py mergeqc` -> results/apertureQc.json.
- Smoke checkpoints removed. STOP-2: awaiting greenlight to `sbatch code/apFull.sbatch`.

### Task B HEAVY RUN LAUNCHED -- GREENLIT (job array 974003_[0-3], -t 2:00:00)
- sbatch code/apFull.sbatch : array 0-3 -> snapshots 771/743/692/660 (817 skipped), each
  -N1 -c8 --mem=64G. Submitted; awaiting completion. Will report row counts vs catalog,
  wall-clock, peak mem, per-galaxy sanity, then mergeqc.

### Task B HEAVY RUN COMPLETE (array 974003) + STOP-3 verification + 771 anomaly RESOLVED
- WALL-CLOCK (sacct, measured): 771 1:02:39, 743 0:57:29, 692 0:50:36, 660 0:46:47 ->
  array wall ~1h03m. PEAK RSS 18.9-19.7 GB/task (req 64G). All COMPLETED.
- ROW COUNTS == catalog exactly: 771 1,002,773; 743 983,697; 692 939,605; 660 913,248.
  index join exact (set match, no dupes) all snaps. (measured)
- 771-ONLY ANOMALY diagnosed (STOP-3):
  - relDev(mStarTotalStars vs catalog mStar): clean p99.9=1.3e-7 (f4 noise); a tail of 201 at
    >1e-3 (up to 0.292) + 5 at 1e-4..7.5e-4, ALL in index 9.880-9.888M = FOF groups 13377/13378
    (21.9% through the subhalo array). 743/692/660: max relDev 1.4e-7, ZERO anomalies. (measured)
  - Pathology was emptyAperture (155), NOT rHalf<=0: bad set has rHalf 1.25-7.54 (all>0),
    nStar 4.7k-267k, mStar up to 5.4e10 -> substantial galaxies with ZERO stars in 2xrHalf.
  - DECISIVE (subfind srun, diagBad771.py): catalog faithful (off/len/massType mism=0). PER-SEEK
    re-slice (pilot method, independent of chunking) REPRODUCES the mismatch: e.g. idx9880149
    starMassSum 1.81e9 vs SubhaloMassType[4] 2.05e9 (ratio 0.885). Stars coherent (single
    GroupID/SubgroupID) but member stars 296-590 ckpc/h from their own SubhaloPos. -> SubhaloPos
    & SubhaloMassType are internally INCONSISTENT with star membership for this localized region.
  - VERDICT: DATA quirk in PIG_771_subfind, NOT a chunk/code bug (per-seek == chunked path; other
    snaps clean). Per pre-stated rule -> flag+exclude, NO RERUN. (measured + decided)
  - ACTION (flagBad771.py, reversible): pathology='massMismatch' for the 208 relDev>1e-5 subhalos
    (clean gap: f4 noise <=1e-6, then nothing until 1e-4). apertureExtra771.parquet clean rows
    now 1,002,565 with max relDev 1.42e-7 (pure f4). results/bad771.json documents indices+groups.
    apertureQc_771.json regenerated excluding flagged.
- FINAL SANITY (clean rows, all 4 snaps): A1<=A2<=Total, 0<massWtAge<t_snap, sfr100/sfr10>=0,
  nStarAper<=nStar, NaN only where flagged -> ALL PASS. Medians sane; massWtAge falls with z
  (771 4.53 -> 660 2.88 Gyr). results/apertureQc.json merged (4 snaps). (measured)
- NOT PACKAGED yet (per STOP-3); _ckpt_* dirs kept. Awaiting greenlight.

### Task C: package phase2sync2.tar.gz (GREENLIT 2026-06-15)
- Built at workspace root with: data/apertureExtra{771,743,692,660}.parquet (12-col),
  results/apertureQc.json + apertureQc_{771,743,692,660}.json, results/bad771.json (208-excl
  provenance), results/snapshotGrid.json (Task A grid), code/apertureMass.py, code/snapshotGrid.py,
  code/apFull.sbatch, veraLog.md, logs/apFull_{0..3}.out. 19 files.
- _ckpt_* dirs RETAINED pending local join-verification all-clear (tar-rebuild safety).
