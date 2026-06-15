"""Task B: exact stellar aperture masses + per-galaxy temporal/physics summaries.

Two modes (argv[1]): 'pilot' (this run) and 'full' (gated, after STOP-2 approval).

PILOT: ~1000 subhalos/snapshot, STRATIFIED across log M* from the 1e9 floor to the max
(so near-floor galaxies are covered, not just giants). Validates the half-mass QC gate
(mStarAper1/mStarTotalStars ~= 0.5), slice correctness vs catalog mStar, periodic wrap,
and measures throughput for the heavy-run extrapolation. Writes results/apertureQcPilot.json.

FULL: chunk-contiguous over EVERY Mstar>1e9 subhalo, writes data/apertureExtra{s}.parquet.
Implemented but NOT launched in the pilot run.

Reuses extractCatalogs.openSubfind EXACTLY so subhalo index stays aligned to the local
catalogs. The catalog's `index` column IS the subfind subhalo index; catalog ROW POSITION is
NOT -- subfind reads must use idx[row], validated here against catalog starOffset/nStar/pos/
rHalf/idMostbound. Periodic minimum-image distance to SubhaloPos (box 250000 ckpc/h). Star age
via a per-snapshot a_form -> t(a_form) interpolation table (astropy ~5000x/snap, never per star).
"""
import json
import os
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import bigfile
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

WS = "/hildafs/projects/phy200026p/akumar45"
CAT = f"{WS}/data/catalogs/catalog{{}}.parquet"
H = 0.6774
MUNIT = 1e10 / H
BOX = 250000.0  # ckpc/h, periodic
SNAPS = ["817", "771", "743", "692", "660"]
SUBFIND = ["/hildafs/datasets/Asterix/PIG2/PIG_{s}_subfind",
           "/hildafs/datasets/Asterix/PIG3/PIG_{s}_subfind_reassign"]
COSMO = FlatLambdaCDM(H0=67.74 * u.km / u.s / u.Mpc, Om0=0.3089, Ob0=0.0486)
ASNAP = {"817": 1.0, "771": 0.9090909090909091, "743": 0.8333333333333334,
         "692": 0.7142857142857143, "660": 0.6681055622823606}
SPANSTARS = {"817": 68_612_624_770, "771": 63_820_798_766, "743": 59_299_421_444,
             "692": 50_687_474_150, "660": 46_929_382_247}
SEED = 42


def openSubfind(s):
    for tpl in SUBFIND:
        path = tpl.format(s=s)
        try:
            f = bigfile.File(path)
            f["SubGroups/SubhaloMass"]
            return f, path
        except Exception as e:
            print(f"  {path}: {e}")
    raise RuntimeError(f"no readable subfind for {s}")


def ageTable(aSnap, npts=5000):
    aGrid = np.linspace(0.005, aSnap, npts)
    tGrid = COSMO.age(1.0 / aGrid - 1.0).to(u.Gyr).value
    tSnap = COSMO.age(1.0 / aSnap - 1.0).to(u.Gyr).value
    return aGrid, tGrid, tSnap


def computeGalaxy(pos, massMsun, sft, met, subPos, rHalf, aGrid, tGrid, tSnap, wantWrapDiag=True):
    """All per-galaxy quantities for one star slice. pos f8 (N,3) ckpc/h."""
    d = pos - subPos
    if wantWrapDiag:
        naiveMaxR = float(np.sqrt(np.einsum("ij,ij->i", d, d).max()))
        rawSpanMax = float((pos.max(0) - pos.min(0)).max())
    else:
        naiveMaxR = rawSpanMax = np.nan
    d -= BOX * np.round(d / BOX)               # periodic minimum image (in place)
    r = np.sqrt(np.einsum("ij,ij->i", d, d))
    minImageMaxR = float(r.max())

    rA2, rA1 = 2.0 * rHalf, 1.0 * rHalf
    inA2 = r < rA2
    inA1 = r < rA1
    mTot = float(massMsun.sum())
    mA2 = float(massMsun[inA2].sum())
    mA1 = float(massMsun[inA1].sum())
    nAper = int(inA2.sum())

    tForm = np.interp(sft, aGrid, tGrid)
    ageStar = np.clip(tSnap - tForm, 0.0, None)
    mwAgeAll = float((massMsun * ageStar).sum() / mTot) if mTot > 0 else np.nan
    if mA2 > 0:
        mA2mass = massMsun[inA2]
        mwAgeAper = float((mA2mass * ageStar[inA2]).sum() / mA2)
        mwMetAper = float((mA2mass * met[inA2]).sum() / mA2)
        sfr100 = float(massMsun[inA2 & (ageStar < 0.1)].sum() / 1e8)   # Msun/yr, 100 Myr window
        sfr10 = float(massMsun[inA2 & (ageStar < 0.01)].sum() / 1e7)   # Msun/yr, 10 Myr window
    else:
        mwAgeAper = mwMetAper = sfr100 = sfr10 = np.nan
    return dict(mStarAper2=mA2, mStarAper1=mA1, mStarTotalStars=mTot,
                massWtAgeGyr=mwAgeAll, massWtAgeAperGyr=mwAgeAper, massWtMetalAper=mwMetAper,
                sfr100=sfr100, sfr10=sfr10, nStarAper=nAper,
                naiveMaxR=naiveMaxR, minImageMaxR=minImageMaxR, rawSpanMax=rawSpanMax)


def readStars(f, off, n):
    pos = np.asarray(f["4/Position"][off:off + n], dtype="f8")
    mass = np.asarray(f["4/Mass"][off:off + n], dtype="f8") * MUNIT
    sft = np.asarray(f["4/StarFormationTime"][off:off + n], dtype="f8")
    met = np.asarray(f["4/Metallicity"][off:off + n], dtype="f8")
    return pos, mass, sft, met


def loadCat(s):
    cat = pq.read_table(CAT.format(s),
                        columns=["index", "mStar", "posX", "posY", "posZ",
                                 "rHalfStar", "nStar", "starOffset", "idMostbound"])
    return dict(
        idx=cat["index"].to_numpy().astype("i8"),
        mStar=cat["mStar"].to_numpy().astype("f8"),
        pos=np.stack([cat["posX"].to_numpy(), cat["posY"].to_numpy(),
                      cat["posZ"].to_numpy()], axis=1).astype("f8"),
        rHalf=cat["rHalfStar"].to_numpy().astype("f8"),
        nStar=cat["nStar"].to_numpy().astype("i8"),
        off=cat["starOffset"].to_numpy().astype("i8"),
        idmb=cat["idMostbound"].to_numpy())


def stratifiedSample(logM, rng):
    def perBin(lo):
        if lo < 10.0:
            return 140
        if lo < 11.0:
            return 75
        if lo < 11.5:
            return 25
        if lo < 12.0:
            return 12
        return 6
    edges = np.arange(9.0, logM.max() + 0.25, 0.25)
    picks, report = [], []
    for lo in edges:
        inBin = np.flatnonzero((logM >= lo) & (logM < lo + 0.25))
        if len(inBin) == 0:
            continue
        k = min(perBin(lo), len(inBin))
        picks.append(rng.choice(inBin, size=k, replace=False))
        report.append((round(float(lo), 2), len(inBin), int(k)))
    return np.sort(np.concatenate(picks)), report


def pctl(a, q):
    a = np.asarray(a, dtype="f8")
    a = a[np.isfinite(a)]
    return float(np.percentile(a, q)) if a.size else float("nan")


def runPilotSnap(s, cat):
    print(f"\n{'='*72}\nPILOT snapshot {s}\n{'='*72}")
    idx, mStarCat = cat["idx"], cat["mStar"]
    posCat, rHalfCat, nStarCat, offCat, idmb = (cat["pos"], cat["rHalf"], cat["nStar"],
                                                cat["off"], cat["idmb"])
    logM = np.log10(mStarCat)
    rng = np.random.default_rng(SEED + int(s))
    sampleRows, perBinReport = stratifiedSample(logM, rng)
    print(f"  catalog rows {len(idx):,}; sampled {len(sampleRows)} subhalos "
          f"(logM {logM.min():.2f}..{logM.max():.2f})")
    print(f"  catalog nStar: min={int(nStarCat.min())} p1={int(np.percentile(nStarCat,1))} "
          f"median={int(np.median(nStarCat))} max={int(nStarCat.max()):,}")

    f, path = openSubfind(s)
    try:
        _ = f["4/Mass"].dtype
    except Exception:
        print(f"  FATAL: snapshot {s} ({path}) has NO 4/Mass star block")
        f.close()
        return None
    bOff, bLen = f["SubGroups/SubhaloOffsetType"], f["SubGroups/SubhaloLenType"]
    bPos, bRh = f["SubGroups/SubhaloPos"], f["SubGroups/SubhaloHalfmassRadType"]
    bIdmb = f["SubGroups/SubhaloIDMostbound"]
    aGrid, tGrid, tSnap = ageTable(ASNAP[s])
    print(f"  a_snap={ASNAP[s]:.6f} t_snap={tSnap:.4f} Gyr; age table {len(aGrid)} pts")

    recs = []
    joinOffMism = joinLenMism = joinIdMism = 0
    joinPosMax = joinRhMax = 0.0
    starsRead = 0
    straddlersInSample = []
    t0 = time.perf_counter()
    for row in sampleRows:
        row = int(row)
        gi = int(idx[row])                       # TRUE subfind subhalo index
        off = int(bOff[gi:gi + 1][0][4])
        n = int(bLen[gi:gi + 1][0][4])
        subPos = np.asarray(bPos[gi:gi + 1][0], dtype="f8")
        rHalf = float(bRh[gi:gi + 1][0][4])
        idmbF = bIdmb[gi:gi + 1][0]
        # join validation: fresh subfind (at idx[row]) vs catalog (at row)
        joinOffMism += (off != int(offCat[row]))
        joinLenMism += (n != int(nStarCat[row]))
        joinIdMism += (int(idmbF) != int(idmb[row]))
        joinPosMax = max(joinPosMax, float(np.abs(subPos - posCat[row]).max()))
        joinRhMax = max(joinRhMax, abs(rHalf - float(rHalfCat[row])))

        pos, mass, sft, met = readStars(f, off, n)
        starsRead += n
        g = computeGalaxy(pos, mass, sft, met, subPos, rHalf, aGrid, tGrid, tSnap)
        flag = ("rHalf<=0" if rHalf <= 0 else
                "emptyAperture" if g["nStarAper"] == 0 else
                "tinyNstar" if n < 300 else "")
        g.update(index=gi, idMostbound=int(idmb[row]), mStarCat=float(mStarCat[row]),
                 nStar=int(n), rHalfStar=rHalf, pathology=flag)
        if g["rawSpanMax"] > BOX / 2:             # true periodic straddler
            straddlersInSample.append(g)
        recs.append(g)
    dt = time.perf_counter() - t0
    f.close()

    print(f"  JOIN (fresh subfind@idx[row] vs catalog@row): offMism={joinOffMism} "
          f"lenMism={joinLenMism} idMostboundMism={joinIdMism} / {len(sampleRows)}; "
          f"max|dPos|={joinPosMax:.3e} max|dRHalf|={joinRhMax:.3e} ckpc/h")

    good = [r for r in recs if r["mStarTotalStars"] > 0 and r["rHalfStar"] > 0]
    relDev = np.array([abs(r["mStarTotalStars"] - r["mStarCat"]) / r["mStarCat"] for r in good])
    ratio1 = np.array([r["mStarAper1"] / r["mStarTotalStars"] for r in good])
    ratio2 = np.array([r["mStarAper2"] / r["mStarTotalStars"] for r in good])
    nStarArr = np.array([r["nStar"] for r in good])
    nAperArr = np.array([r["nStarAper"] for r in good])

    nbins = [0, 2000, 5000, 10000, 30000, 100000, 300000, 1_000_000, 10 ** 12]
    gateByN = []
    for a, b in zip(nbins[:-1], nbins[1:]):
        m = (nStarArr >= a) & (nStarArr < b)
        if m.sum() == 0:
            continue
        gateByN.append(dict(nStarLo=a, nStarHi=(b if b < 10 ** 12 else None), count=int(m.sum()),
                            medRatio1=float(np.median(ratio1[m])),
                            p25=float(np.percentile(ratio1[m], 25)),
                            p75=float(np.percentile(ratio1[m], 75))))

    nPath = dict(rHalfNonPos=sum(r["pathology"] == "rHalf<=0" for r in recs),
                 emptyAperture=sum(r["pathology"] == "emptyAperture" for r in recs),
                 tinyNstar=sum(r["pathology"] == "tinyNstar" for r in recs))

    print(f"  GATE mStarAper1/total: median={np.median(ratio1):.4f} "
          f"IQR=[{np.percentile(ratio1,25):.4f},{np.percentile(ratio1,75):.4f}]  (n={len(good)})")
    for gb in gateByN:
        hi = gb["nStarHi"] if gb["nStarHi"] else "inf"
        print(f"     nStar[{gb['nStarLo']:>7},{str(hi):>7}) n={gb['count']:>3} "
              f"med={gb['medRatio1']:.4f} IQR=[{gb['p25']:.4f},{gb['p75']:.4f}]")
    print(f"  DIAG mStarAper2/total: median={np.median(ratio2):.4f} "
          f"IQR=[{np.percentile(ratio2,25):.4f},{np.percentile(ratio2,75):.4f}]")
    print(f"  SLICE max relDev(totalStars vs catalog mStar)={relDev.max():.3e} "
          f"(median {np.median(relDev):.3e})")
    print(f"  nStarAper: min={nAperArr.min()} median={int(np.median(nAperArr))} "
          f"max={nAperArr.max():,}; pathologies={nPath}; sampleStraddlers={len(straddlersInSample)}")
    print(f"  THROUGHPUT (per-subhalo seeks): {starsRead:,} stars in {dt:.1f}s "
          f"= {starsRead/dt:,.0f} stars/s")

    return dict(
        snap=int(s), subfindPath=path, aSnap=ASNAP[s], tSnapGyr=tSnap,
        nSampled=len(sampleRows), nGood=len(good), perBinAllocation=perBinReport,
        catalogNstar=dict(min=int(nStarCat.min()), p1=int(np.percentile(nStarCat, 1)),
                          median=int(np.median(nStarCat)), max=int(nStarCat.max())),
        join=dict(offMismatch=int(joinOffMism), lenMismatch=int(joinLenMism),
                  idMostboundMismatch=int(joinIdMism), maxPosDiff=joinPosMax, maxRHalfDiff=joinRhMax),
        sliceMaxRelDev=float(relDev.max()), sliceMedRelDev=float(np.median(relDev)),
        gate=dict(medRatio1=float(np.median(ratio1)), p25=float(np.percentile(ratio1, 25)),
                  p75=float(np.percentile(ratio1, 75))),
        gateByNstar=gateByN,
        diagRatio2=dict(median=float(np.median(ratio2)), p25=float(np.percentile(ratio2, 25)),
                        p75=float(np.percentile(ratio2, 75))),
        nStarAper=dict(min=int(nAperArr.min()), median=float(np.median(nAperArr)),
                       max=int(nAperArr.max()), p10=pctl(nAperArr, 10), p90=pctl(nAperArr, 90)),
        pathologies=nPath, nSampleStraddlers=len(straddlersInSample),
        throughputPerSubhaloSeek=dict(starsRead=int(starsRead), seconds=dt,
                                      starsPerSec=starsRead / dt),
        galaxies=recs)


def wrapTest(s, cat, nWant=6):
    """Deterministic periodic-wrap check: scan catalog for galaxies whose CENTER is within
    1xrHalf of a box face (their outer stars must wrap), slice, and show min-image fixes it."""
    print(f"\n{'='*72}\nPERIODIC-WRAP TEST snapshot {s}\n{'='*72}")
    pos, rHalf, off, n, idx = cat["pos"], cat["rHalf"], cat["off"], cat["nStar"], cat["idx"]
    distEdge = np.minimum(pos, BOX - pos).min(axis=1)               # nearest face distance
    cand = np.flatnonzero((distEdge < rHalf) & (rHalf > 0) & (n > 2000) & (n < 300000))
    print(f"  near-face candidates (center within 1xrHalf of a face, 2k<nStar<300k): {len(cand)}")
    f, _ = openSubfind(s)
    aGrid, tGrid, tSnap = ageTable(ASNAP[s])
    out = []
    for row in cand[:nWant]:
        row = int(row)
        gi = int(idx[row])
        pp, mm, ss, me = readStars(f, int(off[row]), int(n[row]))
        subPos = pos[row]
        d0 = pp - subPos
        naiveR = np.sqrt(np.einsum("ij,ij->i", d0, d0))
        dmi = d0 - BOX * np.round(d0 / BOX)
        miR = np.sqrt(np.einsum("ij,ij->i", dmi, dmi))
        nWrapped = int((np.abs(np.round(d0 / BOX)).sum(axis=1) > 0).sum())
        rec = dict(index=gi, nStar=int(n[row]), rHalf=float(rHalf[row]),
                   center=[float(x) for x in subPos], distToFace=float(distEdge[row]),
                   naiveMaxR=float(naiveR.max()), minImageMaxR=float(miR.max()),
                   rawSpanMax=float((pp.max(0) - pp.min(0)).max()), nStarsWrapped=nWrapped,
                   mAper2_minImage=float(mm[miR < 2 * rHalf[row]].sum() * 1.0),
                   mAper2_naive=float(mm[naiveR < 2 * rHalf[row]].sum() * 1.0))
        out.append(rec)
        print(f"  idx={gi} nStar={rec['nStar']} center=({subPos[0]:.0f},{subPos[1]:.0f},"
              f"{subPos[2]:.0f}) distFace={rec['distToFace']:.1f} rHalf={rec['rHalf']:.1f} "
              f"| naiveMaxR={rec['naiveMaxR']:.0f} minImageMaxR={rec['minImageMaxR']:.1f} "
              f"wrapped={nWrapped} | mAper2 naive={rec['mAper2_naive']:.3e} "
              f"minImage={rec['mAper2_minImage']:.3e}")
    f.close()
    return out


def contiguousProbe(s, cat, targetStars=100_000_000):
    print(f"\n{'='*72}\nCONTIGUOUS PROBE snapshot {s} (target {targetStars:,} stars)\n{'='*72}")
    off, n, mStar, rHalf, posC = (cat["off"], cat["nStar"], cat["mStar"], cat["rHalf"], cat["pos"])
    order = np.argsort(off)
    off, n, rHalf, posC = off[order], n[order], rHalf[order], posC[order]
    jstart = int(np.searchsorted(off, off.min() + (off.max() - off.min()) // 2))
    j = jstart
    while j < len(off) - 1 and (off[j] + n[j] - off[jstart]) < targetStars:
        j += 1
    blockLo, blockHi = int(off[jstart]), int(off[j] + n[j])
    nBlock = blockHi - blockLo
    nUseful = int(n[jstart:j + 1].sum())
    aGrid, tGrid, tSnap = ageTable(ASNAP[s])
    f, _ = openSubfind(s)
    tr = time.perf_counter()
    pos, mass, sft, met = readStars(f, blockLo, nBlock)
    readT = time.perf_counter() - tr
    tc = time.perf_counter()
    for k in range(jstart, j + 1):
        lo = int(off[k] - blockLo)
        hi = lo + int(n[k])
        computeGalaxy(pos[lo:hi], mass[lo:hi], sft[lo:hi], met[lo:hi],
                      posC[k], float(rHalf[k]), aGrid, tGrid, tSnap, wantWrapDiag=False)
    compT = time.perf_counter() - tc
    f.close()
    del pos, mass, sft, met
    bps = 8 * 3 + 4 + 4 + 4
    readGBs = nBlock * bps / 1e9 / readT
    combined = nBlock / (readT + compT)
    print(f"  block {nBlock:,} stars ({nUseful:,} useful, {100*nUseful/nBlock:.1f}% eff), "
          f"{nBlock*bps/1e9:.1f} GB")
    print(f"  read {readT:.1f}s ({readGBs:.2f} GB/s, {nBlock/readT:,.0f} st/s); "
          f"compute {compT:.1f}s ({nBlock/compT:,.0f} st/s) [no-wrap-diag]; "
          f"COMBINED {combined:,.0f} st/s")
    return dict(snap=int(s), nBlock=int(nBlock), nUseful=nUseful, gbRead=nBlock * bps / 1e9,
                readSec=readT, computeSec=compT, readGBperSec=readGBs,
                readStarsPerSec=nBlock / readT, computeStarsPerSec=nBlock / compT,
                combinedStarsPerSec=combined)


def jsonify(o):
    if isinstance(o, dict):
        return {k: jsonify(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonify(v) for v in o]
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


def pilot():
    cats = {s: loadCat(s) for s in SNAPS}
    snaps = [runPilotSnap(s, cats[s]) for s in SNAPS]
    snaps = [x for x in snaps if x is not None]
    wrap = {s: wrapTest(s, cats[s]) for s in ["817", "660"]}   # z=0 and the highest-z snap
    probe = contiguousProbe("817", cats["817"], 100_000_000)

    rate = probe["combinedStarsPerSec"]
    extrap, totSec = {}, 0.0
    for s in SNAPS:
        sec = SPANSTARS[s] / rate
        extrap[s] = dict(spanStars=SPANSTARS[s], estHours=sec / 3600.0)
        totSec += sec
    extrap["TOTAL"] = dict(estHours=totSec / 3600.0)
    extrap["TOTAL_no817"] = dict(estHours=(totSec - SPANSTARS["817"] / rate) / 3600.0)

    out = dict(mode="pilot", seed=SEED, cosmology="FlatLambdaCDM h=0.6774 Om0=0.3089 Ob0=0.0486",
               snapshots=snaps, wrapTest=wrap, contiguousProbe=probe, fullRunExtrapolation=extrap)
    outPath = f"{WS}/results/apertureQcPilot.json"
    with open(outPath, "w") as fh:
        json.dump(jsonify(out), fh, indent=2)
    print(f"\n{'='*72}\nFULL-RUN EXTRAPOLATION (combined {rate:,.0f} stars/s, serial read+compute)")
    for s in SNAPS:
        print(f"  snap {s}: {extrap[s]['estHours']:.2f} h")
    print(f"  TOTAL 5: {extrap['TOTAL']['estHours']:.2f} h | "
          f"no-817: {extrap['TOTAL_no817']['estHours']:.2f} h")
    print(f"wrote {outPath}\n{'='*72}")


# ---------------------------------------------------------------------------
# FULL RUN (gated; launched only after STOP-2 greenlight via apFull.sbatch).
# Parallelism: a multiprocessing Pool of NPROC workers, each reading one contiguous
# chunk and computing it with the PILOT-VALIDATED scalar computeGalaxy() loop. The
# 8-way concurrency overlaps read and compute (some workers reading while others
# compute). Scalar-per-core was MEASURED ~2x faster than the vectorized bincount path
# (verify mode: 7.8M vs 3.5M st/s single core -- the vectorized full-array gathers are
# memory-bandwidth bound), so it is the inner loop. processChunk() (vectorized) is kept
# only as the verify-mode reference. Output is identical regardless of NPROC: each chunk
# is deterministic, checkpointed (resumable), then concatenated and sorted by catalog index.
# ---------------------------------------------------------------------------
CHUNK_STARS = 50_000_000         # ~1.8 GB read/worker; x8 workers ~15 GB; giants (<=48.6M) fit
NPROC = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
OUTCOLS = ["index", "idMostbound", "mStarAper2", "mStarAper1", "mStarTotalStars",
           "massWtAgeGyr", "massWtAgeAperGyr", "massWtMetalAper", "sfr100", "sfr10",
           "nStarAper", "pathology"]


def buildChunks(off, n, chunkStars):
    """Group offset-sorted rows into contiguous star blocks of ~chunkStars (a lone giant
    forms its own chunk). Deterministic given (catalog, chunkStars) -> resume-safe."""
    chunks, i, N = [], 0, len(off)
    while i < N:
        r0, startOff = i, off[i]
        while i < N - 1 and (off[i + 1] + n[i + 1] - startOff) <= chunkStars:
            i += 1
        chunks.append((int(r0), int(i)))
        i += 1
    return chunks


def processChunk(pos, mass, sft, met, off, n, posCat, rHalf, blockLo, aGrid, tGrid, tSnap):
    """Vectorized: all per-subhalo quantities for one contiguous block via bincount.
    Gap stars (unmasked subhalos between masked ones) map to a junk bin and are dropped."""
    nBlock = len(mass)
    starts = (off - blockLo)
    ends = starts + n
    m = len(starts)
    q = np.arange(nBlock)
    k = np.searchsorted(starts, q, side="right") - 1
    k = np.clip(k, 0, m - 1)
    valid = q < ends[k]                              # False => gap star
    subId = np.where(valid, k, m)                    # junk bin = m

    d = pos - posCat[k]                              # (N,3); junk stars masked by `valid`
    d -= BOX * np.round(d / BOX)                     # periodic minimum image
    r2 = np.einsum("ij,ij->i", d, d)
    rh = rHalf[k]
    inA2 = valid & (r2 < (2.0 * rh) ** 2)
    inA1 = valid & (r2 < rh ** 2)
    tForm = np.interp(sft, aGrid, tGrid)
    age = np.clip(tSnap - tForm, 0.0, None)

    def seg(w):
        return np.bincount(subId, weights=w, minlength=m + 1)[:m]

    vm = mass * valid
    mTot = seg(vm)
    mA2 = seg(mass * inA2)
    mA1 = seg(mass * inA1)
    nAper = seg(inA2.astype("f8")).astype("i8")
    sAge = seg(vm * age)
    sAgeA = seg(mass * inA2 * age)
    sMetA = seg(mass * inA2 * met)
    sSfr100 = seg(mass * inA2 * (age < 0.1))
    sSfr10 = seg(mass * inA2 * (age < 0.01))

    with np.errstate(invalid="ignore", divide="ignore"):
        massWtAge = sAge / mTot
        a2safe = np.where(mA2 > 0, mA2, np.nan)
        massWtAgeAper = sAgeA / a2safe
        massWtMetalAper = sMetA / a2safe
        sfr100 = sSfr100 / 1e8
        sfr10 = sSfr10 / 1e7

    patho = np.where(rHalf <= 0, "rHalf<=0",
                     np.where(nAper == 0, "emptyAperture", "")).astype(object)
    bad = rHalf <= 0                                 # rHalf<=0 -> NaN aperture cols, never 0
    for arr in (mA2, mA1, massWtAgeAper, massWtMetalAper, sfr100, sfr10):
        arr[bad] = np.nan
    return dict(mStarAper2=mA2, mStarAper1=mA1, mStarTotalStars=mTot,
                massWtAgeGyr=massWtAge, massWtAgeAperGyr=massWtAgeAper,
                massWtMetalAper=massWtMetalAper, sfr100=sfr100, sfr10=sfr10,
                nStarAper=nAper, pathology=patho)


def writeQc(s, idxAll, mTotAll, mA2All):
    """results/apertureQc_{s}.json: aper2/mStar ratio distribution per mass bin."""
    ratio = mA2All / mTotAll
    lm = np.log10(mTotAll)
    edges = np.arange(9.0, np.nanmax(lm) + 0.5, 0.5)
    bins = []
    for lo in edges:
        msk = (lm >= lo) & (lm < lo + 0.5) & np.isfinite(ratio)
        if msk.sum() == 0:
            continue
        rr = ratio[msk]
        bins.append(dict(logMlo=round(float(lo), 2), count=int(msk.sum()),
                         median=float(np.median(rr)), p10=float(np.percentile(rr, 10)),
                         p25=float(np.percentile(rr, 25)), p75=float(np.percentile(rr, 75)),
                         p90=float(np.percentile(rr, 90))))
    finite = np.isfinite(ratio)
    out = dict(snap=int(s), nGalaxies=int(len(idxAll)),
               nNaNaperture=int((~finite).sum()),
               overall=dict(median=float(np.median(ratio[finite])),
                            p25=float(np.percentile(ratio[finite], 25)),
                            p75=float(np.percentile(ratio[finite], 75))),
               byMassBin=bins)
    with open(f"{WS}/results/apertureQc_{s}.json", "w") as fh:
        json.dump(out, fh, indent=2)


GW = {}   # per-process worker globals (fork-inherited arrays + per-worker bigfile handle)


def _initWorker(s):
    GW["f"] = openSubfind(s)[0]


def _doChunk(c):
    G = GW
    cp = f"{G['ckptDir']}/chunk{c:05d}.parquet"
    if os.path.exists(cp):
        return (c, 0, True)
    off, n, rHalf, posC = G["off"], G["n"], G["rHalf"], G["posC"]
    r0, r1 = G["chunks"][c]
    blockLo, nBlock = int(off[r0]), int(off[r1] + n[r1] - off[r0])
    pos, mass, sft, met = readStars(G["f"], blockLo, nBlock)
    m = r1 - r0 + 1
    cols = {k: np.empty(m, dtype="f8") for k in OUTCOLS[2:10]}   # 8 float columns
    nAper = np.empty(m, dtype="i8")
    patho = np.empty(m, dtype=object)
    for ii, k in enumerate(range(r0, r1 + 1)):
        lo = int(off[k] - blockLo)
        hi = lo + int(n[k])
        rh = float(rHalf[k])
        g = computeGalaxy(pos[lo:hi], mass[lo:hi], sft[lo:hi], met[lo:hi],
                          posC[k], rh, G["aGrid"], G["tGrid"], G["tSnap"], wantWrapDiag=False)
        nAper[ii] = g["nStarAper"]
        if rh <= 0:                                  # never silently 0: NaN + flag
            patho[ii] = "rHalf<=0"
            for kk in ("mStarAper2", "mStarAper1", "massWtAgeAperGyr", "massWtMetalAper",
                       "sfr100", "sfr10"):
                g[kk] = np.nan
        else:
            patho[ii] = "emptyAperture" if g["nStarAper"] == 0 else ""
        for kk in cols:
            cols[kk][ii] = g[kk]
    pa_tbl = pa.table({"index": G["idx"][r0:r1 + 1], "idMostbound": G["idmb"][r0:r1 + 1],
                       **cols, "nStarAper": nAper, "pathology": patho})
    pq.write_table(pa_tbl, cp)
    return (c, int(n[r0:r1 + 1].sum()), False)


def full(s, maxChunks=None):
    import multiprocessing as mp
    cat = loadCat(s)
    order = np.argsort(cat["off"])
    off, n = cat["off"][order], cat["nStar"][order]
    aGrid, tGrid, tSnap = ageTable(ASNAP[s])
    chunks = buildChunks(off, n, CHUNK_STARS)
    smoke = maxChunks is not None
    ckptDir = f"{WS}/data/_ckpt_apertureExtra{s}" + ("_smoke" if smoke else "")
    os.makedirs(ckptDir, exist_ok=True)
    GW.update(off=off, n=n, posC=cat["pos"][order], rHalf=cat["rHalf"][order],
              idx=cat["idx"][order], idmb=cat["idmb"][order], chunks=chunks,
              aGrid=aGrid, tGrid=tGrid, tSnap=tSnap, ckptDir=ckptDir)
    todo = list(range(len(chunks)))[: int(maxChunks)] if smoke else list(range(len(chunks)))
    print(f"FULL snap {s}: {len(off):,} subhalos, {len(chunks)} chunks "
          f"(CHUNK_STARS={CHUNK_STARS:,}), processing {len(todo)} [{'SMOKE' if smoke else 'FULL'}]; "
          f"{NPROC} workers; ckpt {ckptDir}", flush=True)

    t0 = time.perf_counter()
    done = stars = 0
    with mp.get_context("fork").Pool(NPROC, initializer=_initWorker, initargs=(s,)) as pool:
        for (c, ns, skipped) in pool.imap_unordered(_doChunk, todo):
            done += 1
            stars += ns
            if done % 25 == 0 or done == len(todo):
                el = time.perf_counter() - t0
                print(f"  {done}/{len(todo)} chunks; {stars:,} new stars in {el:.0f}s "
                      f"= {stars/max(el,1):,.0f} st/s aggregate", flush=True)
    el = time.perf_counter() - t0
    print(f"  DONE {stars:,} stars in {el:.0f}s = {stars/max(el,1):,.0f} st/s "
          f"aggregate ({NPROC} workers)", flush=True)

    if smoke:
        import glob
        tt = pa.concat_tables([pq.read_table(p) for p in
                               sorted(glob.glob(f"{ckptDir}/chunk*.parquet"))])
        mt = tt["mStarTotalStars"].to_numpy()
        ii = tt["index"].to_numpy()
        cmap = dict(zip(cat["idx"].tolist(), cat["mStar"].tolist()))
        rel = np.abs(mt - np.array([cmap[int(x)] for x in ii])) / np.array([cmap[int(x)] for x in ii])
        print(f"  SMOKE correctness: {tt.num_rows:,} rows; "
              f"max relDev(totalStars vs catalog mStar)={rel.max():.3e}", flush=True)
        readSt = 0.37e9 / 36
        print(f"  read-bound check: aggregate {stars/el:,.0f} st/s vs single-stream read "
              f"{readSt:,.0f} st/s -> {'READ-BOUND (good)' if stars/el >= readSt else 'still climbing'}",
              flush=True)
        return

    parts = [pq.read_table(f"{ckptDir}/chunk{c:05d}.parquet") for c in range(len(chunks))]
    full_t = pa.concat_tables(parts)
    full_t = full_t.take(pa.array(np.argsort(full_t["index"].to_numpy())))
    outPath = f"{WS}/data/apertureExtra{s}.parquet"
    pq.write_table(full_t, outPath)
    writeQc(s, full_t["index"].to_numpy(), full_t["mStarTotalStars"].to_numpy(),
            full_t["mStarAper2"].to_numpy())
    print(f"  wrote {outPath}: {full_t.num_rows:,} rows; results/apertureQc_{s}.json", flush=True)


def verify(s="817", targetStars=100_000_000):
    """Adversarial cross-check before the heavy run: vectorized processChunk() MUST equal the
    pilot-validated scalar computeGalaxy() on a real contiguous block; also times the vectorized
    compute to back the read-bound wall-clock estimate."""
    print(f"\n{'='*72}\nVERIFY vectorized processChunk vs scalar computeGalaxy (snap {s})\n{'='*72}")
    cat = loadCat(s)
    order = np.argsort(cat["off"])
    off, n = cat["off"][order], cat["nStar"][order]
    posC, rHalf = cat["pos"][order], cat["rHalf"][order]
    jstart = int(np.searchsorted(off, off.min() + (off.max() - off.min()) // 2))
    j = jstart
    while j < len(off) - 1 and (off[j] + n[j] - off[jstart]) < targetStars:
        j += 1
    blockLo, nBlock = int(off[jstart]), int(off[j] + n[j] - off[jstart])
    aGrid, tGrid, tSnap = ageTable(ASNAP[s])
    f, _ = openSubfind(s)
    pos, mass, sft, met = readStars(f, blockLo, nBlock)
    f.close()
    m = j - jstart + 1
    print(f"  block {nBlock:,} stars over {m} subhalos")

    tv = time.perf_counter()
    res = processChunk(pos, mass, sft, met, off[jstart:j + 1], n[jstart:j + 1],
                       posC[jstart:j + 1], rHalf[jstart:j + 1], blockLo, aGrid, tGrid, tSnap)
    tvec = time.perf_counter() - tv

    # scalar reference (the validated path)
    ref = {k: np.empty(m) for k in ["mStarAper2", "mStarAper1", "mStarTotalStars", "massWtAgeGyr",
                                    "massWtAgeAperGyr", "massWtMetalAper", "sfr100", "sfr10"]}
    refN = np.empty(m, dtype="i8")
    for ii, k in enumerate(range(jstart, j + 1)):
        lo = int(off[k] - blockLo)
        hi = lo + int(n[k])
        g = computeGalaxy(pos[lo:hi], mass[lo:hi], sft[lo:hi], met[lo:hi],
                          posC[k], float(rHalf[k]), aGrid, tGrid, tSnap, wantWrapDiag=False)
        for kk in ref:
            ref[kk][ii] = g[kk]
        refN[ii] = g["nStarAper"]

    print("  column                 maxAbsDiff      maxRelDiff")
    allOk = True
    for kk in ref:
        a, b = res[kk], ref[kk]
        fin = np.isfinite(a) & np.isfinite(b)
        nanMatch = np.array_equal(np.isfinite(a), np.isfinite(b))
        absd = float(np.max(np.abs(a[fin] - b[fin]))) if fin.any() else 0.0
        denom = np.maximum(np.abs(b[fin]), 1e-30)
        reld = float(np.max(np.abs(a[fin] - b[fin]) / denom)) if fin.any() else 0.0
        ok = nanMatch and reld < 1e-6
        allOk &= ok
        print(f"  {kk:22s} {absd:12.4e}   {reld:12.4e}  {'OK' if ok else 'FAIL'}"
              f"{'' if nanMatch else ' (NaN-pattern mismatch)'}")
    nOk = bool(np.array_equal(res["nStarAper"], refN))
    allOk &= nOk
    print(f"  {'nStarAper':22s} {'exact match' if nOk else 'MISMATCH':>26s}  {'OK' if nOk else 'FAIL'}")

    rate = nBlock / tvec
    print(f"\n  VECTORIZED compute: {nBlock:,} stars in {tvec:.2f}s = {rate:,.0f} st/s")
    readRate = 0.37e9 / (8 * 3 + 4 + 4 + 4)  # measured probe read rate in stars/s
    print(f"  read rate (measured) = {readRate:,.0f} st/s -> "
          f"{'READ-BOUND' if rate > readRate else 'COMPUTE-BOUND'} "
          f"(compute/read = {rate/readRate:.1f}x)")
    print(f"\n  RESULT: {'ALL COLUMNS MATCH -- processChunk verified' if allOk else 'MISMATCH -- DO NOT RUN'}")
    no817Span = sum(SPANSTARS[x] for x in SNAPS if x != "817")
    print(f"  read-bound wall: per-snap span/readRate -> 771 {SPANSTARS['771']/readRate/3600:.2f}h "
          f"743 {SPANSTARS['743']/readRate/3600:.2f}h 692 {SPANSTARS['692']/readRate/3600:.2f}h "
          f"660 {SPANSTARS['660']/readRate/3600:.2f}h; 4-task array wall ~= "
          f"{SPANSTARS['771']/readRate/3600:.2f}h; serial-1proc no-817 {no817Span/readRate/3600:.2f}h")
    return allOk


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "pilot"
    if mode == "pilot":
        pilot()
    elif mode == "full":
        full(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    elif mode == "verify":
        sys.exit(0 if verify(*sys.argv[2:]) else 1)
    elif mode == "mergeqc":                       # combine per-snap qc -> results/apertureQc.json
        merged = {}
        for s in SNAPS:
            p = f"{WS}/results/apertureQc_{s}.json"
            if os.path.exists(p):
                merged[s] = json.load(open(p))
        with open(f"{WS}/results/apertureQc.json", "w") as fh:
            json.dump(dict(perSnapshot=merged,
                           note="aper2/mStarTotalStars ratio per snapshot and 0.5-dex mass bin"),
                      fh, indent=2)
        print(f"merged {list(merged)} -> results/apertureQc.json")
    else:
        sys.exit(f"unknown mode {mode!r}")
