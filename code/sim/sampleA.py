"""
Sample A (demographics-matched) candidate draw (preRegistration §12; plan §5.3).

Per observed-z bin, draw ASTRID galaxies from the mapped snapshot to reproduce the real
southern anchor's per-z-bin demographics, with the ONE-ENTRY rule via phase-space chainRoot
(a physical galaxy enters at most once across ALL bins -- also covers the two 743-fed bin
groups), continuous-z assignment within the bin (anti-comb), and per-bin starvation (feeds P3).

Two matching modes (--mode):
- rank  : the plan's PRIMARY scale-invariant draw -- representative within-z sample of the
          available pool, count-matched per z-bin. Within-z absolute mass diff vs real is
          handled at analysis by the importance-weighted AUC_cond (plan §1.3).
- absolute : (z, M*)-histogram match on an aligned mass scale -- reproduces the real within-z
          mass distribution and yields P3 starvation-by-mass-cell. REQUIRES the real masses on
          a trustworthy absolute scale (--realmass zou) + the sim/real alignment (--align);
          colour-mass is elpetro-extrapolated above z~0.15 so absolute+colour is smoke-only.

Mass: sim = aperture mStarAper2 (exact, h=0.6774 logMsun); 'massMismatch' rows (PIG_771 quirk)
excluded. Pre-selection candidate sample; r<19 selection + refill loop is S3.
Run: python code/sim/sampleA.py --mode rank
     python code/sim/sampleA.py --mode absolute --realmass zou --align <offset>   # after Zou
"""
import argparse
import json
import numpy as np
import pandas as pd

EXT = "vera/data/apertureExtra{}.parquet"
CAT = "vera/data/catalogs/catalog{}.parquet"
CHAINS = "data/dedupChainsPhaseSpace.parquet"
NSAMPLE = 50000
SEED = 0
ZW = 0.05
MASSW = 0.1
BINS = [(0.075, "771", "771"), (0.125, "771", "771"),
        (0.175, "743", "743"), (0.225, "743", "743"),
        (0.275, "743r", "743"), (0.325, "743r", "743"),
        (0.375, "692", "692"), (0.425, "692", "692"), (0.475, "692", "692")]


def loadCandidates():
    chains = pd.read_parquet(CHAINS)[["snap", "index", "chainRoot"]]
    cand = {}
    for cs in ["771", "743", "692", "660"]:
        e = pd.read_parquet(EXT.format(cs), columns=["index", "mStarAper2", "pathology"])
        c = pd.read_parquet(CAT.format(cs), columns=["index", "idMostbound", "isCentral"])
        e = e[(e.pathology == "") & np.isfinite(e.mStarAper2)]
        df = e.merge(c, on="index").merge(
            chains[chains.snap == cs][["index", "chainRoot"]], on="index", how="left")
        df["logM"] = np.log10(df.mStarAper2)
        cand[cs] = df.reset_index(drop=True)
    return cand


def realBins(source):
    cov = pd.read_parquet("data/anchorCovariates.parquet")[["dr8_id", "footprint"]]
    base = pd.read_parquet("data/anchorMass.parquet")[["dr8_id", "z", "logM_colour"]]
    if source == "zou":
        z = pd.read_parquet("data/anchorMassZou.parquet")[["dr8_id", "logM_zou"]]
        base = base.merge(z, on="dr8_id", how="inner").rename(columns={"logM_zou": "logM"})
    else:
        base = base.rename(columns={"logM_colour": "logM"})
    d = base.merge(cov, on="dr8_id").query("footprint=='south'")
    d = d[np.isfinite(d.z) & np.isfinite(d.logM)]
    out = {zlo: dict(n=int(((d.z >= zlo) & (d.z < zlo + ZW)).sum()),
                     z=d.z[(d.z >= zlo) & (d.z < zlo + ZW)].to_numpy(),
                     logM=d.logM[(d.z >= zlo) & (d.z < zlo + ZW)].to_numpy())
           for zlo, *_ in BINS}
    return out, sum(o["n"] for o in out.values())


def drawRank(avail, nTarget, rng):
    n = min(nTarget, len(avail))
    return avail.iloc[rng.choice(len(avail), size=n, replace=False)], []


def drawAbsolute(avail, realLogM, align, nTarget, scale, rng):
    """(z,M*)-histogram match on aligned scale; returns (picked, starvation cells)."""
    target = realLogM + align
    edges = np.arange(8.5, 13.01, MASSW)
    th = np.round(np.histogram(target, bins=edges)[0] * scale).astype(int)
    am = avail.logM.to_numpy()
    picks, starv = [], []
    for j in range(len(edges) - 1):
        if th[j] == 0:
            continue
        inbin = np.flatnonzero((am >= edges[j]) & (am < edges[j + 1]))
        k = min(th[j], len(inbin))
        if k:
            picks.append(avail.iloc[rng.choice(inbin, size=k, replace=False)])
        if k < th[j]:
            starv.append(dict(logMlo=round(float(edges[j]), 2), want=int(th[j]), got=int(k)))
    return (pd.concat(picks) if picks else avail.iloc[:0]), starv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["rank", "absolute"], default="rank")
    ap.add_argument("--realmass", choices=["colour", "zou"], default="colour")
    ap.add_argument("--align", type=float, default=0.0,
                    help="logM offset added to real masses to match sim aperture scale")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    cand = loadCandidates()
    real, totalReal = realBins(args.realmass)
    scale = NSAMPLE / totalReal
    print(f"mode={args.mode} realmass={args.realmass} align={args.align} "
          f"realMatched={totalReal} scale={scale:.4f}")

    used, rows, report = set(), [], []
    for zlo, rsnap, csnap in sorted(BINS, key=lambda b: -b[0]):        # scarce high-z first
        nTarget = int(round(real[zlo]["n"] * scale))
        avail = cand[csnap][~cand[csnap].chainRoot.isin(used)]
        if args.mode == "rank":
            pick, starvCells = drawRank(avail, nTarget, rng)
        else:
            pick, starvCells = drawAbsolute(avail, real[zlo]["logM"], args.align, nTarget, scale, rng)
        used.update(pick.chainRoot.tolist())
        zobs = (rng.choice(real[zlo]["z"], size=len(pick), replace=True)
                if real[zlo]["n"] else np.full(len(pick), zlo + ZW / 2))
        for (_, g), zz in zip(pick.iterrows(), zobs):
            rows.append((rsnap, csnap, zlo, int(g["index"]), int(g["idMostbound"]),
                         g["chainRoot"], float(g["logM"]), bool(g["isCentral"]), float(zz)))
        report.append(dict(zlo=zlo, snap=rsnap, nTarget=nTarget, nDrawn=int(len(pick)),
                           starvation=float(1 - len(pick) / nTarget) if nTarget else 0.0,
                           nAvail=int(len(avail)), starvCells=starvCells,
                           simLogM_median=float(np.median(pick.logM)) if len(pick) else None,
                           realLogM_median=float(np.median(real[zlo]["logM"])) if real[zlo]["n"] else None))

    sa = pd.DataFrame(rows, columns=["renderSnap", "catalogSnap", "zbin", "index",
                                     "idMostbound", "chainRoot", "logMaper", "isCentral", "zObs"])
    tag = f"{args.mode}_{args.realmass}"
    sa.to_parquet(f"data/sampleA_{tag}.parquet")
    summary = dict(
        mode=args.mode, realmass=args.realmass, align=args.align,
        nDrawn=int(len(sa)), nTargetTotal=NSAMPLE, scale=float(scale),
        dedupChainReuse=int(sa.chainRoot.duplicated().sum()),
        uniquePhysicalGalaxies=int(sa.chainRoot.nunique()),
        perBin=sorted(report, key=lambda r: r["zlo"]),
        note=("rank: scale-invariant primary; within-z mass diff -> importance-weighted AUC_cond"
              if args.mode == "rank" else
              "absolute: (z,M*) match + P3 starvation-by-cell; needs realmass=zou + correct align"),
    )
    with open(f"results/sampleA_{tag}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Sample A [{tag}]: {len(sa)} drawn, {sa.chainRoot.nunique()} chains, "
          f"reuse={int(sa.chainRoot.duplicated().sum())}")
    print(f"{'zbin':>6}{'snap':>6}{'nTarget':>8}{'nDrawn':>8}{'starv':>7}{'simMed':>8}{'realMed':>8}")
    for r in sorted(report, key=lambda r: r["zlo"]):
        print(f"{r['zlo']:>6.3f}{r['snap']:>6}{r['nTarget']:>8}{r['nDrawn']:>8}"
              f"{r['starvation']:>7.3f}{(r['simLogM_median'] or 0):>8.2f}{(r['realLogM_median'] or 0):>8.2f}")
    print(f"wrote data/sampleA_{tag}.parquet + results/sampleA_{tag}.json")


if __name__ == "__main__":
    main()
