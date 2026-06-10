import json

import numpy as np
import bigfile

PIG = "/hildafs/datasets/Asterix/PIG2/PIG_817_subfind"
RES = "/hildafs/projects/phy200026p/akumar45/results"
H = 0.6774
SSFR_Q = 1e-11  # yr^-1, SFR assumed Msun/yr
EDGES = 10 ** np.arange(8.0, 12.01, 0.5)


def stats(new, old, mstar, sel):
    n = int(sel.sum())
    if n == 0:
        return {"n": 0}
    nw, od, ms = new[sel], old[sel], mstar[sel]
    mx = np.maximum(nw, od)
    mn = np.minimum(nw, od)
    rd = np.zeros_like(mx)
    nz = mx > 0
    rd[nz] = (mx[nz] - mn[nz]) / mx[nz]
    diff = rd > 0
    qNew = ms > 0
    return {
        "n": n,
        "bothZero": int((mx == 0).sum()),
        "fracNewLowerOfDiffering": float((nw[diff] < od[diff]).mean()) if diff.any() else None,
        "fracDiffGt5pct": float((rd > 0.05).mean()),
        "fracDiffGt20pct": float((rd > 0.20).mean()),
        "fracDiffGt2x": float((rd > 0.50).mean()),
        "medianRelDiff": float(np.median(rd)),
        "p90RelDiff": float(np.percentile(rd, 90)),
        "quiescentFracNew": float((nw[qNew] / ms[qNew] < SSFR_Q).mean()) if qNew.any() else None,
        "quiescentFracOld": float((od[qNew] / ms[qNew] < SSFR_Q).mean()) if qNew.any() else None,
    }


def main():
    f = bigfile.File(PIG)
    mstar = (f["SubGroups/SubhaloMassType"][:][:, 4] * (1e10 / H)).astype("f8")
    sfrMain = f["SubGroups/SubhaloSFR"][:].astype("f8")
    sfrOld = f["SubGroups/SubhaloSFR_old"][:].astype("f8")
    sfrNewgal = f["NewgalSFR/SubhaloSFR"][:].astype("f8")
    f.close()

    ident = {
        "mainEqualsNewgal": int((sfrMain == sfrNewgal).sum()),
        "mainEqualsOld": int((sfrMain == sfrOld).sum()),
        "maxAbsDiffMainNewgal": float(np.abs(sfrMain - sfrNewgal).max()),
        "negativeSFRs": {k: int((v < 0).sum()) for k, v in
                         [("main", sfrMain), ("old", sfrOld), ("newgal", sfrNewgal)]},
        "nTotal": int(mstar.size),
    }
    print(f"identity checks: {ident}")
    new = sfrNewgal

    res = {"assumptions": "SFR in Msun/yr; new=NewgalSFR/SubhaloSFR, old=SubGroups/SubhaloSFR_old; "
                          "relDiff=(max-min)/max, both-zero counted as equal; quiescent sSFR<1e-11/yr",
           "identity": ident, "overall": {}, "massBins": []}

    res["overall"]["allSubhalos"] = stats(new, sfrOld, mstar, np.ones(mstar.size, bool))
    res["overall"]["mstar1e8to1e12"] = stats(new, sfrOld, mstar, (mstar >= EDGES[0]) & (mstar < EDGES[-1]))

    print(f"\n{'bin':>22s} {'n':>9s} {'>5%':>7s} {'>20%':>7s} {'>2x':>7s} {'medRD':>8s} "
          f"{'p90RD':>8s} {'qfNew':>7s} {'qfOld':>7s} {'newLow':>7s}")
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        s = stats(new, sfrOld, mstar, (mstar >= lo) & (mstar < hi))
        s["logMstarLo"], s["logMstarHi"] = float(np.log10(lo)), float(np.log10(hi))
        res["massBins"].append(s)
        print(f"{np.log10(lo):10.1f}-{np.log10(hi):>4.1f} dex {s['n']:9d} {s['fracDiffGt5pct']:7.4f} "
              f"{s['fracDiffGt20pct']:7.4f} {s['fracDiffGt2x']:7.4f} {s['medianRelDiff']:8.4f} "
              f"{s['p90RelDiff']:8.4f} {s['quiescentFracNew']:7.4f} {s['quiescentFracOld']:7.4f} "
              f"{s['fracNewLowerOfDiffering']:7.4f}")

    o = res["overall"]["allSubhalos"]
    print(f"\noverall (all {o['n']}): >5% {o['fracDiffGt5pct']:.4f}, >20% {o['fracDiffGt20pct']:.4f}, "
          f">2x {o['fracDiffGt2x']:.4f}, median {o['medianRelDiff']:.4f}, p90 {o['p90RelDiff']:.4f}")

    with open(f"{RES}/sfrConventionDiff.json", "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"wrote {RES}/sfrConventionDiff.json")


if __name__ == "__main__":
    main()
