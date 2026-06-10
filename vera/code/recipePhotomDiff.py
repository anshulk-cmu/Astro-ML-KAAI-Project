import json

import numpy as np
import bigfile

RES = "/hildafs/projects/phy200026p/akumar45/results"
NEW = "/hildafs/datasets/Asterix/photometric/PIG_817_photometric_newSFa_z01"
ORI = "/hildafs/datasets/Asterix/photometric/PIG_817_photometric_oriSFa_z01"
PIG = "/hildafs/datasets/Asterix/PIG2/PIG_817_subfind"
H = 0.6774
BANDS = ["u", "g", "r", "i", "z"]
EDGES = 10 ** np.arange(9.0, 12.51, 0.5)


def dstats(d):
    return {
        "n": int(d.size), "median": float(np.median(d)), "mean": float(d.mean()),
        "std": float(d.std()), "p5": float(np.percentile(d, 5)),
        "p95": float(np.percentile(d, 95)),
        "fracAbsGt0p02": float((np.abs(d) > 0.02).mean()),
        "fracAbsGt0p05": float((np.abs(d) > 0.05).mean()),
        "fracAbsGt0p1": float((np.abs(d) > 0.1).mean()),
    }


def main():
    fs = bigfile.File(PIG)
    mstar = fs["SubGroups/SubhaloMassType"][:][:, 4] * (1e10 / H)
    fs.close()
    sel = mstar > 1e9
    print(f"subhalos with Mstar>1e9: {sel.sum()}")

    fn, fo = bigfile.File(NEW), bigfile.File(ORI)
    assert fn["SubGroups/sdss_g"].size == fo["SubGroups/sdss_g"].size == mstar.size
    mags = {}
    for b in BANDS:
        mags[f"new_{b}"] = fn[f"SubGroups/sdss_{b}"][:][sel].astype("f8")
        mags[f"ori_{b}"] = fo[f"SubGroups/sdss_{b}"][:][sel].astype("f8")
    fn.close(), fo.close()

    ms = mstar[sel]
    finite = np.ones(ms.size, bool)
    for v in mags.values():
        finite &= np.isfinite(v)
    print(f"finite in all 10 columns: {finite.sum()} (dropped {(~finite).sum()})")

    out = {"selection": "Mstar>1e9 Msun", "nSelected": int(sel.sum()),
           "nFinite": int(finite.sum()), "deltaConvention": "newSFa - oriSFa [mag]",
           "bands": {}, "gMinusRShift": {}, "massBins": []}

    print(f"\n{'band':>5s} {'median':>9s} {'std':>8s} {'p5':>9s} {'p95':>9s} "
          f"{'|d|>0.02':>9s} {'|d|>0.05':>9s} {'|d|>0.1':>8s}")
    delta = {}
    for b in BANDS:
        d = (mags[f"new_{b}"] - mags[f"ori_{b}"])[finite]
        delta[b] = d
        s = dstats(d)
        out["bands"][b] = s
        print(f"{b:>5s} {s['median']:9.4f} {s['std']:8.4f} {s['p5']:9.4f} {s['p95']:9.4f} "
              f"{s['fracAbsGt0p02']:9.4f} {s['fracAbsGt0p05']:9.4f} {s['fracAbsGt0p1']:8.4f}")

    dgr = ((mags["new_g"] - mags["new_r"]) - (mags["ori_g"] - mags["ori_r"]))[finite]
    out["gMinusRShift"] = dstats(dgr)
    print(f"\ng-r shift: median {np.median(dgr):+.4f}, std {dgr.std():.4f}, "
          f"|d|>0.02 frac {(np.abs(dgr) > 0.02).mean():.4f}")

    msf = ms[finite]
    print(f"\n{'Mstar bin':>16s} {'n':>8s} {'med dg':>8s} {'p95|dg|':>8s} {'med d(g-r)':>10s}")
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        m = (msf >= lo) & (msf < hi)
        if not m.any():
            continue
        rec = {"logMstarLo": float(np.log10(lo)), "logMstarHi": float(np.log10(hi)),
               "n": int(m.sum()),
               "bands": {b: {"median": float(np.median(delta[b][m])),
                             "p95Abs": float(np.percentile(np.abs(delta[b][m]), 95))}
                         for b in BANDS},
               "gMinusR": {"median": float(np.median(dgr[m])),
                           "p95Abs": float(np.percentile(np.abs(dgr[m]), 95))}}
        out["massBins"].append(rec)
        print(f"  {np.log10(lo):5.1f}-{np.log10(hi):4.1f} dex {m.sum():8d} "
              f"{np.median(delta['g'][m]):+8.4f} {np.percentile(np.abs(delta['g'][m]),95):8.4f} "
              f"{np.median(dgr[m]):+10.4f}")

    with open(f"{RES}/recipePhotomDiff.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {RES}/recipePhotomDiff.json")


if __name__ == "__main__":
    main()
