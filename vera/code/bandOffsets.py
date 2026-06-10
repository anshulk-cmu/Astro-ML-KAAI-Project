import json

import numpy as np
import bigfile

RES = "/hildafs/projects/phy200026p/akumar45/results"
PHOT = "/hildafs/datasets/Asterix/photometric"
# per-star des+lsst columns exist ONLY for 544 and 817; 771's 4/ block is sdss-only,
# so 771 is checked at per-subhalo level (SubGroups has all 16 bands there)
SNAPS = [
    {"tag": "817_perStar_absolute", "path": f"{PHOT}/PIG_817_photometric", "grp": "4", "sample": True},
    {"tag": "544_perStar_apparent_z1", "path": f"{PHOT}/PIG_544_photometric", "grp": "4", "sample": True},
    {"tag": "771_perSubhalo_apparent", "path": f"{PHOT}/PIG_771_photometric", "grp": "SubGroups", "sample": False},
    {"tag": "817_perSubhalo_absolute", "path": f"{PHOT}/PIG_817_photometric", "grp": "SubGroups", "sample": False},
]
BANDS = ["g", "r", "i", "z"]
NCHUNK, CHUNK = 200, 10000
SEED = 42


def readColumn(f, name, starts):
    blk = f[name]
    if starts is None:
        return blk[:].astype("f8")
    return np.concatenate([blk[int(s): int(s) + CHUNK] for s in starts]).astype("f8")


def main():
    out = {"sampling": f"perStar: {NCHUNK} chunks x {CHUNK}, seed {SEED}; perSubhalo: all finite rows",
           "snapshots": {}}
    for cfg in SNAPS:
        f = bigfile.File(cfg["path"])
        n = f[f"{cfg['grp']}/lsst_g"].size
        starts = None
        if cfg["sample"]:
            rng = np.random.default_rng(SEED)
            starts = np.sort(rng.choice(n - CHUNK, NCHUNK, replace=False))
        cols = {}
        for b in BANDS:
            cols[f"des_{b}"] = readColumn(f, f"{cfg['grp']}/des_{b}", starts)
            cols[f"lsst_{b}"] = readColumn(f, f"{cfg['grp']}/lsst_{b}", starts)
        f.close()

        nRead = next(iter(cols.values())).size
        finite = np.ones(nRead, bool)
        for v in cols.values():
            finite &= np.isfinite(v)
        color = cols["lsst_g"][finite] - cols["lsst_z"][finite]
        cEdges = np.linspace(np.percentile(color, 1), np.percentile(color, 99), 11)
        cBin = np.digitize(color, cEdges) - 1
        snap = {"nRows": int(n), "nRead": int(nRead), "nFinite": int(finite.sum()),
                "colorBinEdges_lsstGminusZ": cEdges.tolist(), "bands": {}}
        print(f"\n##### {cfg['tag']}  (rows={n}, read={nRead}, finite={finite.sum()})")
        print(f"{'band':>6s} {'median':>9s} {'std':>8s} {'p5':>9s} {'p95':>9s} {'resid std':>10s}")
        for b in BANDS:
            d = (cols[f"des_{b}"] - cols[f"lsst_{b}"])[finite]
            perBin = []
            resid = np.full_like(d, np.nan)
            for i in range(10):
                m = cBin == i
                med = float(np.median(d[m])) if m.any() else None
                perBin.append({"colorCenter": float((cEdges[i] + cEdges[i + 1]) / 2),
                               "n": int(m.sum()), "medianOffset": med,
                               "std": float(d[m].std()) if m.any() else None})
                if m.any():
                    resid[m] = d[m] - med
            residStd = float(np.nanstd(resid))
            snap["bands"][b] = {
                "median": float(np.median(d)), "std": float(d.std()),
                "p5": float(np.percentile(d, 5)), "p95": float(np.percentile(d, 95)),
                "residStdAfterColorTerm": residStd, "vsColor": perBin,
            }
            print(f"{b:>6s} {np.median(d):9.4f} {d.std():8.4f} {np.percentile(d,5):9.4f} "
                  f"{np.percentile(d,95):9.4f} {residStd:10.4f}")
            print("       color-bin medians: " +
                  " ".join(f"{p['medianOffset']:+.3f}" if p["medianOffset"] is not None else "  -  "
                           for p in perBin))
        out["snapshots"][cfg["tag"]] = snap

    with open(f"{RES}/bandOffsets.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {RES}/bandOffsets.json")


if __name__ == "__main__":
    main()
