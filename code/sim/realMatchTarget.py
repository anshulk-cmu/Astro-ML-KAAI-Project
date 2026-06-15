"""
Sample A matching target: the real southern anchor's (observed-z, M*) distribution
(preRegistration §12; plan §5.3). Sample A draws ASTRID galaxies to reproduce this joint
distribution, so demographic terms are removed by construction.

Stratify on observed-z bins of 0.05 (mapped to snapshots) x logM bins of 0.1 dex. The headline
match is RANK matching within z-bins (invariant to any monotone mass recalibration), so the
per-z-bin COUNT + the real logM rank distribution are the essential targets; the absolute 2-D
histogram supports the absolute-mass robustness variant (which needs the anchor-mass scale
aligned to the sim aperture-mass h=0.6774 convention -- applied at draw time, flagged here).

Observed-z = the anchor redshift (mostly photo-z); masses = colour-mass (Track 2; recalibrate
absolute strata when Zou DR9 lands). Matched range z >= 0.075 (no mocks below z=0.1; z<0.07
excluded from matched-pair analysis).
"""
import json
import numpy as np
import pandas as pd

ZEDGES = np.round(np.arange(0.075, 0.55, 0.05), 3)     # 0.075..0.525 -> bins of 0.05
MEDGES = np.round(np.arange(9.3, 11.85, 0.1), 2)        # 0.1-dex logM bins over [9.3,11.8]
SNAP_OF_ZBIN = {0.075: "771", 0.125: "771", 0.15: "743", 0.20: "743",
                0.25: "743r", 0.30: "743r", 0.35: "692", 0.40: "692", 0.45: "692",
                0.50: "660"}   # 0.05-bin lower edge -> rendering snapshot (724 excluded)
NSAMPLE = 50000


def snapOf(zlo):
    for edge in sorted(SNAP_OF_ZBIN, reverse=True):
        if zlo + 1e-9 >= edge:
            return SNAP_OF_ZBIN[edge]
    return "771"


def main():
    m = pd.read_parquet("data/anchorMass.parquet")[["dr8_id", "z", "logM_colour"]]
    cov = pd.read_parquet("data/anchorCovariates.parquet")[["dr8_id", "footprint"]]
    d = m.merge(cov, on="dr8_id", how="inner")
    south = d[d.footprint == "south"].copy()
    south = south[np.isfinite(south.z) & np.isfinite(south.logM_colour)]
    matched = south[(south.z >= ZEDGES[0]) & (south.z < ZEDGES[-1])].copy()
    print(f"south={len(south)} matched(z>={ZEDGES[0]})={len(matched)} "
          f"(below-range z<{ZEDGES[0]}: {int((south.z < ZEDGES[0]).sum())})")

    matched["zbin"] = pd.cut(matched.z, ZEDGES, right=False, labels=ZEDGES[:-1])
    scale = NSAMPLE / len(matched)

    perBin = {}
    for zlo in ZEDGES[:-1]:
        s = matched[matched.zbin == zlo]
        if len(s) == 0:
            continue
        lm = np.sort(s.logM_colour.values)
        perBin[f"{zlo:.3f}"] = {
            "snap": snapOf(zlo), "nReal": int(len(s)),
            "nSimTarget": int(round(len(s) * scale)),
            "logM_q": {q: float(np.percentile(lm, int(q)))
                       for q in ["5", "25", "50", "75", "95"]},
        }

    # 2-D (z 0.05 x logM 0.1) target histogram, scaled to NSAMPLE
    H, _, _ = np.histogram2d(matched.z, matched.logM_colour, bins=[ZEDGES, MEDGES])
    Ht = H * scale
    cells = []
    zc = ZEDGES[:-1]
    mc = MEDGES[:-1]
    for i, zlo in enumerate(zc):
        for j, mlo in enumerate(mc):
            if H[i, j] > 0:
                cells.append({"zlo": float(zlo), "mlo": float(mc[j]),
                              "snap": snapOf(zlo), "nReal": int(H[i, j]),
                              "nSimTarget": float(Ht[i, j])})
    pd.DataFrame(cells).to_parquet("data/matchTargetCells.parquet")

    summary = {
        "nSouth": int(len(south)), "nMatched": int(len(matched)),
        "nSimTarget": NSAMPLE, "scale": float(scale),
        "zEdges": [float(x) for x in ZEDGES], "mEdges": [float(x) for x in MEDGES],
        "perZbin": perBin,
        "snapCounts": {s: int(sum(b["nReal"] for b in perBin.values() if b["snap"] == s))
                       for s in ["771", "743", "743r", "692", "660"]},
        "massScale": "colour-mass (elpetro/NSA Chabrier); rank matching is convention-free; "
                     "absolute-variant binning needs h=0.6774 alignment at draw time",
        "redshift": "observed-z = anchor redshift (mostly photo-z)",
    }
    with open("results/matchTarget.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("snap target counts:", summary["snapCounts"])
    print("per-z-bin (zlo: snap nReal->nSimTarget medianLogM):")
    for z, b in perBin.items():
        print(f"  {z}: {b['snap']:4s} {b['nReal']:6d} -> {b['nSimTarget']:5d}  "
              f"med logM {b['logM_q']['50']:.2f}")
    print("wrote data/matchTargetCells.parquet + results/matchTarget.json")


if __name__ == "__main__":
    main()
