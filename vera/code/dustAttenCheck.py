import json

import numpy as np
import bigfile

RES = "/hildafs/projects/phy200026p/akumar45/results"
DUST = "/hildafs/datasets/Asterix/Fatemeh/LSST/dust"
INTR = "/hildafs/datasets/Asterix/Fatemeh/LSST/PIG_771_photometric"
PIG = "/hildafs/datasets/Asterix/PIG2/PIG_771_subfind"
NCHUNK, CHUNK = 200, 10000
SEED = 42


def sampleColumn(path, block, starts):
    f = bigfile.File(path)
    blk = f[block]
    v = np.concatenate([blk[int(s): int(s) + CHUNK] for s in starts]).astype("f8")
    f.close()
    return v


def main():
    fd = bigfile.File(DUST)
    nDust = fd["4/z0.1/lsst_g/kappa_2_9/lsst_g"].size
    fd.close()
    fi = bigfile.File(INTR)
    nIntr = fi["4/lsst_g"].size
    fi.close()
    fp = bigfile.File(PIG)
    nStar = fp["4/Metallicity"].size
    fp.close()
    print(f"row counts: dust z0.1={nDust}, intrinsic 771={nIntr}, subfind 771 stars={nStar}")
    assert nDust == nIntr == nStar

    rng = np.random.default_rng(SEED)
    starts = np.sort(rng.choice(nDust - CHUNK, NCHUNK, replace=False))
    att = sampleColumn(DUST, "4/z0.1/lsst_g/kappa_2_9/lsst_g", starts)
    intr = sampleColumn(INTR, "4/lsst_g", starts)
    zmet = sampleColumn(PIG, "4/Metallicity", starts)

    finite = np.isfinite(att) & np.isfinite(intr)
    ag = att[finite] - intr[finite]
    zf = zmet[finite]
    pct = {f"p{q}": float(np.percentile(ag, q)) for q in [1, 5, 25, 50, 75, 95, 99]}
    out = {
        "sampling": f"{NCHUNK} chunks x {CHUNK} stars, seed {SEED}, identical row indices",
        "nSampled": int(NCHUNK * CHUNK), "nFinite": int(finite.sum()),
        "A_g": {
            **pct, "mean": float(ag.mean()), "min": float(ag.min()), "max": float(ag.max()),
            "fracBelow0p01": float((ag < 0.01).mean()),
            "fracNegativeBeyondNoise": float((ag < -0.01).mean()),
            "fracNegativeAny": float((ag < 0).mean()),
        },
        "vsMetallicityDeciles": [],
    }
    print(f"A_g stats: {out['A_g']}")

    qEdges = np.percentile(zf, np.linspace(0, 100, 11))
    for i in range(10):
        m = (zf >= qEdges[i]) & (zf <= qEdges[i + 1] if i == 9 else zf < qEdges[i + 1])
        rec = {"decile": i + 1, "zMin": float(qEdges[i]), "zMax": float(qEdges[i + 1]),
               "n": int(m.sum()), "medianAg": float(np.median(ag[m])),
               "p95Ag": float(np.percentile(ag[m], 95))}
        out["vsMetallicityDeciles"].append(rec)
        print(f"Z decile {i+1:2d} [{qEdges[i]:.2e},{qEdges[i+1]:.2e}] n={m.sum():7d} "
              f"medianA_g={np.median(ag[m]):.4f} p95={np.percentile(ag[m],95):.4f}")

    neg = ag < -0.01
    verdict = "ROW-ALIGNED (negative tail negligible)" if neg.mean() < 1e-3 else \
        "WARNING: significant negative tail -> products may NOT be row-aligned"
    out["verdict"] = verdict
    print(f"\nverdict: {verdict}")

    with open(f"{RES}/dustAttenCheck.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"wrote {RES}/dustAttenCheck.json")


if __name__ == "__main__":
    main()
