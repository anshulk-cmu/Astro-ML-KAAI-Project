import json

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import bigfile

PHOT = "/hildafs/datasets/Asterix/photometric/PIG_817_photometric"
PIG = "/hildafs/datasets/Asterix/PIG2/PIG_817_subfind"
RES = "/hildafs/projects/phy200026p/akumar45/results"
H = 0.6774
BINW = 0.25
EDGES = np.arange(-24.0, -14.0 + BINW / 2, BINW)
VOL = (250.0 / H) ** 3  # comoving Mpc^3


def main():
    fs = bigfile.File(PIG)
    mstar = fs["SubGroups/SubhaloMassType"][:][:, 4] * (1e10 / H)
    fs.close()
    print(f"subhalos: {mstar.size}, with Mstar>0: {(mstar > 0).sum()}, Mstar>1e8: {(mstar > 1e8).sum()}")

    fp = bigfile.File(PHOT)
    curves = {}
    qc = {}
    for band in ["sdss_g", "des_g"]:
        mag = fp[f"SubGroups/{band}"][:]
        finite = np.isfinite(mag)
        qc[band] = {
            "nonFinite": int((~finite).sum()),
            "zeroExactly": int((mag == 0).sum()),
            "nonNegative": int((mag[finite] >= 0).sum()),
            "minMag": float(mag[finite].min()),
        }
        print(f"{band} QC: {qc[band]}")
        for cut, sel in [("all", finite), ("mstarGt1e8", finite & (mstar > 1e8))]:
            cnt, _ = np.histogram(mag[sel], bins=EDGES)
            curves[f"{band}_{cut}"] = {
                "counts": cnt.astype(int).tolist(),
                "phi": (cnt / (VOL * BINW)).tolist(),
            }
    fp.close()

    centers = (EDGES[:-1] + BINW / 2).tolist()
    out = {
        "snapshot": "PIG_817 (z=0)",
        "sources": {"mags": PHOT + "/SubGroups", "mass": PIG + "/SubGroups/SubhaloMassType"},
        "h": H,
        "volumeMpc3": VOL,
        "binWidthMag": BINW,
        "binCenters": centers,
        "phiUnits": "Mpc^-3 mag^-1 (comoving)",
        "magConvention": "absolute (product stores negative values at z=0)",
        "qc": qc,
        "curves": curves,
    }
    with open(f"{RES}/lumFn817.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"wrote {RES}/lumFn817.json")

    fig, ax = plt.subplots(figsize=(7, 5))
    for key, c in curves.items():
        phi = np.array(c["phi"])
        ax.plot(centers, np.where(phi > 0, phi, np.nan), label=key)
    ax.set_yscale("log")
    ax.set_xlabel("absolute g magnitude")
    ax.set_ylabel(r"$\phi$ [Mpc$^{-3}$ mag$^{-1}$]")
    ax.set_title("ASTRID PIG_817 (z=0) g-band luminosity function")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{RES}/lumFn817.png", dpi=130)
    print(f"wrote {RES}/lumFn817.png")

    print(f"\n{'Mg':>7s} {'N(sdss,all)':>12s} {'phi(sdss,all)':>13s} {'N(des,all)':>12s}")
    for i, m in enumerate(centers):
        print(f"{m:7.2f} {curves['sdss_g_all']['counts'][i]:12d} "
              f"{curves['sdss_g_all']['phi'][i]:13.4e} {curves['des_g_all']['counts'][i]:12d}")


if __name__ == "__main__":
    main()
