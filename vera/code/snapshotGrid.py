"""Task A: exact temporal backbone for the five working snapshots.

For 817, 771, 743, 692, 660 open the SAME subfind catalog extractCatalogs.py used
(openSubfind fallback), read Header attrs, cross-check against Snapshots.txt, and with
astropy compute z, cosmic age, lookback time, kpc/arcsec and adjacent-snapshot dt.

Light job: Header attrs are metadata only (no particle reads) -> login node OK.
Writes results/snapshotGrid.json and prints a table. Numbers tagged measured vs interpreted
in the log, not here.
"""
import json
import os

import numpy as np
import bigfile
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

WS = "/hildafs/projects/phy200026p/akumar45"
OUT = f"{WS}/results/snapshotGrid.json"
SNAPSTXT = "/hildafs/datasets/Asterix/Snapshots.txt"
SNAPS = ["817", "771", "743", "692", "660"]  # descending a (z=0 -> z~0.5)
# adjacency the brief asked for: 817->771->743->692->660
ADJ = [("817", "771"), ("771", "743"), ("743", "692"), ("692", "660")]
SUBFIND = ["/hildafs/datasets/Asterix/PIG2/PIG_{s}_subfind",
           "/hildafs/datasets/Asterix/PIG3/PIG_{s}_subfind_reassign"]


def openSubfind(s):
    """Identical to extractCatalogs.py: PIG2 first, PIG3 _reassign fallback."""
    for tpl in SUBFIND:
        path = tpl.format(s=s)
        try:
            f = bigfile.File(path)
            f["SubGroups/SubhaloMass"]
            return f, path
        except Exception as e:
            print(f"  {path}: {e}")
    raise RuntimeError(f"no readable subfind for {s}")


def scalar(v):
    """BigFile attrs are arrays; coerce length-1 arrays to python scalars."""
    a = np.asarray(v)
    if a.ndim == 0 or a.size == 1:
        x = a.reshape(-1)[0]
        return x.item() if hasattr(x, "item") else x
    return a.tolist()


def getAttr(attrs, names, default=None):
    for nm in names:
        if nm in attrs:
            return scalar(attrs[nm])
    return default


def loadSnapsTxt():
    """snap number -> scale factor a (measured from Snapshots.txt)."""
    grid = {}
    with open(SNAPSTXT) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) >= 2:
                grid[int(parts[0])] = float(parts[1])
    return grid


def main():
    snapsTxt = loadSnapsTxt()
    print(f"Snapshots.txt: {len(snapsTxt)} snapshots, "
          f"a in [{min(snapsTxt.values()):.4f}, {max(snapsTxt.values()):.6f}]\n")

    # ---- read every header first; pin cosmology from snapshot 817 and verify others match ----
    hdr = {}
    for s in SNAPS:
        f, path = openSubfind(s)
        a = f["Header"].attrs
        rec = {
            "path": path,
            "attrKeys": sorted(a.keys()),
            "time": getAttr(a, ["Time"]),
            "redshiftHeader": getAttr(a, ["Redshift"]),  # may be absent
            "hubbleParam": getAttr(a, ["HubbleParam"]),
            "omega0": getAttr(a, ["Omega0", "OmegaM", "OmegaMatter"]),
            "omegaLambda": getAttr(a, ["OmegaLambda"]),
            "omegaBaryon": getAttr(a, ["OmegaBaryon"]),
            "boxSize": getAttr(a, ["BoxSize"]),
            "massTable": scalar(a["MassTable"]) if "MassTable" in a else None,
            "unitLengthCgs": getAttr(a, ["UnitLength_in_cm", "UnitLength_in_cgs"]),
            "unitMassCgs": getAttr(a, ["UnitMass_in_g", "UnitMass_in_cgs"]),
            "unitVelocityCgs": getAttr(a, ["UnitVelocity_in_cm_per_s", "UnitVelocity_in_cgs"]),
        }
        print(f"snapshot {s}: {path}")
        for k in rec["attrKeys"]:
            print(f"    {k} = {scalar(a[k])}")
        print()
        f.close()
        hdr[s] = rec

    # ---- cosmology pinned from 817 header ----
    h817 = hdr["817"]
    hub = float(h817["hubbleParam"])
    om0 = float(h817["omega0"])
    obar = h817["omegaBaryon"]
    cosmo = FlatLambdaCDM(H0=100.0 * hub * u.km / u.s / u.Mpc, Om0=om0,
                          Ob0=(float(obar) if obar is not None else None))
    omLamHeader = h817["omegaLambda"]
    flatResidual = None if omLamHeader is None else abs((om0 + float(omLamHeader)) - 1.0)
    print(f"COSMOLOGY (from snap 817 header): H0={100*hub:.4f} km/s/Mpc (h={hub}), Om0={om0}, "
          f"OmegaLambda_header={omLamHeader}, OmegaBaryon={obar}")
    if flatResidual is not None:
        print(f"  flatness check |Om0+OmegaLambda-1| = {flatResidual:.3e} "
              f"({'OK flat' if flatResidual < 1e-3 else 'NOT FLAT -- review'})")
    print()

    # ---- per-snapshot temporal quantities ----
    rows = []
    for s in SNAPS:
        sn = int(s)
        aHeader = float(hdr[s]["time"])
        aTxt = snapsTxt.get(sn)
        agree = (aTxt is not None) and (abs(aHeader - aTxt) < 1e-6)
        z = 1.0 / aHeader - 1.0
        zHeader = hdr[s]["redshiftHeader"]
        age = cosmo.age(z).to(u.Gyr).value
        lookback = cosmo.lookback_time(z).to(u.Gyr).value
        if z > 0:
            kpcPerArcsec = (cosmo.kpc_proper_per_arcmin(z).to(u.kpc / u.arcmin).value) / 60.0
            dA = cosmo.angular_diameter_distance(z).to(u.Mpc).value
        else:
            kpcPerArcsec = 0.0  # z=0: angular-diameter distance -> 0, scale degenerate
            dA = 0.0
        rows.append({
            "snap": sn,
            "scaleFactorHeader": aHeader,
            "scaleFactorSnapsTxt": aTxt,
            "headerVsTxtAgree": bool(agree),
            "headerVsTxtAbsDiff": (None if aTxt is None else abs(aHeader - aTxt)),
            "redshift": z,
            "redshiftHeaderAttr": zHeader,
            "ageGyr": age,
            "lookbackGyr": lookback,
            "angularDiameterDistMpc": dA,
            "kpcPerArcsec": kpcPerArcsec,
            "boxSizeCkpcH": hdr[s]["boxSize"],
        })

    # ---- adjacent dt (cosmic time gap), feeds dedup v_pec*dt bound ----
    ageBySnap = {r["snap"]: r["ageGyr"] for r in rows}
    adj = []
    for hi, lo in ADJ:
        dt = ageBySnap[int(hi)] - ageBySnap[int(lo)]  # hi is older (larger a) -> positive
        adj.append({"pair": f"{hi}->{lo}", "deltaTGyr": dt})

    # ---- print table ----
    print("=" * 104)
    print(f"{'snap':>5} {'a':>10} {'z':>9} {'age/Gyr':>9} {'lookback/Gyr':>13} "
          f"{'D_A/Mpc':>10} {'kpc/arcsec':>11} {'a==txt':>7}")
    print("-" * 104)
    for r in rows:
        print(f"{r['snap']:>5} {r['scaleFactorHeader']:>10.6f} {r['redshift']:>9.5f} "
              f"{r['ageGyr']:>9.4f} {r['lookbackGyr']:>13.4f} "
              f"{r['angularDiameterDistMpc']:>10.3f} {r['kpcPerArcsec']:>11.5f} "
              f"{str(r['headerVsTxtAgree']):>7}")
    print("=" * 104)
    print("adjacent cosmic-time gaps (older->younger):")
    for a in adj:
        print(f"    {a['pair']:>10}: dt = {a['deltaTGyr']:.4f} Gyr")
    print("=" * 104)

    out = {
        "cosmology": {
            "model": "FlatLambdaCDM",
            "H0_km_s_Mpc": 100.0 * hub,
            "h": hub,
            "Om0": om0,
            "OmegaLambda_header": omLamHeader,
            "OmegaBaryon": obar,
            "flatnessResidual": flatResidual,
            "source": "snap 817 Header (HubbleParam, Omega0); OmegaLambda=1-Om0 enforced by FlatLambdaCDM",
        },
        "snapshots": rows,
        "adjacentDeltaT": adj,
        "headerRaw": {s: {k: hdr[s][k] for k in
                          ["path", "time", "redshiftHeader", "hubbleParam", "omega0",
                           "omegaLambda", "omegaBaryon", "boxSize", "massTable",
                           "unitLengthCgs", "unitMassCgs", "unitVelocityCgs", "attrKeys"]}
                      for s in SNAPS},
        "snapsTxtContext": {str(k): snapsTxt[k] for k in sorted(snapsTxt)
                            if k in (660, 692, 743, 771, 817) or 650 <= k <= 662
                            or 815 <= k <= 818},
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
