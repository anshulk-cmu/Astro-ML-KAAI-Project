import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import bigfile

OUT = "/hildafs/projects/phy200026p/akumar45/data/catalogs"
H = 0.6774
MCUT = 1e9
CHUNK = 8_000_000
SNAPS = ["817", "771", "743", "692", "660"]
SUBFIND = ["/hildafs/datasets/Asterix/PIG2/PIG_{s}_subfind",
           "/hildafs/datasets/Asterix/PIG3/PIG_{s}_subfind_reassign"]
PHOT = "/hildafs/datasets/Asterix/photometric/PIG_{s}_photometric"
PHOTBANDS = ["des_g", "des_r", "des_i", "des_z", "sdss_g", "lsst_g", "lsst_r", "lsst_i", "lsst_z"]
META = {b"units": b"masses Msun (1e10/h applied, h=0.6774); pos ckpc/h; vel km/s as stored "
                  b"(UsePeculiarVelocity=1); SFR Msun/yr; rHalfStar ckpc/h; mags as stored "
                  b"(absolute for 817, apparent intrinsic for z>0 snaps); "
                  b"sfrNew=SubhaloSFR(newSFa), sfrOld=SubhaloSFR_old; isCentral=RankInGr==0"}


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


def extract(s):
    fs, sfPath = openSubfind(s)
    fp = bigfile.File(PHOT.format(s=s))
    n = fs["SubGroups/SubhaloMass"].size
    assert fp[f"SubGroups/{PHOTBANDS[0]}"].size == n, "photometric not row-aligned"
    print(f"snapshot {s}: {sfPath}, {n} subhalos")

    acc = {k: [] for k in
           ["index", "mStar", "mGas", "mDM", "posX", "posY", "posZ", "velX", "velY", "velZ",
            "idMostbound", "rHalfStar", "sfrNew", "sfrOld", "vmax", "velDisp", "nStar",
            "starOffset", "rankInGr"] + PHOTBANDS}
    for lo in range(0, n, CHUNK):
        hi = min(lo + CHUNK, n)
        mt = fs["SubGroups/SubhaloMassType"][lo:hi] * (1e10 / H)
        m = mt[:, 4] > MCUT
        if not m.any():
            continue
        acc["index"].append(lo + np.flatnonzero(m).astype("i8"))
        acc["mStar"].append(mt[m, 4].astype("f4"))
        acc["mGas"].append(mt[m, 0].astype("f4"))
        acc["mDM"].append(mt[m, 1].astype("f4"))
        pos = fs["SubGroups/SubhaloPos"][lo:hi][m]
        vel = fs["SubGroups/SubhaloVel"][lo:hi][m]
        for i, ax in enumerate("XYZ"):
            acc[f"pos{ax}"].append(pos[:, i].astype("f4"))
            acc[f"vel{ax}"].append(vel[:, i].astype("f4"))
        acc["idMostbound"].append(fs["SubGroups/SubhaloIDMostbound"][lo:hi][m])
        acc["rHalfStar"].append(fs["SubGroups/SubhaloHalfmassRadType"][lo:hi][m][:, 4].astype("f4"))
        acc["sfrNew"].append(fs["SubGroups/SubhaloSFR"][lo:hi][m].astype("f4"))
        acc["sfrOld"].append(fs["SubGroups/SubhaloSFR_old"][lo:hi][m].astype("f4"))
        acc["vmax"].append(fs["SubGroups/SubhaloVmax"][lo:hi][m].astype("f4"))
        acc["velDisp"].append(fs["SubGroups/SubhaloVelDisp"][lo:hi][m].astype("f4"))
        acc["nStar"].append(fs["SubGroups/SubhaloLenType"][lo:hi][m][:, 4].astype("i8"))
        acc["starOffset"].append(fs["SubGroups/SubhaloOffsetType"][lo:hi][m][:, 4].astype("i8"))
        acc["rankInGr"].append(fs["SubGroups/SubhaloRankInGr"][lo:hi][m].astype("i4"))
        for b in PHOTBANDS:
            acc[b].append(fp[f"SubGroups/{b}"][lo:hi][m].astype("f4"))
    fs.close(), fp.close()

    cols = {k: np.concatenate(v) for k, v in acc.items()}
    cols["isCentral"] = cols["rankInGr"] == 0
    names = {b: b.replace("_g", "G").replace("_r", "R").replace("_i", "I").replace("_z", "Z")
             for b in PHOTBANDS}
    table = pa.table({names.get(k, k): v for k, v in cols.items()}).replace_schema_metadata(META)
    path = f"{OUT}/catalog{s}.parquet"
    pq.write_table(table, path)
    sz = os.path.getsize(path) / 1e6
    print(f"  -> {path}: {table.num_rows} rows ({cols['isCentral'].sum()} centrals), {sz:.1f} MB")
    return sz


def main():
    os.makedirs(OUT, exist_ok=True)
    total = sum(extract(s) for s in SNAPS)
    print(f"\ntotal catalog size: {total:.1f} MB")


if __name__ == "__main__":
    main()
