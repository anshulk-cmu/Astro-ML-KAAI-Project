"""
Track 1 (preRegistration §5.3 PRIMARY): independent SPS stellar masses for the
anchor from the Zou et al. 2022 DESI DR9 photo-z + stellar-mass catalog
(china-vo paperdata; DOI 10.12149/101090). ~320M galaxies, log M* sigma ~0.2 dex,
full z<1, both hemispheres.

Stream-crossmatch: for each RA-range shard, download -> read ra/dec/photoz/mass ->
nearest-match the anchor galaxies in that RA range (within 1") -> keep -> DELETE the
shard. Peak disk ~ one shard (~3 GB).

NETWORK NOTE (verified on Vera 2026-06-15): the china-vo directory LISTING is a dead
JS/cookie portal (HTTP 200, empty body) -> cannot scrape it. Direct shard GETs work and
need a browser User-Agent (default Python-urllib is what the portal rejects). Naming is
fixed: desidr9_galaxy_cspcat_ra{LLL}_{HHH}.fits, 10-deg bins, 3-digit zero-pad -> 36
shards (ra000_010 .. ra350_360). Malformed names return Content-Length 0 (not 404).
So the shard list is GENERATED, not discovered, and downloads carry a UA + are validated.

RUN ON VERA from the workspace root: python -u code/crossmatchZouMass.py
Resumable via zouDone.txt. Needs astropy, scipy, pandas, numpy. Anchor: data/anchorRADec.csv.
"""
import os
import re
import sys
import shutil
import urllib.request
import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.spatial import cKDTree

BASE = "https://paperdata.china-vo.org/zouhu/pz_cluster/photoz_desidr9/"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36"
TMP = "zou_shard.fits"
OUT = "data/anchorMassZou.parquet"
DONE = "data/zouDone.txt"
MATCH_AS = 1.0
SHARDS = [f"desidr9_galaxy_cspcat_ra{i:03d}_{i+10:03d}.fits" for i in range(0, 360, 10)]
RA_CANDS = ["ra", "RA", "raj2000", "RAJ2000"]
DEC_CANDS = ["dec", "DEC", "dej2000", "DEJ2000", "decl"]
Z_CANDS = ["photoz", "photo_z", "z_phot", "zphot", "zphot_best", "z_best", "zp", "redshift"]
M_CANDS = ["logmass", "logmass_best", "mass", "mass_best", "mass_median", "logm",
           "mstar", "logmstar", "stellar_mass", "logms", "smass", "logsm", "sm"]


def pick(cols, cands):
    low = {c.lower(): c for c in cols}
    for c in cands:
        if c.lower() in low:
            return low[c.lower()]
    return None


def unit_vecs(ra, dec):
    r, d = np.radians(ra), np.radians(dec)
    return np.column_stack([np.cos(d) * np.cos(r), np.cos(d) * np.sin(r), np.sin(d)])


def download(url, dest):
    """Streamed download with a browser UA; returns bytes written. 0 = bad/empty name."""
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=600) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f, length=4 * 1024 * 1024)
    return os.path.getsize(dest)


def isFits(path):
    with open(path, "rb") as f:
        return f.read(6) == b"SIMPLE"


def main():
    anc = pd.read_csv("data/anchorRADec.csv")
    print(f"anchor n={len(anc)}; {len(SHARDS)} generated shards (ra000_010..ra350_360)")
    done = set(open(DONE).read().split()) if os.path.exists(DONE) else set()
    results = [pd.read_parquet(OUT)] if os.path.exists(OUT) else []
    cmap = None

    for sh in SHARDS:
        if sh in done:
            continue
        lo, hi = map(int, re.findall(r"ra(\d+)_(\d+)", sh)[0])
        sub = anc[(anc.ra >= lo - 0.02) & (anc.ra < hi + 0.02)]
        if len(sub) == 0:
            done.add(sh); open(DONE, "a").write(sh + "\n"); continue

        print(f"{sh}: anchor-in-range {len(sub)} ... downloading", flush=True)
        nbytes = download(BASE + sh, TMP)
        if nbytes == 0 or not isFits(TMP):
            print(f"  !! {sh}: bytes={nbytes}, not valid FITS -- skipping (bad name?)")
            os.remove(TMP) if os.path.exists(TMP) else None
            done.add(sh); open(DONE, "a").write(sh + "\n"); continue
        print(f"  downloaded {nbytes/1e9:.2f} GB", flush=True)

        with fits.open(TMP, memmap=True) as hd:
            cols = hd[1].columns.names
            if cmap is None:
                print("FITS columns:", cols)
                cmap = {"ra": pick(cols, RA_CANDS), "dec": pick(cols, DEC_CANDS),
                        "z": pick(cols, Z_CANDS), "m": pick(cols, M_CANDS)}
                print("column map:", cmap)
                if not all([cmap["ra"], cmap["dec"], cmap["m"]]):
                    sys.exit(f"could not map ra/dec/mass from {cols} - edit *_CANDS")
            t = hd[1].data
            tra = np.asarray(t[cmap["ra"]], float)
            tdec = np.asarray(t[cmap["dec"]], float)
            tz = np.asarray(t[cmap["z"]], float) if cmap["z"] else np.full(len(tra), np.nan)
            tm = np.asarray(t[cmap["m"]], float)

        tree = cKDTree(unit_vecs(tra, tdec))
        dist, idx = tree.query(unit_vecs(sub.ra.values, sub.dec.values), k=1)
        sep_as = np.degrees(2 * np.arcsin(np.clip(dist / 2, 0, 1))) * 3600.0
        ok = sep_as <= MATCH_AS
        results.append(pd.DataFrame({
            "dr8_id": sub.dr8_id.values[ok], "sep": sep_as[ok],
            "zphot_zou": tz[idx[ok]], "logM_zou": tm[idx[ok]],
        }))
        pd.concat(results, ignore_index=True).drop_duplicates("dr8_id").to_parquet(OUT)
        os.remove(TMP)
        done.add(sh); open(DONE, "a").write(sh + "\n")
        got = pd.concat(results, ignore_index=True).drop_duplicates("dr8_id")
        print(f"  matched {int(ok.sum())}/{len(sub)}  cumulative {len(got)}", flush=True)

    got = pd.concat(results, ignore_index=True).drop_duplicates("dr8_id")
    print(f"DONE: {len(got)}/{len(anc)} anchor galaxies have Zou masses")


if __name__ == "__main__":
    main()
