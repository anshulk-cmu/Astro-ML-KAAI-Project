"""
Track 1 (preRegistration §5.3 PRIMARY): independent SPS stellar masses for the
anchor from the Zou et al. 2022 DESI DR9 photo-z + stellar-mass catalog
(china-vo paperdata; DOI 10.12149/101090). ~320M galaxies, log M* sigma ~0.2 dex,
full z<1, both hemispheres.

Stream-crossmatch: discover the RA-range shards, and for each shard download ->
read ra/dec/photoz/mass -> nearest-match the anchor galaxies in that RA range
(within 1") -> keep -> DELETE the shard. Peak disk ~ one shard.

RUN ON VERA (astropy + bandwidth; ~78 GB total transfer, deleted as it goes).
Resumable: shards already in the done-list are skipped.
Needs: astropy, scipy, pandas, numpy, requests. Anchor file: data/anchorRADec.csv
(dr8_id,ra,dec) — export it from sample.parquet[ok_index] before running.
"""
import os
import re
import sys
import urllib.request
import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.spatial import cKDTree

BASE = "https://paperdata.china-vo.org/zouhu/pz_cluster/photoz_desidr9/"
TMP = "zou_shard.fits"
OUT = "data/anchorMassZou.parquet"
DONE = "data/zouDone.txt"
MATCH_AS = 1.0
RA_CANDS = ["ra", "RA", "raj2000", "RAJ2000"]
DEC_CANDS = ["dec", "DEC", "dej2000", "DEJ2000", "decl"]
Z_CANDS = ["photoz", "photo_z", "z_phot", "zphot", "zphot_best", "z_best", "zp"]
M_CANDS = ["logmass", "logmass_best", "mass", "mass_best", "mass_median",
           "logm", "mstar", "logmstar", "stellar_mass", "logms"]


def pick(cols, cands):
    low = {c.lower(): c for c in cols}
    for c in cands:
        if c.lower() in low:
            return low[c.lower()]
    return None


def unit_vecs(ra, dec):
    r, d = np.radians(ra), np.radians(dec)
    return np.column_stack([np.cos(d) * np.cos(r), np.cos(d) * np.sin(r), np.sin(d)])


def main():
    anc = pd.read_csv("data/anchorRADec.csv")
    listing = urllib.request.urlopen(BASE, timeout=120).read().decode("latin1")
    shards = sorted(set(re.findall(r"desidr9_galaxy_cspcat_ra\d+_\d+\.fits", listing)))
    print(f"discovered {len(shards)} shards; anchor n={len(anc)}")
    if not shards:
        sys.exit("no shards parsed from listing - check BASE / page format")

    done = set(open(DONE).read().split()) if os.path.exists(DONE) else set()
    results = [pd.read_parquet(OUT)] if os.path.exists(OUT) else []
    cmap = None

    for sh in shards:
        if sh in done:
            continue
        lo, hi = map(int, re.findall(r"ra(\d+)_(\d+)", sh)[0])
        sub = anc[(anc.ra >= lo - 0.02) & (anc.ra < hi + 0.02)]
        if len(sub) == 0:
            done.add(sh); open(DONE, "a").write(sh + "\n"); continue

        print(f"{sh}: anchor-in-range {len(sub)} ... downloading")
        urllib.request.urlretrieve(BASE + sh, TMP)
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
        rad = 2 * np.sin(np.radians(MATCH_AS / 3600.0) / 2)
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
        print(f"  matched {ok.sum()}/{len(sub)}  cumulative {len(got)}")

    got = pd.concat(results, ignore_index=True).drop_duplicates("dr8_id")
    print(f"DONE: {len(got)}/{len(anc)} anchor galaxies have Zou masses")


if __name__ == "__main__":
    main()
