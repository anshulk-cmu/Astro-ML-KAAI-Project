"""
DESI DR1 FastSpecFit VAC (iron v3.0, main-bright): download the 12 healpix
catalog FITS files (raws kept under data/vac/fastspecfit/), then crossmatch
against (a) the 48k anchor and (b) the faint sample (if present) by position.
Keeps AGE, ZZSUN, LOGMSTAR, SFR (+_IVAR each) + DN4000 -- columns VERIFIED
present in HDU2 (FASTSPEC) by header range-read before any download.

Outputs: data/anchorFastspec.parquet (+ data/faintFastspec.parquet if the faint
catalog exists). Resumable (skips downloaded files).
Run:  python code/anchor/fetchFastspec.py
"""
import os
import time
import urllib.request
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u

BASE = "https://data.desi.lbl.gov/public/dr1/vac/dr1/fastspecfit/iron/v3.0/catalogs"
FILES = [f"fastspec-iron-main-bright-nside1-hp{h:02d}.fits" for h in range(12)]
RAW = "data/vac/fastspecfit"
KEEP = ["AGE", "AGE_IVAR", "ZZSUN", "ZZSUN_IVAR", "LOGMSTAR", "LOGMSTAR_IVAR",
        "SFR", "SFR_IVAR", "DN4000", "DN4000_IVAR"]
MATCH_AS = 1.5
os.makedirs(RAW, exist_ok=True)


def download():
    for f in FILES:
        dst = f"{RAW}/{f}"
        if os.path.exists(dst) and os.path.getsize(dst) > 1e6:
            continue
        t0 = time.time()
        for att in range(4):
            try:
                urllib.request.urlretrieve(f"{BASE}/{f}", dst)
                print(f"{f}: {os.path.getsize(dst)/1e6:.0f} MB in {time.time()-t0:.0f}s", flush=True)
                break
            except Exception as e:
                print(f"{f} attempt {att}: {e}", flush=True)
                time.sleep(10 * (att + 1))


def crossmatch(targets, tname):
    """targets: DataFrame with dr8-style id column 'key', ra, dec (deg)."""
    tc = SkyCoord(targets.ra.to_numpy(float) * u.deg, targets.dec.to_numpy(float) * u.deg)
    outs = []
    for f in FILES:
        with fits.open(f"{RAW}/{f}", memmap=True) as hd:
            meta = hd[1].data                      # METADATA: RA, DEC, TARGETID
            spec = hd[2].data                      # FASTSPEC: AGE, ZZSUN, ...
            sc = SkyCoord(np.asarray(meta["RA"], float) * u.deg,
                          np.asarray(meta["DEC"], float) * u.deg)
            idx, sep, _ = tc.match_to_catalog_sky(sc)
            m = sep.arcsec <= MATCH_AS
            if m.sum() == 0:
                continue
            rows = {"key": targets.key.values[m], "sep": sep.arcsec[m],
                    "TARGETID": np.asarray(meta["TARGETID"])[idx[m]],
                    "Z": np.asarray(meta["Z"], float)[idx[m]] if "Z" in meta.names else np.nan}
            for c in KEEP:
                rows[c.lower()] = np.asarray(spec[c], float)[idx[m]]
            outs.append(pd.DataFrame(rows))
            print(f"  {f}: {int(m.sum())} matches for {tname}", flush=True)
    if not outs:
        print(f"no matches for {tname}", flush=True)
        return
    df = pd.concat(outs, ignore_index=True).sort_values("sep").drop_duplicates("key")
    df.to_parquet(f"data/{tname}Fastspec.parquet")
    print(f"{tname}: {len(df)} unique matches -> data/{tname}Fastspec.parquet", flush=True)


def main():
    download()
    df = pd.read_parquet("data/sample.parquet")
    ok = np.load("data/ok_index.npy")
    d = df.iloc[ok].reset_index(drop=True)
    anchor = pd.DataFrame({"key": d["dr8_id"].astype(str).values,
                           "ra": pd.to_numeric(d["ra"], errors="coerce").to_numpy(float),
                           "dec": pd.to_numeric(d["dec"], errors="coerce").to_numpy(float)}).dropna()
    crossmatch(anchor, "anchor")
    if os.path.exists("data/faintSample.parquet"):
        ft = pd.read_parquet("data/faintSample.parquet").reset_index()
        ft["key"] = ft["index"].astype(str)
        crossmatch(ft[["key", "ra", "dec"]], "faint")
    else:
        print("(faintSample.parquet not present yet -- rerun later for the faint match)", flush=True)


if __name__ == "__main__":
    main()
