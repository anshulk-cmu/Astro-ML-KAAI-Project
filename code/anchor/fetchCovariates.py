"""
Bulk survey-condition covariates for the 48,398 anchor galaxies from
ls_dr10.tractor (NOIRLab Astro Data Lab). preRegistration §11.3 / 12.2 / 17.3.

Pulls psfsize_{g,r,i,z}, psfdepth_{g,r,i,z}, ebv, brick, type, fluxes, and the
release code (the footprint flag: 10000 = DR10/DECam South = AION's training
domain; 9011 = DR9 North BASS+MzLS = the instrument-shift negative control).

Method: batched OR-of-bounding-boxes queries (POST) to the public TAP sync
endpoint, then nearest-neighbour match per anchor within 1.44". No q3c, no extra
packages, anonymous. Resumable (checkpoints every batch).
Run:  python code/anchor/fetchCovariates.py --limit 300   # quick check
      python code/anchor/fetchCovariates.py                # full 48k
"""
import argparse
import time
import urllib.request
import urllib.parse
import numpy as np
import pandas as pd
from io import StringIO

TAP = "https://datalab.noirlab.edu/tap/sync"
MATCH_AS = 1.44                 # arcsec
HALF = MATCH_AS / 2 / 3600.0    # box half-width in deg (dec)
BATCH = 150
PARTIAL = "data/anchorCovariates_partial.parquet"
FINAL = "data/anchorCovariates.parquet"
COLS = ("release brickid brickname type ebv "
        "psfsize_g psfsize_r psfsize_i psfsize_z "
        "psfdepth_g psfdepth_r psfdepth_i psfdepth_z "
        "flux_g flux_r flux_i flux_z").split()


def anchor():
    df = pd.read_parquet("data/sample.parquet")
    ok = np.load("data/ok_index.npy")
    d = df.iloc[ok].reset_index(drop=True)
    return pd.DataFrame({
        "dr8_id": d["dr8_id"].astype(str).values,
        "ra": pd.to_numeric(d["ra"], errors="coerce").to_numpy(float),
        "dec": pd.to_numeric(d["dec"], errors="coerce").to_numpy(float),
    })


def query(sub):
    boxes = []
    for ra, dec in zip(sub.ra, sub.dec):
        dra = HALF / max(np.cos(np.radians(dec)), 1e-3)
        boxes.append(f"(ra BETWEEN {ra-dra:.6f} AND {ra+dra:.6f} "
                     f"AND dec BETWEEN {dec-HALF:.6f} AND {dec+HALF:.6f})")
    sql = (f"SELECT ra,dec,{','.join(COLS)} FROM ls_dr10.tractor "
           f"WHERE {' OR '.join(boxes)}")
    data = urllib.parse.urlencode({"REQUEST": "doQuery", "LANG": "ADQL",
                                   "FORMAT": "csv", "QUERY": sql}).encode()
    raw = urllib.request.urlopen(TAP, data=data, timeout=180).read().decode()
    return pd.read_csv(StringIO(raw))


def match(sub, res):
    """nearest tractor row per anchor galaxy within MATCH_AS."""
    if len(res) == 0:
        return pd.DataFrame()
    cd = np.cos(np.radians(sub.dec.values))
    tx, ty = res.ra.values, res.dec.values
    out = []
    for k, (ra, dec, c) in enumerate(zip(sub.ra.values, sub.dec.values, cd)):
        dx = (tx - ra) * c
        sep = np.hypot(dx, ty - dec) * 3600.0
        j = int(np.argmin(sep))
        if sep[j] <= MATCH_AS:
            row = res.iloc[j].to_dict()
            row["sep"] = float(sep[j])
            row["dr8_id"] = sub.dr8_id.values[k]
            out.append(row)
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    a = anchor()
    if args.limit:
        a = a.iloc[:args.limit].reset_index(drop=True)

    done, chunks = set(), []
    try:
        prev = pd.read_parquet(PARTIAL)
        done = set(prev["dr8_id"]); chunks = [prev]
        print(f"resume: {len(done)} done")
    except FileNotFoundError:
        pass

    todo = a[~a["dr8_id"].isin(done)].reset_index(drop=True)
    t0 = time.time()
    for s in range(0, len(todo), BATCH):
        sub = todo.iloc[s:s + BATCH]
        try:
            m = match(sub, query(sub))
        except Exception as e:
            print(f"batch {s} FAILED: {e}"); break
        chunks.append(m)
        pd.concat(chunks, ignore_index=True).to_parquet(PARTIAL)
        if (s // BATCH) % 10 == 0:
            tot = pd.concat(chunks, ignore_index=True)
            print(f"{s+len(sub)}/{len(todo)} matched={len(tot)} "
                  f"({time.time()-t0:.0f}s)")
        time.sleep(0.2)

    full = pd.concat(chunks, ignore_index=True)
    # north == DR9 BASS/MzLS (release 9011); everything else is DECam-south
    # (9010 DECaLS, 9012 DES, 10000/10001/10002 DR10 DECam)
    full["footprint"] = np.where(full["release"] == 9011, "north", "south")
    full.to_parquet(FINAL)
    print(f"DONE {len(full)}/{len(a)} matched; "
          f"south={int((full.footprint=='south').sum())} "
          f"north={int((full.footprint=='north').sum())} "
          f"max_sep={full['sep'].max():.2f}\"")


if __name__ == "__main__":
    main()
