"""
Faint-extension catalog: 150k unbiased DR10-South galaxies at r = 19-21 from
ls_dr10.tractor (Data Lab TAP), sampled via random_id (verified present) --
the target list for the cutout fleet + the depth-robustness tests (Paper 1 F6).

Selection: 22.5-2.5log10(flux_r) in [19,21] => flux_r in [3.981, 25.119];
type != 'PSF' (galaxies, matching the anchor's non-PSF cut); release != 9011
(south = AION's training footprint). Columns: position, type/release, shapes
(+ivars), griz+W1/W2 fluxes, PSF/depth/EBV covariates, random_id.

Paged by random_id windows (random_id is uniform in [0,100]) until 150k rows.
Run:  python code/anchor/fetchFaint.py          (resumable via parquet append)
"""
import time
import urllib.request
import urllib.parse
import numpy as np
import pandas as pd
from io import StringIO

TAP = "https://datalab.noirlab.edu/tap/sync"
OUT = "data/faintSample.parquet"
TARGET = 150_000
COLS = ("ra dec type release brickname ebv random_id "
        "shape_e1 shape_e2 shape_r shape_e1_ivar shape_e2_ivar shape_r_ivar "
        "flux_g flux_r flux_i flux_z flux_w1 flux_w2 "
        "psfsize_g psfsize_r psfsize_i psfsize_z "
        "psfdepth_g psfdepth_r psfdepth_i psfdepth_z").split()
FLO, FHI = 3.981, 25.119                 # r in [19, 21]


def q(sql, tries=4):
    d = urllib.parse.urlencode({"REQUEST": "doQuery", "LANG": "ADQL",
                                "FORMAT": "csv", "QUERY": sql}).encode()
    last = None
    for k in range(tries):
        try:
            return pd.read_csv(StringIO(urllib.request.urlopen(TAP, data=d, timeout=600).read().decode()))
        except Exception as e:
            last = e; time.sleep(3 * (k + 1))
    raise last


def main():
    try:
        acc = pd.read_parquet(OUT)
        lo = float(acc.random_id.max())
        print(f"resume: {len(acc)} rows, random_id > {lo:.4f}")
    except FileNotFoundError:
        acc, lo = pd.DataFrame(), 0.0
    t0, step = time.time(), 0.004         # adaptive window (sync TAP 504s on big scans)
    while len(acc) < TARGET and lo < 100:
        hi = min(lo + step, 100.0)
        sql = (f"SELECT {','.join(COLS)} FROM ls_dr10.tractor "
               f"WHERE random_id > {lo:.6f} AND random_id <= {hi:.6f} "
               f"AND flux_r BETWEEN {FLO} AND {FHI} AND type <> 'PSF' AND release <> 9011")
        tp = time.time()
        try:
            page = q(sql, tries=1)
        except Exception as e:
            step = max(step / 2, 0.0005)
            print(f"window ({lo:.4f},{hi:.4f}] failed ({type(e).__name__}) -> shrink step to {step:.4f}", flush=True)
            continue
        dt = time.time() - tp
        acc = pd.concat([acc, page], ignore_index=True) if len(acc) else page
        acc.to_parquet(OUT)
        print(f"random_id ({lo:.4f},{hi:.4f}]: +{len(page)} -> {len(acc)} total ({dt:.0f}s page)", flush=True)
        lo = hi
        if dt < 45:
            step = min(step * 1.4, 0.02)  # speed up while the service tolerates it
    acc = acc.iloc[:TARGET].reset_index(drop=True)
    acc["magR"] = 22.5 - 2.5 * np.log10(acc.flux_r.clip(lower=1e-9))
    acc.to_parquet(OUT)
    print(f"\nDONE {len(acc)} rows -> {OUT}; magR p[1,50,99] = "
          f"{np.percentile(acc.magR, [1, 50, 99]).round(2)}", flush=True)


if __name__ == "__main__":
    main()
