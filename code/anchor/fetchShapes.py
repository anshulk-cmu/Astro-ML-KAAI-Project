"""
Light-profile shape parameters for the 48,398 anchor galaxies from
ls_dr10.tractor (NOIRLab Data Lab), for Track A: testing whether AION encodes
orientation angles (position angle, inclination) as closed loops in E_img.

Pulls shape_e1, shape_e2, shape_r (+ ivars) and morphological type by positional
match, exactly like fetchCovariates.py. Derives position angle, ellipticity, axis
ratio b/a, inclination, and an ellipticity signal-to-noise for quality cuts.

Run:  python code/anchor/fetchShapes.py --limit 300   # quick check
      python code/anchor/fetchShapes.py                # full 48k (~20 min, resumable)
"""
import argparse, time, urllib.request, urllib.parse
import numpy as np, pandas as pd
from io import StringIO

TAP = "https://datalab.noirlab.edu/tap/sync"
MATCH_AS = 1.44
HALF = MATCH_AS / 2 / 3600.0
BATCH = 150
PARTIAL = "data/anchorShapes_partial.parquet"
FINAL = "data/anchorShapes.parquet"
COLS = "type shape_e1 shape_e2 shape_r shape_e1_ivar shape_e2_ivar".split()


def anchor():
    df = pd.read_parquet("data/sample.parquet")
    ok = np.load("data/ok_index.npy")
    d = df.iloc[ok].reset_index(drop=True)
    return pd.DataFrame({"dr8_id": d["dr8_id"].astype(str).values,
                         "ra": pd.to_numeric(d["ra"], errors="coerce").to_numpy(float),
                         "dec": pd.to_numeric(d["dec"], errors="coerce").to_numpy(float)})


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
    if len(res) == 0:
        return pd.DataFrame()
    cd = np.cos(np.radians(sub.dec.values)); tx, ty = res.ra.values, res.dec.values
    out = []
    for k, (ra, dec, c) in enumerate(zip(sub.ra.values, sub.dec.values, cd)):
        sep = np.hypot((tx - ra) * c, ty - dec) * 3600.0
        j = int(np.argmin(sep))
        if sep[j] <= MATCH_AS:
            row = res.iloc[j].to_dict()
            row["sep"] = float(sep[j]); row["dr8_id"] = sub.dr8_id.values[k]
            out.append(row)
    return pd.DataFrame(out)


def derive(df):
    """position angle [0,180), axis ratio, inclination, ellipticity SNR."""
    e1 = df.shape_e1.to_numpy(float); e2 = df.shape_e2.to_numpy(float)
    e = np.hypot(e1, e2)
    df["ellip"] = e
    df["ba"] = (1 - e) / (1 + e)                                    # axis ratio b/a
    df["paDeg"] = (0.5 * np.degrees(np.arctan2(e2, e1))) % 180.0    # position angle, mod 180
    df["inclDeg"] = np.degrees(np.arccos(np.clip(df["ba"].to_numpy(float), 0, 1)))
    s1 = 1.0 / np.sqrt(np.clip(df.shape_e1_ivar.to_numpy(float), 1e-30, None))
    s2 = 1.0 / np.sqrt(np.clip(df.shape_e2_ivar.to_numpy(float), 1e-30, None))
    df["ellipSnr"] = e / np.clip(np.hypot(s1, s2), 1e-9, None)      # how well PA is measured
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    a = anchor()
    if args.limit:
        a = a.iloc[:args.limit].reset_index(drop=True)

    done, chunks = set(), []
    try:
        prev = pd.read_parquet(PARTIAL); done = set(prev.dr8_id); chunks = [prev]
        print(f"resume: {len(done)} done")
    except FileNotFoundError:
        pass

    todo = a[~a.dr8_id.isin(done)].reset_index(drop=True); t0 = time.time()
    for s in range(0, len(todo), BATCH):
        sub = todo.iloc[s:s + BATCH]
        try:
            m = match(sub, query(sub))
        except Exception as e:
            print(f"batch {s} FAILED: {e}"); break
        chunks.append(m); pd.concat(chunks, ignore_index=True).to_parquet(PARTIAL)
        if (s // BATCH) % 10 == 0:
            print(f"{s+len(sub)}/{len(todo)} ({time.time()-t0:.0f}s)")
        time.sleep(0.2)

    full = derive(pd.concat(chunks, ignore_index=True))
    full.to_parquet(FINAL)
    print(f"\nDONE {len(full)}/{len(a)} matched -> {FINAL}")
    print(f"  ellip>0.3 (well-defined PA): {int((full.ellip > 0.3).sum())}")
    print(f"  ellipSnr>5: {int((full.ellipSnr > 5).sum())}")


if __name__ == "__main__":
    main()
