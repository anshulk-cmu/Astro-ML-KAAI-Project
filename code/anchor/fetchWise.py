"""
WISE W1/W2 fluxes for the anchor from ls_dr10.tractor (Data Lab), to build an
AGN-candidate proxy label for Track D (Matt's "AGN signatures" dictionary
candidate; no AGN label exists locally). Proxy = Stern+12 wedge W1-W2 >= 0.8
(Vega); tractor fluxes are AB nanomaggies, converted via the standard offsets
(W1: +2.699, W2: +3.339 => (W1-W2)_Vega = (W1-W2)_AB + 0.64). APPROXIMATE,
flagged as a proxy -- shallow WISE + no k-correction; good enough to ask whether
any dictionary feature tracks it.

Same machinery as fetchShapes.py, with the 4x retry learned from the ageMetal
fetch failure. Run:  python code/anchor/fetchWise.py [--limit 300]
"""
import argparse, time, urllib.request, urllib.parse
import numpy as np, pandas as pd
from io import StringIO

TAP = "https://datalab.noirlab.edu/tap/sync"
MATCH_AS = 1.44
HALF = MATCH_AS / 2 / 3600.0
BATCH = 250
PARTIAL = "data/anchorWise_partial.parquet"
FINAL = "data/anchorWise.parquet"


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
    sql = ("SELECT ra,dec,flux_w1,flux_w2 FROM ls_dr10.tractor "
           f"WHERE {' OR '.join(boxes)}")
    data = urllib.parse.urlencode({"REQUEST": "doQuery", "LANG": "ADQL",
                                   "FORMAT": "csv", "QUERY": sql}).encode()
    last = None
    for k in range(4):
        try:
            raw = urllib.request.urlopen(TAP, data=data, timeout=300).read().decode()
            return pd.read_csv(StringIO(raw))
        except Exception as e:
            last = e; time.sleep(2 * (k + 1))
    raise last


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
            row["dr8_id"] = sub.dr8_id.values[k]
            out.append(row)
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    a = anchor()
    if args.limit:
        a = a.iloc[:args.limit].reset_index(drop=True)

    done, chunks = set(), []
    try:
        prev = pd.read_parquet(PARTIAL); done = set(prev.dr8_id); chunks = [prev]
        print(f"resume: {len(done)} done", flush=True)
    except FileNotFoundError:
        pass

    todo = a[~a.dr8_id.isin(done)].reset_index(drop=True); t0 = time.time()
    for s in range(0, len(todo), BATCH):
        sub = todo.iloc[s:s + BATCH]
        try:
            m = match(sub, query(sub))
        except Exception as e:
            print(f"batch {s} FAILED after retries: {e}", flush=True); break
        chunks.append(m); pd.concat(chunks, ignore_index=True).to_parquet(PARTIAL)
        if (s // BATCH) % 10 == 0:
            print(f"{s+len(sub)}/{len(todo)} ({time.time()-t0:.0f}s)", flush=True)
        time.sleep(0.2)

    full = pd.concat(chunks, ignore_index=True)
    f1, f2 = full.flux_w1.to_numpy(float), full.flux_w2.to_numpy(float)
    okf = (f1 > 0) & (f2 > 0)
    # m = 22.5 - 2.5 log10(f)  =>  (W1-W2)_AB = 2.5 log10(f2/f1); Vega offset +0.64
    full["w1w2Vega"] = np.where(okf, 2.5 * np.log10(np.where(okf, f2, 1) / np.where(okf, f1, 1)) + 0.64, np.nan)
    full["agnProxy"] = full["w1w2Vega"] > 0.8
    full.to_parquet(FINAL)
    n_ok = int(np.isfinite(full.w1w2Vega).sum())
    print(f"\nDONE {len(full)}/{len(a)} matched; finite W1-W2: {n_ok}; "
          f"AGN-proxy (Stern wedge): {int(full.agnProxy.sum())}", flush=True)


if __name__ == "__main__":
    main()
