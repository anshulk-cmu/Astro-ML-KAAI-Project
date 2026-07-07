"""
Stellar AGE + METALLICITY for the anchor from the SDSS DR16 eBOSS Firefly VAC
(sdss_dr16.sdssebossfirefly, full-spectrum fitting, Chabrier IMF + MILES models),
for Track C's age-metallicity degeneracy angle. Positional crossmatch to the
anchor, same machinery as fetchShapes.py / fetchCovariates.py.

Pulls light- and mass-weighted age and metallicity ([Z/H]). Coverage is the
spectroscopic subset (galaxies with SDSS/BOSS/eBOSS spectra), like Phase 1's
sparse mass/sSFR probes -- enough to fit two probe directions and measure the angle.

Run:  python code/anchor/fetchAgeMetal.py --limit 300   # quick check + overlap
      python code/anchor/fetchAgeMetal.py                # full anchor (resumable)
"""
import argparse, time, urllib.request, urllib.parse
import numpy as np, pandas as pd
from io import StringIO

TAP = "https://datalab.noirlab.edu/tap/sync"
TABLE = "sdss_dr16.sdssebossfirefly"
MATCH_AS = 1.5
HALF = MATCH_AS / 2 / 3600.0
BATCH = 250
PARTIAL = "data/anchorAgeMetal_partial.parquet"
FINAL = "data/anchorAgeMetal.parquet"
COLS = ("chabrier_miles_age_lightw chabrier_miles_metallicity_lightw "
        "chabrier_miles_age_massw chabrier_miles_metallicity_massw").split()
SHORT = {"chabrier_miles_age_lightw": "ageLightW", "chabrier_miles_metallicity_lightw": "metalLightW",
         "chabrier_miles_age_massw": "ageMassW", "chabrier_miles_metallicity_massw": "metalMassW"}


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
    sql = f"SELECT ra,dec,{','.join(COLS)} FROM {TABLE} WHERE {' OR '.join(boxes)}"
    data = urllib.parse.urlencode({"REQUEST": "doQuery", "LANG": "ADQL",
                                   "FORMAT": "csv", "QUERY": sql}).encode()
    last = None
    for k in range(4):                                   # retry transient network timeouts
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
            row["sep"] = float(sep[j]); row["dr8_id"] = sub.dr8_id.values[k]
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
            print(f"{s+len(sub)}/{len(todo)} matched={sum(len(x) for x in chunks)} ({time.time()-t0:.0f}s)")
        time.sleep(0.2)

    full = pd.concat(chunks, ignore_index=True).rename(columns=SHORT)
    for cnew in SHORT.values():                              # -9999 fill -> NaN
        full[cnew] = np.where(full[cnew].to_numpy(float) < -50, np.nan, full[cnew])
    full.to_parquet(FINAL)
    n_ok = int(np.isfinite(full["ageLightW"]).sum())
    print(f"\nDONE {len(full)}/{len(a)} matched -> {FINAL}; finite age+metal: {n_ok}")
    if n_ok:
        print(f"  ageLightW range [{full.ageLightW.min():.2f},{full.ageLightW.max():.2f}], "
              f"metalLightW range [{full.metalLightW.min():.2f},{full.metalLightW.max():.2f}]")


if __name__ == "__main__":
    main()
