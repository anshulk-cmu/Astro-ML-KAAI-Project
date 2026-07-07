"""
Anchor spectra via SPARCL (NOIRLab): crossmatch the FULL 48k anchor against
sparcl.main (TAP), then retrieve every matching spectrum with sparclclient.
Covers SDSS-DR17 + BOSS-DR17 + DESI-EDR + DESI-DR1 in one service (verified via
tap_schema). The manifest's redshift column doubles as the DESI spec-z upgrade.

Stage 1 -> data/anchorSparclManifest.parquet (dr8_id, sparcl_id, specid,
data_release, redshift, redshift_warning, spectype, specprimary, sep)
Stage 2 -> data/anchorSpectra/shard_####.npz (sparcl_id, flux, wavelength, ivar)
Both stages resumable.  Run: python code/anchor/fetchSparclSpectra.py
"""
import os
import time
import urllib.request
import urllib.parse
import numpy as np
import pandas as pd
from io import StringIO

TAP = "https://datalab.noirlab.edu/tap/sync"
MATCH_AS = 1.5
HALF = MATCH_AS / 2 / 3600.0
BATCH = 50                                # sync TAP 504s on heavy OR-of-box queries
MANIFEST = "data/anchorSparclManifest.parquet"
MPARTIAL = "data/anchorSparclManifest_partial.parquet"
OUTDIR = "data/anchorSpectra"
SHARD = 500                    # DO NOT change: shard index = resume key
os.makedirs(OUTDIR, exist_ok=True)


def anchor():
    df = pd.read_parquet("data/sample.parquet")
    ok = np.load("data/ok_index.npy")
    d = df.iloc[ok].reset_index(drop=True)
    return pd.DataFrame({"dr8_id": d["dr8_id"].astype(str).values,
                         "ra": pd.to_numeric(d["ra"], errors="coerce").to_numpy(float),
                         "dec": pd.to_numeric(d["dec"], errors="coerce").to_numpy(float)})


def q(sql, tries=4):
    d = urllib.parse.urlencode({"REQUEST": "doQuery", "LANG": "ADQL",
                                "FORMAT": "csv", "QUERY": sql}).encode()
    last = None
    for k in range(tries):
        try:
            return pd.read_csv(StringIO(urllib.request.urlopen(TAP, data=d, timeout=300).read().decode()))
        except Exception as e:
            last = e; time.sleep(3 * (k + 1))
    raise last


def stage1():
    a = anchor()
    done, chunks = set(), []
    try:
        prev = pd.read_parquet(MPARTIAL); done = set(prev.dr8_id); chunks = [prev]
        print(f"stage1 resume: {len(done)} anchors done", flush=True)
    except FileNotFoundError:
        pass
    todo = a[~a.dr8_id.isin(done)].reset_index(drop=True)
    t0 = time.time()

    def boxq(sub):
        boxes = []
        for ra, dec in zip(sub.ra, sub.dec):
            dra = HALF / max(np.cos(np.radians(dec)), 1e-3)
            boxes.append(f"(ra BETWEEN {ra-dra:.6f} AND {ra+dra:.6f} "
                         f"AND dec BETWEEN {dec-HALF:.6f} AND {dec+HALF:.6f})")
        sql = ("SELECT ra,dec,sparcl_id,specid,data_release,redshift,redshift_warning,"
               "spectype,specprimary FROM sparcl.main WHERE specprimary=1 AND "
               f"({' OR '.join(boxes)})")
        return q(sql)

    for s in range(0, len(todo), BATCH):
        sub = todo.iloc[s:s + BATCH]
        try:
            res = boxq(sub)
        except Exception:                              # split, never silently skip
            parts = []
            for k in range(0, len(sub), 17):
                parts.append(boxq(sub.iloc[k:k + 17]))
                time.sleep(1)
            res = pd.concat(parts, ignore_index=True)
        rows = []
        if len(res):
            tx, ty = res.ra.to_numpy(float), res.dec.to_numpy(float)
            for k, (ra, dec) in enumerate(zip(sub.ra.values, sub.dec.values)):
                sep = np.hypot((tx - ra) * np.cos(np.radians(dec)), ty - dec) * 3600.0
                near = np.where(sep <= MATCH_AS)[0]
                for j in near:                       # keep ALL releases per galaxy
                    r = res.iloc[j].to_dict(); r["sep"] = float(sep[j])
                    r["dr8_id"] = sub.dr8_id.values[k]
                    rows.append(r)
        # mark the whole batch done even if no match (dr8_id sentinel rows with no sparcl_id)
        matched_ids = {r["dr8_id"] for r in rows}
        for d_id in sub.dr8_id.values:
            if d_id not in matched_ids:
                rows.append({"dr8_id": d_id, "sparcl_id": None})
        chunks.append(pd.DataFrame(rows))
        pd.concat(chunks, ignore_index=True).to_parquet(MPARTIAL)
        if (s // BATCH) % 20 == 0:
            tot = pd.concat(chunks, ignore_index=True)
            print(f"stage1 {s+len(sub)}/{len(todo)} matches={int(tot.sparcl_id.notna().sum())} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        time.sleep(0.15)
    man = pd.concat(chunks, ignore_index=True)
    man = man[man.sparcl_id.notna()].reset_index(drop=True)
    man.to_parquet(MANIFEST)
    print(f"stage1 DONE: {len(man)} spectra for {man.dr8_id.nunique()} galaxies; by release:", flush=True)
    print(man.data_release.value_counts().to_string(), flush=True)
    return man


def stage2(man):
    import sparcl.client as sc_mod
    sc_mod.MAX_CONNECT_TIMEOUT = 60.0        # library hard-clamps to 3.1s; slow TLS handshakes need more
    from sparcl.client import SparclClient
    client = SparclClient(connect_timeout=60, read_timeout=5400)
    for attr in dir(client):              # belt-and-braces: some paths ignore ctor timeouts
        if "timeout" in attr.lower() and isinstance(getattr(client, attr, None), (int, float)):
            setattr(client, attr, max(getattr(client, attr), 300.0))
    ids = man.sparcl_id.astype(str).tolist()
    for s in range(0, len(ids), SHARD):
        out = f"{OUTDIR}/shard_{s//SHARD:04d}.npz"
        if os.path.exists(out):
            continue
        batch = ids[s:s + SHARD]
        for att in range(4):
            try:
                res = client.retrieve(uuid_list=batch, include=["sparcl_id", "flux", "wavelength", "ivar"],
                                      dataset_list=None)
                recs = res.records
                np.savez_compressed(out,
                                    sparcl_id=np.array([r.sparcl_id for r in recs]),
                                    flux=np.array([np.asarray(r.flux, dtype=np.float32) for r in recs], dtype=object),
                                    wavelength=np.array([np.asarray(r.wavelength, dtype=np.float32) for r in recs], dtype=object),
                                    ivar=np.array([np.asarray(r.ivar, dtype=np.float32) for r in recs], dtype=object),
                                    allow_pickle=True)
                print(f"stage2 shard {s//SHARD}: {len(recs)}/{len(batch)} spectra", flush=True)
                break
            except Exception as e:
                print(f"stage2 shard {s//SHARD} attempt {att}: {e}", flush=True)
                time.sleep(5 * (att + 1))
    print("stage2 DONE", flush=True)


if __name__ == "__main__":
    if os.path.exists(MANIFEST):
        man = pd.read_parquet(MANIFEST)
        print(f"manifest exists: {len(man)} spectra", flush=True)
    else:
        man = stage1()
    stage2(man)
