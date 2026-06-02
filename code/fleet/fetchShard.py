import os
import io
import time
import subprocess
import numpy as np
import pandas as pd
import requests
from astropy.io import fits
from concurrent.futures import ThreadPoolExecutor, as_completed

SHARD, TOTAL = int(os.environ["SHARD"]), int(os.environ["TOTAL"])
WORKERS = int(os.environ.get("WORKERS", "8"))
CKPT = int(os.environ.get("CKPT", "250"))          # upload to S3 every CKPT images
BUCKET = os.environ.get("BUCKET", "")              # if set, fetch_shard self-uploads
SIZE, PIX, LAYER = 96, 0.262, "ls-dr10"
URL = "https://www.legacysurvey.org/viewer/fits-cutout"
AWS = "/usr/local/bin/aws"

df = pd.read_parquet("sample.parquet")
N = len(df)
a, b = SHARD * N // TOTAL, (SHARD + 1) * N // TOTAL
sub = df.iloc[a:b].reset_index(drop=True)
n = len(sub)
IMG, ST = f"images_{SHARD}.npy", f"status_{SHARD}.npy"
imgs = np.lib.format.open_memmap(IMG, mode="w+", dtype=np.float32, shape=(n, 4, SIZE, SIZE))
status = np.zeros(n, dtype=np.int8)


def push():
    imgs.flush()
    np.save(ST, status)
    if BUCKET:
        subprocess.run([AWS, "s3", "cp", IMG, f"s3://{BUCKET}/shards/{IMG}"], check=False)
        subprocess.run([AWS, "s3", "cp", ST, f"s3://{BUCKET}/shards/{ST}"], check=False)


def fetch(i, ra, dec):
    p = dict(ra=ra, dec=dec, layer=LAYER, pixscale=PIX, bands="griz", size=SIZE)
    for at in range(5):
        try:
            r = requests.get(URL, params=p, timeout=30)
            if r.status_code == 200 and r.content[:6] == b"SIMPLE":
                with fits.open(io.BytesIO(r.content)) as hd:
                    arr = hd[0].data
                if arr is not None and arr.shape == (4, SIZE, SIZE):
                    return i, np.nan_to_num(arr.astype(np.float32)), 1
                return i, None, 2
            if r.status_code == 429:
                time.sleep(2 ** at)
                continue
            return i, None, 2
        except Exception:
            time.sleep(1.5 * (at + 1))
    return i, None, 2


t0, done = time.time(), 0
with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    futs = [ex.submit(fetch, i, r.ra, r.dec) for i, r in enumerate(sub.itertuples())]
    for f in as_completed(futs):
        i, arr, st = f.result()
        status[i] = st
        if st == 1:
            imgs[i] = arr
        done += 1
        if done % CKPT == 0:
            push()
            print(f"shard {SHARD} {done}/{n} ok={int((status == 1).sum())} {time.time() - t0:.0f}s", flush=True)
push()
print(f"SHARD {SHARD} DONE rows[{a}:{b}] n={n} ok={int((status == 1).sum())} time={time.time() - t0:.0f}s", flush=True)
