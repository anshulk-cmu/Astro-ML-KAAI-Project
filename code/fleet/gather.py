"""Concatenate fleet shards into images.npy / status.npy, aligned to sample.parquet.

Run on the GPU box after `aws s3 cp --recursive s3://<bucket>/shards/ <D>/shards/`.
Shard s holds rows [s*N//TOTAL : (s+1)*N//TOTAL] of sample.parquet, so stacking shards
0..TOTAL-1 in order reproduces the exact sample.parquet row order. A missing shard is left
as zeros (status 0) and reported, so a partial run still gathers everything that landed.

Usage:  python gather.py <TOTAL>   (default 48)
"""
import os
import sys
import numpy as np
import pandas as pd

D = "/home/ubuntu/data"
SHARDS = f"{D}/shards"
SIZE = 96
TOTAL = int(sys.argv[1]) if len(sys.argv) > 1 else 48

df = pd.read_parquet(f"{D}/sample.parquet")
N = len(df)

images = np.lib.format.open_memmap(f"{D}/images.npy", mode="w+", dtype=np.float32, shape=(N, 4, SIZE, SIZE))
status = np.zeros(N, dtype=np.int8)
missing = []
for s in range(TOTAL):
    a, b = s * N // TOTAL, (s + 1) * N // TOTAL
    ip, sp = f"{SHARDS}/images_{s}.npy", f"{SHARDS}/status_{s}.npy"
    if not os.path.exists(ip):
        missing.append(s)
        continue
    im = np.load(ip, mmap_mode="r")
    assert im.shape == (b - a, 4, SIZE, SIZE), (s, im.shape, b - a)
    images[a:b] = im[:]
    if os.path.exists(sp):
        status[a:b] = np.load(sp)
    else:  # derive: ok where the image has any nonzero pixel
        status[a:b] = np.where(np.abs(im).reshape(b - a, -1).max(1) > 0, 1, 2).astype(np.int8)
    print(f"shard {s:>2} rows[{a}:{b}] ok={int((status[a:b] == 1).sum())}/{b - a}", flush=True)

images.flush()
np.save(f"{D}/status.npy", status)
ok = int((status == 1).sum())
print(f"DONE images.npy {images.shape} ok={ok}/{N} ({100 * ok / N:.1f}%) missing_shards={missing}")
