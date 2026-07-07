"""
Concatenate the faint-extension fleet shards into faintImages.npy / faintStatus.npy,
aligned to data/faintSample.parquet row order (local adaptation of code/fleet/gather.py:
shard s holds rows [s*N//TOTAL : (s+1)*N//TOTAL]).

Run:  python code/anchor/gatherFaint.py
"""
import os
import numpy as np
import pandas as pd

SHARDS, TOTAL, SIZE = "data/faintShards", 48, 96

df = pd.read_parquet("data/faintSample.parquet")
N = len(df)
images = np.lib.format.open_memmap("data/faintImages.npy", mode="w+",
                                   dtype=np.float32, shape=(N, 4, SIZE, SIZE))
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
    status[a:b] = np.load(sp) if os.path.exists(sp) else \
        np.where(np.abs(im).reshape(b - a, -1).max(1) > 0, 1, 2).astype(np.int8)
    print(f"shard {s:>2} rows[{a}:{b}] ok={int((status[a:b] == 1).sum())}/{b - a}", flush=True)

images.flush()
np.save("data/faintStatus.npy", status)
ok = int((status == 1).sum())
print(f"DONE faintImages.npy {images.shape} ok={ok}/{N} ({100 * ok / N:.1f}%) missing={missing}")
