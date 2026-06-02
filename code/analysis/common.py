import json
import os
import time
import numpy as np
import pandas as pd
import torch

D = os.environ.get("AION_DATA", "/home/ubuntu/data")
RES = os.environ.get("AION_RESULTS", "/home/ubuntu/results")
os.makedirs(RES, exist_ok=True)
_t0 = time.time()


def log(msg):
    print(f"[{time.time() - _t0:7.1f}s] {msg}", flush=True)


def load():
    ef = np.load(f"{D}/E_full.npy")
    ei = np.load(f"{D}/E_img.npy")
    ok = np.load(f"{D}/ok_index.npy")
    df = pd.read_parquet(f"{D}/sample.parquet").iloc[ok].reset_index(drop=True)
    return ef, ei, ok, df


def save_json(name, obj):
    with open(f"{RES}/{name}.json", "w") as f:
        json.dump(obj, f, indent=2, default=float)
    log(f"saved {name}.json")


def save_npy(name, arr):
    np.save(f"{RES}/{name}.npy", arr)
    log(f"saved {name}.npy {tuple(arr.shape)}")


def knn(X, k, tile=2048):
    """GPU k-nearest-neighbour distances/indices (self excluded), sorted ascending."""
    Xt = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)).cuda()
    n = Xt.shape[0]
    dists = np.empty((n, k), np.float32)
    idxs = np.empty((n, k), np.int64)
    for s in range(0, n, tile):
        dd, ii = torch.cdist(Xt[s:s + tile], Xt).topk(k + 1, largest=False)
        dists[s:s + tile] = dd[:, 1:].cpu().numpy()
        idxs[s:s + tile] = ii[:, 1:].cpu().numpy()
    return dists, idxs
