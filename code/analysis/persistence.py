"""Persistence diagrams for the H1 resolution check.

Uses the same ten paired 2,000-galaxy subsamples for Euclidean, Fermat, and the
alpha=1 self-tuned diffusion metric. Keeps every replicate's maximum persistence
and the worst-case diagram. The old 20-draw, top-300-bar bootstrap was not a
calibrated Fasy confidence set, so it is removed rather than presented as one.
The fixed 0.1 line is retained only as a descriptive reference.
"""
import json
import os
import time
import numpy as np
import ripser
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix, diags
from scipy.sparse.csgraph import shortest_path
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import kneighbors_graph, NearestNeighbors

ROOT = r"d:/AstroML-Project"
RES = os.path.join(ROOT, "results")
N_SUB, N_REP, K, SEED = 2000, 10, 15, 0
THRESH = 0.1            # the fixed significance threshold used in topology.py
METRICS = ["euclidean", "fermat", "diffusion"]
_t0 = time.time()


def log(m):
    print(f"[{time.time()-_t0:7.1f}s] {m}", flush=True)


def zscore(E):
    return ((E - E.mean(0)) / (E.std(0) + 1e-8)).astype(np.float32)


def geodesic(S, k, power):
    G = kneighbors_graph(S, k, mode="distance")
    if power != 1:
        G.data = G.data ** power
    return shortest_path(G.maximum(G.T), directed=False)


def diffusion_dist(S, k):
    dist, idx = NearestNeighbors(n_neighbors=k + 1).fit(S).kneighbors(S)
    dist, idx = dist[:, 1:], idx[:, 1:]
    sigma = dist[:, 6] + 1e-9
    w = np.exp(-(dist ** 2) / (sigma[:, None] * sigma[idx]))
    n = len(S)
    W0 = csr_matrix((w.ravel(), (np.repeat(np.arange(n), k), idx.ravel())),
                    shape=(n, n))
    W = W0.maximum(W0.T)
    q = np.asarray(W.sum(1)).ravel()
    K1 = diags(1.0 / q) @ W @ diags(1.0 / q)
    deg = np.asarray(K1.sum(1)).ravel()
    Di = diags(1.0 / np.sqrt(deg))
    vals, vecs = eigsh((Di @ K1 @ Di).tocsr(), k=40, which="LA")
    o = np.argsort(vals)[::-1]
    psi = np.asarray(Di @ vecs[:, o])
    psi = psi[:, 1:] * vals[o][1:]
    return cdist(psi, psi)


def distmat(S, nm):
    if nm == "euclidean":
        D = cdist(S, S)
    elif nm == "fermat":
        D = geodesic(S, K, 2)
    else:
        D = diffusion_dist(S, K)
    return (D / D.max()).astype(np.float64)


def h1(D):
    dg = ripser.ripser(D, distance_matrix=True, maxdim=1)["dgms"][1]
    return dg if len(dg) else np.zeros((0, 2))


E = zscore(np.load(os.path.join(ROOT, "data", "E_full.npy")))
log(f"loaded E_full z-scored {E.shape}")

# One paired subsample list for all metrics.
rng = np.random.default_rng(SEED)
subs = [rng.choice(len(E), N_SUB, replace=False) for _ in range(N_REP)]

out = {}
for nm in METRICS:
    maxp, sig, nbars = [], [], []
    best_p, best_dg, best_idx = -1.0, None, -1
    for r in range(N_REP):
        S = E[subs[r]]
        dg = h1(distmat(S, nm))
        p = (dg[:, 1] - dg[:, 0]) if len(dg) else np.array([0.0])
        maxp.append(float(p.max())); sig.append(int((p > THRESH).sum())); nbars.append(int(len(dg)))
        if float(p.max()) > best_p:
            best_p, best_dg, best_idx = float(p.max()), dg, r
    np.save(os.path.join(RES, f"persistenceDiag_{nm}.npy"), best_dg.astype(np.float32))
    log(f"{nm}: max_pers={max(maxp):.3f} sig(>0.1)={np.mean(sig):.1f} range={[min(sig),max(sig)]} "
        f"mean_bars={np.mean(nbars):.0f} worst_rep={best_idx} -> diagram saved ({len(best_dg)} bars)")
    out[nm] = dict(
        max_persistence=float(max(maxp)),
        per_rep_max=[float(x) for x in maxp],
        sig_loops_mean=float(np.mean(sig)),
        sig_loops_range=[int(min(sig)), int(max(sig))],
        mean_bars=float(np.mean(nbars)),
        worst_rep=int(best_idx),
        n_bars_worst=int(len(best_dg)),
        subsamples_with_any_over_0p1=int(np.sum(np.asarray(sig) > 0)),
        bottleneck_to_empty=float(best_p / 2.0),
        threshold=THRESH,
        verdict=("descriptive H1 candidates present" if np.any(np.asarray(sig) > 0)
                 else "no candidate exceeds the descriptive 0.1 line"),
    )

out["_design"] = dict(
    paired_subsamples_across_metrics=True,
    seed=SEED, n_rep=N_REP, n_sub=N_SUB,
    diffusion="alpha=1 self-tuned, matching diffusionMap.py",
    inference=("no formal confidence band is claimed; the previous B=20 truncated "
               "bootstrap was removed as uncalibrated"),
)

with open(os.path.join(RES, "persistence.json"), "w") as f:
    json.dump(out, f, indent=2)
log("DONE -> results/persistence.json + results/persistenceDiag_*.npy")
