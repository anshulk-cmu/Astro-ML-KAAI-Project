"""Persistence diagrams for the beta_1 check, under Euclidean / Fermat / diffusion metrics.

Reproduces topology.py's betti1 exactly (same seed, same 10 x 2000 subsamples, same draw order,
same three metrics, same diameter normalisation) but KEEPS the H1 birth-death diagram instead of
discarding it. For each metric it also estimates a Fasy/Chazal bootstrap 95% confidence band c_n on
the worst-case (max-persistence) subsample, so the noise band is statistical, not just the fixed 0.1
threshold. Saves the worst-case diagram per metric (.npy) and a summary (.json). Plotting is separate.
"""
import json
import os
import time
import numpy as np
import ripser
from persim import bottleneck
from scipy.spatial.distance import cdist
from scipy.sparse import diags
from scipy.sparse.csgraph import shortest_path
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import kneighbors_graph

ROOT = r"d:/AstroML-Project"
RES = os.path.join(ROOT, "results")
N_SUB, N_REP, K, SEED = 2000, 10, 15, 0
THRESH = 0.1            # the fixed significance threshold used in topology.py
B_BOOT = 20            # bootstrap resamples for the confidence band
TOPK_BOTTLE = 300      # cap bars by persistence for the bottleneck (near-diagonal bars are negligible)
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
    G = kneighbors_graph(S, k, mode="distance"); G = G.maximum(G.T)
    eps = np.median(G.data) ** 2
    W = G.copy(); W.data = np.exp(-(G.data ** 2) / eps)
    Di = diags(1.0 / np.sqrt(np.asarray(W.sum(1)).ravel()))
    vals, vecs = eigsh((Di @ W @ Di).tocsr(), k=40, which="LM")
    o = np.argsort(vals)[::-1]
    psi = vecs[:, o][:, 1:] * vals[o][1:]
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


def topk(dg, k=TOPK_BOTTLE):
    if len(dg) <= k:
        return dg
    p = dg[:, 1] - dg[:, 0]
    return dg[np.argsort(p)[::-1][:k]]


def bootstrap_cn(S, ref_dg, nm, rng):
    """Fasy/Chazal bootstrap: resample the cloud with replacement, recompute the diagram,
    take the 95th percentile of the bottleneck distance to the reference diagram = c_n.
    A feature is significant if its distance to the diagonal exceeds c_n, i.e. persistence > 2 c_n."""
    ref = topk(ref_dg)
    n = len(S)
    ds = []
    for b in range(B_BOOT):
        Sb = S[rng.integers(0, n, n)]
        dgb = topk(h1(distmat(Sb, nm)))
        ds.append(float(bottleneck(ref, dgb)))
    return float(np.percentile(ds, 95)), ds


E = zscore(np.load(os.path.join(ROOT, "data", "E_full.npy")))
log(f"loaded E_full z-scored {E.shape}")

# reproduce topology.py's exact subsample order: euclidean x10, then fermat x10, then diffusion x10
rng = np.random.default_rng(SEED)
subs = {nm: [rng.choice(len(E), N_SUB, replace=False) for _ in range(N_REP)] for nm in METRICS}

out = {}
for nm in METRICS:
    maxp, sig, nbars = [], [], []
    best_p, best_dg, best_idx, best_S = -1.0, None, -1, None
    for r in range(N_REP):
        S = E[subs[nm][r]]
        dg = h1(distmat(S, nm))
        p = (dg[:, 1] - dg[:, 0]) if len(dg) else np.array([0.0])
        maxp.append(float(p.max())); sig.append(int((p > THRESH).sum())); nbars.append(int(len(dg)))
        if float(p.max()) > best_p:
            best_p, best_dg, best_idx, best_S = float(p.max()), dg, r, S
    np.save(os.path.join(RES, f"persistenceDiag_{nm}.npy"), best_dg.astype(np.float32))
    log(f"{nm}: max_pers={max(maxp):.3f} sig(>0.1)={np.mean(sig):.1f} range={[min(sig),max(sig)]} "
        f"mean_bars={np.mean(nbars):.0f} worst_rep={best_idx} -> diagram saved ({len(best_dg)} bars)")
    cn, ds = bootstrap_cn(best_S, best_dg, nm, rng)
    log(f"{nm}: bootstrap c_n(95%)={cn:.4f} -> band persistence={2*cn:.4f} (over B={B_BOOT})")
    out[nm] = dict(
        max_persistence=float(max(maxp)),
        per_rep_max=[float(x) for x in maxp],
        sig_loops_mean=float(np.mean(sig)),
        sig_loops_range=[int(min(sig)), int(max(sig))],
        mean_bars=float(np.mean(nbars)),
        worst_rep=int(best_idx),
        n_bars_worst=int(len(best_dg)),
        bootstrap_cn=float(cn),
        band_persistence=float(2 * cn),
        bottleneck_to_empty=float(best_p / 2.0),
        threshold=THRESH,
    )

with open(os.path.join(RES, "persistence.json"), "w") as f:
    json.dump(out, f, indent=2)
log("DONE -> results/persistence.json + results/persistenceDiag_*.npy")
