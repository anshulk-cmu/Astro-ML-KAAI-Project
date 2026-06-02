import numpy as np
from sklearn.neighbors import kneighbors_graph
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import shortest_path
from scipy.sparse.linalg import eigsh
from scipy.sparse import diags
from common import load, save_json, log

N_SUB, K, SEED = 2000, 15, 0


def zscore(E):
    return ((E - E.mean(0)) / (E.std(0) + 1e-8)).astype(np.float64)


def stats(D):
    off = D[np.triu_indices_from(D, k=1)]
    nn = np.partition(D + np.eye(len(D)) * 1e18, 1, axis=1)[:, 1]
    return dict(rdr=float((off.max() - off.min()) / off.min()),
                nn_over_mean=float(nn.mean() / off.mean()),
                inf=int(np.isinf(off).sum()))


def geodesic(X, k, power):
    G = kneighbors_graph(X, k, mode="distance")
    if power != 1:
        G.data = G.data ** power
    G = G.maximum(G.T)
    return shortest_path(G, method="D", directed=False)


def diffusion(X, k):
    G = kneighbors_graph(X, k, mode="distance")
    G = G.maximum(G.T)
    eps = np.median(G.data) ** 2
    W = G.copy()
    W.data = np.exp(-(G.data ** 2) / eps)
    dd = np.asarray(W.sum(1)).ravel()
    Di = diags(1.0 / np.sqrt(dd))
    vals, vecs = eigsh((Di @ W @ Di).tocsr(), k=50, which="LM")
    o = np.argsort(vals)[::-1]
    psi = vecs[:, o][:, 1:] * vals[o][1:]
    return cdist(psi, psi)


ef, ei, ok, df = load()
idx = np.random.default_rng(SEED).choice(len(ef), N_SUB, replace=False)
out = {}
for name, E in [("E_full", ef), ("E_img", ei)]:
    S = zscore(E)[idx]
    Xn = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-12)
    res = dict(
        euclidean=stats(cdist(S, S)),
        cosine=stats(1.0 - Xn @ Xn.T),
        isomap=stats(geodesic(S, K, 1)),
        fermat_p2=stats(geodesic(S, K, 2)),
        diffusion=stats(diffusion(S, K)),
    )
    out[name] = res
    for m, r in res.items():
        log(f"{name} {m:10s} RDR={r['rdr']:8.3f}  NN/mean={r['nn_over_mean']:.3f}  inf={r['inf']}")
save_json("metric", out)
log("DONE")
