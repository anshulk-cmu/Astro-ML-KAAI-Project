import numpy as np
import ripser
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix, diags
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components, shortest_path
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import kneighbors_graph
from common import load, save_json, log, knn

N_SUB, N_REP, K, SEED = 2000, 10, 15, 0


def zscore(E):
    return ((E - E.mean(0)) / (E.std(0) + 1e-8)).astype(np.float32)


def betti0(X, k=K):
    dist, idx = knn(X, k)
    n = len(X)
    G = csr_matrix((dist.ravel() + 1e-9, (np.repeat(np.arange(n), k), idx.ravel())), shape=(n, n))
    G = G.maximum(G.T)
    ncomp = connected_components(G, directed=False)[0]
    mst = minimum_spanning_tree(G).tocoo()
    order = np.argsort(mst.data)[::-1]
    w = mst.data[order]
    splits = {}
    for t in (1, 2, 3):
        keep = np.ones(len(mst.data), bool); keep[order[:t]] = False
        Mt = csr_matrix((mst.data[keep], (mst.row[keep], mst.col[keep])), shape=(n, n))
        _, lab = connected_components(Mt.maximum(Mt.T), directed=False)
        splits[str(t)] = sorted(np.bincount(lab).tolist(), reverse=True)[:4]
    return dict(knn_components=int(ncomp), top_mst_edges=[float(x) for x in w[:8]],
                top_gaps=[float(x) for x in (w[:-1] - w[1:])[:8]], split_sizes=splits)


def geodesic(S, k, power):
    G = kneighbors_graph(S, k, mode="distance")
    if power != 1:
        G.data = G.data ** power
    return shortest_path(G.maximum(G.T), directed=False)


def diffusion_dist(S, k):
    G = kneighbors_graph(S, k, mode="distance"); G = G.maximum(G.T)
    eps = np.median(G.data) ** 2
    W = G.copy(); W.data = np.exp(-(G.data ** 2) / eps)
    Di = diags(1 / np.sqrt(np.asarray(W.sum(1)).ravel()))
    vals, vecs = eigsh((Di @ W @ Di).tocsr(), k=40, which="LM")
    o = np.argsort(vals)[::-1]
    return cdist(vecs[:, o][:, 1:] * vals[o][1:], vecs[:, o][:, 1:] * vals[o][1:])


def betti1(X):
    rng = np.random.default_rng(SEED)
    out = {}
    for nm in ("euclidean", "fermat", "diffusion"):
        nbars, maxp, sig = [], [], []
        for _ in range(N_REP):
            S = X[rng.choice(len(X), N_SUB, replace=False)]
            D = cdist(S, S) if nm == "euclidean" else geodesic(S, K, 2) if nm == "fermat" else diffusion_dist(S, K)
            D = D / D.max()
            dg = ripser.ripser(D, distance_matrix=True, maxdim=1)["dgms"][1]
            p = (dg[:, 1] - dg[:, 0]) if len(dg) else np.array([0.0])
            nbars.append(len(dg)); maxp.append(float(p.max())); sig.append(int((p > 0.1).sum()))
        out[nm] = dict(mean_bars=float(np.mean(nbars)), max_persistence=float(np.max(maxp)),
                       sig_loops_mean=float(np.mean(sig)), sig_loops_range=[int(min(sig)), int(max(sig))])
    return out


ef, ei, ok, df = load()
Zf = zscore(ef)
out = {}
log("== beta0 EXACT (full 48398, single-linkage MST) ==")
out["beta0"] = betti0(Zf)
log(f"beta0: knn_components={out['beta0']['knn_components']} split@1edge={out['beta0']['split_sizes']['1']} "
    f"split@2={out['beta0']['split_sizes']['2']} top_gaps={[round(g, 3) for g in out['beta0']['top_gaps'][:4]]}")

log("== beta1 (10x 2k subsamples; Euclidean + Fermat + diffusion; diameter-normalized) ==")
out["beta1"] = betti1(Zf)
for nm, r in out["beta1"].items():
    log(f"beta1 {nm:10s} mean_bars={r['mean_bars']:.0f} sig_loops(pers>0.1)={r['sig_loops_mean']:.1f} "
        f"range={r['sig_loops_range']} max_pers={r['max_persistence']:.3f}")

save_json("topology", out)
log("DONE")
