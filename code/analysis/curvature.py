import numpy as np
import torch
import ot
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from sklearn.neighbors import kneighbors_graph
from common import load, save_json, log, knn

SEED, N_QUAD, K_FORMAN, N_OLL, K_OLL, ALPHA = 0, 1_000_000, 15, 2000, 10, 0.5


def zscore(E):
    return ((E - E.mean(0)) / (E.std(0) + 1e-8)).astype(np.float32)


def delta_hyper(X, metric="euclidean", n_quad=N_QUAD, batch=200000):
    Xt = torch.from_numpy(X).cuda().float()
    if metric == "cosine":
        Xt = Xt / (Xt.norm(dim=1, keepdim=True) + 1e-9)

    def pdist(A, B):
        if metric == "cosine":
            return 1.0 - (A * B).sum(1)
        return (A - B).norm(dim=1)

    n, rng = len(Xt), np.random.default_rng(SEED)
    ia, ib = rng.integers(0, n, 100000), rng.integers(0, n, 100000)
    diam = float(pdist(Xt[ia], Xt[ib]).max())
    res, done = [], 0
    while done < n_quad:
        m = min(batch, n_quad - done); done += m
        q = torch.from_numpy(rng.integers(0, n, (m, 4))).cuda()
        A, B, C, D = (Xt[q[:, j]] for j in range(4))
        s = torch.stack([pdist(A, B) + pdist(C, D),
                         pdist(A, C) + pdist(B, D),
                         pdist(A, D) + pdist(B, C)], 1).sort(dim=1, descending=True)[0]
        res.append(((s[:, 0] - s[:, 1]) / 2).cpu().numpy())
    d = np.concatenate(res) / diam
    return dict(metric=metric, diameter_random_pair_estimate=diam,
                mean=float(d.mean()), median=float(np.median(d)),
                p95=float(np.percentile(d, 95)))


def tree_distance(i, j):
    """Exact path distance in the binary heap tree on node ids 0..n-1."""
    a, b = np.asarray(i, np.int64) + 1, np.asarray(j, np.int64) + 1
    da = np.floor(np.log2(a)).astype(np.int64)
    db = np.floor(np.log2(b)).astype(np.int64)
    aa = a >> np.maximum(da - db, 0)
    bb = b >> np.maximum(db - da, 0)
    dl = np.minimum(da, db)
    neq = aa != bb
    while np.any(neq):
        aa[neq] >>= 1
        bb[neq] >>= 1
        dl[neq] -= 1
        neq = aa != bb
    return da + db - 2 * dl


def delta_hyper_tree(n, n_quad=N_QUAD, batch=200000):
    rng = np.random.default_rng(SEED)
    diameter = float(2 * int(np.floor(np.log2(n))))
    vals = []
    done = 0
    while done < n_quad:
        m = min(batch, n_quad - done)
        done += m
        q = rng.integers(0, n, (m, 4))
        A, B, C, D = (q[:, j] for j in range(4))
        s = np.stack([tree_distance(A, B) + tree_distance(C, D),
                      tree_distance(A, C) + tree_distance(B, D),
                      tree_distance(A, D) + tree_distance(B, C)], 1)
        s.sort(axis=1)
        vals.append((s[:, 2] - s[:, 1]) / 2.0)
    d = np.concatenate(vals) / diameter
    return dict(metric="exact_binary_tree_path", diameter_exact=diameter,
                mean=float(d.mean()), median=float(np.median(d)),
                p95=float(np.percentile(d, 95)))


def forman(X, k):
    _, idx = knn(X, k)
    n = len(X)
    A = csr_matrix((np.ones(n * k), (np.repeat(np.arange(n), k), idx.ravel())), shape=(n, n))
    A = (A + A.T)
    A.data[:] = 1.0
    deg = np.asarray(A.sum(1)).ravel()
    edges = A.tocoo()
    u, v = edges.row, edges.col
    tri = np.asarray((A @ A)[u, v]).ravel()                # triangle count for every edge
    F = 4 - deg[u] - deg[v] + 3 * tri                       # augmented Forman-Ricci (triangle-aware)
    return dict(frac_negative=float((F < 0).mean()), mean=float(F.mean()),
                p5=float(np.percentile(F, 5)), p50=float(np.percentile(F, 50)), p95=float(np.percentile(F, 95)))


def ollivier_distance(D, k, alpha=ALPHA):
    idx = np.argsort(D, axis=1)[:, 1:k + 1]
    kap = []
    for u in range(len(D)):
        for v in idx[u]:
            nu, nv = np.r_[u, idx[u]], np.r_[v, idx[v]]
            mu = np.r_[alpha, np.full(k, (1 - alpha) / k)]
            mv = np.r_[alpha, np.full(k, (1 - alpha) / k)]
            kap.append(1 - ot.emd2(mu, mv, D[np.ix_(nu, nv)]) / D[u, v])
    kap = np.array(kap)
    return dict(frac_negative=float((kap < 0).mean()), mean=float(kap.mean()),
                p5=float(np.percentile(kap, 5)), p95=float(np.percentile(kap, 95)))


def fermat_distance(X, graph_k=15, power=2):
    G = kneighbors_graph(X, graph_k, mode="distance")
    G.data = G.data ** power
    D = shortest_path(G.maximum(G.T), directed=False)
    if not np.isfinite(D).all():
        raise RuntimeError("Fermat graph is disconnected")
    return D


def matched_gaussian(Z, seed=SEED):
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(np.cov(Z.T) + 1e-4 * np.eye(Z.shape[1]))
    return (rng.standard_normal(Z.shape).astype(np.float32) @ L.T).astype(np.float32)


def tree_cloud(n, dim=1024, branch=2, seed=SEED):
    rng = np.random.default_rng(seed)
    pts, frontier = [np.zeros(dim, np.float32)], [0]
    while len(pts) < n and frontier:
        nf = []
        for p in frontier:
            for _ in range(branch):
                if len(pts) >= n:
                    break
                d = rng.standard_normal(dim); d /= np.linalg.norm(d)
                pts.append(pts[p] + d.astype(np.float32)); nf.append(len(pts) - 1)
        frontier = nf
    return np.array(pts[:n])


ef, ei, ok, df = load()
Zf, Zi = zscore(ef), zscore(ei)
out = {"delta": {}, "forman": {}, "ollivier": {}}

log("== delta-hyperbolicity (1e6 quads, diameter-normalized) ==")
out["delta"]["E_full_euclid"] = delta_hyper(Zf)
out["delta"]["E_full_cosine"] = delta_hyper(Zf, metric="cosine")
out["delta"]["E_img_euclid"] = delta_hyper(Zi)
out["delta"]["gaussian_anchor"] = delta_hyper(matched_gaussian(Zf))
out["delta"]["tree_validation"] = delta_hyper_tree(len(Zf))
for nmk, r in out["delta"].items():
    log(f"delta {nmk:18s} median={r['median']:.4f} mean={r['mean']:.4f} p95={r['p95']:.4f}")

log("== Forman-Ricci (full kNN graph) ==")
out["forman"]["E_full"] = forman(Zf, K_FORMAN)
log(f"Forman E_full frac_neg={out['forman']['E_full']['frac_negative']:.3f} "
    f"mean={out['forman']['E_full']['mean']:.1f} p5={out['forman']['E_full']['p5']:.0f}")

log("== Ollivier-Ricci (2k subsample, exact EMD, alpha=0.5) ==")
sub = Zf[np.random.default_rng(SEED).choice(len(Zf), N_OLL, replace=False)]
D_eu = cdist(sub, sub).astype(np.float64)
out["ollivier"]["E_full"] = ollivier_distance(D_eu, K_OLL)
out["ollivier"]["euclidean_k_sensitivity"] = {
    str(k): ollivier_distance(D_eu, k) for k in (8, 10, 15)
}
D_fermat = fermat_distance(sub, graph_k=15, power=2)
out["ollivier"]["fermat_p2_k10"] = ollivier_distance(D_fermat, 10)
out["ollivier"]["note"] = ("one fixed 2k sample; k sensitivity and Fermat metric check, "
                            "not a population bootstrap")
log(f"Ollivier E_full frac_neg={out['ollivier']['E_full']['frac_negative']:.3f} "
    f"mean={out['ollivier']['E_full']['mean']:.3f} p5={out['ollivier']['E_full']['p5']:.3f} p95={out['ollivier']['E_full']['p95']:.3f}")

save_json("curvature", out)
log("DONE")
