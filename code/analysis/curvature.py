import numpy as np
import torch
import ot
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from common import load, save_json, log, knn

SEED, N_QUAD, K_FORMAN, N_OLL, K_OLL, ALPHA = 0, 1_000_000, 15, 2000, 10, 0.5


def zscore(E):
    return ((E - E.mean(0)) / (E.std(0) + 1e-8)).astype(np.float32)


def delta_hyper(X, normalize=False, n_quad=N_QUAD, batch=200000):
    Xt = torch.from_numpy(X).cuda().float()
    if normalize:
        Xt = Xt / (Xt.norm(dim=1, keepdim=True) + 1e-9)
    n, rng = len(Xt), np.random.default_rng(SEED)
    ia, ib = rng.integers(0, n, 100000), rng.integers(0, n, 100000)
    diam = float((Xt[ia] - Xt[ib]).norm(dim=1).max())
    res, done = [], 0
    while done < n_quad:
        m = min(batch, n_quad - done); done += m
        q = torch.from_numpy(rng.integers(0, n, (m, 4))).cuda()
        A, B, C, D = (Xt[q[:, j]] for j in range(4))
        s = torch.stack([(A - B).norm(dim=1) + (C - D).norm(dim=1),
                         (A - C).norm(dim=1) + (B - D).norm(dim=1),
                         (A - D).norm(dim=1) + (B - C).norm(dim=1)], 1).sort(dim=1, descending=True)[0]
        res.append(((s[:, 0] - s[:, 1]) / 2).cpu().numpy())
    d = np.concatenate(res) / diam
    return dict(mean=float(d.mean()), median=float(np.median(d)), p95=float(np.percentile(d, 95)))


def forman(X, k):
    _, idx = knn(X, k)
    n = len(X)
    A = csr_matrix((np.ones(n * k), (np.repeat(np.arange(n), k), idx.ravel())), shape=(n, n))
    A = (A + A.T)
    A.data[:] = 1.0
    deg = np.asarray(A.sum(1)).ravel()
    co = A.multiply(A @ A).tocoo()                          # triangle count per edge
    u, v, tri = co.row, co.col, co.data
    F = 4 - deg[u] - deg[v] + 3 * tri                       # augmented Forman-Ricci (triangle-aware)
    return dict(frac_negative=float((F < 0).mean()), mean=float(F.mean()),
                p5=float(np.percentile(F, 5)), p50=float(np.percentile(F, 50)), p95=float(np.percentile(F, 95)))


def ollivier(X, k, alpha=ALPHA):
    _, idx = knn(X, k)
    D = cdist(X, X).astype(np.float64)
    kap = []
    for u in range(len(X)):
        for v in idx[u]:
            nu, nv = np.r_[u, idx[u]], np.r_[v, idx[v]]
            mu = np.r_[alpha, np.full(k, (1 - alpha) / k)]
            mv = np.r_[alpha, np.full(k, (1 - alpha) / k)]
            kap.append(1 - ot.emd2(mu, mv, D[np.ix_(nu, nv)]) / D[u, v])
    kap = np.array(kap)
    return dict(frac_negative=float((kap < 0).mean()), mean=float(kap.mean()),
                p5=float(np.percentile(kap, 5)), p95=float(np.percentile(kap, 95)))


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
out["delta"]["E_full_cosine"] = delta_hyper(Zf, normalize=True)
out["delta"]["E_img_euclid"] = delta_hyper(Zi)
out["delta"]["gaussian_anchor"] = delta_hyper(matched_gaussian(Zf))
out["delta"]["tree_validation"] = delta_hyper(tree_cloud(len(Zf)))
for nmk, r in out["delta"].items():
    log(f"delta {nmk:18s} median={r['median']:.4f} mean={r['mean']:.4f} p95={r['p95']:.4f}")

log("== Forman-Ricci (full kNN graph) ==")
out["forman"]["E_full"] = forman(Zf, K_FORMAN)
log(f"Forman E_full frac_neg={out['forman']['E_full']['frac_negative']:.3f} "
    f"mean={out['forman']['E_full']['mean']:.1f} p5={out['forman']['E_full']['p5']:.0f}")

log("== Ollivier-Ricci (2k subsample, Sinkhorn/EMD, alpha=0.5) ==")
sub = Zf[np.random.default_rng(SEED).choice(len(Zf), N_OLL, replace=False)]
out["ollivier"]["E_full"] = ollivier(sub, K_OLL)
log(f"Ollivier E_full frac_neg={out['ollivier']['E_full']['frac_negative']:.3f} "
    f"mean={out['ollivier']['E_full']['mean']:.3f} p5={out['ollivier']['E_full']['p5']:.3f} p95={out['ollivier']['E_full']['p95']:.3f}")

save_json("curvature", out)
log("DONE")
