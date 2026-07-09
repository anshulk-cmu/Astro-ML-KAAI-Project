import numpy as np
import ripser
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix, diags
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components, shortest_path
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import kneighbors_graph, NearestNeighbors
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
    return dict(
        estimator=f"symmetric_{k}NN_graph_MST_not_full_Euclidean_MST",
        k=int(k), knn_components=int(ncomp),
        top_mst_edges=[float(x) for x in w[:8]],
        top_gaps=[float(x) for x in (w[:-1] - w[1:])[:8]],
        split_sizes=splits,
        interpretation=("connectivity and largest-edge split diagnostic; not an exact "
                        "persistent beta0 theorem for the underlying manifold"),
    )


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
    W = csr_matrix((w.ravel(), (np.repeat(np.arange(n), k), idx.ravel())),
                   shape=(n, n)).maximum(
        csr_matrix((w.ravel(), (np.repeat(np.arange(n), k), idx.ravel())),
                   shape=(n, n)).T)
    q = np.asarray(W.sum(1)).ravel()
    K1 = diags(1.0 / q) @ W @ diags(1.0 / q)              # alpha=1
    deg = np.asarray(K1.sum(1)).ravel()
    Di = diags(1 / np.sqrt(deg))
    vals, vecs = eigsh((Di @ K1 @ Di).tocsr(), k=40, which="LA")
    o = np.argsort(vals)[::-1]
    psi = np.asarray(Di @ vecs[:, o])
    coords = psi[:, 1:] * vals[o][1:]
    return cdist(coords, coords)


def betti1(X):
    rng = np.random.default_rng(SEED)
    subs = [rng.choice(len(X), N_SUB, replace=False) for _ in range(N_REP)]
    raw = {nm: dict(nbars=[], maxp=[], sig=[])
           for nm in ("euclidean", "fermat", "diffusion")}
    for rep, sub_idx in enumerate(subs):
        S = X[sub_idx]
        for nm in raw:
            D = cdist(S, S) if nm == "euclidean" else geodesic(S, K, 2) if nm == "fermat" else diffusion_dist(S, K)
            if not np.isfinite(D).all():
                raise RuntimeError(f"{nm} distance matrix disconnected/non-finite at rep {rep}")
            D = D / D.max()
            dg = ripser.ripser(D, distance_matrix=True, maxdim=1)["dgms"][1]
            p = (dg[:, 1] - dg[:, 0]) if len(dg) else np.array([0.0])
            raw[nm]["nbars"].append(len(dg))
            raw[nm]["maxp"].append(float(p.max()))
            raw[nm]["sig"].append(int((p > 0.1).sum()))
    out = {}
    for nm, r in raw.items():
        out[nm] = dict(
            mean_bars=float(np.mean(r["nbars"])),
            max_persistence=float(np.max(r["maxp"])),
            per_rep_max_persistence=r["maxp"],
            per_rep_sig_count=r["sig"],
            subsamples_with_any_over_0p1=int(np.sum(np.asarray(r["sig"]) > 0)),
            sig_loops_mean=float(np.mean(r["sig"])),
            sig_loops_range=[int(min(r["sig"])), int(max(r["sig"]))],
        )
    out["_design"] = dict(seed=SEED, n_rep=N_REP, n_sub=N_SUB,
                          paired_subsamples_across_metrics=True,
                          diffusion="alpha=1 self-tuned, matching diffusionMap.py",
                          threshold_note="0.1 is descriptive; persistence resampling supplies the band")
    return out


ef, ei, ok, df = load()
Zf = zscore(ef)
out = {}
log("== full-sample kNN connectivity + sparse-graph MST split diagnostic ==")
out["beta0"] = betti0(Zf)
out["beta0_k_sensitivity"] = {str(k): betti0(Zf, k) for k in (10, 15, 20)}
log(f"beta0: knn_components={out['beta0']['knn_components']} split@1edge={out['beta0']['split_sizes']['1']} "
    f"split@2={out['beta0']['split_sizes']['2']} top_gaps={[round(g, 3) for g in out['beta0']['top_gaps'][:4]]}")

log("== beta1 (10x 2k subsamples; Euclidean + Fermat + diffusion; diameter-normalized) ==")
out["beta1"] = betti1(Zf)
for nm, r in out["beta1"].items():
    if nm.startswith("_"):
        continue
    log(f"beta1 {nm:10s} mean_bars={r['mean_bars']:.0f} sig_loops(pers>0.1)={r['sig_loops_mean']:.1f} "
        f"range={r['sig_loops_range']} max_pers={r['max_persistence']:.3f}")

save_json("topology", out)
log("DONE")
