import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh
from scipy.sparse.csgraph import connected_components
from common import load, save_json, save_npy, log, knn

K, NEIG, SCALE_NN = 64, 30, 7
K_SENSITIVITY = [32, 48, 64, 96]


def zscore(E):
    return ((E - E.mean(0)) / (E.std(0) + 1e-8)).astype(np.float32)


def diffmap(X, k, neig):
    dist, idx = knn(X, k)
    sigma = dist[:, SCALE_NN - 1] + 1e-9                       # self-tuning local scale
    w = np.exp(-(dist.astype(np.float64) ** 2) / (sigma[:, None] * sigma[idx]))
    n = len(X)
    W = csr_matrix((w.ravel(), (np.repeat(np.arange(n), k), idx.ravel())), shape=(n, n))
    W = W.maximum(W.T)
    ncomp = connected_components(W, directed=False)[0]
    q = np.asarray(W.sum(1)).ravel()
    K1 = diags(1.0 / q) @ W @ diags(1.0 / q)                   # alpha = 1 (Laplace-Beltrami)
    deg = np.asarray(K1.sum(1)).ravel()
    Di = diags(1.0 / np.sqrt(deg))
    vals, vecs = eigsh((Di @ K1 @ Di).tocsr(), k=neig, which="LA")
    o = np.argsort(vals)[::-1]
    phi = np.asarray(Di @ vecs[:, o])
    return vals[o], phi, ncomp


def harmonic_r2(coords):
    base = coords[:, :3]
    P = np.column_stack([np.ones(len(coords)), base, base ** 2,
                         base[:, 0] * base[:, 1], base[:, 0] * base[:, 2], base[:, 1] * base[:, 2]])
    out = []
    for k in range(3, coords.shape[1]):
        y = coords[:, k]
        b, *_ = np.linalg.lstsq(P, y, rcond=None)
        out.append(float(1 - np.var(y - P @ b) / np.var(y)))
    return out


ef, ei, ok, df = load()
out = {}
for name, E in [("E_full", ef), ("E_img", ei)]:
    X = zscore(E)
    vals, phi, ncomp = diffmap(X, K, NEIG)
    coords = phi[:, 1:] * vals[1:]                             # diffusion coords at t=1
    save_npy(f"diffcoords_{name}", coords[:, :10].astype(np.float32))
    gaps = (vals[:-1] - vals[1:]).tolist()
    sensitivity = {}
    for ks in K_SENSITIVITY:
        if ks == K:
            vs, nc = vals, ncomp
        else:
            vs, _, nc = diffmap(X, ks, 12)
        sensitivity[str(ks)] = {
            "n_components": int(nc),
            "eigenvalues": vs[:12].tolist(),
            "largest_nontrivial_gap_index_1based": int(np.argmax(vs[1:-1] - vs[2:]) + 1),
            "largest_nontrivial_gap": float(np.max(vs[1:-1] - vs[2:])),
        }
    out[name] = dict(
        primary_k=K, local_scale_neighbor=SCALE_NN,
        n_components=int(ncomp), eigenvalues=vals.tolist(), gaps=gaps,
        harmonic_r2=harmonic_r2(coords[:, :12]),
        k_sensitivity=sensitivity,
        coordinate_note=("right diffusion eigenvectors from the alpha=1 self-tuned operator; "
                         "no extra per-coordinate Euclidean renormalization"),
    )
    log(f"{name}: components={ncomp} lambda1-8={[round(v, 4) for v in vals[1:9]]} "
        f"biggest_gap_among_1-10=#{int(np.argmax(gaps[1:11])) + 1}")
save_json("diffusionMap", out)
log("DONE")
