"""Named nulls. One implementation each, so no diagnostic invents its own."""
import numpy as np
from sklearn.decomposition import PCA

import config as C


def shuffled(theta_rad, mask, seed=C.SEED):
    rng = np.random.default_rng(seed)
    out = theta_rad.copy()
    idx = np.where(mask & np.isfinite(theta_rad))[0]
    out[idx] = theta_rad[rng.permutation(idx)]
    return out


def pc_basis(Z, n_components, seed=C.SEED):
    p = PCA(n_components=n_components, svd_solver="randomized", random_state=seed)
    return p.fit_transform(Z), p.explained_variance_ratio_


def matched_norm_directions(w, n, seed=C.SEED):
    rng = np.random.default_rng(seed)
    V = rng.standard_normal((n, len(w)))
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    return V * np.linalg.norm(w)
