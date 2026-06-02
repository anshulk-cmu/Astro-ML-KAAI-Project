import numpy as np
from common import load, save_json, log


def zscore(E):
    return ((E - E.mean(0)) / (E.std(0) + 1e-8)).astype(np.float32)


def whiten(Z):
    C = np.cov(Z.T) + 1e-3 * np.eye(Z.shape[1])
    w, V = np.linalg.eigh(np.linalg.inv(C))
    return Z @ (V * np.sqrt(np.maximum(w, 0))) @ V.T   # Z * C^{-1/2}


def direction(W, pos, neg):
    v = W[pos].mean(0) - W[neg].mean(0)
    return v / (np.linalg.norm(v) + 1e-12)


ef, ei, ok, df = load()
c = lambda k: df[k].values
W = whiten(zscore(ef))
smooth = c("smooth-or-featured_smooth_fraction") > 0.6
featured = c("smooth-or-featured_featured-or-disk_fraction") > 0.6
spiral = (c("has-spiral-arms_yes_fraction") > 0.5) & featured

v_parent = direction(W, featured, smooth)              # featured-vs-smooth concept axis
v_child = direction(W, spiral, featured & ~spiral)     # spiral within the featured branch
cos_pc = abs(float(v_parent @ v_child))
rng = np.random.default_rng(0)                          # null: random spiral-sized subset within featured
fi = np.where(featured)[0]
nulls = []
for _ in range(200):
    s = np.zeros(len(featured), bool)
    s[rng.choice(fi, int(spiral.sum()), replace=False)] = True
    nulls.append(abs(float(v_parent @ direction(W, s, featured & ~s))))
p95 = float(np.percentile(nulls, 95))
out = dict(cos_parent_child=cos_pc, null_p95=p95, significant=bool(cos_pc > p95),
           interpretation="Park et al. linear hierarchy: a child (spiral) parent-component above the shuffle null "
                          "supports child = featured-direction + spiral-specific orthogonal part. Exploratory.")
log(f"cos(parent=featured, child=spiral)={cos_pc:.3f} null_p95={p95:.3f} significant={cos_pc > p95}")
save_json("geometryOfConcepts", out)
log("DONE")
