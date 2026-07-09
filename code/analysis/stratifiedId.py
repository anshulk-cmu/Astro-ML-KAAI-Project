import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import gamma
from common import load, save_json, log, knn

SEED = 0


def zscore(E):
    return ((E - E.mean(0)) / (E.std(0) + 1e-8)).astype(np.float32)


def id_boot(X, n_boot=200):
    d, _ = knn(X, 4)
    mu = d[:, 1] / np.maximum(d[:, 0], 1e-12)
    mu = mu[mu > 1]
    mle = (len(mu) - 1) / np.log(mu).sum()
    rng = np.random.default_rng(SEED)
    boot = np.array([(len(mu) - 1) / np.log(rng.choice(mu, len(mu))).sum() for _ in range(n_boot)])
    return float(mle), boot


ef, ei, ok, df = load()
Zf = zscore(ef)
gr = (df["mag_g_desi"] - df["mag_r_desi"]).values
fin = np.isfinite(gr)
gm = GaussianMixture(2, random_state=SEED).fit(gr[fin].reshape(-1, 1))
lab = np.full(len(gr), -1)
lab[fin] = gm.predict(gr[fin].reshape(-1, 1))
order = np.argsort(gm.means_.ravel())               # low g-r = star-forming (blue), high = passive (red)
sf = (lab == order[0]) & fin
passive = (lab == order[1]) & fin
# decision boundary actually used by predict (posterior crossing), NOT the midpoint of means
grid = np.linspace(gm.means_.ravel().min(), gm.means_.ravel().max(), 200001).reshape(-1, 1)
flip = np.nonzero(np.diff(gm.predict(grid)))[0]
cut_boundary = float(grid[flip[0] + 1, 0]) if len(flip) else float(gm.means_.mean())

mleP, bP = id_boot(Zf[passive])
mleS, bS = id_boot(Zf[sf])
dID = bP - bS
out = dict(
    gmm_means=[float(x) for x in np.sort(gm.means_.ravel())],
    cut=cut_boundary, cut_midpoint_of_means=float(gm.means_.mean()),
    cut_note="cut is the GMM posterior decision boundary that predict() actually uses; "
             "the midpoint of means is kept for provenance only and labels no population",
    ID_passive=dict(n=int(passive.sum()), mle=mleP, ci=[float(np.percentile(bP, 2.5)), float(np.percentile(bP, 97.5))]),
    ID_sf=dict(n=int(sf.sum()), mle=mleS, ci=[float(np.percentile(bS, 2.5)), float(np.percentile(bS, 97.5))]),
    delta_ID=dict(mean=float(dID.mean()), ci=[float(np.percentile(dID, 2.5)), float(np.percentile(dID, 97.5))],
                  excludes_zero=bool(np.percentile(dID, 2.5) > 0 or np.percentile(dID, 97.5) < 0)),
    uncertainty_note="CIs are conditional bootstraps of the precomputed TwoNN neighbour ratios on the "
                     "fixed kNN graph; they do not resample galaxies, refit the GMM, or recompute "
                     "neighbours, so split and graph uncertainty are not included")
log(f"passive n={passive.sum()} ID={mleP:.2f} | SF n={sf.sum()} ID={mleS:.2f} | "
    f"dID={dID.mean():.2f} CI={out['delta_ID']['ci']} excl0={out['delta_ID']['excludes_zero']}")
save_json("stratifiedId", out)
log("DONE")
