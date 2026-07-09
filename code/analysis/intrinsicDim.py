import numpy as np
import torch
from scipy.stats import gamma
from scipy.special import digamma
from common import load, save_json, log, knn
from synthetic import datasets

KMAX = 512
KGRID = [10, 14, 20, 28, 40, 56, 80, 113, 160, 226]                  # local-MLE neighbourhood sweep
SCALES = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256]        # Gride scale curve (orders n, 2n)
SEED = 0
N_BOOT = 200


def zscore(E):
    return ((E - E.mean(0)) / (E.std(0) + 1e-8)).astype(np.float32)


def twonn(d):
    mu = d[:, 1] / np.maximum(d[:, 0], 1e-12)
    mu = mu[mu > 1]
    v, N = np.log(mu).sum(), len(mu)
    lo, hi = gamma.ppf([0.025, 0.975], a=N, scale=1.0 / v)
    rng = np.random.default_rng(SEED)
    boot = [(N - 1) / np.log(rng.choice(mu, N)).sum() for _ in range(200)]
    s = np.sort(mu); F = (np.arange(1, N + 1) - 0.5) / N; m = F < 0.9
    x = np.log(s[m]); lin = float((x @ -np.log(1 - F[m])) / (x @ x))
    return dict(mle=float((N - 1) / v), ci_post=[float(lo), float(hi)],
                ci_boot=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                linear=lin, n=N)


def boot_ci(values, estimator, seed=SEED, n_boot=N_BOOT):
    rng = np.random.default_rng(seed)
    n = len(values)
    vals = [estimator(values[rng.integers(0, n, n)]) for _ in range(n_boot)]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def local_mle(d, K):
    sl = (np.log(d[:, K - 1:K]) - np.log(np.maximum(d[:, :K - 1], 1e-12))).sum(1)
    m = (K - 2) / np.maximum(sl, 1e-12)
    estimator = lambda v: float(1.0 / np.mean(1.0 / v))
    return estimator(m), boot_ci(m, estimator, seed=SEED + K)


def gride(d):
    out, cis = {}, {}
    for n1 in SCALES:
        n2 = 2 * n1
        if n2 > d.shape[1]:
            break
        mu = d[:, n2 - 1] / np.maximum(d[:, n1 - 1], 1e-12)
        logmu = np.log(mu[mu > 1])
        scale = float(digamma(n2) - digamma(n1))
        estimator = lambda v: float(scale / v.mean())
        out[str(n1)] = estimator(logmu)
        cis[str(n1)] = boot_ci(logmu, estimator, seed=SEED + n1)
    return out, cis


def pca_pr(X):
    Xt = torch.from_numpy(X).cuda().float()
    s = torch.linalg.svdvals(Xt - Xt.mean(0)).cpu().numpy().astype(np.float64)
    return float((s ** 2).sum() ** 2 / (s ** 4).sum())


def estimate(X):
    d, _ = knn(X, KMAX)
    g, gci = gride(d)
    lm = {str(K): local_mle(d, K) for K in KGRID}
    return dict(
        twonn=twonn(d),
        gride=g,
        gride_ci=gci,
        local_mle={k: v[0] for k, v in lm.items()},
        local_mle_ci={k: v[1] for k, v in lm.items()},
        pca_pr=pca_pr(X),
        uncertainty_note=("Gride/local-MLE intervals are conditional row bootstraps of the "
                          "fixed neighbour-distance graph; they do not refit neighbours"),
    )


ef, ei, ok, df = load()
out = {"synthetic": {}, "aion": {}}

log(f"== synthetic validation (matched N={len(ef)}) ==")
for name, (X, truth) in datasets(len(ef), SEED).items():
    r = estimate(zscore(X))
    out["synthetic"][name] = dict(truth=truth, **r)
    log(f"{name:15s} truth={truth} twonn={r['twonn']['mle']:.2f} "
        f"CI={[round(c, 2) for c in r['twonn']['ci_post']]} pca_pr={r['pca_pr']:.1f}")

log("== AION ==")
for name, E in [("E_full", ef), ("E_img", ei)]:
    r = estimate(zscore(E))
    gk, lk = str(SCALES[-1]), str(KGRID[-1])
    r["headline_scale_summary"] = {
        "gride_largest_tested_scale": dict(n1=SCALES[-1], estimate=r["gride"][gk],
                                             ci=r["gride_ci"][gk]),
        "local_mle_largest_tested_scale": dict(K=KGRID[-1], estimate=r["local_mle"][lk],
                                                ci=r["local_mle_ci"][lk]),
        "wording": "largest tested scales; both curves are still declining, so not a proven plateau",
    }
    out["aion"][name] = r
    log(f"{name} twonn={r['twonn']['mle']:.2f} CI={[round(c, 2) for c in r['twonn']['ci_post']]} "
        f"gride={ {k: round(v, 2) for k, v in r['gride'].items()} } pca_pr={r['pca_pr']:.1f}")

save_json("intrinsicDim", out)
log("DONE")
