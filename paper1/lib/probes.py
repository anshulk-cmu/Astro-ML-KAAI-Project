"""Ridge probes, bootstrap intervals, rank correlations."""
import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import train_test_split

import config as C


def ci(vals, lo=2.5, hi=97.5):
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    if len(v) < 10:
        return [None, None]
    return [float(np.percentile(v, lo)), float(np.percentile(v, hi))]


def boot_stat(x, fn=np.median, n_boot=C.N_BOOT, seed=C.SEED):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    return ci([fn(x[rng.integers(0, len(x), len(x))]) for _ in range(n_boot)])


def split(mask, seed=C.SEED, test_size=C.TEST_SIZE):
    return train_test_split(np.where(mask)[0], test_size=test_size, random_state=seed)


def r2(y, p):
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def probe(Z, y, mask=None, n_boot=C.N_BOOT, seed=C.SEED):
    ok = np.isfinite(y) if mask is None else (np.isfinite(y) & mask)
    tr, te = split(ok, seed)
    m = RidgeCV(alphas=C.ALPHAS).fit(Z[tr], y[tr])
    p, yte = m.predict(Z[te]), y[te]
    rng = np.random.default_rng(seed)
    boot = [r2(yte[b], p[b]) for b in
            (rng.integers(0, len(te), len(te)) for _ in range(n_boot))]
    return dict(r2=r2(yte, p), r2_ci=ci(boot), n=int(ok.sum()), n_test=int(len(te)),
                alpha=float(m.alpha_))


def spearman(x, y, n_boot=C.N_BOOT, seed=C.SEED):
    fin = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x, float)[fin], np.asarray(y, float)[fin]
    rho, p = spearmanr(x, y)
    rng = np.random.default_rng(seed)
    boot = [spearmanr(x[b], y[b]).statistic for b in
            (rng.integers(0, len(x), len(x)) for _ in range(n_boot))]
    return dict(rho=float(rho), rho_ci=ci(boot), p_value=float(p), n=int(len(x)))


def partial_spearman(x, y, z, n_boot=C.N_BOOT, seed=C.SEED):
    """Spearman(x, y) after removing rank(z) from both, so an elongation trend is not
    reattributed to brightness or size."""
    fin = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = [np.asarray(v, float)[fin] for v in (x, y, z)]

    def resid(a, b):
        A = rankdata(a)
        B = np.column_stack([np.ones(len(b)), rankdata(b)])
        return A - B @ np.linalg.lstsq(B, A, rcond=None)[0]

    rho = spearmanr(resid(x, z), resid(y, z)).statistic
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        b = rng.integers(0, len(x), len(x))
        boot.append(spearmanr(resid(x[b], z[b]), resid(y[b], z[b])).statistic)
    return dict(rho=float(rho), rho_ci=ci(boot), n=int(len(x)))


def direction(Z, y, mask=None, alpha=C.ALPHA_DIRECTION):
    ok = np.isfinite(y) if mask is None else (np.isfinite(y) & mask)
    w = Ridge(alpha=alpha).fit(Z[ok], y[ok]).coef_
    return w / (np.linalg.norm(w) + 1e-12)
