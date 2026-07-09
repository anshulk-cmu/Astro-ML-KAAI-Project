"""
Shared utilities for Tracks A/B/C: z-score, RidgeCV probes, concept directions,
angles, and bootstrap confidence intervals. Bootstrap design: angle_with_ci
resamples each direction's own population INDEPENDENTLY per replicate (and the
label-null population separately), so shared sampling covariance between the two
directions is not propagated; probe/circ_recover CIs resample fixed held-out
predictions (test-set sampling variation only, not train/split/alpha variation).
"""
import numpy as np
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.model_selection import train_test_split

ALPHAS = np.logspace(-2, 4, 7)
SEED = 0
N_BOOT = 200


def zscore(E):
    return ((E - E.mean(0)) / (E.std(0) + 1e-8)).astype(np.float32)


def ci(vals, lo=2.5, hi=97.5):
    v = np.asarray(vals, float); v = v[np.isfinite(v)]
    if len(v) < 10:
        return [None, None]
    return [float(np.percentile(v, lo)), float(np.percentile(v, hi))]


def basic_ci(point, boot):
    """Basic (pivotal) bootstrap interval [2p - hi, 2p - lo]. Reported alongside the
    percentile interval for ANGLE quantities: bootstrap-refit directions are noisier
    than the full-sample fit, which biases bootstrap angles toward 90 deg; the basic
    interval un-biases to first order (percentile can exclude its own point estimate
    for the noisiest directions)."""
    lo, hi = ci(boot)
    return [None, None] if lo is None else [2 * point - hi, 2 * point - lo]


def probe(X, y, mask=None, n_boot=N_BOOT, seed=SEED):
    """RidgeCV R2 on a held-out split, with a bootstrap CI over the split's
    resampled residuals (matches Phase 1's boot_ci: resample held-out predictions)."""
    ok = np.isfinite(y) if mask is None else (np.isfinite(y) & mask)
    idx = np.where(ok)[0]
    tr, te = train_test_split(idx, test_size=0.2, random_state=seed)
    m = RidgeCV(alphas=ALPHAS).fit(X[tr], y[tr])
    p = m.predict(X[te]); yte = y[te]
    r2 = float(1 - ((yte - p) ** 2).sum() / ((yte - yte.mean()) ** 2).sum())
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for j in range(n_boot):
        b = rng.integers(0, len(te), len(te))
        boot[j] = 1 - ((yte[b] - p[b]) ** 2).sum() / ((yte[b] - yte[b].mean()) ** 2).sum()
    return dict(r2=r2, ci=ci(boot), n=int(len(idx)), alpha=float(m.alpha_))


def direction(X, y, mask=None):
    ok = np.isfinite(y) if mask is None else (np.isfinite(y) & mask)
    w = Ridge(alpha=100.0).fit(X[ok], y[ok]).coef_
    return w / (np.linalg.norm(w) + 1e-12)


def angle(a, b):
    return float(np.degrees(np.arccos(np.clip(a @ b, -1, 1))))


def label_null_angle(ya, yb):
    fin = np.isfinite(ya) & np.isfinite(yb)
    r = np.corrcoef(ya[fin], yb[fin])[0, 1]
    return float(np.degrees(np.arccos(np.clip(r, -1, 1))))


def angle_with_ci(X, labels, key_a, key_b, mask_a=None, mask_b=None, n_boot=N_BOOT, seed=SEED):
    """Bootstrap CI for the embedding angle AND the label-null angle between two
    concept directions. Each direction is fit on ITS OWN natural population
    (isfinite(label) & its own mask) -- NOT a joint intersection of both masks,
    which would silently force a globally-meaningful direction (e.g. the main
    sequence, fit on ~48k galaxies) to be refit on a much smaller population just
    because the OTHER concept (e.g. bar) has sparser catalog coverage. Each
    bootstrap replicate independently resamples (with replacement) from each
    direction's own eligible population; the label-null angle uses the population
    where BOTH labels are finite AND both masks hold (point estimate and bootstrap
    on the same population). CAVEAT: arccos(corr) equals the probe angle only in an
    idealized whitened linear model; the embedding covariance is anisotropic and
    Ridge shrinkage rotates directions, so the 'excess' is a DESCRIPTIVE contrast,
    not a calibrated null for disentanglement/independence/hierarchy claims."""
    ya, yb = labels[key_a], labels[key_b]
    oka = np.isfinite(ya) if mask_a is None else (np.isfinite(ya) & mask_a)
    okb = np.isfinite(yb) if mask_b is None else (np.isfinite(yb) & mask_b)
    idx_a, idx_b = np.where(oka)[0], np.where(okb)[0]
    ea = angle(direction(X, ya, oka), direction(X, yb, okb))
    okab = oka & okb
    idx_ab = np.where(okab)[0]
    na = label_null_angle(ya[idx_ab], yb[idx_ab])   # same masked population as its bootstrap
    rng = np.random.default_rng(seed)
    eb, nb = np.empty(n_boot), np.empty(n_boot)
    for j in range(n_boot):
        ba = rng.choice(idx_a, len(idx_a), replace=True)
        bb = rng.choice(idx_b, len(idx_b), replace=True)
        va = direction(X[ba], ya[ba]); vb = direction(X[bb], yb[bb])
        eb[j] = angle(va, vb)
        bab = rng.choice(idx_ab, len(idx_ab), replace=True)
        nb[j] = label_null_angle(ya[bab], yb[bab])
    return dict(embed_angle=ea, embed_ci=ci(eb), embed_ci_basic=basic_ci(ea, eb),
                label_null=na, label_null_ci=ci(nb),
                excess=ea - na, excess_ci=ci(eb - nb), excess_ci_basic=basic_ci(ea - na, eb - nb),
                n_a=int(len(idx_a)), n_b=int(len(idx_b)))


def circ_recover(X, theta_rad, k, mask, seed=SEED, full_proj=False, n_boot=N_BOOT):
    """Recover a period-(2pi/k) angle from X via Ridge on (cos k.theta, sin k.theta).
    Held-out circular error in DEGREES on the theta scale, with a bootstrap CI on
    the median error (resampling the held-out set)."""
    ok = mask & np.isfinite(theta_rad)
    idx = np.where(ok)[0]
    tr, te = train_test_split(idx, test_size=0.2, random_state=seed)
    c, s = np.cos(k * theta_rad), np.sin(k * theta_rad)
    mc = RidgeCV(alphas=ALPHAS).fit(X[tr], c[tr])
    ms = RidgeCV(alphas=ALPHAS).fit(X[tr], s[tr])
    pc, ps = mc.predict(X[te]), ms.predict(X[te])
    r2 = lambda y, p: float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    d = np.angle(np.exp(1j * (np.arctan2(ps, pc) - np.arctan2(s[te], c[te]))))
    err = np.degrees(np.abs(d)) / k
    rng = np.random.default_rng(seed)
    boot = np.array([np.median(err[rng.integers(0, len(err), len(err))]) for _ in range(n_boot)])
    res = dict(n=int(len(idx)), r2_cos=r2(c[te], pc), r2_sin=r2(s[te], ps),
               med_err_deg=float(np.median(err)), med_err_ci=ci(boot),
               frac_within_20=float((err < 20).mean()))
    if full_proj:
        res["_proj"] = np.column_stack([np.degrees(theta_rad[idx]) % (360.0 / k),
                                        mc.predict(X[idx]), ms.predict(X[idx]),
                                        np.isin(idx, te).astype(float)])   # col 3: 1 = held-out row
    return res
