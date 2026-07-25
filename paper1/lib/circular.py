"""Circular readouts for quantities that wrap: doubled-angle probes, circular error, loop radius."""
import numpy as np
from sklearn.linear_model import RidgeCV

import config as C
from probes import boot_stat, ci, r2, split


class CircProbe:
    """Ridge probes on (cos k.theta, sin k.theta). Fit once, then applied unchanged."""

    def __init__(self, Z, theta_rad, k, train_idx):
        self.k = k
        c, s = np.cos(k * theta_rad), np.sin(k * theta_rad)
        self.mc = RidgeCV(alphas=C.ALPHAS).fit(Z[train_idx], c[train_idx])
        self.ms = RidgeCV(alphas=C.ALPHAS).fit(Z[train_idx], s[train_idx])
        self.alpha = (float(self.mc.alpha_), float(self.ms.alpha_))

    def predict(self, Z):
        return self.mc.predict(Z), self.ms.predict(Z)


def circ_error(pc, ps, theta_rad, k):
    d = np.angle(np.exp(1j * (np.arctan2(ps, pc) - k * theta_rad)))
    return np.degrees(np.abs(d)) / k


def radius(pc, ps):
    return np.hypot(pc, ps)


def evaluate(pr, Z, theta_rad, idx, n_boot=C.N_BOOT, seed=C.SEED):
    pc, ps = pr.predict(Z[idx])
    err, rad = circ_error(pc, ps, theta_rad[idx], pr.k), radius(pc, ps)
    c, s = np.cos(pr.k * theta_rad[idx]), np.sin(pr.k * theta_rad[idx])
    rng = np.random.default_rng(seed)
    f20 = [(err[b] < 20).mean() for b in
           (rng.integers(0, len(err), len(err)) for _ in range(n_boot))]
    return dict(n=int(len(idx)),
                med_err_deg=float(np.median(err)), med_err_ci=boot_stat(err, np.median, n_boot, seed),
                loop_radius=float(np.median(rad)), loop_radius_ci=boot_stat(rad, np.median, n_boot, seed),
                loop_radius_rms=float(np.sqrt((rad ** 2).mean())),
                frac_within_20=float((err < 20).mean()), frac_within_20_ci=ci(f20),
                r2_cos=r2(c, pc), r2_sin=r2(s, ps),
                chance_floor_deg=90.0 / pr.k)


def fit_evaluate(Z, theta_rad, k, mask, seed=C.SEED, n_boot=C.N_BOOT):
    ok = mask & np.isfinite(theta_rad)
    tr, te = split(ok, seed)
    pr = CircProbe(Z, theta_rad, k, tr)
    out = evaluate(pr, Z, theta_rad, te, n_boot, seed)
    out["alpha_cos_sin"] = pr.alpha
    out["n_train"], out["n_pool"] = int(len(tr)), int(ok.sum())
    return pr, tr, te, out


def linear_angle_probe(Z, theta_deg, period, mask, seed=C.SEED, n_boot=C.N_BOOT):
    """Plain scalar ridge on the raw angle, scored with the same wrapped error."""
    ok = mask & np.isfinite(theta_deg)
    tr, te = split(ok, seed)
    m = RidgeCV(alphas=C.ALPHAS).fit(Z[tr], theta_deg[tr])
    p = m.predict(Z[te])
    d = (p - theta_deg[te] + period / 2) % period - period / 2
    err = np.abs(d)
    return dict(n=int(len(te)), r2=r2(theta_deg[te], p),
                med_err_deg=float(np.median(err)), med_err_ci=boot_stat(err, np.median, n_boot, seed),
                frac_within_20=float((err < 20).mean()), alpha=float(m.alpha_))
