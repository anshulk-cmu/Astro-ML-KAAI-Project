import numpy as np
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.model_selection import train_test_split
from common import load, save_json, log, knn

ALPHAS = np.logspace(-2, 4, 7)
SEED = 0


def zscore(E):
    return ((E - E.mean(0)) / (E.std(0) + 1e-8)).astype(np.float32)


def boot_ci(yte, p, n=1000):
    rng = np.random.default_rng(SEED)
    r = np.empty(n)
    for j in range(n):
        i = rng.integers(0, len(yte), len(yte))
        r[j] = 1 - ((yte[i] - p[i]) ** 2).sum() / ((yte[i] - yte[i].mean()) ** 2).sum()
    return [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]


def probe(X, y):
    ok = np.isfinite(y)
    Xtr, Xte, ytr, yte = train_test_split(X[ok], y[ok], test_size=0.2, random_state=SEED)
    m = RidgeCV(alphas=ALPHAS).fit(Xtr, ytr)
    p = m.predict(Xte)
    r2 = 1 - ((yte - p) ** 2).sum() / ((yte - yte.mean()) ** 2).sum()
    return dict(r2=float(r2), ci=boot_ci(yte, p), alpha=float(m.alpha_), n=int(ok.sum()))


def direction(X, y):
    ok = np.isfinite(y)
    w = Ridge(alpha=100.0).fit(X[ok], y[ok]).coef_
    return w / (np.linalg.norm(w) + 1e-12)


ef, ei, ok_idx, df = load()
Zf, Zi = zscore(ef), zscore(ei)
c = lambda k: df[k].values
g_r, r_z = c("mag_g_desi") - c("mag_r_desi"), c("mag_r_desi") - c("mag_z_desi")
featured = c("smooth-or-featured_featured-or-disk_fraction")
branch = (featured > 0.5) & (c("disk-edge-on_yes_fraction") < 0.5)

full = {"redshift": c("redshift"), "g_r": g_r, "r_z": r_z,
        "smooth": c("smooth-or-featured_smooth_fraction"), "featured": featured,
        "merger": c("merging_merger_fraction")}
branched = {"spiral": c("has-spiral-arms_yes_fraction"), "bar": c("bar_strong_fraction")}
sparse = {"mass": c("elpetro_mass_log"), "sSFR": c("total_ssfr_median"), "sersic": c("sersic_n")}

out = {"E_img": {}, "modality": {}, "sparse_E_full": {}, "sparse_E_img": {}, "disentangle": {}, "knn_purity": {}}

log("== leakage-free probes on E_img (full-N) ==")
for n, y in {**full, **{k: np.where(branch, v, np.nan) for k, v in branched.items()}}.items():
    out["E_img"][n] = probe(Zi, y)
    r = out["E_img"][n]
    log(f"E_img {n:9s} R2={r['r2']:+.3f} CI={[round(x, 3) for x in r['ci']]} n={r['n']}")

log("== modality ablation (E_img leakage-free vs E_full leakage) ==")
for n in ["redshift", "g_r", "r_z"]:
    out["modality"][n] = dict(E_img=out["E_img"][n]["r2"], E_full=probe(Zf, full[n])["r2"])
    log(f"{n}: E_img={out['modality'][n]['E_img']:+.3f}  E_full={out['modality'][n]['E_full']:+.3f}")

log("== non-input physics (sparse ~4k, exploratory): E_full (has flux+z) vs E_img (image-only) ==")
for n, y in sparse.items():
    out["sparse_E_full"][n] = probe(Zf, y)
    out["sparse_E_img"][n] = probe(Zi, y)
    log(f"{n:7s} E_full R2={out['sparse_E_full'][n]['r2']:+.3f} | E_img R2={out['sparse_E_img'][n]['r2']:+.3f} "
        f"CI_img={[round(x, 3) for x in out['sparse_E_img'][n]['ci']]} n={out['sparse_E_full'][n]['n']}")

log("== disentanglement angles on E_img (full-N) ==")
dn = ["redshift", "g_r", "r_z", "smooth", "merger"]
W = {n: direction(Zi, full[n]) for n in dn}
for i, a in enumerate(dn):
    for b in dn[i + 1:]:
        th = float(np.degrees(np.arccos(np.clip(W[a] @ W[b], -1, 1))))
        fin = np.isfinite(full[a]) & np.isfinite(full[b])
        nul = float(np.degrees(np.arccos(np.clip(np.corrcoef(full[a][fin], full[b][fin])[0, 1], -1, 1))))
        out["disentangle"][f"{a}-{b}"] = dict(theta=th, null=nul, excess=th - nul)
        log(f"{a}-{b}: theta={th:.0f} null={nul:.0f} excess={th - nul:+.0f}")

log("== kNN purity (model-free, E_full, k=20) ==")
_, idx = knn(Zf, 20)
lab = np.where(c("smooth-or-featured_smooth_fraction") > 0.7, 1, np.where(featured > 0.7, 0, -1))
for cls, nm in [(1, "smooth"), (0, "featured")]:
    sel = np.where(lab == cls)[0]
    nb = lab[idx[sel]]
    pur = (nb == cls).sum(1) / np.maximum((nb != -1).sum(1), 1)
    out["knn_purity"][nm] = dict(n=int(len(sel)), purity=float(pur.mean()))
    log(f"{nm}: n={len(sel)} kNN-purity={pur.mean():.3f}")

save_json("probes", out)
log("DONE")
