"""
Track C (audit-grade): redshift as a translation direction + property-angle
battery + the age-metallicity degeneracy, with the honesty fixes from the
2026-06-30 audit applied:

  - Every decode R2 and every angle carries a 200x bootstrap 95% CI (trackUtils).
  - Metallicity is log10-transformed (it spans 0.01-2.0 Zsun, a 200x range;
    age was already log10 -- the two are now treated consistently).
  - The linear "steer +dz and read out" cross-response is RELABELED for what it
    is: with linear probes it is algebraically equivalent to the angle battery
    (steer along w_z, read with w_k -> response = cos(angle) x norms), so it is
    reported as a DERIVED restatement for intuition, NOT independent causal
    evidence. The genuine (nonlinear) causal test is the kNN manifold
    translation, which now carries a bootstrap CI and two nulls.
  - Selection-vs-evolution (Matt's explicit diagnostic): (a) z decodability
    within apparent-magnitude tertiles (does it survive at fixed flux?);
    (b) partial-out test -- how much z decoding survives after removing the
    obvious imaging channels (apparent mag, angular size, colours)?;
    (c) downsizing (exploratory): true stellar age vs z at fixed mass.
  - Robustness variants on the degeneracy angle: mass-weighted age/metal, and
    a z>=0.15 rerun (resolution-sensitivity convention from the main project).
  - Morphology-preservation now has NULLS: the kNN sampling floor (dz=0) and a
    matched-norm random direction.

Run from the project root:  python code/analysis/trackC.py
Needs data/{E_img.npy, ok_index.npy, sample.parquet, anchorAgeMetal.parquet,
anchorShapes.parquet}. Writes results/trackC.json + results/trackC_steer.npy.
"""
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr
from trackUtils import zscore, probe, angle_with_ci, ci, ALPHAS, SEED

DATA, RES = "data", "results"


def fit_full(X, y):
    """RidgeCV on all finite rows -> (w, b) for decoding steered points."""
    ok = np.isfinite(y)
    m = RidgeCV(alphas=ALPHAS).fit(X[ok], y[ok])
    return m.coef_.astype(np.float64), float(m.intercept_)


def main():
    ei = np.load(f"{DATA}/E_img.npy")
    ok = np.load(f"{DATA}/ok_index.npy")
    df = pd.read_parquet(f"{DATA}/sample.parquet").iloc[ok].reset_index(drop=True)
    df["dr8_id"] = df["dr8_id"].astype(str)
    am = pd.read_parquet(f"{DATA}/anchorAgeMetal.parquet"); am["dr8_id"] = am["dr8_id"].astype(str)
    sh = pd.read_parquet(f"{DATA}/anchorShapes.parquet"); sh["dr8_id"] = sh["dr8_id"].astype(str)
    df = df.merge(am[["dr8_id", "ageLightW", "metalLightW", "ageMassW", "metalMassW"]]
                  .drop_duplicates("dr8_id"), on="dr8_id", how="left")
    df = df.merge(sh[["dr8_id", "shape_r"]].drop_duplicates("dr8_id"), on="dr8_id", how="left")
    Zi = zscore(ei)
    c = lambda k: df[k].to_numpy(float)

    z = c("redshift")
    g_r, r_z = c("mag_g_desi") - c("mag_r_desi"), c("mag_r_desi") - c("mag_z_desi")
    mass = np.where((c("elpetro_mass_log") > 5) & np.isfinite(c("elpetro_mass_log")), c("elpetro_mass_log"), np.nan)
    ssfr = np.where(c("total_ssfr_median") > -90, c("total_ssfr_median"), np.nan)
    featured = c("smooth-or-featured_featured-or-disk_fraction")
    log_or_nan = lambda v: np.log10(np.where(v > 0, v, np.nan))
    age = log_or_nan(c("ageLightW"))              # log10(age/yr), Firefly light-weighted
    metal = log_or_nan(c("metalLightW"))          # log10(Z/Zsun)  [AUDIT FIX: was linear]
    age_mw = log_or_nan(c("ageMassW"))            # mass-weighted robustness variants
    metal_mw = log_or_nan(c("metalMassW"))
    L = {"redshift": z, "g_r": g_r, "r_z": r_z, "mass": mass, "sSFR": ssfr,
         "featured": featured, "age": age, "metal": metal}
    out = {"conventions": "age=log10(yr) lightW Firefly; metal=log10(Z/Zsun) lightW (log FIX applied)"}

    # ---- 1. decodability, all with CIs
    out["decode"] = {k: probe(Zi, v) for k, v in L.items()}

    # ---- 2. property-angle battery, all pairs, with CIs
    keys = list(L)
    out["angles"] = {}
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            out["angles"][f"{a}-{b}"] = angle_with_ci(Zi, L, a, b)

    # ---- 3. the age-metallicity degeneracy: headline + robustness variants
    d = out["angles"]["age-metal"]
    out["age_metallicity"] = dict(headline=d,
                                  age_r2=out["decode"]["age"]["r2"], metal_r2=out["decode"]["metal"]["r2"])
    Lmw = {"age": age_mw, "metal": metal_mw}
    out["age_metallicity"]["massweighted"] = angle_with_ci(Zi, Lmw, "age", "metal")
    hz = z >= 0.15                                 # resolution-sensitivity convention
    Lhz = {"age": np.where(hz, age, np.nan), "metal": np.where(hz, metal, np.nan)}
    out["age_metallicity"]["z_ge_0p15"] = angle_with_ci(Zi, Lhz, "age", "metal")

    # ---- 4. linear cross-response (DERIVED, kept for intuition only)
    wz, bz = fit_full(Zi, z)
    vz, nz = wz / np.linalg.norm(wz), float(np.linalg.norm(wz))
    DZ = 0.1
    out["z_steer_response_DERIVED"] = {"_note": ("linear steer + linear readout is algebraically "
        "the angle battery restated (response ~ cos(angle) x norms); NOT independent causal evidence")}
    for k, v in L.items():
        wk, _ = fit_full(Zi, v)
        fv = v[np.isfinite(v)]
        delta = float((DZ / nz) * (vz @ wk))
        out["z_steer_response_DERIVED"][k] = dict(delta=delta, delta_per_sigma=float(delta / (fv.std() + 1e-9)))

    # ---- 5. manifold translation (the GENUINE nonlinear causal test) + CI + nulls
    rng = np.random.default_rng(SEED)
    finz = np.isfinite(z)
    nn = NearestNeighbors(n_neighbors=21, algorithm="brute").fit(Zi[finz])
    zref = z[finz]
    lo = finz & (z >= 0.06) & (z < 0.11); hi = finz & (z >= 0.16) & (z < 0.21)
    d_emb = Zi[hi].mean(0) - Zi[lo].mean(0)
    dz_ref = float(np.median(z[hi]) - np.median(z[lo]))
    pool = np.where(finz & (z > 0.10) & (z < 0.16))[0]
    samp = rng.choice(pool, min(1500, len(pool)), replace=False)
    vr = rng.standard_normal(Zi.shape[1]); vr *= np.linalg.norm(d_emb) / np.linalg.norm(vr)
    grid = [-0.06, -0.03, 0.0, 0.03, 0.06, 0.1]
    nbrz = {"emb": [], "rand": []}                 # per-dz, per-query neighbour-mean z
    feat_shift = {}
    for dz in grid:
        for vec, key in [(d_emb, "emb"), (vr, "rand")]:
            nbr = nn.kneighbors(Zi[samp] + (dz / dz_ref) * vec, return_distance=False)[:, 1:]
            nbrz[key].append(zref[nbr].mean(1))
        if dz in (0.0, 0.1):                       # morphology preservation + its nulls
            for vec, key in [(d_emb, f"emb_{dz}"), (vr, f"rand_{dz}")]:
                nbr = nn.kneighbors(Zi[samp] + (dz / dz_ref) * vec, return_distance=False)[:, 1:]
                idx = np.where(finz)[0][nbr]
                feat_shift[key] = float(np.nanmean(np.abs(featured[idx].mean(1) - featured[samp])))
    nbrz = {k: np.array(v) for k, v in nbrz.items()}          # (n_dz, n_query)
    i0 = grid.index(0.0)

    def slope_of(mat, qsel):
        med = np.median(mat[:, qsel], axis=1)
        return float(np.polyfit(grid, med - med[i0], 1)[0])
    nq = nbrz["emb"].shape[1]
    slope = slope_of(nbrz["emb"], np.arange(nq))
    boots = [slope_of(nbrz["emb"], rng.integers(0, nq, nq)) for _ in range(200)]
    out["translation"] = dict(
        neighbor_z_slope=slope, slope_ci=ci(boots), dz_ref=dz_ref, n_sample=int(nq),
        rand_slope=slope_of(nbrz["rand"], np.arange(nq)),
        featured_shift_at_0p1=feat_shift["emb_0.1"],
        null_knn_floor_dz0=feat_shift["emb_0.0"],
        null_random_dir_0p1=feat_shift["rand_0.1"])

    # ---- 6. selection vs evolution (Matt's diagnostic)
    magr = c("mag_r_desi")
    out["selection"] = {"z_decode_by_magr_tertile": {}}
    fin_m = np.isfinite(magr)
    terts = np.nanpercentile(magr[fin_m], [33.3, 66.7])
    for nm, msk in [("bright", magr < terts[0]),
                    ("mid", (magr >= terts[0]) & (magr < terts[1])),
                    ("faint", magr >= terts[1])]:
        out["selection"]["z_decode_by_magr_tertile"][nm] = probe(Zi, z, fin_m & msk)
    # partial-out: remove apparent mag, angular size, colours from z; re-probe the residual
    size = log_or_nan(c("shape_r"))
    C = np.column_stack([magr, size, g_r, r_z])
    okc = np.isfinite(C).all(1) & finz
    Cz = (C[okc] - C[okc].mean(0)) / (C[okc].std(0) + 1e-9)
    z_res = np.full_like(z, np.nan)
    z_res[okc] = z[okc] - Ridge(alpha=10.0).fit(Cz, z[okc]).predict(Cz)
    r_full = out["decode"]["redshift"]["r2"]
    p_res = probe(Zi, z_res)
    out["selection"]["z_partial_out"] = dict(
        covariates="mag_r, log shape_r, g-r, r-z", n=p_res["n"],
        r2_full=r_full, r2_covariates_only=float(1 - np.var(z_res[okc]) / np.var(z[okc])),
        r2_residual_from_Eimg=p_res["r2"], residual_ci=p_res["ci"])
    # downsizing (exploratory, label-level): true age vs z at fixed mass
    out["selection"]["downsizing_exploratory"] = {}
    for lo_m, hi_m in [(10.0, 10.5), (10.5, 11.0), (11.0, 11.5)]:
        msk = np.isfinite(age) & np.isfinite(mass) & (mass >= lo_m) & (mass < hi_m) & finz
        if msk.sum() > 100:
            sp = spearmanr(age[msk], z[msk])
            out["selection"]["downsizing_exploratory"][f"{lo_m}-{hi_m}"] = dict(
                n=int(msk.sum()), spearman_age_z=float(sp.correlation), p=float(sp.pvalue))

    np.save(f"{RES}/trackC_steer.npy",
            np.column_stack([grid, np.median(nbrz["emb"], 1), np.median(nbrz["rand"], 1)]))
    with open(f"{RES}/trackC.json", "w") as f:
        json.dump(out, f, indent=2, default=float)

    # ---- report
    print("decode R2 (with 95% CI):")
    for k in L:
        r = out["decode"][k]
        print(f"  {k:9s} {r['r2']:+.3f}  CI[{r['ci'][0]:.3f},{r['ci'][1]:.3f}]  n={r['n']}")
    amh = out["age_metallicity"]
    for tag in ("headline", "massweighted", "z_ge_0p15"):
        a = amh[tag] if tag != "headline" else amh["headline"]
        print(f"age-metal angle [{tag:12s}]: embed {a['embed_angle']:.1f} CI{[round(x,1) for x in a['embed_ci']]} "
              f"| null {a['label_null']:.1f} | excess {a['excess']:+.1f} CI{[round(x,1) for x in a['excess_ci']]}")
    t = out["translation"]
    print(f"manifold translation: slope {t['neighbor_z_slope']:.2f} CI{[round(x,2) for x in t['slope_ci']]} "
          f"(random dir {t['rand_slope']:.2f})")
    print(f"  morphology shift under +0.1 z-move: {t['featured_shift_at_0p1']:.3f} "
          f"(kNN floor {t['null_knn_floor_dz0']:.3f}; random-dir {t['null_random_dir_0p1']:.3f})")
    s = out["selection"]
    tz = s["z_decode_by_magr_tertile"]
    print("selection: z decode within apparent-mag tertiles: " +
          " | ".join(f"{k} {tz[k]['r2']:.2f}" for k in tz))
    po = s["z_partial_out"]
    print(f"  z partial-out ({po['covariates']}): full R2 {po['r2_full']:.2f} -> residual-z R2 "
          f"{po['r2_residual_from_Eimg']:.2f} CI{[round(x,2) for x in po['residual_ci']]}")
    print("  downsizing (true age vs z at fixed mass, exploratory): " +
          " | ".join(f"{k}: rho={v['spearman_age_z']:+.2f} (n={v['n']})"
                     for k, v in s["downsizing_exploratory"].items()))
    print("(linear z-steer cross-response kept in JSON as DERIVED-only; see _note)")
    print(f"wrote {RES}/trackC.json + {RES}/trackC_steer.npy")


if __name__ == "__main__":
    main()
