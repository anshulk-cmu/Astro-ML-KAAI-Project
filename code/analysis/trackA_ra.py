"""
Track A, RA control follow-up: is RA's weak decodability (R2~0.41 from E_img) just
sky position leaking through observing conditions, rather than AION reading
coordinates? Partial the observing-condition covariates (E(B-V), depth, PSF size)
out of RA and re-probe E_img. If the residual RA signal collapses, it was nuisance.

Run from the project root:  python code/analysis/trackA_ra.py
Needs data/E_img.npy, data/ok_index.npy, data/sample.parquet, data/anchorCovariates.parquet.
"""
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.model_selection import train_test_split

DATA, RES = "data", "results"
ALPHAS = np.logspace(-2, 4, 7)
SEED = 0
COV = ("ebv psfsize_g psfsize_r psfsize_i psfsize_z "
       "psfdepth_g psfdepth_r psfdepth_i psfdepth_z").split()


def zscore(M):
    return ((M - np.nanmean(M, 0)) / (np.nanstd(M, 0) + 1e-8))


def r2_probe(X, y):
    ok = np.isfinite(y) & np.isfinite(X).all(1)
    Xtr, Xte, ytr, yte = train_test_split(X[ok], y[ok], test_size=0.2, random_state=SEED)
    p = RidgeCV(alphas=ALPHAS).fit(Xtr, ytr).predict(Xte)
    return float(1 - ((yte - p) ** 2).sum() / ((yte - yte.mean()) ** 2).sum())


def direction(X, y):
    ok = np.isfinite(y) & np.isfinite(X).all(1)
    w = Ridge(alpha=100.0).fit(X[ok], y[ok]).coef_
    return w / (np.linalg.norm(w) + 1e-12)


def main():
    ei = np.load(f"{DATA}/E_img.npy")
    ok = np.load(f"{DATA}/ok_index.npy")
    df = pd.read_parquet(f"{DATA}/sample.parquet").iloc[ok].reset_index(drop=True)
    df["dr8_id"] = df["dr8_id"].astype(str)
    cov = pd.read_parquet(f"{DATA}/anchorCovariates.parquet")
    cov["dr8_id"] = cov["dr8_id"].astype(str)
    m = df.merge(cov[["dr8_id"] + COV].drop_duplicates("dr8_id"), on="dr8_id", how="left")

    Zi = zscore(ei).astype(np.float32)
    ra = np.radians(m["ra"].to_numpy(float))
    C = m[COV].to_numpy(float)
    C = np.where(np.isfinite(C), C, np.nanmedian(C, 0))      # impute missing (e.g. i-depth=0)
    Cz = zscore(C)
    out = {"covariates": COV}

    for nm, tgt in [("cosRA", np.cos(ra)), ("sinRA", np.sin(ra))]:
        r_emb = r2_probe(Zi, tgt)                            # E_img predicts RA component
        r_cov = r2_probe(Cz, tgt)                            # observing conditions predict it
        ok = np.isfinite(tgt) & np.isfinite(Cz).all(1)
        resid = tgt.copy()
        resid[ok] = tgt[ok] - Ridge(alpha=10.0).fit(Cz[ok], tgt[ok]).predict(Cz[ok])
        r_resid = r2_probe(Zi, resid)                        # E_img predicts RA BEYOND conditions
        out[nm] = dict(r2_from_Eimg=r_emb, r2_from_covariates=r_cov,
                       r2_residual_from_Eimg=r_resid,
                       explained_frac=float(1 - r_resid / r_emb) if r_emb > 0 else None)

    # does E_img encode the observing conditions, and does the RA direction align with them?
    out["cond_in_Eimg"] = {c: r2_probe(Zi, m[c].to_numpy(float)) for c in ["ebv", "psfdepth_r", "psfsize_r"]}
    d_ra = direction(Zi, np.cos(ra))
    out["align_cosRA_dir"] = {c: float(abs(d_ra @ direction(Zi, m[c].to_numpy(float))))
                              for c in ["ebv", "psfdepth_r", "psfsize_r"]}

    with open(f"{RES}/trackA_ra.json", "w") as f:
        json.dump(out, f, indent=2, default=float)

    for nm in ("cosRA", "sinRA"):
        o = out[nm]
        print(f"{nm}: E_img R2={o['r2_from_Eimg']:.2f} | from covariates R2={o['r2_from_covariates']:.2f} | "
              f"residual-after-covariates R2={o['r2_residual_from_Eimg']:.2f} "
              f"({100*(o['explained_frac'] or 0):.0f}% explained by conditions)")
    print(f"observing conditions in E_img: ebv R2={out['cond_in_Eimg']['ebv']:.2f}, "
          f"depth_r R2={out['cond_in_Eimg']['psfdepth_r']:.2f}, psfsize_r R2={out['cond_in_Eimg']['psfsize_r']:.2f}")
    print(f"|cos(RA-dir, condition-dir)|: {out['align_cosRA_dir']}")
    print(f"wrote {RES}/trackA_ra.json")


if __name__ == "__main__":
    main()
