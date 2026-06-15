"""
Full-anchor photometric stellar masses via a self-calibrated colour-luminosity
relation (preRegistration Track 2 / fallback, §11.5).

Fits log M* on the ~3.7k anchor galaxies that carry NSA elpetro masses, using
DESI g/r/z colours + distance-corrected z-band magnitude + redshift, then applies
to all 48,398. The primary use (rank matching within z-bins) is invariant to a
monotone recalibration, so rank fidelity is the headline metric; absolute scale
inherits elpetro (NSA Chabrier) and is recalibrated when the Zou DR9 SPS masses
land. Output is on the elpetro scale, no h-conversion applied here.
"""
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from scipy.stats import spearmanr

H0, OM, OL = 67.74, 0.3089, 0.6911       # ASTRID/Planck; distance scale only
C_KMS = 299792.458
SEED = 0
ZBINS = np.arange(0.0, 0.55, 0.05)


def distmod(z):
    """Distance modulus for a flat LCDM, vectorised numerical integral."""
    zg = np.linspace(0.0, float(np.nanmax(z)) + 1e-3, 4000)
    Ez = np.sqrt(OM * (1 + zg) ** 3 + OL)
    dc = (C_KMS / H0) * np.concatenate([[0.0], np.cumsum(0.5 * (1 / Ez[1:] + 1 / Ez[:-1]) * np.diff(zg))])
    dL = (1 + z) * np.interp(z, zg, dc)       # Mpc
    return 5 * np.log10(np.clip(dL, 1e-6, None) * 1e5)


def load():
    df = pd.read_parquet("data/sample.parquet")
    ok = np.load("data/ok_index.npy")
    d = df.iloc[ok].reset_index(drop=True)
    num = lambda c: pd.to_numeric(d[c], errors="coerce").to_numpy(float)
    out = pd.DataFrame({
        "dr8_id": d["dr8_id"].astype(str).values,
        "ra": num("ra"), "dec": num("dec"), "z": num("redshift"),
        "g": num("mag_g_desi"), "r": num("mag_r_desi"), "zmag": num("mag_z_desi"),
        "elpetro": num("elpetro_mass_log"),
    })
    return out


def features(d):
    dm = distmod(np.clip(d["z"].values, 1e-4, None))
    gr = d["g"].values - d["r"].values
    rz = d["r"].values - d["zmag"].values
    Mz = d["zmag"].values - dm                 # absolute z mag (no K-corr); z soaks K
    return np.column_stack([gr, rz, Mz, d["z"].values])


def main():
    d = load()
    X = features(d)
    base_ok = np.all(np.isfinite(X), axis=1)
    train = base_ok & np.isfinite(d["elpetro"].values) & (d["elpetro"].values > 0)
    print(f"anchor={len(d)} usable_features={base_ok.sum()} calib(elpetro)={train.sum()}")

    Xtr, ytr = X[train], d["elpetro"].values[train]
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(Xtr))
    cut = int(0.8 * len(idx))
    tr, te = idx[:cut], idx[cut:]

    model = make_pipeline(
        PolynomialFeatures(degree=2, include_bias=False),
        StandardScaler(),
        RidgeCV(alphas=np.logspace(-3, 3, 13)),
    )
    model.fit(Xtr[tr], ytr[tr])
    pred_te = model.predict(Xtr[te])
    r2 = 1 - np.sum((ytr[te] - pred_te) ** 2) / np.sum((ytr[te] - ytr[te].mean()) ** 2)
    rmse = float(np.sqrt(np.mean((ytr[te] - pred_te) ** 2)))
    rho_all = float(spearmanr(ytr[te], pred_te).statistic)

    # rank fidelity within z-bins (the preReg acceptance metric, >= 0.8)
    zte = Xtr[te][:, 3]
    per_zbin = {}
    for lo, hi in zip(ZBINS[:-1], ZBINS[1:]):
        m = (zte >= lo) & (zte < hi)
        if m.sum() >= 30:
            per_zbin[f"{lo:.2f}-{hi:.2f}"] = {
                "n": int(m.sum()),
                "spearman": float(spearmanr(ytr[te][m], pred_te[m]).statistic),
            }

    # refit on ALL calibration data, apply to the full anchor
    model.fit(Xtr, ytr)
    logm = np.full(len(d), np.nan)
    logm[base_ok] = model.predict(X[base_ok])

    # loose training-hull flag (1-99 pct per feature)
    lo = np.percentile(Xtr, 1, axis=0)
    hi = np.percentile(Xtr, 99, axis=0)
    in_hull = base_ok & np.all((X >= lo) & (X <= hi), axis=1)

    d["logM_colour"] = logm
    d["inTrainHull"] = in_hull
    d[["dr8_id", "ra", "dec", "z", "g", "r", "zmag", "elpetro",
       "logM_colour", "inTrainHull"]].to_parquet("data/anchorMass.parquet")

    summary = {
        "n_anchor": int(len(d)),
        "n_with_colourmass": int(np.isfinite(logm).sum()),
        "n_calibration_elpetro": int(train.sum()),
        "holdout_r2": float(r2),
        "holdout_rmse_dex": rmse,
        "holdout_spearman_overall": rho_all,
        "holdout_spearman_per_zbin": per_zbin,
        "ridge_alpha": float(model.named_steps["ridgecv"].alpha_),
        "features": ["g-r", "r-z", "Mz_abs_noK", "z", "(degree-2 poly)"],
        "in_train_hull_frac": float(in_hull.sum() / base_ok.sum()),
        "mass_scale": "NSA elpetro (Chabrier IMF); not h-converted; recalibrate to Zou later",
        "cosmology": "flat LCDM H0=67.74 Om0=0.3089 (distance scale only)",
        "preReg_acceptance": "rank Spearman >= 0.8 within z-bins",
    }
    with open("results/anchorMass.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
