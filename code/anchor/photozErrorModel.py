"""
Anchor photo-z error model (preRegistration §12 / 17.7).

Calibrated on the ~6.7k anchor galaxies with both spec-z and photo-z. Provides
(a) the measured photo-z scatter the anchor carries, and (b) the stratum-matched
noise model used to perturb each mock's assigned redshift into an observed-z
(both domains binned on observed-z, never true z). Normalised residual
dz = (photo_z - spec_z) / (1 + spec_z).
"""
import json
import numpy as np
import pandas as pd

ZBINS = np.arange(0.0, 0.55, 0.05)
OUTLIER = 0.15


def nmad(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def main():
    df = pd.read_parquet("data/sample.parquet")
    ok = np.load("data/ok_index.npy")
    d = df.iloc[ok].reset_index(drop=True)
    sz = pd.to_numeric(d["spec_z"], errors="coerce").to_numpy(float)
    pz = pd.to_numeric(d["photo_z"], errors="coerce").to_numpy(float)
    m = np.isfinite(sz) & (sz > 0) & np.isfinite(pz)
    sz, pz = sz[m], pz[m]
    dz = (pz - sz) / (1 + sz)
    print(f"spec&photo pairs = {m.sum()}")

    glob = {
        "n": int(m.sum()),
        "bias_median": float(np.median(dz)),
        "sigma_nmad": float(nmad(dz)),
        "outlier_frac": float(np.mean(np.abs(dz) > OUTLIER)),
        "raw_rms_dz": float(np.sqrt(np.mean((pz - sz) ** 2))),
    }
    per_bin = {}
    for lo, hi in zip(ZBINS[:-1], ZBINS[1:]):
        b = (sz >= lo) & (sz < hi)
        if b.sum() >= 30:
            per_bin[f"{lo:.2f}-{hi:.2f}"] = {
                "n": int(b.sum()),
                "bias_median": float(np.median(dz[b])),
                "sigma_nmad": float(nmad(dz[b])),
                "outlier_frac": float(np.mean(np.abs(dz[b]) > OUTLIER)),
            }

    # save residuals (per spec-z bin) for empirical resampling of the noise model
    np.save("data/photozResiduals.npy",
            np.column_stack([sz, dz]).astype(np.float32))

    summary = {
        "definition": "dz = (photo_z - spec_z)/(1+spec_z); sigma = 1.4826*MAD",
        "global": glob,
        "per_specz_bin": per_bin,
        "outlier_threshold": OUTLIER,
        "usage": "draw observed_z = assigned_z + N(bias,sigma) or resample residuals, per z-bin, both domains",
        "caveat": "spec-z subset is brighter/lower-z than the full anchor; high-z bins sparse",
    }
    with open("results/photozError.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
