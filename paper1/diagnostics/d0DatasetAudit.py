"""Dataset and substrate audit. Reference block for report sections 1 and 2.

Every count, shape, range and coverage figure quoted in the report's data and model sections
comes from here, so none of them is transcribed by hand.

    python paper1/diagnostics/d0DatasetAudit.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

import numpy as np

import config as C
import data as D
import provenance

NAME = "d0DatasetAudit"


def describe(x, qs=(1, 50, 99)):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return {"n_finite": int(len(x)), "percentiles": {str(q): float(np.percentile(x, q)) for q in qs},
            "min": float(x.min()), "max": float(x.max())}


def substrate_stats(name):
    Z, path = D.substrate(name)
    E = np.load(path)
    norms = np.linalg.norm(E, axis=1)
    sd = E.std(0)
    return {"path": str(path), "shape": list(E.shape), "dtype": str(E.dtype),
            "n_nan": int(np.isnan(E).sum()), "n_inf": int(np.isinf(E).sum()),
            "norm_mean": float(norms.mean()), "norm_std": float(norms.std()),
            "per_dim_std_min": float(sd.min()), "per_dim_std_max": float(sd.max()),
            "per_dim_std_ratio": float(sd.max() / sd.min())}


def main():
    t0 = time.time()
    df = D.anchor()
    ok = np.load(C.OK_INDEX)
    n = len(df)

    out = {"anchor": {
        "n_rows": int(n),
        "n_unique_dr8_id": int(df["dr8_id"].nunique()),
        "ok_index_len": int(len(ok)),
        "row_contract": "E_img row i is sample.parquet row ok_index[i]; catalog tables left-joined on dr8_id"}}

    out["recipe"] = {"model_path": str(C.MODEL), "encoder_tokens": C.ENCODER_TOKENS,
                     "pooling": C.POOLING, "pixel_scale_arcsec": C.PIXEL_SCALE,
                     "cutout_pixels": C.CUTOUT_PIX, "bands": list(C.BANDS),
                     "zeropoint": C.ZEROPOINT,
                     "field_of_view_arcsec": round(C.PIXEL_SCALE * C.CUTOUT_PIX, 3)}

    out["substrates"] = {k: substrate_stats(k) for k in ("img", "full")
                         if {"img": C.E_IMG, "full": C.E_FULL}[k].exists()}

    out["analysis_constants"] = {"seed": C.SEED, "test_size": C.TEST_SIZE,
                                 "alphas": [float(a) for a in C.ALPHAS],
                                 "alpha_direction": C.ALPHA_DIRECTION,
                                 "n_boot": C.N_BOOT, "ellip_cut": C.ELLIP_CUT}

    ty = df["type"].astype(str)
    pa_ok = D.pa_defined(df)
    ellip = df["ellip"].to_numpy(float)
    out["shape_catalog"] = {
        "n_with_shape_row": int(np.isfinite(df["paDeg"].to_numpy(float)).sum()),
        "n_pa_defined": int(pa_ok.sum()),
        "n_circular_models": int((np.isfinite(df["paDeg"].to_numpy(float)) & ~pa_ok).sum()),
        "type_counts": {t: int((ty == t).sum()) for t in sorted(ty.unique())},
        "n_above_ellip_cut": int((pa_ok & np.isfinite(ellip) & (ellip > C.ELLIP_CUT)).sum()),
        "ellipticity": describe(ellip[pa_ok]),
        "sigma_pa_deg": describe(D.pa_uncertainty(df)[pa_ok])}

    labels = {"redshift": "redshift", "spec_z": "spec_z",
              "mag_r_desi": "mag_r_desi", "elpetro_mass_log": "elpetro_mass_log",
              "total_ssfr_median": "total_ssfr_median", "sersic_n": "sersic_n",
              "smooth_fraction": "smooth-or-featured_smooth_fraction",
              "featured_fraction": "smooth-or-featured_featured-or-disk_fraction",
              "edgeon_fraction": "disk-edge-on_yes_fraction"}
    out["label_coverage"] = {k: {"n_finite": int(np.isfinite(df[c].to_numpy(float)).sum()),
                                 "coverage": round(float(np.isfinite(df[c].to_numpy(float)).mean()), 4)}
                             for k, c in labels.items() if c in df}

    cov = {"psfsize_r": "psfsize_r", "psfdepth_r": "psfdepth_r", "ebv": "ebv"}
    out["covariates"] = {k: describe(df[c].to_numpy(float)) for k, c in cov.items() if c in df}
    fp = df["footprint"].astype(str)
    out["covariates"]["footprint_counts"] = {t: int((fp == t).sum()) for t in sorted(fp.unique())}

    out["redshift"] = describe(df["redshift"].to_numpy(float))

    path = provenance.write(NAME, out, [C.E_IMG, C.E_FULL, C.OK_INDEX, C.SAMPLE, C.SHAPES,
                                        C.COVARIATES], t0)
    a = out["anchor"]
    print(f"anchor {a['n_rows']} rows, {a['n_unique_dr8_id']} unique ids")
    for k, v in out["substrates"].items():
        print(f"  E_{k}: {v['shape']} {v['dtype']} nan={v['n_nan']} inf={v['n_inf']} "
              f"|x|={v['norm_mean']:.2f}+-{v['norm_std']:.2f} per-dim std ratio {v['per_dim_std_ratio']:.1f}")
    s = out["shape_catalog"]
    print(f"  shapes: {s['n_with_shape_row']} rows, {s['n_pa_defined']} PA defined, "
          f"{s['n_circular_models']} circular models, {s['n_above_ellip_cut']} above cut")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
