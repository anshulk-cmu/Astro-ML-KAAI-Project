"""Aligned loading of substrates, labels and covariates.

Row contract: E_img row i is sample.parquet row ok_index[i]. Catalog tables are left-joined
on dr8_id, so a missing galaxy becomes NaN rather than shifting a row.
"""
import numpy as np
import pandas as pd

import config as C


def anchor():
    ok = np.load(C.OK_INDEX)
    df = pd.read_parquet(C.SAMPLE).iloc[ok].reset_index(drop=True)
    df["dr8_id"] = df["dr8_id"].astype(str)

    sh = pd.read_parquet(C.SHAPES)
    sh["dr8_id"] = sh["dr8_id"].astype(str)
    df = df.merge(sh[["dr8_id", "paDeg", "ellip", "ba", "inclDeg", "shape_r", "shape_e1",
                      "shape_e2", "shape_e1_ivar", "shape_e2_ivar", "ellipSnr", "type"]]
                  .drop_duplicates("dr8_id"), on="dr8_id", how="left")

    cv = pd.read_parquet(C.COVARIATES)
    cv["dr8_id"] = cv["dr8_id"].astype(str)
    df = df.merge(cv[["dr8_id", "psfsize_r", "psfdepth_r", "ebv", "footprint"]]
                  .drop_duplicates("dr8_id"), on="dr8_id", how="left")

    assert len(df) == len(ok), "merge changed row count"
    return df


def substrate(name="img"):
    path = {"img": C.E_IMG, "full": C.E_FULL, "img_base": C.E_IMG_BASE}[name]
    E = np.load(path)
    return ((E - E.mean(0)) / (E.std(0) + 1e-8)).astype(np.float32), path


def pa_defined(df):
    """Position angle exists only where the catalog fitted an ellipticity. REX, PSF and DUP
    models are circular by construction, carry shape_e1 = shape_e2 = 0 exactly, and would
    read as a placeholder PA of 0."""
    e1, e2 = df["shape_e1"].to_numpy(float), df["shape_e2"].to_numpy(float)
    return (np.isfinite(df["paDeg"].to_numpy(float)) & np.isfinite(e1) & np.isfinite(e2)
            & ~((e1 == 0) & (e2 == 0)))


def pa_uncertainty(df):
    """Catalog sigma_PA in degrees, propagated from the ellipticity component variances."""
    e1, e2 = df["shape_e1"].to_numpy(float), df["shape_e2"].to_numpy(float)
    i1, i2 = df["shape_e1_ivar"].to_numpy(float), df["shape_e2_ivar"].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        var_phi = (e2 ** 2 / i1 + e1 ** 2 / i2) / (e1 ** 2 + e2 ** 2) ** 2
        sig = np.degrees(np.sqrt(var_phi)) / 2.0
    sig[~np.isfinite(sig)] = np.nan
    return sig
