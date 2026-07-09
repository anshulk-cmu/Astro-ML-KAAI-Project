"""
Track A cross-modal check (Matt's ask): do IMAGE-derived and CATALOG-derived
position-angle loops agree, for the same galaxies?

AION natively accepts the Legacy Survey shape parameters (e1, e2, r) as a
catalog-side modality (tok_shape_e1/e2/r) distinct from the image pixels. We feed
the anchor's own Tractor shape values (data/anchorShapes.parquet -- the same
shape_e1/e2/r used to DEFINE the true position angle) through AION's encoder to
get a catalog-derived embedding E_shape, fit the SAME doubled-angle loop probe on
it, and then check whether E_shape's recovered angle agrees with E_img's
recovered angle for the SAME held-out galaxies -- not just whether each agrees
with the (somewhat tautological, since e1/e2 define PA) ground truth.

Run (needs local GPU + polymathic-aion installed):
    python code/analysis/trackA_crossmodal.py
Writes results/trackA_crossmodal.json.
"""
import json
import time
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from aion import AION
from aion.codecs import CodecManager
from aion.modalities import LegacySurveyShapeE1, LegacySurveyShapeE2, LegacySurveyShapeR

import sys
sys.path.insert(0, "code/analysis")
from trackUtils import zscore, ALPHAS, SEED, ci
from sklearn.linear_model import RidgeCV

DATA, RES = "data", "results"
DEV = "cuda"
ELLIP_MIN = 0.3
BATCH = 2048


def circ_fit(X, theta_rad, tr, te, k=2):
    c, s = np.cos(k * theta_rad), np.sin(k * theta_rad)
    mc = RidgeCV(alphas=ALPHAS).fit(X[tr], c[tr])
    ms = RidgeCV(alphas=ALPHAS).fit(X[tr], s[tr])
    return mc.predict(X[te]), ms.predict(X[te]), c[te], s[te]


def circ_err_deg(pc, ps, ct, st, k=2):
    d = np.angle(np.exp(1j * (np.arctan2(ps, pc) - np.arctan2(st, ct))))
    return np.degrees(np.abs(d)) / k


def main():
    ei = np.load(f"{DATA}/E_img.npy")
    ok = np.load(f"{DATA}/ok_index.npy")
    df = pd.read_parquet(f"{DATA}/sample.parquet").iloc[ok].reset_index(drop=True)
    df["dr8_id"] = df["dr8_id"].astype(str)
    sh = pd.read_parquet(f"{DATA}/anchorShapes.parquet")
    sh["dr8_id"] = sh["dr8_id"].astype(str)
    m = df.merge(sh[["dr8_id", "paDeg", "ellip", "shape_e1", "shape_e2", "shape_r"]]
                .drop_duplicates("dr8_id"), on="dr8_id", how="left")
    assert len(m) == len(df)

    pa = np.radians(m["paDeg"].to_numpy(float))
    ellip = m["ellip"].to_numpy(float)
    e1, e2, r = m["shape_e1"].to_numpy(float), m["shape_e2"].to_numpy(float), m["shape_r"].to_numpy(float)
    elong = (np.isfinite(ellip) & (ellip > ELLIP_MIN) & np.isfinite(pa)
            & np.isfinite(e1) & np.isfinite(e2) & np.isfinite(r) & (r > 0))
    idx = np.where(elong)[0]
    print(f"n_elong_with_shapes = {len(idx)}")

    # --- encode the catalog shape triple through AION -> E_shape ---
    print("loading AION...")
    model = AION.from_pretrained(f"{DATA}/models/aion-large").to(DEV).eval()
    cm = CodecManager(device=DEV)
    E_shape = np.zeros((len(idx), 1024), np.float32)
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, len(idx), BATCH):
            sub = idx[s:s + BATCH]
            m1 = LegacySurveyShapeE1(value=torch.tensor(e1[sub], device=DEV, dtype=torch.float32))
            m2 = LegacySurveyShapeE2(value=torch.tensor(e2[sub], device=DEV, dtype=torch.float32))
            m3 = LegacySurveyShapeR(value=torch.tensor(r[sub], device=DEV, dtype=torch.float32))
            tok = cm.encode(m1, m2, m3)
            E_shape[s:s + len(sub)] = model.encode(tok, num_encoder_tokens=3).mean(1).float().cpu().numpy()
            if s % (BATCH * 4) == 0:
                print(f"  {s}/{len(idx)} ({time.time()-t0:.0f}s)")
    print(f"E_shape encoded: {E_shape.shape} in {time.time()-t0:.0f}s")

    Zi = zscore(ei)[idx]
    Zs = zscore(E_shape)
    pa_sub = pa[idx]
    tr, te = train_test_split(np.arange(len(idx)), test_size=0.2, random_state=SEED)

    pc_i, ps_i, ct, st = circ_fit(Zi, pa_sub, tr, te)
    pc_s, ps_s, _, _ = circ_fit(Zs, pa_sub, tr, te)
    err_img_vs_true = circ_err_deg(pc_i, ps_i, ct, st)
    err_shape_vs_true = circ_err_deg(pc_s, ps_s, ct, st)
    err_img_vs_shape = circ_err_deg(pc_i, ps_i, pc_s, ps_s)   # the actual cross-modal check

    rng = np.random.default_rng(SEED)
    def boot_med(e):
        return ci([np.median(e[rng.integers(0, len(e), len(e))]) for _ in range(200)])

    out = dict(
        n_elong_with_shapes=int(len(idx)), n_test=int(len(te)),
        img_vs_true=dict(med_err_deg=float(np.median(err_img_vs_true)), ci=boot_med(err_img_vs_true)),
        shape_vs_true=dict(med_err_deg=float(np.median(err_shape_vs_true)), ci=boot_med(err_shape_vs_true)),
        img_vs_shape=dict(med_err_deg=float(np.median(err_img_vs_shape)), ci=boot_med(err_img_vs_shape)),
        claim_note=("both readouts come from SEPARATELY target-trained probes; their agreement shows "
                    "both modalities decode PA consistently, NOT that the two raw modalities occupy "
                    "one shared oriented 2-D subspace (no cross-probe/Procrustes test was run)"),
    )
    with open(f"{RES}/trackA_crossmodal.json", "w") as f:
        json.dump(out, f, indent=2, default=float)

    print(f"\nimage-embed vs TRUE PA:   median err {out['img_vs_true']['med_err_deg']:.1f} deg CI{out['img_vs_true']['ci']}")
    print(f"shape-embed vs TRUE PA:   median err {out['shape_vs_true']['med_err_deg']:.1f} deg CI{out['shape_vs_true']['ci']}")
    print(f"image-embed vs shape-embed (CROSS-MODAL): median err {out['img_vs_shape']['med_err_deg']:.1f} deg CI{out['img_vs_shape']['ci']}")
    print(f"wrote {RES}/trackA_crossmodal.json")


if __name__ == "__main__":
    main()
