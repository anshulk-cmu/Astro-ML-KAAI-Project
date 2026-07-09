"""
Track A mirror-flip causal test: completes the O(2) group (rotations verified in
trackA_causal.py; this adds the reflection). Physically mirror every elongated
galaxy's real cutout, re-encode through AION's image encoder, and check that the
FIXED doubled-angle probe's readout reflects: theta -> -theta (mod 180).

Geometry (fixed before running): np.flip on the column axis maps the array-frame
angle theta_arr -> -theta_arr (mod 180). Catalog PA = theta_arr - 90 (same-handed,
verified runLog 2026-07-01), and the 90-deg offset cancels mod 180, so the catalog
readout must go to -PA. Fixed points at PA in {0, 90} (axes parallel/perpendicular
to the flip axis); maximal displacement (+-90) at 45/135. If the true frame offset
c differs slightly from 90, the flip readout carries a constant -2(c-90) shift
(reflections conjugate the offset; rotations cancel it) -- estimated from the data
and reported. Unlike rotation the flip is INTERPOLATION-FREE (exact pixel
permutation), so this is the cleanest input-space intervention in the battery.

A second pass (flip, then rotate +30 deg) tests the group COMPOSITION law:
rotation shifts the readout by -phi on top of the reflection (verified rotation
convention), so the expected readout is -PA - 30 (a reflection about a different
axis).

No subsampling: all elongated galaxies (ellip>0.3), same population as Track A.

Run (needs local GPU + polymathic-aion): python code/analysis/trackA_flip.py
Optional smoke test first:                python code/analysis/trackA_flip.py --smoke 100
Checkpoints per-op to results/trackA_flip_ckpt/ (resumable; with both checkpoints
present a re-run is CPU-only). Writes results/trackA_flip.json.
"""
import argparse
import json
import os
import time
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, "code/analysis")
from trackUtils import zscore, ALPHAS, SEED, ci
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split

DATA, RES = "data", "results"
CKPT = f"{RES}/trackA_flip_ckpt"
os.makedirs(CKPT, exist_ok=True)
DEV = "cuda"
BATCH = 32
ELLIP_MIN = 0.3
OPS = [("flip", 0.0), ("flip_rot30", 30.0)]   # (name, rotation applied AFTER the flip)


def load_anchor():
    ei = np.load(f"{DATA}/E_img.npy")
    ok = np.load(f"{DATA}/ok_index.npy")
    df = pd.read_parquet(f"{DATA}/sample.parquet").iloc[ok].reset_index(drop=True)
    df["dr8_id"] = df["dr8_id"].astype(str)
    sh = pd.read_parquet(f"{DATA}/anchorShapes.parquet")
    sh["dr8_id"] = sh["dr8_id"].astype(str)
    m = df.merge(sh[["dr8_id", "paDeg", "ellip"]].drop_duplicates("dr8_id"),
                 on="dr8_id", how="left")
    assert len(m) == len(df)
    return ei, ok, m


def encode_op(model, cm, imgs, gal_idx, rot_deg, batch=BATCH, log_every=40):
    import torch
    from scipy.ndimage import rotate as sp_rotate
    from aion.modalities import LegacySurveyImage
    n = len(gal_idx)
    out = np.zeros((n, 1024), np.float32)
    t0 = time.time()
    for s in range(0, n, batch):
        sub = gal_idx[s:s + batch]
        cube = np.ascontiguousarray(np.flip(imgs[sub], axis=-1))   # exact mirror
        if rot_deg != 0.0:
            rot = np.empty_like(cube)
            for i in range(cube.shape[0]):
                for b in range(4):
                    rot[i, b] = sp_rotate(cube[i, b], rot_deg, reshape=False,
                                          order=1, mode="constant", cval=0.0)
            cube = rot
        with torch.no_grad():
            flux = torch.from_numpy(cube).to(DEV)
            img = LegacySurveyImage(flux=flux, bands=["DES-G", "DES-R", "DES-I", "DES-Z"])
            tok = cm.encode(img)
            out[s:s + len(sub)] = model.encode(tok, num_encoder_tokens=600).mean(1).float().cpu().numpy()
        if (s // batch) % log_every == 0:
            print(f"    rot_after_flip={rot_deg:.0f} {s+len(sub)}/{n} ({time.time()-t0:.0f}s)", flush=True)
    return out


def wrap2(deg):
    """fold a PA-scale difference to (-90, 90]"""
    return (np.asarray(deg) + 90.0) % 180.0 - 90.0


def circ_err(a_deg, b_deg):
    """circular |a-b| on the mod-180 PA scale, degrees"""
    return np.abs(wrap2(a_deg - b_deg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0, help="encode only the first N galaxies, no checkpoints")
    args = ap.parse_args()

    ei, ok, m = load_anchor()
    Zi_full = zscore(ei)
    pa = m["paDeg"].to_numpy(float)                       # degrees, [0,180)
    ellip = m["ellip"].to_numpy(float)
    elong = np.isfinite(ellip) & (ellip > ELLIP_MIN) & np.isfinite(pa)
    idx = np.where(elong)[0]
    if args.smoke:
        idx = idx[:args.smoke]
    gal_idx = ok[idx]
    print(f"n = {len(idx)}" + (" (SMOKE)" if args.smoke else " (full elongated population)"), flush=True)

    # fixed probe, fit once on real un-transformed embeddings (identical to trackA_causal)
    tr, te = train_test_split(np.arange(len(idx)), test_size=0.2, random_state=SEED)
    par = np.radians(pa[idx])
    mc = RidgeCV(alphas=ALPHAS).fit(Zi_full[idx][tr], np.cos(2 * par[tr]))
    ms = RidgeCV(alphas=ALPHAS).fit(Zi_full[idx][tr], np.sin(2 * par[tr]))
    readout = lambda X: np.degrees(0.5 * np.arctan2(ms.predict(X), mc.predict(X)))
    base_err = circ_err(readout(Zi_full[idx][te]), pa[idx][te])
    print(f"baseline (untransformed, held-out) median err = {np.median(base_err):.2f} deg", flush=True)
    pa0_hat = readout(Zi_full[idx])                       # unflipped readout, all galaxies

    imgs = np.load(f"{DATA}/images.npy", mmap_mode="r")
    need = [(nm, rd) for nm, rd in OPS
            if args.smoke or not os.path.exists(f"{CKPT}/{nm}.npy")]
    model = cm = None
    if need:
        import torch
        from aion import AION
        from aion.codecs import CodecManager
        model = AION.from_pretrained(f"{DATA}/models/aion-large").to(DEV).eval()
        cm = CodecManager(device=DEV)

    hats = {}
    for nm, rot_deg in OPS:
        ckf = f"{CKPT}/{nm}.npy"
        if not args.smoke and os.path.exists(ckf):
            print(f"{nm}: loading checkpoint", flush=True)
            emb = np.load(ckf)
        else:
            print(f"{nm}: encoding {len(idx)} mirrored galaxies...", flush=True)
            emb = encode_op(model, cm, imgs, gal_idx, rot_deg)
            if not args.smoke:
                np.save(ckf, emb)
        Zr = ((emb - ei.mean(0)) / (ei.std(0) + 1e-8)).astype(np.float32)
        hats[nm] = readout(Zr)

    rng = np.random.default_rng(SEED)
    boot_med = lambda e: ci([np.median(e[rng.integers(0, len(e), len(e))]) for _ in range(200)])
    rng_ho = np.random.default_rng(SEED + 1)   # separate stream: keeps all-population numbers identical
    boot_ho = lambda e: ci([np.median(e[rng_ho.integers(0, len(e), len(e))]) for _ in range(200)])
    out = {"n": int(len(idx)), "smoke": bool(args.smoke), "n_heldout": int(len(te)),
           "population_note": ("plain fields summarize ALL elongated galaxies, which include the probe's "
                               "80% training rows; the *_heldout fields use only the 20% test split the "
                               "probe never saw and are the leakage-free summaries"),
           "baseline_median_err_deg": float(np.median(base_err)),
           "sign_convention": ("np.flip on the column axis maps the array-frame angle to its negative "
                               "(mod 180); catalog PA = array angle - 90 same-handed (verified runLog "
                               "2026-07-01) and the 90-deg offset cancels mod 180 -> expected readout "
                               "-PA, fixed points PA in {0,90}. A residual frame offset c-90 would "
                               "appear as a constant -2(c-90) shift (reflections conjugate the offset; "
                               "rotations cancel it) -> estimated below as offset_est_deg. Composition: "
                               "rotate(+30) after the flip shifts the readout by an additional -30.")}

    # --- flip: readout must be -PA ---
    h = hats["flip"]
    err0 = circ_err(h, -pa0_hat)                          # vs the probe's own unflipped readout
    errt = circ_err(h, -pa[idx])                          # vs the catalog truth
    offset = float(np.median(wrap2(h - (-pa0_hat))))      # signed; ~ -2(c-90)
    err0c = circ_err(h, -pa0_hat + offset)
    d_rec = wrap2(h - pa0_hat)                            # recovered displacement
    d_pred = wrap2(-2.0 * pa[idx])                        # predicted -2*theta (0 at fixed points)
    keep = np.abs(d_pred) <= 80.0                         # exclude the +-90 wrap boundary
    slope, intercept = np.polyfit(d_pred[keep], d_rec[keep], 1)
    resid = d_rec[keep] - (slope * d_pred[keep] + intercept)
    near = lambda t, c: np.abs(wrap2(t - c)) < 5.0
    fp = near(pa[idx], 0.0) | near(pa[idx], 90.0)         # fixed points
    an = near(pa[idx], 45.0) | near(pa[idx], 135.0)       # antinodes
    te_mask = np.zeros(len(idx), bool); te_mask[te] = True
    keep_ho = keep & te_mask
    slope_ho, icpt_ho = np.polyfit(d_pred[keep_ho], d_rec[keep_ho], 1)
    out["flip"] = dict(
        median_err_vs_minus_readout_deg=float(np.median(err0)), err_ci=boot_med(err0),
        median_err_vs_minus_true_deg=float(np.median(errt)), err_true_ci=boot_med(errt),
        offset_est_deg=offset,
        median_err_offset_corrected_deg=float(np.median(err0c)),
        disp_slope=float(slope), disp_intercept=float(intercept),
        disp_median_abs_resid=float(np.median(np.abs(resid))),
        disp_p95_abs_resid=float(np.percentile(np.abs(resid), 95)),
        n_slope=int(keep.sum()), slope_excluded="|predicted displacement| > 80 (mod-180 wrap boundary)",
        fixedpoint_median_absdisp_deg=float(np.median(np.abs(d_rec[fp]))), n_fixedpoint=int(fp.sum()),
        antinode_median_absdisp_deg=float(np.median(np.abs(d_rec[an]))), n_antinode=int(an.sum()),
        median_err_vs_minus_readout_deg_heldout=float(np.median(err0[te])), err_ci_heldout=boot_ho(err0[te]),
        median_err_vs_minus_true_deg_heldout=float(np.median(errt[te])),
        disp_slope_heldout=float(slope_ho), disp_intercept_heldout=float(icpt_ho),
        n_slope_heldout=int(keep_ho.sum()))

    # --- flip then rotate 30: readout must be -PA - 30 (group composition) ---
    h2 = hats["flip_rot30"]
    errc0 = circ_err(h2, -pa0_hat - 30.0)
    errct = circ_err(h2, -pa[idx] - 30.0)
    out["flip_rot30"] = dict(
        expected="-(unflipped readout) - 30 deg",
        median_err_vs_expected_deg=float(np.median(errc0)), err_ci=boot_med(errc0),
        median_err_vs_true_deg=float(np.median(errct)),
        median_err_offset_corrected_deg=float(np.median(circ_err(h2, -pa0_hat - 30.0 + offset))),
        median_err_vs_expected_deg_heldout=float(np.median(errc0[te])), err_ci_heldout=boot_ho(errc0[te]))

    if not args.smoke:
        np.save(f"{RES}/trackA_flip_disp.npy",
                np.column_stack([pa[idx], d_pred, d_rec]).astype(np.float32))
        with open(f"{RES}/trackA_flip.json", "w") as f:
            json.dump(out, f, indent=2, default=float)

    f_ = out["flip"]
    print(f"\nFLIP: median circular err vs -readout = {f_['median_err_vs_minus_readout_deg']:.2f} deg "
          f"CI{f_['err_ci']} (vs -truth {f_['median_err_vs_minus_true_deg']:.2f}; baseline {out['baseline_median_err_deg']:.2f})")
    print(f"  frame-offset estimate {f_['offset_est_deg']:+.2f} deg -> offset-corrected err "
          f"{f_['median_err_offset_corrected_deg']:.2f} deg")
    print(f"  displacement law: slope {f_['disp_slope']:.3f} (expect +1), intercept {f_['disp_intercept']:+.2f}, "
          f"median|resid| {f_['disp_median_abs_resid']:.2f}, p95 {f_['disp_p95_abs_resid']:.2f} (n={f_['n_slope']})")
    print(f"  fixed points (PA~0/90, n={f_['n_fixedpoint']}): median |disp| = {f_['fixedpoint_median_absdisp_deg']:.2f} deg | "
          f"antinodes (PA~45/135, n={f_['n_antinode']}): {f_['antinode_median_absdisp_deg']:.2f} deg (expect ~90)")
    c_ = out["flip_rot30"]
    print(f"FLIP+ROT30 (composition): median err vs (-readout - 30) = "
          f"{c_['median_err_vs_expected_deg']:.2f} deg CI{c_['err_ci']}")
    if not args.smoke:
        print(f"wrote {RES}/trackA_flip.json + trackA_flip_disp.npy")


if __name__ == "__main__":
    main()
