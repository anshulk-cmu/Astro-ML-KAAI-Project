"""
Track A causal test (Matt's ask): rotate a real galaxy image by a KNOWN angle,
re-embed through AION's actual image encoder, and check whether the readout
shifts by the rotation applied -- the direct causal "rotate along the loop, check
the readout" test, as opposed to Track A's original purely observational probe.

No subsampling: uses ALL elongated anchor galaxies (ellip>0.3, the same physical
restriction applied everywhere else in Track A -- PA is undefined for round
galaxies, this is not a convenience subsample).

Method: fit the doubled-angle probe (cos2theta,sin2theta ~ E_img) ONCE on the
real, UNROTATED embeddings (already computed, data/E_img.npy). Then, for each of
several known rotation angles, rotate every elongated galaxy's real cutout by
that angle (scipy CPU rotation, bilinear, reshape=False), re-encode through AION
locally (GPU, batch=32 -- the measured steady-state throughput sweet spot on this
12GB card), and apply the SAME fixed probe to the new embeddings. If AION's loop
is causally faithful, the recovered PA should shift by (applied rotation) mod 180.
180 deg is included deliberately: a 180-deg image rotation maps a galaxy's true
axis onto itself, so the readout should return to (very nearly) its original
value -- a built-in test of the mod-180 symmetry itself, using real intervention
rather than just doubling the angle in the regression target.

Run (needs local GPU + polymathic-aion): python code/analysis/trackA_causal.py
Checkpoints per-angle to results/trackA_causal_ckpt/ (resumable; with all six
checkpoints present a re-run is CPU-only post-processing -- the model is only
loaded when a checkpoint is missing).
Writes results/trackA_causal.json.
"""
import json
import os
import time
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import rotate as sp_rotate
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from aion import AION
from aion.codecs import CodecManager
from aion.modalities import LegacySurveyImage

import sys
sys.path.insert(0, "code/analysis")
from trackUtils import zscore, ALPHAS, SEED, ci

DATA, RES = "data", "results"
CKPT = f"{RES}/trackA_causal_ckpt"
os.makedirs(CKPT, exist_ok=True)
DEV = "cuda"
BATCH = 32
ELLIP_MIN = 0.3
ANGLES = [30.0, 60.0, 90.0, 120.0, 150.0, 180.0]   # applied rotation, degrees


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


def fit_probe(Zi_train, pa_train):
    c, s = np.cos(2 * pa_train), np.sin(2 * pa_train)
    mc = RidgeCV(alphas=ALPHAS).fit(Zi_train, c)
    ms = RidgeCV(alphas=ALPHAS).fit(Zi_train, s)
    return mc, ms


def encode_rotated(model, cm, imgs, gal_idx, angle_deg, batch=BATCH, log_every=20):
    n = len(gal_idx)
    out = np.zeros((n, 1024), np.float32)
    t0 = time.time()
    for s in range(0, n, batch):
        sub = gal_idx[s:s + batch]
        cube = np.ascontiguousarray(imgs[sub])            # (b,4,96,96)
        if angle_deg != 0.0:
            rot = np.empty_like(cube)
            for i in range(cube.shape[0]):
                for b in range(4):
                    rot[i, b] = sp_rotate(cube[i, b], angle_deg, reshape=False,
                                          order=1, mode="constant", cval=0.0)
        else:
            rot = cube
        with torch.no_grad():
            flux = torch.from_numpy(rot).to(DEV)
            img = LegacySurveyImage(flux=flux, bands=["DES-G", "DES-R", "DES-I", "DES-Z"])
            tok = cm.encode(img)
            emb = model.encode(tok, num_encoder_tokens=600).mean(1).float().cpu().numpy()
        out[s:s + len(sub)] = emb
        if (s // batch) % log_every == 0:
            print(f"    angle={angle_deg:.0f} {s+len(sub)}/{n} ({time.time()-t0:.0f}s)", flush=True)
    return out


def circ_shift_deg(pa_hat_rad, pa0_rad):
    """Signed shift theta_hat - theta0, wrapped to (-90, 90] (the PA scale, mod 180)."""
    d = np.angle(np.exp(1j * 2 * (pa_hat_rad - pa0_rad)))
    return np.degrees(d) / 2.0


def main():
    ei, ok, m = load_anchor()
    Zi_full = zscore(ei)
    pa = np.radians(m["paDeg"].to_numpy(float))
    ellip = m["ellip"].to_numpy(float)
    elong = np.isfinite(ellip) & (ellip > ELLIP_MIN) & np.isfinite(pa)
    idx = np.where(elong)[0]                # indices into the E_img/sample.parquet array
    gal_idx = ok[idx]                        # indices into images.npy (the raw cutout array)
    print(f"n_elong = {len(idx)} (no subsampling: full elongated population)", flush=True)

    # fit the probe ONCE on real, unrotated embeddings (held-out split for an honest baseline)
    tr, te = train_test_split(np.arange(len(idx)), test_size=0.2, random_state=SEED)
    mc, ms = fit_probe(Zi_full[idx][tr], pa[idx][tr])
    pa0_hat_te = 0.5 * np.arctan2(ms.predict(Zi_full[idx][te]), mc.predict(Zi_full[idx][te]))
    pa0_true_te = pa[idx][te]
    base_err = np.degrees(np.abs(np.angle(np.exp(1j * 2 * (pa0_hat_te - pa0_true_te))))) / 2.0
    print(f"baseline (unrotated, held-out) median err = {np.median(base_err):.1f} deg", flush=True)

    imgs = np.load(f"{DATA}/images.npy", mmap_mode="r")
    missing = [a for a in ANGLES if not os.path.exists(f"{CKPT}/rot_{int(a)}.npy")]
    model = cm = None
    if missing:
        model = AION.from_pretrained(f"{DATA}/models/aion-large").to(DEV).eval()
        cm = CodecManager(device=DEV)

    results = {"n_elong": int(len(idx)), "baseline_median_err_deg": float(np.median(base_err)),
              "sign_convention": ("scipy rotate(+phi) shifts the array-frame angle by -phi, and "
                                  "catalog PA is same-handed with the array frame (both verified, "
                                  "runLog 2026-07-01) -> a physical rotation by +phi must move the "
                                  "readout by -phi. expected_shift_deg carries that minus sign; "
                                  "applied_fold_deg is the raw applied rotation folded to (-90,90]."),
              "population_note": ("plain fields summarize ALL elongated galaxies, which include the "
                                  "probe's 80% training rows; the *_heldout fields use only the 20% "
                                  "test split the probe never saw and are the leakage-free summaries"),
              "n_heldout": int(len(te)),
              "angles": {}}
    rng = np.random.default_rng(SEED)
    rng_ho = np.random.default_rng(SEED + 1)   # separate stream: keeps all-population numbers identical

    for ang in ANGLES:
        ck = f"{CKPT}/rot_{int(ang)}.npy"
        if os.path.exists(ck):
            print(f"angle {ang}: loading checkpoint", flush=True)
            emb_rot = np.load(ck)
        else:
            print(f"angle {ang}: encoding {len(idx)} rotated galaxies...", flush=True)
            emb_rot = encode_rotated(model, cm, imgs, gal_idx, ang)
            np.save(ck, emb_rot)
        Zr = (emb_rot - ei.mean(0)) / (ei.std(0) + 1e-8)     # same z-score convention as Zi_full
        pa_hat = 0.5 * np.arctan2(ms.predict(Zr.astype(np.float32)), mc.predict(Zr.astype(np.float32)))
        pa0_hat_full = 0.5 * np.arctan2(ms.predict(Zi_full[idx]), mc.predict(Zi_full[idx]))
        shift = circ_shift_deg(pa_hat, pa0_hat_full)          # recovered shift, degrees
        applied_fold = ang if ang <= 90 else ang - 180         # raw fold to (-90,90]
        # sign-corrected expectation (see sign_convention); +/-90 is the same mod-180 antipode
        expected = 90.0 if applied_fold == 90.0 else -applied_fold
        resid = circ_shift_deg(np.radians(shift), np.radians(expected))  # circular err vs expectation
        boot = [np.median(np.abs(resid)[rng.integers(0, len(resid), len(resid))]) for _ in range(200)]
        resid_ho = resid[te]
        boot_ho = [np.median(np.abs(resid_ho)[rng_ho.integers(0, len(resid_ho), len(resid_ho))]) for _ in range(200)]
        results["angles"][str(ang)] = dict(
            applied_fold_deg=float(applied_fold),
            expected_shift_deg=float(expected),
            median_recovered_shift_deg=float(np.median(shift)),
            median_abs_error_deg=float(np.median(np.abs(resid))),
            error_ci=ci(boot),
            median_recovered_shift_deg_heldout=float(np.median(shift[te])),
            median_abs_error_deg_heldout=float(np.median(np.abs(resid_ho))),
            error_ci_heldout=ci(boot_ho))
        print(f"  applied={ang:.0f} (fold {applied_fold:+.0f}, expected shift {expected:+.0f}) -> "
              f"recovered median={np.median(shift):+.1f} deg (heldout {np.median(shift[te]):+.1f}), "
              f"circular err vs expected={np.median(np.abs(resid)):.2f} deg "
              f"(heldout {np.median(np.abs(resid_ho)):.2f})", flush=True)

    # slope of recovered shift vs APPLIED rotation (-1.0 = perfectly faithful in the verified
    # sign convention). |fold|=90 excluded: at the mod-180 antipode the SIGNED median is
    # ill-defined (per-galaxy shifts wrap across the boundary), though its per-galaxy
    # circular error above is as small as every other angle's.
    keep = [a for a in ANGLES if abs(results["angles"][str(a)]["applied_fold_deg"]) != 90.0]
    exp_arr = np.array([results["angles"][str(a)]["applied_fold_deg"] for a in keep])
    rec_arr = np.array([results["angles"][str(a)]["median_recovered_shift_deg"] for a in keep])
    slope, intercept = np.polyfit(exp_arr, rec_arr, 1)
    results["slope_vs_applied"] = float(slope)
    results["intercept_vs_applied"] = float(intercept)
    results["max_abs_fit_residual_deg"] = float(np.abs(rec_arr - (slope * exp_arr + intercept)).max())
    results["slope_excluded_angles"] = [90.0]
    rec_ho = np.array([results["angles"][str(a)]["median_recovered_shift_deg_heldout"] for a in keep])
    slope_ho, icpt_ho = np.polyfit(exp_arr, rec_ho, 1)
    results["slope_vs_applied_heldout"] = float(slope_ho)
    results["intercept_vs_applied_heldout"] = float(icpt_ho)
    results["max_abs_fit_residual_deg_heldout"] = float(np.abs(rec_ho - (slope_ho * exp_arr + icpt_ho)).max())

    with open(f"{RES}/trackA_causal.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSLOPE (recovered shift vs applied rotation, |fold|=90 excluded) = {slope:.3f} "
          f"(-1.0 = perfectly faithful); max fit residual {results['max_abs_fit_residual_deg']:.2f} deg")
    print(f"wrote {RES}/trackA_causal.json")


if __name__ == "__main__":
    main()
