"""Diagnostic 2 - O(2) equivariance. Tier: input-space causal.

Does the internal angle coordinate transform correctly under known transformations of the
input, and do the transformations compose the way they compose in the plane?

The probe is fit ONCE on real untransformed embeddings and frozen. Inputs are then rotated or
mirrored, re-encoded through the frozen model, and read with that unmodified probe. Nothing
is refit to a transformed condition, which is what makes this causal rather than correlational.

Sign conventions are pinned in tests/testTransforms.py before this runs. rotate(+phi) shifts
the array-frame angle by -phi, the array frame is same-handed with the catalog frame at an
offset of 90 degrees, and the mirror negates the angle exactly.

Blocks named EXTENSION go beyond the scoping document and are reported as such.

    python paper1/diagnostics/d2Equivariance.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

import json
import numpy as np
from sklearn.linear_model import RidgeCV

import config as C
import data as D
import encode as E
import nulls as N
import probes as P
import provenance
import transforms as T
from circular import fit_evaluate

NAME = "d2Equivariance"
ROT_ANGLES = [30.0, 60.0, 90.0, 120.0, 150.0, 180.0]
N_REENCODE = 64
N_RANDOM_DIR = 20
N_FRAME = 600
FIXED_POINT_TOL = 5.0

# Physical quantities that a rotation or a reflection cannot change. The invariance
# complement reads these out of the TRANSFORMED embeddings with probes fit on untransformed
# ones: a model that moved them would be writing orientation into quantities that have none.
INVARIANTS = [("colour_g_minus_r", "derived: mag_g_desi - mag_r_desi"),
              ("smooth-or-featured_smooth_fraction", "Galaxy Zoo DESI vote fraction"),
              ("smooth-or-featured_featured-or-disk_fraction", "Galaxy Zoo DESI vote fraction"),
              ("elpetro_mass_log", "NSA elpetro crossmatch")]


def signed_circ_mean(shift_deg):
    """Circular mean of a signed axial shift, in (-90, 90], with its resultant length.

    The plain median is what the pre-harness code used and is reported beside this one. They
    agree away from the wrap; at +/-90 the signed centre is ill-defined for either statistic,
    which is why that angle is excluded from the slope fit.
    """
    z = np.exp(2j * np.radians(np.asarray(shift_deg, float))).mean()
    return float(np.degrees(np.angle(z)) / 2.0), float(np.abs(z))


def readout_deg(pr, Z):
    pc, ps = pr.predict(Z)
    return np.degrees(0.5 * np.arctan2(ps, pc))


def slope_of(folds, shifts):
    s, i = np.polyfit(np.asarray(folds, float), np.asarray(shifts, float), 1)
    return float(s), float(i)


def slope_ci(folds, per_angle_shift, sel, n_boot=C.N_BOOT, seed=C.SEED):
    """Bootstrap the slope with the galaxy as the resampling unit: resample galaxies, recompute
    every angle's median shift on the same resampled galaxies, refit."""
    rng = np.random.default_rng(seed)
    n = len(sel)
    out = []
    for _ in range(n_boot):
        b = sel[rng.integers(0, n, n)]
        out.append(slope_of(folds, [float(np.median(s[b])) for s in per_angle_shift])[0])
    return P.ci(out)


def measure_frame_offset(df, idx, seed=C.SEED, n=N_FRAME):
    """Measure the array-to-catalog frame offset directly from the pixels.

    Under reflection a frame offset c does not cancel: it appears as a constant -2(c - 90)
    shift in the readout. Measuring c from the images turns the mirror offset from a fitted
    nuisance into a prediction that the flip result can agree or disagree with.
    """
    ok = np.load(C.OK_INDEX)
    ellip = df["ellip"].to_numpy(float)
    magr = df["mag_r_desi"].to_numpy(float)
    sel = np.where((ellip[idx] > 0.5) & (magr[idx] < 18.5))[0]
    rng = np.random.default_rng(seed)
    sel = sel[rng.choice(len(sel), min(n, len(sel)), replace=False)]
    imgs = np.load(C.IMAGES, mmap_mode="r")
    pa = df["paDeg"].to_numpy(float)
    th, conv = [], []
    for j in sel:
        t, _, c = T.adaptive_moments(imgs[ok[idx[j]], 1])
        th.append(t)
        conv.append(c)
    th = np.array(th, float)
    good = np.array(conv, bool) & np.isfinite(th)
    d = T.wrap_axial(th[good] - pa[idx][sel][good])
    r_same, off = T.axial_concentration(d)
    r_opp, _ = T.axial_concentration(T.wrap_axial(th[good] + pa[idx][sel][good]))
    se = float(np.degrees(np.sqrt(-2 * np.log(r_same) / good.sum())) / 2.0)
    resid = T.wrap_axial(d - off)
    return {"n_attempted": int(len(sel)), "n_converged": int(good.sum()),
            "offset_deg": off, "offset_se_deg": se,
            "resultant_same_handed": r_same, "resultant_opposite_handed": r_opp,
            "median_abs_residual_deg": float(np.median(np.abs(resid))),
            "deviation_from_90_deg": float(off - 90.0),
            "deviation_in_sigma": float((off - 90.0) / se) if se > 0 else None,
            "predicted_mirror_offset_deg": float(-2.0 * (off - 90.0)),
            "predicted_mirror_offset_se_deg": float(2.0 * se),
            "selection": "elongated, catalog ellipticity > 0.5, r < 18.5",
            "note": ("iterated Gaussian-weighted moments on the r band; the handedness is "
                     "decided by which pairing concentrates, not by documentation")}


def main():
    t0 = time.time()
    df = D.anchor()
    Zi, p_img = D.substrate("img")
    ei = np.load(C.E_IMG)
    mu, sd = ei.mean(0), ei.std(0) + 1e-8

    pa_deg = df["paDeg"].to_numpy(float)
    pa = np.radians(pa_deg)
    ellip = df["ellip"].to_numpy(float)
    elong = D.pa_defined(df) & np.isfinite(ellip) & (ellip > C.ELLIP_CUT)
    idx = np.where(elong)[0]

    pr, tr, te, head = fit_evaluate(Zi, pa, 2, elong)
    pos_te = np.searchsorted(idx, te)
    assert np.array_equal(idx[pos_te], te), "held-out rows are not a subset of the population"
    te_mask = np.zeros(len(idx), bool)
    te_mask[pos_te] = True

    out = {"population": {
        "n_anchor": int(len(df)), "n_elongated": int(elong.sum()),
        "n_train": int(len(tr)), "n_heldout": int(len(te)),
        "ellip_cut": C.ELLIP_CUT, "substrate": "E_img (image tokens only)",
        "period_deg": 180.0, "k": 2,
        "row_order": "ascending anchor index; verified identical to the cached encode order",
        "note": ("plain fields summarize all elongated galaxies, which include the probe's "
                 "training rows; heldout fields use only the 20 percent the probe never saw "
                 "and are the leakage-free summaries")}}

    out["fixed_probe"] = {
        "fit_on": "untransformed E_img, elongated training rows only, never refit",
        "baseline_heldout": head,
        "alpha_cos_sin": head["alpha_cos_sin"],
        "conventions": {
            "rotate_plus_phi_shifts_array_angle_by": -1.0,
            "expected_readout_shift_for_rotation_deg": "-phi (mod 180)",
            "mirror_maps_readout_to": "-theta (mod 180)",
            "array_to_catalog_offset_deg": T.ARRAY_TO_CATALOG_OFFSET_DEG,
            "pinned_by": "paper1/tests/testTransforms.py"}}

    pa0 = readout_deg(pr, Zi[idx])
    pa_true = pa_deg[idx]

    out["EXTENSION_frame_offset_from_pixels"] = measure_frame_offset(df, idx)

    # ---------------- rotation ----------------
    reencode, adopted = {}, {}
    per_angle_shift_all, per_angle_shift_ho, folds = [], [], []
    angles = {}
    imgs = np.load(C.IMAGES, mmap_mode="r")
    ok = np.load(C.OK_INDEX)
    rng_chk = np.random.default_rng(C.SEED)
    chk = np.sort(rng_chk.choice(len(idx), N_REENCODE, replace=False))
    cubes = np.stack([np.asarray(imgs[ok[idx[j]]], np.float32) for j in chk])

    def verify(tag, Z, op):
        try:
            reencode[tag] = E.reencode_check(cubes, op, Z[chk])
        except Exception as exc:
            reencode[tag] = {"error": f"{type(exc).__name__}: {exc}"}

    for phi in ROT_ANGLES:
        tag = f"rot_{int(phi)}"
        Zl, checks = E.adopt_legacy(tag, C.LEGACY_ROT_CKPT / f"{tag}.npy", ei[idx],
                                    {"operator": "rotate", "phi_deg": phi, "rows": "elongated"})
        adopted[tag] = checks
        verify(tag, Zl, lambda a, p=phi: T.rotate(a, p))
        h = readout_deg(pr, ((Zl - mu) / sd).astype(np.float32))
        shift = T.wrap_axial(h - pa0)
        fold = float(T.wrap_axial(phi))
        expected = float(T.wrap_axial(-phi))
        resid = T.wrap_axial(shift - expected)
        cm_all, r_all = signed_circ_mean(shift)
        cm_ho, r_ho = signed_circ_mean(shift[te_mask])
        angles[str(phi)] = {
            "applied_deg": phi, "applied_fold_deg": fold, "expected_shift_deg": expected,
            "median_shift_deg": float(np.median(shift)),
            "circ_mean_shift_deg": cm_all, "shift_resultant": r_all,
            "median_abs_error_deg": float(np.median(np.abs(resid))),
            "median_abs_error_ci": P.boot_stat(np.abs(resid), np.median),
            "median_shift_deg_heldout": float(np.median(shift[te_mask])),
            "circ_mean_shift_deg_heldout": cm_ho, "shift_resultant_heldout": r_ho,
            "median_abs_error_deg_heldout": float(np.median(np.abs(resid[te_mask]))),
            "median_abs_error_ci_heldout": P.boot_stat(np.abs(resid[te_mask]), np.median),
            "in_slope_fit": abs(fold) != 90.0,
            "signed_centre_note": ("ill-defined at |fold| = 90: +90 and -90 are one point on a "
                                   "mod-180 loop, so only the per-galaxy error is meaningful")
            if abs(fold) == 90.0 else None}
        if abs(fold) != 90.0:
            folds.append(fold)
            per_angle_shift_all.append(shift)
            per_angle_shift_ho.append(shift[te_mask])

    med_all = [float(np.median(s)) for s in per_angle_shift_all]
    med_ho = [float(np.median(s)) for s in per_angle_shift_ho]
    s_all, i_all = slope_of(folds, med_all)
    s_ho, i_ho = slope_of(folds, med_ho)
    all_sel = np.arange(len(idx))
    out["rotation"] = {
        "angles": angles,
        "grid_deg": ROT_ANGLES,
        "slope_vs_applied": s_all, "intercept_vs_applied": i_all,
        "slope_ci": slope_ci(folds, per_angle_shift_all, all_sel),
        "max_abs_fit_residual_deg": float(np.abs(np.array(med_all)
                                                 - (s_all * np.array(folds) + i_all)).max()),
        "slope_vs_applied_heldout": s_ho, "intercept_vs_applied_heldout": i_ho,
        "slope_ci_heldout": slope_ci(folds, per_angle_shift_ho, np.arange(len(pos_te))),
        "max_abs_fit_residual_deg_heldout": float(np.abs(np.array(med_ho)
                                                         - (s_ho * np.array(folds) + i_ho)).max()),
        "slope_excluded_angles_deg": [90.0],
        "expected_slope": -1.0,
        "note": ("slope of recovered shift against applied rotation folded to (-90, 90]; -1 is "
                 "exact equivariance in the verified sign convention")}

    # EXTENSION: what floor should the shift errors be compared against? The per-angle error
    # is a difference of two readouts of the SAME galaxy, not a readout against truth, so the
    # baseline is not directly the right yardstick.
    out["EXTENSION_shift_error_floor"] = {
        "baseline_readout_error_deg": head["med_err_deg"],
        "independent_errors_reference_deg": float(head["med_err_deg"] * np.sqrt(2)),
        "measured_heldout_error_deg": {str(a): angles[str(a)]["median_abs_error_deg_heldout"]
                                       for a in ROT_ANGLES},
        "note": ("if the untransformed and transformed readouts erred independently with the "
                 "same distribution, the median absolute difference would be about sqrt(2) "
                 "times the single-readout median. Measured values below that reference mean "
                 "the two readouts err in the same direction on the same galaxy, which is what "
                 "a shared coordinate should do")}

    out["self_map_180"] = {
        **{k: angles["180.0"][k] for k in
           ("median_shift_deg", "circ_mean_shift_deg", "median_abs_error_deg",
            "median_abs_error_ci", "median_shift_deg_heldout",
            "median_abs_error_deg_heldout", "median_abs_error_ci_heldout")},
        "expected_shift_deg": 0.0,
        "why_it_matters": ("the probe is built on a mod-180 assumption; a half turn maps a "
                           "galaxy's axis onto itself, so this element tests the periodicity "
                           "by intervention instead of assuming it")}

    # ---------------- reflection ----------------
    Zf, checks = E.adopt_legacy("flip", C.LEGACY_FLIP_CKPT / "flip.npy", ei[idx],
                                {"operator": "mirror", "rows": "elongated"})
    adopted["flip"] = checks
    verify("flip", Zf, T.mirror)
    hf = readout_deg(pr, ((Zf - mu) / sd).astype(np.float32))

    err_vs_readout = np.abs(T.wrap_axial(hf - (-pa0)))
    err_vs_truth = np.abs(T.wrap_axial(hf - (-pa_true)))
    offset_meas, offset_r = signed_circ_mean(T.wrap_axial(hf - (-pa0)))
    err_corrected = np.abs(T.wrap_axial(hf - (-pa0 + offset_meas)))

    d_rec = T.wrap_axial(hf - pa0)
    d_pred = T.wrap_axial(-2.0 * pa_true)
    keep = np.abs(d_pred) <= 80.0
    sl, ic = slope_of(d_pred[keep], d_rec[keep])
    fit_resid = d_rec[keep] - (sl * d_pred[keep] + ic)
    near = lambda t, c: np.abs(T.wrap_axial(t - c)) < FIXED_POINT_TOL
    fp = near(pa_true, 0.0) | near(pa_true, 90.0)
    an = near(pa_true, 45.0) | near(pa_true, 135.0)
    keep_ho = keep & te_mask
    sl_ho, ic_ho = slope_of(d_pred[keep_ho], d_rec[keep_ho])
    rng_d = np.random.default_rng(C.SEED)
    kd = np.where(keep)[0]
    sl_boot = [slope_of(d_pred[b], d_rec[b])[0]
               for b in (kd[rng_d.integers(0, len(kd), len(kd))] for _ in range(C.N_BOOT))]

    pred_off = out["EXTENSION_frame_offset_from_pixels"]["predicted_mirror_offset_deg"]
    pred_se = out["EXTENSION_frame_offset_from_pixels"]["predicted_mirror_offset_se_deg"]
    out["reflection"] = {
        "expected_readout": "-theta, exactly, if the frame offset is exactly 90 degrees",
        "median_err_vs_minus_readout_deg": float(np.median(err_vs_readout)),
        "median_err_ci": P.boot_stat(err_vs_readout, np.median),
        "median_err_vs_minus_truth_deg": float(np.median(err_vs_truth)),
        "median_err_vs_minus_truth_ci": P.boot_stat(err_vs_truth, np.median),
        "median_err_vs_minus_readout_deg_heldout": float(np.median(err_vs_readout[te_mask])),
        "median_err_ci_heldout": P.boot_stat(err_vs_readout[te_mask], np.median),
        "median_err_vs_minus_truth_deg_heldout": float(np.median(err_vs_truth[te_mask])),
        "measured_offset_deg": offset_meas, "measured_offset_resultant": offset_r,
        "median_err_offset_corrected_deg": float(np.median(err_corrected)),
        "predicted_offset_from_pixels_deg": pred_off,
        "predicted_offset_se_deg": pred_se,
        "offset_measured_minus_predicted_deg": float(offset_meas - pred_off),
        "offset_agreement_in_sigma": (float((offset_meas - pred_off) / pred_se)
                                      if pred_se > 0 else None),
        "displacement_slope": sl, "displacement_slope_ci": P.ci(sl_boot),
        "displacement_intercept": ic,
        "displacement_median_abs_resid_deg": float(np.median(np.abs(fit_resid))),
        "displacement_p95_abs_resid_deg": float(np.percentile(np.abs(fit_resid), 95)),
        "n_displacement_fit": int(keep.sum()),
        "displacement_slope_heldout": sl_ho, "displacement_intercept_heldout": ic_ho,
        "n_displacement_fit_heldout": int(keep_ho.sum()),
        "displacement_excluded": "|predicted displacement| > 80 deg, the mod-180 wrap boundary",
        "expected_displacement_slope": 1.0,
        "fixed_points": {"selection": f"catalog PA within {FIXED_POINT_TOL} deg of 0 or 90",
                         "n": int(fp.sum()),
                         "median_abs_displacement_deg": float(np.median(np.abs(d_rec[fp]))),
                         "median_abs_displacement_ci": P.boot_stat(np.abs(d_rec[fp]), np.median),
                         "expected_at_exact_point_deg": 0.0,
                         "analytic_expected_given_tolerance_deg": FIXED_POINT_TOL,
                         "median_abs_predicted_displacement_deg":
                             float(np.median(np.abs(d_pred[fp]))),
                         "median_abs_residual_vs_prediction_deg":
                             float(np.median(np.abs(T.wrap_axial(d_rec[fp] - d_pred[fp])))),
                         "tolerance_note": (
                             "a galaxy selected within tol of a fixed point has a PREDICTED "
                             "displacement of -2 PA spanning +/- 2 tol, so the median predicted "
                             "displacement is tol, not zero. The model-quality number is the "
                             "residual against the per-galaxy prediction, not the raw "
                             "displacement")},
        "antinodes": {"selection": f"catalog PA within {FIXED_POINT_TOL} deg of 45 or 135",
                      "n": int(an.sum()),
                      "median_abs_displacement_deg": float(np.median(np.abs(d_rec[an]))),
                      "median_abs_displacement_ci": P.boot_stat(np.abs(d_rec[an]), np.median),
                      "expected_at_exact_point_deg": 90.0,
                      "analytic_expected_given_tolerance_deg": 90.0 - FIXED_POINT_TOL,
                      "median_abs_predicted_displacement_deg":
                          float(np.median(np.abs(d_pred[an]))),
                      "median_abs_residual_vs_prediction_deg":
                          float(np.median(np.abs(T.wrap_axial(d_rec[an] - d_pred[an]))))},
        "interpolation_note": ("the mirror is an exact pixel permutation, so its error carries "
                               "no resampling component; comparing it with the rotation errors "
                               "separates representational error from interpolation artifact")}

    # ---------------- composition ----------------
    Zc, checks = E.adopt_legacy("flip_rot30", C.LEGACY_FLIP_CKPT / "flip_rot30.npy", ei[idx],
                                {"operator": "mirror_then_rotate", "phi_deg": 30.0,
                                 "rows": "elongated"})
    adopted["flip_rot30"] = checks
    verify("flip_rot30", Zc, lambda a: T.rotate(T.mirror(a), 30.0))
    hc = readout_deg(pr, ((Zc - mu) / sd).astype(np.float32))
    err_comp = np.abs(T.wrap_axial(hc - (-pa0 - 30.0)))
    err_comp_true = np.abs(T.wrap_axial(hc - (-pa_true - 30.0)))
    err_comp_corr = np.abs(T.wrap_axial(hc - (-pa0 - 30.0 + offset_meas)))
    out["composition"] = {
        "operation": "mirror, then rotate by +30 degrees",
        "expected_readout": "-(untransformed readout) - 30 deg",
        "median_err_vs_expected_deg": float(np.median(err_comp)),
        "median_err_ci": P.boot_stat(err_comp, np.median),
        "median_err_vs_expected_deg_heldout": float(np.median(err_comp[te_mask])),
        "median_err_ci_heldout": P.boot_stat(err_comp[te_mask], np.median),
        "median_err_vs_truth_deg": float(np.median(err_comp_true)),
        "median_err_offset_corrected_deg": float(np.median(err_comp_corr)),
        "why_it_matters": ("composition upgrades the claim from responding to two separate "
                           "transformations to representing the group action")}

    # ---------------- invariance complement ----------------
    transformed = {f"rotate_{int(a)}": (C.LEGACY_ROT_CKPT / f"rot_{int(a)}.npy")
                   for a in ROT_ANGLES}
    transformed["mirror"] = C.LEGACY_FLIP_CKPT / "flip.npy"
    transformed["mirror_then_rotate_30"] = C.LEGACY_FLIP_CKPT / "flip_rot30.npy"
    zt_cache = {k: ((np.load(v) - mu) / sd).astype(np.float32) for k, v in transformed.items()}

    inv = {}
    for col, source in INVARIANTS:
        if col == "colour_g_minus_r":
            y = df["mag_g_desi"].to_numpy(float) - df["mag_r_desi"].to_numpy(float)
            valid = np.isfinite(y)
        else:
            valid, y = D.valid(df, col)
        fit_rows = np.intersect1d(tr, np.where(valid)[0])
        ev_rows = np.intersect1d(te, np.where(valid)[0])
        if len(fit_rows) < 100 or len(ev_rows) < 30:
            inv[col] = {"declared_source": source, "n_fit": int(len(fit_rows)),
                        "n_heldout": int(len(ev_rows)),
                        "reported": False,
                        "reason": "too few elongated galaxies carry this label to fit or score"}
            continue
        m = RidgeCV(alphas=C.ALPHAS).fit(Zi[fit_rows], y[fit_rows])
        pos_ev = np.searchsorted(idx, ev_rows)
        p0 = m.predict(Zi[ev_rows])
        y_ev = y[ev_rows]
        spread = float(np.std(y_ev))
        pred_sd = float(np.std(p0))
        resid_sd = float(np.std(y_ev - p0))
        w = m.coef_
        rng_r = np.random.default_rng(C.SEED)
        boot_idx = [rng_r.integers(0, len(ev_rows), len(ev_rows)) for _ in range(C.N_BOOT)]
        rows = {}
        for name, Zt in zt_cache.items():
            pt = m.predict(Zt[pos_ev])
            d = np.abs(pt - p0)
            rows[name] = {
                "r2_untransformed": P.r2(y_ev, p0), "r2_transformed": P.r2(y_ev, pt),
                "delta_r2": float(P.r2(y_ev, pt) - P.r2(y_ev, p0)),
                "delta_r2_ci": P.ci([P.r2(y_ev[b], pt[b]) - P.r2(y_ev[b], p0[b])
                                     for b in boot_idx]),
                "median_abs_change": float(np.median(d)),
                "median_abs_change_ci": P.boot_stat(d, np.median),
                "median_abs_change_over_label_sd": float(np.median(d) / spread),
                "median_abs_change_over_prediction_sd": float(np.median(d) / pred_sd),
                "median_abs_change_over_residual_sd": float(np.median(d) / resid_sd),
                "pearson_pred_vs_pred": float(np.corrcoef(p0, pt)[0, 1])}
        # The null must be compared on a scale both readouts share. Matching the norm of w
        # does NOT match the output scale: a random direction of the same norm has a far wider
        # output spread than a fitted probe, so only movement relative to each readout's own
        # spread is a like-for-like comparison.
        V = N.matched_norm_directions(w, N_RANDOM_DIR)
        null_rows = {}
        for name, Zt in zt_cache.items():
            rel, absolute = [], []
            for v in V:
                q0, qt = Zi[ev_rows] @ v, Zt[pos_ev] @ v
                absolute.append(float(np.median(np.abs(qt - q0))))
                rel.append(float(np.median(np.abs(qt - q0)) / np.std(q0)))
            null_rows[name] = {"median_abs_change": float(np.median(absolute)),
                               "median_abs_change_over_label_sd":
                                   float(np.median(absolute) / spread),
                               "median_abs_change_over_prediction_sd": float(np.median(rel)),
                               "spread_over_directions_relative":
                                   [float(np.min(rel)), float(np.max(rel))]}
        inv[col] = {"declared_source": source, "reported": True,
                    "n_fit": int(len(fit_rows)), "n_heldout": int(len(ev_rows)),
                    "label_sd": spread, "prediction_sd": pred_sd, "residual_sd": resid_sd,
                    "alpha": float(m.alpha_),
                    "under_transform": rows,
                    "NULL_matched_norm_random_directions": null_rows}
    # EXTENSION: rotation resamples, the mirror does not, and bilinear resampling is nearly
    # exact at multiples of 90 degrees where the sample points land on pixel centres. If the
    # small loss in the invariant readouts is a resampling artifact rather than a
    # representational failure, it must track grid alignment and not the size of the rotation.
    GRID_ALIGNED = ["mirror", "rotate_90", "rotate_180"]
    OFF_AXIS = ["rotate_30", "rotate_60", "rotate_120", "rotate_150"]
    split_rows = {}
    for col, v in inv.items():
        if not v["reported"]:
            continue
        g = [v["under_transform"][t]["delta_r2"] for t in GRID_ALIGNED]
        o = [v["under_transform"][t]["delta_r2"] for t in OFF_AXIS]
        split_rows[col] = {"grid_aligned_mean_delta_r2": float(np.mean(g)),
                           "off_axis_mean_delta_r2": float(np.mean(o)),
                           "grid_aligned_ops": GRID_ALIGNED, "off_axis_ops": OFF_AXIS}
    out["EXTENSION_resampling_split"] = {
        "by_label": split_rows,
        "rationale": ("the mirror is an exact pixel permutation and 90 and 180 degree "
                      "rotations resample onto pixel centres, so all three are effectively "
                      "interpolation-free; 30, 60, 120 and 150 degrees are not. A loss that "
                      "appears only in the second group is a resampling artifact, since the "
                      "first group changes orientation just as much"),
        "reflection_error_vs_rotation_error_deg": {
            "mirror_heldout": float(np.median(err_vs_readout[te_mask])),
            "off_axis_rotation_heldout": [angles[str(a)]["median_abs_error_deg_heldout"]
                                          for a in (30.0, 60.0, 120.0, 150.0)]}}

    out["invariance_complement"] = {
        "probes": inv,
        "n_random_directions": N_RANDOM_DIR,
        "contrast": {
            "angle_readout_shift_under_rotate_30_deg":
                abs(angles["30.0"]["median_shift_deg"]),
            "angle_period_deg": 180.0,
            "angle_shift_as_fraction_of_period":
                abs(angles["30.0"]["median_shift_deg"]) / 180.0},
        "note": ("probes fit once on untransformed embeddings and applied unchanged; the null "
                 "is what an arbitrary direction of the same norm moves by under the same "
                 "transformation, which is the scale an invariant readout must beat")}

    # ---------------- nulls ----------------
    wc, ws = pr.mc.coef_, pr.ms.coef_
    Vc = N.matched_norm_directions(wc, N_RANDOM_DIR, seed=C.SEED)
    Vs = N.matched_norm_directions(ws, N_RANDOM_DIR, seed=C.SEED + 1)
    rand_slopes, rand_err = [], []
    Zt_rot = {a: ((np.load(C.LEGACY_ROT_CKPT / f"rot_{int(a)}.npy") - mu) / sd).astype(np.float32)
              for a in ROT_ANGLES}
    for vc, vs in zip(Vc, Vs):
        r0 = np.degrees(0.5 * np.arctan2(Zi[idx] @ vs, Zi[idx] @ vc))
        meds, errs = [], []
        for a in ROT_ANGLES:
            rt = np.degrees(0.5 * np.arctan2(Zt_rot[a] @ vs, Zt_rot[a] @ vc))
            sh = T.wrap_axial(rt - r0)
            errs.append(float(np.median(np.abs(T.wrap_axial(sh - T.wrap_axial(-a))))))
            if abs(T.wrap_axial(a)) != 90.0:
                meds.append(float(np.median(sh)))
        rand_slopes.append(slope_of(folds, meds)[0])
        rand_err.append(float(np.median(errs)))

    out["nulls"] = {
        "unperturbed_baseline": {
            "type": "population control",
            "median_circular_error_deg": head["med_err_deg"],
            "median_circular_error_ci": head["med_err_ci"],
            "loop_radius": head["loop_radius"], "n": head["n"],
            "role": ("the noise floor the transformed per-galaxy errors should match; a "
                     "transformed error at this level means the transformation cost nothing"),
            "slope_axis": ("degenerate by construction: with no transformation the readout is "
                           "compared with itself and the shift is exactly 0")},
        "matched_norm_random_directions": {
            "type": "matched-norm random directions",
            "n_directions": N_RANDOM_DIR,
            "slopes": [float(v) for v in rand_slopes],
            "errors_deg": [float(v) for v in rand_err],
            "slope_median": float(np.median(rand_slopes)),
            "slope_ci": P.ci(rand_slopes),
            "slope_min_max": [float(np.min(rand_slopes)), float(np.max(rand_slopes))],
            "analytic_expected_slope": 0.0,
            "median_circular_error_deg": float(np.median(rand_err)),
            "error_ci": P.ci(rand_err),
            "analytic_expected_error_deg": 45.0,
            "role": ("a direction pair of the same norm as the fitted probe, carrying no fitted "
                     "information; calibrates both axes the result is reported on")}}

    # ---------------- consistency checks ----------------
    legacy = {}
    for f, key in [("trackA_causal.json", "rotation"), ("trackA_flip.json", "reflection")]:
        p = C.ROOT / "results" / f
        if p.exists():
            legacy[key] = json.loads(p.read_text())
    lc = legacy.get("rotation", {})
    lf = legacy.get("reflection", {})
    d1p = C.RESULTS / "d1AngleReadout.json"
    d1_base = (json.loads(d1p.read_text())["result"]["readout"]["E_img"]["med_err_deg"]
               if d1p.exists() else None)
    rot_errs = [angles[str(a)]["median_abs_error_deg_heldout"] for a in ROT_ANGLES]
    out["consistency_checks"] = {
        "cached_encodes_adopted": adopted,
        "reencode_verification": reencode,
        "reencode_note": ("a subsample re-encoded through the frozen model under the operators "
                          "in lib/transforms.py and compared with the cached rows; this is what "
                          "establishes that each file is the operator its name claims and is "
                          "aligned row for row"),
        "baseline_matches_d1": {"d2": head["med_err_deg"], "d1": d1_base,
                                "delta": (None if d1_base is None
                                          else float(head["med_err_deg"] - d1_base))},
        "legacy_slope": lc.get("slope_vs_applied"),
        "legacy_slope_delta": (None if "slope_vs_applied" not in lc else
                               float(s_all - lc["slope_vs_applied"])),
        "legacy_baseline_delta": (None if "baseline_median_err_deg" not in lc else
                                  float(head["med_err_deg"] - lc["baseline_median_err_deg"])),
        "legacy_flip_slope": lf.get("flip", {}).get("disp_slope"),
        "legacy_flip_slope_delta": (None if "flip" not in lf else
                                    float(sl - lf["flip"]["disp_slope"])),
        "self_map_returns_to_zero": {"measured": angles["180.0"]["median_shift_deg"],
                                     "expected": 0.0},
        "reflection_offset_predicted_vs_measured": {
            "predicted_from_pixels": pred_off, "measured_from_flip": offset_meas,
            "delta": float(offset_meas - pred_off), "predicted_se": pred_se},
        "interpolation_free_beats_interpolated": {
            "mirror_median_err_deg": float(np.median(err_vs_readout[te_mask])),
            "rotation_median_err_deg_range": [float(np.min(rot_errs)), float(np.max(rot_errs))],
            "expectation": ("the mirror resamples nothing, so its error should not exceed the "
                            "rotation errors, which carry a bilinear interpolation component")},
        "random_direction_slope_reaches_zero": {
            "measured": float(np.median(rand_slopes)), "expected": 0.0},
        "random_direction_error_reaches_chance": {
            "measured": float(np.median(rand_err)), "expected": 45.0}}

    np.savez(C.RESULTS / f"{NAME}Arrays.npz",
             pa_true=pa_true.astype(np.float32),
             heldout=te_mask,
             mirror_d_pred=d_pred.astype(np.float32),
             mirror_d_rec=d_rec.astype(np.float32),
             rot_shift=np.column_stack([T.wrap_axial(readout_deg(
                 pr, ((np.load(C.LEGACY_ROT_CKPT / f"rot_{int(a)}.npy") - mu) / sd
                      ).astype(np.float32)) - pa0) for a in ROT_ANGLES]).astype(np.float32),
             rot_angles=np.array(ROT_ANGLES, np.float32))

    inputs = [p_img, C.OK_INDEX, C.SAMPLE, C.SHAPES, C.COVARIATES, C.IMAGES]
    inputs += [C.LEGACY_ROT_CKPT / f"rot_{int(a)}.npy" for a in ROT_ANGLES]
    inputs += [C.LEGACY_FLIP_CKPT / "flip.npy", C.LEGACY_FLIP_CKPT / "flip_rot30.npy"]
    path = provenance.write(NAME, out, inputs, t0)

    print(f"n_elong={out['population']['n_elongated']} n_heldout={out['population']['n_heldout']}")
    print(f"baseline {head['med_err_deg']:.4f} deg (d1 delta "
          f"{out['consistency_checks']['baseline_matches_d1']['delta']})")
    for a in ROT_ANGLES:
        r = angles[str(a)]
        print(f"  rot {a:5.0f}  fold {r['applied_fold_deg']:+6.1f}  expect {r['expected_shift_deg']:+6.1f}"
              f"  shift {r['median_shift_deg']:+8.3f}  err {r['median_abs_error_deg_heldout']:6.3f}"
              f"{'' if r['in_slope_fit'] else '   (excluded from slope)'}")
    print(f"slope {s_all:.4f} CI{[round(v,4) for v in out['rotation']['slope_ci']]} "
          f"(heldout {s_ho:.4f}), expected -1")
    print(f"mirror  err {out['reflection']['median_err_vs_minus_readout_deg']:.3f} deg, "
          f"offset measured {offset_meas:+.3f} vs predicted {pred_off:+.3f} +/- {pred_se:.3f}")
    for nm in ("fixed_points", "antinodes"):
        b = out["reflection"][nm]
        print(f"  {nm:12} n={b['n']:5d} displacement {b['median_abs_displacement_deg']:6.3f} "
              f"vs predicted {b['median_abs_predicted_displacement_deg']:6.3f} "
              f"(analytic {b['analytic_expected_given_tolerance_deg']:.1f} at this tolerance), "
              f"residual {b['median_abs_residual_vs_prediction_deg']:.3f}")
    print(f"composition err {out['composition']['median_err_vs_expected_deg']:.3f} deg")
    print(f"null    random-direction slope {np.median(rand_slopes):+.4f} (expect 0), "
          f"error {np.median(rand_err):.2f} deg (expect 45)")
    for col, v in inv.items():
        if v["reported"]:
            r30 = v["under_transform"]["rotate_30"]
            print(f"  invariant {col[:38]:40} n={v['n_heldout']:5d} "
                  f"R2 {r30['r2_untransformed']:.4f} -> {r30['r2_transformed']:.4f} "
                  f"move {r30['median_abs_change_over_label_sd']:.4f} sd")
        else:
            print(f"  invariant {col[:38]:40} not reported: {v['reason']}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
