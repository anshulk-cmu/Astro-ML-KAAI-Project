"""Diagnostic 1 - Angle readout characterization. Tier: descriptive.

Does the model store a quantity that wraps on a closed loop rather than a line, and does the
fidelity of that storage track the observable's own measurability?

Implements the four stress axes and three nulls of paperScopingV1.md. Blocks named EXTENSION
go beyond the scoping document and are reported as such.

    python paper1/diagnostics/d1AngleReadout.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

import json
import numpy as np
from scipy.stats import kstest
from sklearn.linear_model import RidgeCV

import config as C
import data as D
import nulls as N
import probes as P
import provenance
from circular import circ_error, evaluate, fit_evaluate, linear_angle_probe, radius

NAME = "d1AngleReadout"
ELLIP_BINS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 1.00]
PC_SWEEP = [2, 5, 10, 20, 50, 100, 200, 512]


def scalar_abs_err(Z, y, mask, seed=C.SEED):
    ok = mask & np.isfinite(y)
    tr, te = P.split(ok, seed)
    m = RidgeCV(alphas=C.ALPHAS).fit(Z[tr], y[tr])
    err = np.abs(m.predict(Z[te]) - y[te])
    return dict(med_abs_err=float(np.median(err)), med_abs_err_ci=P.boot_stat(err, np.median))


def main():
    t0 = time.time()
    df = D.anchor()
    Zi, p_img = D.substrate("img")
    Zf, p_full = D.substrate("full")

    pa_deg = df["paDeg"].to_numpy(float)
    pa = np.radians(pa_deg)
    ellip = df["ellip"].to_numpy(float)
    pa_ok = D.pa_defined(df)
    elong = pa_ok & np.isfinite(ellip) & (ellip > C.ELLIP_CUT)
    sig_pa = D.pa_uncertainty(df)

    ty = df["type"].astype(str).to_numpy()
    circ_model = np.isfinite(pa_deg) & ~pa_ok
    out = {"population": {
        "n_anchor": int(len(df)), "n_elongated": int(elong.sum()),
        "ellip_cut": C.ELLIP_CUT,
        "n_pa_finite_in_catalog": int(np.isfinite(pa_deg).sum()),
        "n_pa_defined": int(pa_ok.sum()),
        "n_excluded_circular_model": int(circ_model.sum()),
        "excluded_type_counts": {t: int((ty[circ_model] == t).sum())
                                 for t in np.unique(ty[circ_model])},
        "exclusion_reason": ("shape_e1 = shape_e2 = 0 exactly (REX, PSF, DUP tractor models are "
                             "circular by construction), which would read as a placeholder PA of 0"),
        "substrate": "E_img (image tokens only)", "period_deg": 180.0, "k": 2}}

    pr, tr, te, head = fit_evaluate(Zi, pa, 2, elong)
    out["readout"] = {"E_img": head, "E_full": fit_evaluate(Zf, pa, 2, elong)[3]}

    in_train = np.zeros(len(df), bool)
    in_train[tr] = True

    pool = pa_ok & np.isfinite(ellip) & ~in_train
    grading = []
    for lo, hi in zip(ELLIP_BINS[:-1], ELLIP_BINS[1:]):
        idx = np.where(pool & (ellip >= lo) & (ellip < hi))[0]
        if len(idx) < 100:
            continue
        row = evaluate(pr, Zi, pa, idx)
        row.update(ellip_lo=lo, ellip_hi=hi, ellip_median=float(np.median(ellip[idx])),
                   above_working_cut=bool(lo >= C.ELLIP_CUT),
                   catalog_sigma_pa_deg=float(np.nanmedian(sig_pa[idx])),
                   label_uniformity_ks_p=float(kstest((pa_deg[idx] % 180) / 180, "uniform").pvalue))
        grading.append(row)
    out["elongation_grading"] = {
        "note": ("one probe fit on the elongated training rows, evaluated on every galaxy with a "
                 "defined position angle that the probe never saw; bins below the working cut are "
                 "the regime where the angle is genuinely ill-defined"),
        "bins": grading}

    # EXTENSION: how the readout error scales with elongation, against the catalog's own
    # geometric scaling sigma_PA ~ sigma_e / (2 ellip)
    ge = np.array([b["ellip_median"] for b in grading])
    gy = np.array([b["med_err_deg"] for b in grading])
    gs = np.array([b["catalog_sigma_pa_deg"] for b in grading])
    fit = ge < 0.6
    sl_m, ic_m = np.polyfit(np.log10(ge[fit]), np.log10(gy[fit]), 1)
    sl_c, ic_c = np.polyfit(np.log10(ge[fit]), np.log10(gs[fit]), 1)
    out["EXTENSION_error_scaling"] = {
        "model_loglog_slope": float(sl_m), "model_loglog_intercept": float(ic_m),
        "catalog_loglog_slope": float(sl_c), "catalog_loglog_intercept": float(ic_c),
        "model_err_times_ellip_deg": [float(v) for v in gy * ge],
        "model_over_catalog_ratio": [float(v) for v in gy / gs],
        "saturation_bins_ellip_gt_0p6_med_err_deg": [float(v) for v in gy[~fit]],
        "fit_bins_ellip_lt_0p6": int(fit.sum()),
        "note": ("a slope of -1 is the geometric expectation for an axis estimator whose "
                 "component noise is independent of elongation; fitted on bin medians below "
                 "ellipticity 0.6, above which the readout saturates")}

    incl = df["inclDeg"].to_numpy(float)
    out["topology_matching"] = {
        "inclination_scalar": {**P.probe(Zi, incl, elong), **scalar_abs_err(Zi, incl, elong)},
        "axis_ratio_scalar": P.probe(Zi, df["ba"].to_numpy(float), elong),
        "edgeon_vote_scalar": P.probe(Zi, df["disk-edge-on_yes_fraction"].to_numpy(float),
                                      np.ones(len(df), bool)),
        "position_angle_circular": {"med_err_deg": head["med_err_deg"],
                                    "loop_radius": head["loop_radius"]}}

    incl_ok = elong & np.isfinite(incl)
    theta_incl = np.radians(np.clip(incl, 0, 90) * 2.0)
    inc_circ = fit_evaluate(Zi, theta_incl, 2, incl_ok)[3]
    out["topology_matching"]["EXTENSION_inclination_forced_circular"] = {
        **inc_circ,
        "med_err_deg_in_inclination_units": inc_circ["med_err_deg"] / 2.0,
        "note": ("underpowered by design: a bounded quantity has no wrap seam for a linear "
                 "encoding to fail at, so this control cannot fail the way the angle case can")}

    magr = df["mag_r_desi"].to_numpy(float)
    smooth = df["smooth-or-featured_smooth_fraction"].to_numpy(float)
    feat = df["smooth-or-featured_featured-or-disk_fraction"].to_numpy(float)
    te_mask = np.zeros(len(df), bool)
    te_mask[te] = True
    terts = np.nanpercentile(magr[elong & np.isfinite(magr)], [33.3, 66.7])
    strata = {"bright": magr < terts[0],
              "mid": (magr >= terts[0]) & (magr < terts[1]),
              "faint": magr >= terts[1],
              "smooth": smooth > 0.7,
              "featured": feat > 0.7}
    fixed, refit = {}, {}
    for nm, sel in strata.items():
        idx = np.where(elong & sel & te_mask & np.isfinite(pa))[0]
        if len(idx) >= 100:
            fixed[nm] = evaluate(pr, Zi, pa, idx)
        m = elong & sel
        if (m & np.isfinite(pa)).sum() >= 500:
            refit[nm] = fit_evaluate(Zi, pa, 2, m)[3]
    out["population_invariance"] = {
        "fixed_global_probe": fixed,
        "SECONDARY_per_stratum_refit": refit,
        "tertile_edges_mag_r": [float(t) for t in terts],
        "note": ("the fixed-probe block is the invariance evidence; refits measure "
                 "within-stratum decodability and are secondary")}

    pc_te, ps_te = pr.predict(Zi[te])
    err_te = circ_error(pc_te, ps_te, pa[te], 2)
    shape_r = df["shape_r"].to_numpy(float)
    het = {}
    for nm, v in [("apparent_magnitude_r", magr[te]),
                  ("angular_size_shape_r_arcsec", shape_r[te]),
                  ("ellipticity", ellip[te])]:
        het[nm] = {"spearman_vs_error": P.spearman(v, err_te)}
        if nm != "ellipticity":
            het[nm]["partial_spearman_controlling_ellipticity"] = \
                P.partial_spearman(v, err_te, ellip[te])
        q = np.nanpercentile(v, np.linspace(0, 100, 6))
        bins = []
        for j, (lo, hi) in enumerate(zip(q[:-1], q[1:])):
            b = np.isfinite(v) & (v >= lo) & ((v <= hi) if j == 4 else (v < hi))
            if b.sum() >= 50:
                bins.append({"lo": float(lo), "hi": float(hi), "n": int(b.sum()),
                             "med_err_deg": float(np.median(err_te[b])),
                             "med_err_ci": P.boot_stat(err_te[b], np.median),
                             "loop_radius": float(np.median(radius(pc_te[b], ps_te[b])))})
        het[nm]["quintiles"] = bins
    out["heteroscedasticity"] = het

    pa_shuf = N.shuffled(pa, elong)
    sh = fit_evaluate(Zi, pa_shuf, 2, elong)[3]
    pcs, evr = N.pc_basis(Zi, max(PC_SWEEP))
    pc2 = fit_evaluate(pcs[:, :2], pa, 2, elong)[3]
    lin = linear_angle_probe(Zi, pa_deg, 180.0, elong)
    sweep = []
    for k in PC_SWEEP:
        r = fit_evaluate(pcs[:, :k], pa, 2, elong)[3]
        sweep.append({"n_components": k, "cum_variance": float(evr[:k].sum()),
                      "med_err_deg": r["med_err_deg"], "med_err_ci": r["med_err_ci"],
                      "loop_radius": r["loop_radius"],
                      "r2_cos": r["r2_cos"], "r2_sin": r["r2_sin"]})
    out["nulls"] = {"shuffled_labels": sh,
                    "top_2_principal_components": pc2,
                    "plain_linear_probe_raw_angle": lin,
                    "chance_floor_deg": 45.0,
                    "EXTENSION_pc_sweep": sweep}

    # EXTENSION: is the readout a property of the representation or of one lucky split
    sens = [fit_evaluate(Zi, pa, 2, elong, seed=s, n_boot=200)[3] for s in range(10)]
    se = np.array([r["med_err_deg"] for r in sens])
    sr = np.array([r["loop_radius"] for r in sens])
    out["EXTENSION_split_sensitivity"] = {
        "n_seeds": len(sens), "seeds": list(range(10)),
        "med_err_deg": {"values": [float(v) for v in se], "mean": float(se.mean()),
                        "std": float(se.std(ddof=1)), "min": float(se.min()), "max": float(se.max())},
        "loop_radius": {"mean": float(sr.mean()), "std": float(sr.std(ddof=1)),
                        "min": float(sr.min()), "max": float(sr.max())},
        "r2_cos": {"mean": float(np.mean([r["r2_cos"] for r in sens]))},
        "r2_sin": {"mean": float(np.mean([r["r2_sin"] for r in sens]))},
        "note": ("each seed redraws the 80/20 split and refits both probes, so this spread "
                 "covers split and refit variation that the within-split bootstrap does not")}

    e = sig_pa[elong & np.isfinite(sig_pa)]
    out["EXTENSION_label_noise_floor"] = {
        "catalog_sigma_pa_deg_median": float(np.median(e)),
        "catalog_sigma_pa_deg_p25_p75": [float(np.percentile(e, 25)), float(np.percentile(e, 75))],
        "n": int(len(e)), "readout_median_error_deg": head["med_err_deg"],
        "note": ("propagated from shape_e1_ivar and shape_e2_ivar; the readout cannot be scored "
                 "below the uncertainty of the label it is scored against")}

    legacy = None
    lp = C.ROOT / "results" / "trackA.json"
    if lp.exists():
        legacy = json.loads(lp.read_text())["pa_loop_Eimg"]["med_err_deg"]
    rho_bins = P.spearman(np.array([b["ellip_median"] for b in grading]),
                          np.array([b["med_err_deg"] for b in grading]))
    round_bin = grading[0]
    out["consistency_checks"] = {
        "roundest_bin_approaches_chance_from_below": {
            "ellip_range": [round_bin["ellip_lo"], round_bin["ellip_hi"]],
            "med_err_deg": round_bin["med_err_deg"], "chance_floor_deg": 45.0,
            "label_uniformity_ks_p": round_bin["label_uniformity_ks_p"]},
        "shuffle_vs_theoretical_chance": {"measured": sh["med_err_deg"], "theory": 45.0},
        "shuffle_loop_radius_should_collapse": sh["loop_radius"],
        "loop_radius_rms_vs_sqrt_mean_r2": {
            "rms_radius": head["loop_radius_rms"],
            "sqrt_mean_r2": float(np.sqrt((head["r2_cos"] + head["r2_sin"]) / 2))},
        "monotone_error_vs_ellipticity": rho_bins,
        "legacy_trackA_med_err_deg": legacy,
        "legacy_reproduction_delta_deg": (None if legacy is None
                                          else float(head["med_err_deg"] - legacy))}

    np.save(C.RESULTS / f"{NAME}Projection.npy",
            np.column_stack([pa_deg[te], pc_te, ps_te, err_te, ellip[te]]))
    path = provenance.write(NAME, out, [p_img, p_full, C.OK_INDEX, C.SAMPLE, C.SHAPES,
                                        C.COVARIATES], t0)

    h = out["readout"]["E_img"]
    print(f"n_elong={out['population']['n_elongated']}  n_test={h['n']}")
    print(f"readout  {h['med_err_deg']:.4f} deg CI{[round(v, 3) for v in h['med_err_ci']]}  "
          f"radius {h['loop_radius']:.4f}  R2 {h['r2_cos']:.4f}/{h['r2_sin']:.4f}  "
          f"within20 {h['frac_within_20']:.4f}")
    print(f"nulls    shuffle {sh['med_err_deg']:.2f} (radius {sh['loop_radius']:.4f})  "
          f"PC2 {pc2['med_err_deg']:.2f}  linear {lin['med_err_deg']:.2f} (R2 {lin['r2']:.3f})")
    print(f"floor    sigma_PA median "
          f"{out['EXTENSION_label_noise_floor']['catalog_sigma_pa_deg_median']:.3f} deg")
    print(f"legacy   delta {out['consistency_checks']['legacy_reproduction_delta_deg']}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
