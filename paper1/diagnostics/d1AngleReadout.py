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
from itertools import combinations

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
TRACK_A_RADEC = C.ROOT / "results" / "trackA_radec.json"
TRACK_A_RA = C.ROOT / "results" / "trackA_ra.json"
TRACK_A_UNSUP = C.ROOT / "results" / "trackA_unsupervised.json"
TRACK_A_CROSS = C.ROOT / "results" / "trackA_crossmodal.json"


def r2_of(y, X):
    """R2 of the least-squares fit of y on the columns of X."""
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(1 - ((y - X @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum())
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

    # EXTENSION, at Matt's request: sky position is the cleanest possible topology pair.
    # Right ascension is genuinely circular, wrapping at 360 degrees, while declination is
    # bounded and has no seam. Both are defined for every galaxy, so this runs on the whole
    # anchor rather than the elongated cut.
    ra = df["ra"].to_numpy(float)
    dec = df["dec"].to_numpy(float)
    allrows = np.ones(len(df), bool)
    fp = df["footprint"].astype(str).to_numpy()
    south, north = fp == "south", fp == "north"

    ra_circ = fit_evaluate(Zi, np.radians(ra), 1, allrows)[3]
    ra_lin = linear_angle_probe(Zi, ra, 360.0, allrows)
    ra_shuf = fit_evaluate(Zi, N.shuffled(np.radians(ra), allrows), 1, allrows)[3]
    # declination forced through the circular machinery: the 180 degree range is stretched
    # onto a full circle, so an error in that coordinate is twice the error in degrees of arc
    dec_forced = fit_evaluate(Zi, np.radians((dec + 90.0) * 2.0), 1, allrows)[3]

    out["EXTENSION_sky_position_topology"] = {
        "why": ("right ascension wraps at 360 degrees and declination does not, so this is the "
                "topology prediction in its purest form: the periodic coordinate should need "
                "the loop and the bounded one should not"),
        "footprint_covers_the_wrap": {
            "n_within_15deg_of_ra_zero": int(((ra < 15) | (ra > 345)).sum()),
            "occupied_fraction_of_the_ra_circle":
                float((np.histogram(ra, bins=24, range=(0, 360))[0] > 0).mean()),
            "dec_range_deg": [float(dec.min()), float(dec.max())],
            "ra_counts_per_15deg_bin": [int(v) for v in
                                        np.histogram(ra, bins=24, range=(0, 360))[0]],
            "ra_bin_count_min_max": [int(np.histogram(ra, bins=24, range=(0, 360))[0].min()),
                                     int(np.histogram(ra, bins=24, range=(0, 360))[0].max())],
            "coverage_note": ("the circle is spanned but not evenly, which matters for the "
                              "shuffled-label null: a label drawn from a clumped distribution "
                              "beats a uniform guess slightly, so that null sits below 90"),
            "note": ("the circular claim is only testable because the sample spans the whole "
                     "circle; a footprint that stopped short of the seam could not test it")},
        "right_ascension_circular": ra_circ,
        "right_ascension_plain_linear": ra_lin,
        "right_ascension_shuffled_null": ra_shuf,
        "declination_scalar": {**P.probe(Zi, dec, allrows), **scalar_abs_err(Zi, dec, allrows)},
        "declination_forced_circular": {
            **dec_forced,
            "med_err_deg_in_declination_units": dec_forced["med_err_deg"] / 2.0},
        "chance_floor_deg_full_circle": 90.0,
        "INTERPRETATION_WARNING": (
            "decoding sky position from pixels is not physics. It is the survey: depth, "
            "seeing, extinction and the instrument itself all vary across the footprint. A "
            "high score here is a systematic, and the confound controls below separate the "
            "topology claim from the leakage claim")}

    # Declination tracks which telescope took the image, so a declination probe could be
    # reading the instrument rather than the coordinate. Refit inside each hemisphere.
    out["EXTENSION_sky_position_topology"]["confound_controls"] = {
        "hemisphere_from_image": P.probe(Zi, north.astype(float), allrows),
        "declination_within_south": {**P.probe(Zi, dec, south),
                                     **scalar_abs_err(Zi, dec, south)},
        "declination_within_north": {**P.probe(Zi, dec, north),
                                     **scalar_abs_err(Zi, dec, north)},
        "right_ascension_within_south": fit_evaluate(Zi, np.radians(ra), 1, south)[3],
        "note": ("north is BASS and MzLS, south is DECam, so the hemispheres are different "
                 "instruments. If declination decodes only because the instrument does, the "
                 "within-hemisphere refits collapse")}

    # Carried over from the Track A work, and kept because it answers a DIFFERENT question on
    # a DIFFERENT substrate. There the coordinates were fed to the model as its own catalog
    # tokens and read back out, so the topology prediction is tested where the answer is known
    # in advance. Here they are read from the image, where nothing guarantees they are present
    # at all. Both are recorded; neither is a substitute for the other.
    carried = {}
    for name, p in [("codec_readback", TRACK_A_RADEC), ("image_leakage", TRACK_A_RA)]:
        if p.exists():
            carried[name] = json.loads(p.read_text())
    out["EXTENSION_sky_position_topology"]["CARRIED_FROM_TRACK_A"] = {
        "substrate_warning": ("the codec-readback block is NOT the image substrate. RA and Dec "
                              "were supplied to the model as catalog tokens and recovered from "
                              "the resulting embedding, so a high score there confirms the "
                              "tokeniser preserves the coordinate and says nothing about "
                              "whether an image carries it"),
        "codec_readback": carried.get("codec_readback"),
        "image_leakage_control": carried.get("image_leakage"),
        "reading": ("Track A found RA only weakly decodable from the image, and about half of "
                    "that traceable to observing conditions, while the conditions themselves "
                    "decode strongly. That is a preview of Diagnostic 4 and the reason sky "
                    "position cannot be read as a physical result")}

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

    # EXTENSION, carried from Track A and recomputed here. Everything above asks whether the
    # loop is linearly DECODABLE, which uses the true angles to build the readout. A different
    # and harder question is whether the loop is DISCOVERABLE without them: does any pair of
    # unsupervised principal directions already trace the ring?
    c2, s2 = np.cos(2 * pa[elong]), np.sin(2 * pa[elong])
    pc50 = pcs[elong][:, :50]
    corr_c = np.array([abs(np.corrcoef(pc50[:, i], c2)[0, 1]) for i in range(50)])
    corr_s = np.array([abs(np.corrcoef(pc50[:, i], s2)[0, 1]) for i in range(50)])
    best_pair, best_ring = None, -1.0
    for i, j in combinations(range(50), 2):
        Xd = np.column_stack([np.ones(len(pc50)), pc50[:, [i, j]]])
        r2c = r2_of(c2, Xd)
        r2s = r2_of(s2, Xd)
        both = min(r2c, r2s)
        if both > best_ring:
            best_ring, best_pair = both, (i, j)
    legacy_unsup = C.ROOT / "results" / "trackA_unsupervised.json"
    out["EXTENSION_unsupervised_discovery"] = {
        "question": ("the probes above are told the answer and asked to find it. This asks "
                     "whether the ring is already present among directions found without any "
                     "labels at all, which is the stronger claim and the one Matt's original "
                     "method proposed"),
        "n_components_scanned": 50,
        "best_single_component_abs_corr": float(max(corr_c.max(), corr_s.max())),
        "best_component_pair": [int(best_pair[0]), int(best_pair[1])],
        "best_pair_ring_r2": float(best_ring),
        "ring_r2_definition": ("for a pair of components, fit both cos 2 theta and sin 2 theta "
                               "from that pair and take the SMALLER of the two R2 values, "
                               "because a ring needs both coordinates, not one"),
        "supervised_comparison": {"full_embedding_med_err_deg": head["med_err_deg"],
                                  "top_2_components_med_err_deg": pc2["med_err_deg"]},
        "CARRIED_FROM_TRACK_A": (json.loads(legacy_unsup.read_text())
                                 if legacy_unsup.exists() else None),
        "reading": ("a high single-component correlation with one of the two coordinates is "
                    "not a ring. The pair statistic is the one that matters, and it has to be "
                    "read against the supervised readout on the full embedding")}

    # Carried from Track A and NOT recomputed, with the reason recorded. That check fed the
    # catalog shape parameters through the model as its own tokens and compared the resulting
    # angle readout with the image one. It sits on a third substrate, and on that substrate
    # the input defines the label, so it is a readback rather than a measurement of what an
    # image carries. It is cited here rather than rebuilt.
    out["CARRIED_crossmodal_image_versus_shape_tokens"] = {
        "values": (json.loads(TRACK_A_CROSS.read_text()) if TRACK_A_CROSS.exists() else None),
        "substrate": "E_shape, the model's own catalog shape tokens, not E_img and not E_full",
        "recomputed_here": False,
        "why_not": ("shape_e1 and shape_e2 are what DEFINE the catalog position angle, so "
                    "recovering the angle from them measures the tokeniser, not the "
                    "representation. The image side of the comparison is already reproduced "
                    "exactly by the headline readout above"),
        "note": ("the original file carries its own caveat, kept verbatim in the values block: "
                 "the two readouts come from separately trained probes, so their agreement "
                 "shows both modalities decode the angle consistently and NOT that they share "
                 "one oriented subspace")}

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

    # The RA and Dec predictions themselves, so the figure can SHOW the circle rather than
    # summarise it: a periodic coordinate should trace a ring, a bounded one a line.
    ra_pr, ra_tr, ra_te, _ = fit_evaluate(Zi, np.radians(ra), 1, allrows)
    rc, rs = ra_pr.predict(Zi[ra_te])
    dec_tr, dec_te_i = P.split(allrows)
    dec_m = RidgeCV(alphas=C.ALPHAS).fit(Zi[dec_tr], dec[dec_tr])
    np.savez(C.RESULTS / f"{NAME}SkyProjection.npz",
             ra_true=ra[ra_te].astype(np.float32),
             ra_cos=rc.astype(np.float32), ra_sin=rs.astype(np.float32),
             dec_true=dec[dec_te_i].astype(np.float32),
             dec_pred=dec_m.predict(Zi[dec_te_i]).astype(np.float32))
    inputs = [p_img, p_full, C.OK_INDEX, C.SAMPLE, C.SHAPES, C.COVARIATES]
    inputs += [p for p in (TRACK_A_RADEC, TRACK_A_RA, TRACK_A_UNSUP, TRACK_A_CROSS)
               if p.exists()]
    path = provenance.write(NAME, out, inputs, t0)

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
