"""
Track A: are orientation angles stored as closed loops in AION's image embedding?

Tests, on the real anchor only (E_img, image-only, leakage-free):
  1. Position angle as a closed loop. PA wraps at 180 deg, so the faithful encoding
     is the DOUBLED angle: regress (cos 2theta, sin 2theta) from E_img, recover
     theta_hat = 0.5*atan2(sin_hat, cos_hat), measure circular error vs true PA.
  2. Conditioning: PA is only defined for elongated galaxies -> headline on
     ellip>0.3, plus an error-vs-ellipticity sweep.
  3. Controls: permutation null, linear-scalar PA probe, PCA-2D baseline, RA
     negative control, and an ellipSnr>5 quality-cut robustness check.
  4. Invariance: is the loop distorted by brightness or morphology (stratify by
     mag_r tertile and by smooth/featured, compare loop radius + error)?
  5. Inclination (secondary): scalar probes on axis-ratio inclination and edge-on
     vote -> encoded as an arc, not a loop.
All headline numbers carry a 200x bootstrap 95% CI (trackUtils.circ_recover/probe).

Run from the project root:  python code/analysis/trackA.py
Needs data/E_img.npy, data/E_full.npy, data/ok_index.npy, data/sample.parquet,
data/anchorShapes.parquet. Writes results/trackA.json + results/trackA_loop.npy.
"""
import json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from trackUtils import zscore, probe, circ_recover, SEED

DATA, RES = "data", "results"
ELLIP_MIN = 0.3


def load():
    ei = np.load(f"{DATA}/E_img.npy")
    ef = np.load(f"{DATA}/E_full.npy")
    ok = np.load(f"{DATA}/ok_index.npy")
    df = pd.read_parquet(f"{DATA}/sample.parquet").iloc[ok].reset_index(drop=True)
    df["dr8_id"] = df["dr8_id"].astype(str)
    sh = pd.read_parquet(f"{DATA}/anchorShapes.parquet")
    sh["dr8_id"] = sh["dr8_id"].astype(str)
    cols = ["dr8_id", "paDeg", "ba", "inclDeg", "ellip", "ellipSnr", "type"]
    m = df.merge(sh[cols].drop_duplicates("dr8_id"), on="dr8_id", how="left")
    assert len(m) == len(df)
    return ei, ef, m


def loop_radius(X, theta_rad, k, mask, seed=SEED):
    """In-sample loop radius sqrt(cos_hat^2+sin_hat^2): does it stay ~1 (a clean
    circle) or shrink/distort in a stratum (brightness/morphology invariance)?"""
    from sklearn.linear_model import RidgeCV
    from trackUtils import ALPHAS
    ok = mask & np.isfinite(theta_rad)
    if ok.sum() < 300:
        return None
    c, s = np.cos(k * theta_rad), np.sin(k * theta_rad)
    mc = RidgeCV(alphas=ALPHAS).fit(X[ok], c[ok])
    ms = RidgeCV(alphas=ALPHAS).fit(X[ok], s[ok])
    pc, ps = mc.predict(X[ok]), ms.predict(X[ok])
    return dict(n=int(ok.sum()), radius=float(np.median(np.hypot(pc, ps))))


def main():
    ei, ef, m = load()
    Zi, Zf = zscore(ei), zscore(ef)
    pa = np.radians(m["paDeg"].to_numpy(float))
    ellip = m["ellip"].to_numpy(float)
    elong = np.isfinite(ellip) & (ellip > ELLIP_MIN)
    out = {"n_total": int(len(m)), "n_elong": int(elong.sum()), "ellip_min": ELLIP_MIN}

    # 1. headline: PA as a doubled-angle loop in E_img, elongated galaxies
    pa_loop = circ_recover(Zi, pa, k=2, mask=elong, full_proj=True)
    np.save(f"{RES}/trackA_loop.npy", pa_loop.pop("_proj"))
    out["pa_loop_Eimg"] = pa_loop
    out["pa_loop_Efull"] = circ_recover(Zf, pa, k=2, mask=elong)

    # 2. ellipticity conditioning sweep
    out["pa_vs_ellip"] = {}
    for lo, hi in [(0.1, 0.25), (0.25, 0.4), (0.4, 0.55), (0.55, 0.7), (0.7, 1.0)]:
        msk = np.isfinite(ellip) & (ellip >= lo) & (ellip < hi)
        if msk.sum() > 500:
            r = circ_recover(Zi, pa, k=2, mask=msk)
            out["pa_vs_ellip"][f"{lo:.2f}-{hi:.2f}"] = {"n": r["n"],
                "med_err_deg": r["med_err_deg"], "frac_within_20": r["frac_within_20"]}

    # 3a. controls: shuffle null, linear-scalar, PCA-2D, RA negative control
    rng = np.random.default_rng(SEED)
    pa_shuf = pa.copy(); good = np.where(elong)[0]
    pa_shuf[good] = pa[rng.permutation(good)]
    out["pa_null_shuffle"] = circ_recover(Zi, pa_shuf, k=2, mask=elong)
    out["pa_linear_scalar"] = probe(Zi, m["paDeg"].to_numpy(float), elong)
    pcs = PCA(n_components=2, random_state=SEED).fit_transform(Zi)
    out["pa_loop_PCA2"] = circ_recover(pcs, pa, k=2, mask=elong)
    ra = np.radians(m["ra"].to_numpy(float))
    out["ra_control_Eimg"] = circ_recover(Zi, ra, k=1, mask=np.isfinite(ra))

    # 3b. quality-cut robustness: restrict to well-measured ellipticity (ellipSnr>5).
    # NOTE: every elongated galaxy passes this cut (n_excluded_by_cut=0), so the result
    # is identical to the headline -- recorded so it cannot be read as independent evidence.
    snr = m["ellipSnr"].to_numpy(float)
    cut = elong & np.isfinite(snr) & (snr > 5)
    r_cut = circ_recover(Zi, pa, k=2, mask=cut)
    r_cut["n_excluded_by_cut"] = int(elong.sum() - cut.sum())
    r_cut["identical_to_headline"] = bool((cut == elong).all())
    out["pa_loop_snrcut"] = r_cut

    # 4. invariance to brightness and morphology (loop radius + error by stratum)
    magr = m["mag_r_desi"].to_numpy(float)
    feat = m["smooth-or-featured_featured-or-disk_fraction"].to_numpy(float)
    smooth = m["smooth-or-featured_smooth_fraction"].to_numpy(float)
    out["pa_vs_brightness"] = {}
    fin_mag = np.isfinite(magr) & elong
    if fin_mag.sum() > 1000:
        terts = np.nanpercentile(magr[fin_mag], [33.3, 66.7])
        bins = [("bright", magr < terts[0]), ("mid", (magr >= terts[0]) & (magr < terts[1])),
                ("faint", magr >= terts[1])]
        for nm, b in bins:
            msk = elong & b
            r = circ_recover(Zi, pa, k=2, mask=msk)
            lr = loop_radius(Zi, pa, k=2, mask=msk)
            out["pa_vs_brightness"][nm] = {"n": r["n"], "med_err_deg": r["med_err_deg"],
                                           "loop_radius": lr["radius"] if lr else None}
    out["pa_vs_morphology"] = {}
    for nm, b in [("smooth", smooth > 0.7), ("featured", feat > 0.7)]:
        msk = elong & b
        if msk.sum() > 500:
            r = circ_recover(Zi, pa, k=2, mask=msk)
            lr = loop_radius(Zi, pa, k=2, mask=msk)
            out["pa_vs_morphology"][nm] = {"n": r["n"], "med_err_deg": r["med_err_deg"],
                                           "loop_radius": lr["radius"] if lr else None}

    # 5. inclination as an arc (secondary)
    out["inclination_Eimg"] = probe(Zi, m["inclDeg"].to_numpy(float), elong)
    out["axisratio_Eimg"] = probe(Zi, m["ba"].to_numpy(float), elong)
    out["edgeon_vote_Eimg"] = probe(
        Zi, m["disk-edge-on_yes_fraction"].to_numpy(float), np.ones(len(m), bool))

    with open(f"{RES}/trackA.json", "w") as f:
        json.dump(out, f, indent=2, default=float)

    h = out["pa_loop_Eimg"]
    print(f"n_elong={out['n_elong']} (ellip>{ELLIP_MIN})")
    print(f"PA loop (E_img): median err {h['med_err_deg']:.1f} deg CI{h['med_err_ci']}, "
          f"within-20deg {100*h['frac_within_20']:.1f}%, "
          f"R2(cos2t,sin2t)={h['r2_cos']:.2f}/{h['r2_sin']:.2f}")
    print(f"  null(shuffle) err {out['pa_null_shuffle']['med_err_deg']:.1f} | "
          f"PCA-2D err {out['pa_loop_PCA2']['med_err_deg']:.1f} | "
          f"linear-scalar R2 {out['pa_linear_scalar']['r2']:+.2f} | "
          f"ellipSnr>5 cut err {out['pa_loop_snrcut']['med_err_deg']:.1f} "
          f"(excludes {out['pa_loop_snrcut']['n_excluded_by_cut']} galaxies)")
    print(f"  RA control err {out['ra_control_Eimg']['med_err_deg']:.1f} deg, "
          f"R2 {out['ra_control_Eimg']['r2_cos']:+.2f}")
    print(f"  inclination R2 {out['inclination_Eimg']['r2']:+.2f} CI{out['inclination_Eimg']['ci']} | "
          f"edge-on vote R2 {out['edgeon_vote_Eimg']['r2']:+.2f}")
    print("brightness invariance (loop radius + error by tertile):")
    for nm, v in out["pa_vs_brightness"].items():
        print(f"  {nm:6s} n={v['n']:5d} err={v['med_err_deg']:.1f} radius={v['loop_radius']:.2f}")
    print("morphology invariance:")
    for nm, v in out["pa_vs_morphology"].items():
        print(f"  {nm:9s} n={v['n']:5d} err={v['med_err_deg']:.1f} radius={v['loop_radius']:.2f}")
    print(f"wrote {RES}/trackA.json + {RES}/trackA_loop.npy")


if __name__ == "__main__":
    main()
