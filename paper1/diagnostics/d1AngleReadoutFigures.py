"""Figures for Diagnostic 1. Reads only the results artifact, so every plotted value is traceable.

    python paper1/diagnostics/d1AngleReadoutFigures.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config as C

NAME = "d1AngleReadout"
R = json.loads((C.RESULTS / f"{NAME}.json").read_text())["result"]
PROJ = np.load(C.RESULTS / f"{NAME}Projection.npy")
C.FIGURES.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 200, "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.25})


def save(fig, stem):
    p = C.FIGURES / f"{stem}.png"
    fig.tight_layout()
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


def fig_loop():
    pa, pc, ps, err = PROJ[:, 0], PROJ[:, 1], PROJ[:, 2], PROJ[:, 3]
    h = R["readout"]["E_img"]
    fig, ax = plt.subplots(1, 2, figsize=(9, 4.2))
    t = np.linspace(0, 2 * np.pi, 400)
    ax[0].plot(np.cos(t), np.sin(t), color="0.6", lw=1, zorder=1)
    s = ax[0].scatter(pc, ps, c=pa, cmap="hsv", s=2, alpha=0.55, zorder=2)
    ax[0].set_aspect("equal")
    ax[0].set_xlabel(r"predicted $\cos 2\theta$")
    ax[0].set_ylabel(r"predicted $\sin 2\theta$")
    ax[0].set_title(f"held-out predictions, n = {h['n']}\n"
                    f"median error {h['med_err_deg']:.2f}$\\degree$, loop radius {h['loop_radius']:.3f}")
    plt.colorbar(s, ax=ax[0], label="catalog position angle (deg)", ticks=[0, 45, 90, 135, 180])

    rad = np.hypot(pc, ps)
    ax[1].hist(rad, bins=60, color="#4878a8", alpha=0.85)
    ax[1].axvline(1.0, color="0.3", ls="--", lw=1, label="unit circle")
    ax[1].axvline(h["loop_radius"], color="#b8442e", lw=1.5,
                  label=f"median {h['loop_radius']:.3f}")
    ax[1].axvline(R["nulls"]["shuffled_labels"]["loop_radius"], color="#7a7a7a", lw=1.5,
                  label=f"shuffled-label null {R['nulls']['shuffled_labels']['loop_radius']:.3f}")
    ax[1].set_xlabel("loop radius of prediction")
    ax[1].set_ylabel("galaxies")
    ax[1].set_title("radius distribution")
    ax[1].legend(fontsize=8)
    save(fig, "d1Loop")


def fig_elongation():
    b = R["elongation_grading"]["bins"]
    x = np.array([r["ellip_median"] for r in b])
    y = np.array([r["med_err_deg"] for r in b])
    lo = y - np.array([r["med_err_ci"][0] for r in b])
    hi = np.array([r["med_err_ci"][1] for r in b]) - y
    sig = np.array([r["catalog_sigma_pa_deg"] for r in b])
    rad = np.array([r["loop_radius"] for r in b])
    cut = np.array([r["above_working_cut"] for r in b])
    sc = R["EXTENSION_error_scaling"]

    fig, ax = plt.subplots(1, 2, figsize=(9.5, 4.2))
    ax[0].errorbar(x[~cut], y[~cut], yerr=[lo[~cut], hi[~cut]], fmt="o", color="#b8442e",
                   ms=5, label="below working cut (angle ill-defined)")
    ax[0].errorbar(x[cut], y[cut], yerr=[lo[cut], hi[cut]], fmt="o", color="#25506e",
                   ms=5, label="above working cut")
    xx = np.linspace(x.min(), 0.6, 50)
    ax[0].plot(xx, 10 ** sc["model_loglog_intercept"] * xx ** sc["model_loglog_slope"],
               color="#25506e", ls=":", lw=1.2,
               label=f"power law, slope {sc['model_loglog_slope']:.2f}")
    ax[0].plot(x, sig, "s--", color="#5a8f4a", ms=4, label=r"catalog $\sigma_{PA}$")
    ax[0].axhline(45, color="0.4", ls="--", lw=1, label="chance floor 45$\\degree$")
    ax[0].axvline(C.ELLIP_CUT, color="0.7", lw=1)
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("catalog ellipticity")
    ax[0].set_ylabel("median circular error (deg)")
    ax[0].set_title("one fixed probe, evaluated on held-out galaxies")
    ax[0].legend(fontsize=7.5)

    ax[1].plot(x[~cut], rad[~cut], "o-", color="#b8442e", ms=5, label="below working cut")
    ax[1].plot(x[cut], rad[cut], "o-", color="#25506e", ms=5, label="above working cut")
    ax[1].axhline(1.0, color="0.3", ls="--", lw=1, label="unit circle")
    ax[1].axhline(R["nulls"]["shuffled_labels"]["loop_radius"], color="0.6", ls=":", lw=1,
                  label="shuffled-label null")
    ax[1].set_xscale("log")
    ax[1].set_xlabel("catalog ellipticity")
    ax[1].set_ylabel("loop radius")
    ax[1].set_title("confidence of the readout, not just its accuracy")
    ax[1].legend(fontsize=7.5)
    save(fig, "d1Elongation")


def fig_nulls():
    h = R["readout"]["E_img"]
    n = R["nulls"]
    rows = [("readout, E_img", h["med_err_deg"], h["med_err_ci"], "#25506e"),
            ("readout, E_full", R["readout"]["E_full"]["med_err_deg"],
             R["readout"]["E_full"]["med_err_ci"], "#4878a8"),
            ("plain linear probe on raw angle", n["plain_linear_probe_raw_angle"]["med_err_deg"],
             n["plain_linear_probe_raw_angle"]["med_err_ci"], "#8a7a3a"),
            ("top 2 principal components", n["top_2_principal_components"]["med_err_deg"],
             n["top_2_principal_components"]["med_err_ci"], "#7a7a7a"),
            ("shuffled labels", n["shuffled_labels"]["med_err_deg"],
             n["shuffled_labels"]["med_err_ci"], "#9a9a9a")]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    yp = np.arange(len(rows))[::-1]
    for yi, (lab, v, ci, col) in zip(yp, rows):
        ax[0].barh(yi, v, color=col, height=0.62)
        ax[0].plot([ci[0], ci[1]], [yi, yi], color="k", lw=1.2)
        ax[0].text(v + 1.2, yi, f"{v:.2f}", va="center", fontsize=8)
    ax[0].axvline(45, color="0.3", ls="--", lw=1)
    ax[0].text(45.4, len(rows) - 1.2, "chance 45$\\degree$", fontsize=8, color="0.3")
    ax[0].set_yticks(yp, [r[0] for r in rows], fontsize=8)
    ax[0].set_xlabel("median circular error (deg)")
    ax[0].set_title("readout against its three named nulls")

    sw = n["EXTENSION_pc_sweep"]
    k = [s["n_components"] for s in sw]
    e = [s["med_err_deg"] for s in sw]
    ax[1].plot(k, e, "o-", color="#25506e")
    for s in sw:
        ax[1].annotate(f"{s['cum_variance']*100:.0f}%", (s["n_components"], s["med_err_deg"]),
                       textcoords="offset points", xytext=(6, -10), fontsize=7, color="0.35")
    ax[1].axhline(45, color="0.3", ls="--", lw=1, label="chance")
    ax[1].axhline(R["readout"]["E_img"]["med_err_deg"], color="#b8442e", lw=1.2,
                  label="full 1024-d embedding")
    ax[1].set_xscale("log")
    ax[1].set_xlabel("principal components retained")
    ax[1].set_ylabel("median circular error (deg)")
    ax[1].set_title("how much of the variance basis the loop needs\n(labels: cumulative variance)")
    ax[1].legend(fontsize=8)
    save(fig, "d1Nulls")


def fig_invariance():
    f = R["population_invariance"]["fixed_global_probe"]
    s = R["population_invariance"]["SECONDARY_per_stratum_refit"]
    names = list(f)
    x = np.arange(len(names))
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
    y = [f[k]["med_err_deg"] for k in names]
    lo = [y[i] - f[k]["med_err_ci"][0] for i, k in enumerate(names)]
    hi = [f[k]["med_err_ci"][1] - y[i] for i, k in enumerate(names)]
    ax[0].errorbar(x, y, yerr=[lo, hi], fmt="o", color="#25506e", ms=6, label="one fixed global probe")
    ax[0].plot(x, [s[k]["med_err_deg"] if k in s else np.nan for k in names], "s",
               color="#b8442e", ms=5, alpha=0.8, label="per-stratum refit (secondary)")
    ax[0].axhline(R["readout"]["E_img"]["med_err_deg"], color="0.5", ls="--", lw=1,
                  label="all held-out galaxies")
    ax[0].set_xticks(x, [f"{k}\nn={f[k]['n']}" for k in names], fontsize=8)
    ax[0].set_ylabel("median circular error (deg)")
    ax[0].set_title("does one coordinate serve every population")
    ax[0].legend(fontsize=8)

    ax[1].plot(x, [f[k]["loop_radius"] for k in names], "o", color="#25506e", ms=6)
    ax[1].axhline(1.0, color="0.3", ls="--", lw=1, label="unit circle")
    ax[1].set_xticks(x, names, fontsize=8)
    ax[1].set_ylabel("loop radius")
    ax[1].set_ylim(0.9, 1.02)
    ax[1].set_title("loop radius by stratum")
    ax[1].legend(fontsize=8)
    save(fig, "d1Invariance")


def fig_hetero():
    het = R["heteroscedasticity"]
    keys = ["apparent_magnitude_r", "angular_size_shape_r_arcsec", "ellipticity"]
    lab = ["apparent magnitude r", "angular size shape_r (arcsec)", "ellipticity"]
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    for a, k, l in zip(ax, keys, lab):
        q = het[k]["quintiles"]
        xc = [(r["lo"] + r["hi"]) / 2 for r in q]
        y = [r["med_err_deg"] for r in q]
        lo = [y[i] - r["med_err_ci"][0] for i, r in enumerate(q)]
        hi = [r["med_err_ci"][1] - y[i] for i, r in enumerate(q)]
        a.errorbar(xc, y, yerr=[lo, hi], fmt="o-", color="#25506e", ms=5)
        sp = het[k]["spearman_vs_error"]
        txt = f"Spearman {sp['rho']:+.3f} [{sp['rho_ci'][0]:+.3f}, {sp['rho_ci'][1]:+.3f}]"
        if "partial_spearman_controlling_ellipticity" in het[k]:
            p = het[k]["partial_spearman_controlling_ellipticity"]
            txt += f"\npartial, ellipticity held: {p['rho']:+.3f} [{p['rho_ci'][0]:+.3f}, {p['rho_ci'][1]:+.3f}]"
        a.set_title(txt, fontsize=8)
        a.set_xlabel(l)
        a.set_ylabel("median circular error (deg)")
    save(fig, "d1Heteroscedasticity")


def main():
    fig_loop()
    fig_elongation()
    fig_nulls()
    fig_invariance()
    fig_hetero()


if __name__ == "__main__":
    main()
