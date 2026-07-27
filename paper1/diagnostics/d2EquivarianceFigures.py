"""Figures for Diagnostic 2. Reads only the results artifact and its saved arrays, so every
plotted value is traceable to a provenance-stamped number.

    python paper1/diagnostics/d2EquivarianceFigures.py
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

NAME = "d2Equivariance"
R = json.loads((C.RESULTS / f"{NAME}.json").read_text())["result"]
A = np.load(C.RESULTS / f"{NAME}Arrays.npz")
C.FIGURES.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 200, "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.25})

BLUE, RED, MID, GREEN = "#25506e", "#b8442e", "#4878a8", "#5a8f4a"
GRID_ALIGNED = frozenset(R["EXTENSION_resampling_split"]["by_label"]
                         [next(iter(R["EXTENSION_resampling_split"]["by_label"]))]
                         ["grid_aligned_ops"])


def save(fig, stem):
    p = C.FIGURES / f"{stem}.png"
    fig.tight_layout()
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


def fig_rotation():
    rot = R["rotation"]
    ang = rot["angles"]
    keys = [str(a) for a in rot["grid_deg"]]
    fit = [k for k in keys if ang[k]["in_slope_fit"]]
    exc = [k for k in keys if not ang[k]["in_slope_fit"]]

    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))

    x = np.array([ang[k]["applied_fold_deg"] for k in fit])
    y = np.array([ang[k]["median_shift_deg"] for k in fit])
    xx = np.linspace(-95, 95, 50)
    ax[0].plot(xx, -xx, color="0.6", ls="--", lw=1, label="exact equivariance, slope -1")
    ax[0].plot(xx, rot["slope_vs_applied"] * xx + rot["intercept_vs_applied"],
               color=BLUE, lw=1.4,
               label=f"fit {rot['slope_vs_applied']:.4f} "
                     f"[{rot['slope_ci'][0]:.4f}, {rot['slope_ci'][1]:.4f}]")
    ax[0].plot(x, y, "o", color=BLUE, ms=7, zorder=3, label="median recovered shift")
    for k in exc:
        a = ang[k]
        ax[0].plot([a["applied_fold_deg"]], [a["circ_mean_shift_deg"]], "s", color=RED, ms=7,
                   mfc="none", mew=1.6, zorder=3,
                   label="90 deg: circular mean, excluded from fit")
    ax[0].set_xlabel("applied rotation, folded to (-90, 90] (deg)")
    ax[0].set_ylabel("recovered shift of the readout (deg)")
    ax[0].set_title("the readout tracks the intervention\n"
                    f"max fit residual {rot['max_abs_fit_residual_deg']:.3f} deg")
    ax[0].legend(fontsize=7.5, loc="upper right")

    base = R["nulls"]["unperturbed_baseline"]["median_circular_error_deg"]
    ref = R["EXTENSION_shift_error_floor"]["independent_errors_reference_deg"]
    xa = np.arange(len(keys))
    ye = np.array([ang[k]["median_abs_error_deg_heldout"] for k in keys])
    lo = ye - np.array([ang[k]["median_abs_error_ci_heldout"][0] for k in keys])
    hi = np.array([ang[k]["median_abs_error_ci_heldout"][1] for k in keys]) - ye
    cols = [RED if k in exc else BLUE for k in keys]
    for i in range(len(keys)):
        ax[1].errorbar(xa[i], ye[i], yerr=[[lo[i]], [hi[i]]], fmt="o", color=cols[i], ms=6)
    ax[1].axhline(base, color=GREEN, ls="--", lw=1.2,
                  label=f"untransformed baseline {base:.3f} deg")
    ax[1].axhline(ref, color="0.5", ls=":", lw=1.2,
                  label=f"independent-errors reference {ref:.3f} deg")
    ax[1].set_xticks(xa, [f"{float(k):.0f}" for k in keys])
    ax[1].set_xlabel("applied rotation (deg)")
    ax[1].set_ylabel("median per-galaxy circular error (deg)")
    ax[1].set_ylim(0, max(ref, ye.max()) * 1.25)
    ax[1].set_title("held-out error against its two references")
    ax[1].legend(fontsize=7.5, loc="lower right")
    save(fig, "d2Rotation")


def fig_mirror():
    f = R["reflection"]
    dp, dr = A["mirror_d_pred"], A["mirror_d_rec"]
    pa = A["pa_true"]

    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    keep = np.abs(dp) <= 80.0
    ax[0].hexbin(dp[keep], dr[keep], gridsize=70, cmap="Blues", bins="log", mincnt=1)
    xx = np.linspace(-80, 80, 20)
    ax[0].plot(xx, xx, color="0.4", ls="--", lw=1, label="exact reflection, slope 1")
    ax[0].plot(xx, f["displacement_slope"] * xx + f["displacement_intercept"], color=RED, lw=1.2,
               label=f"fit {f['displacement_slope']:.4f}, "
                     f"median |resid| {f['displacement_median_abs_resid_deg']:.2f} deg")
    ax[0].set_xlabel(r"predicted displacement $-2\theta$ (deg)")
    ax[0].set_ylabel("recovered displacement (deg)")
    ax[0].set_title(f"displacement law, n = {f['n_displacement_fit']:,}")
    ax[0].legend(fontsize=7.5)

    edges = np.linspace(0, 180, 37)
    mid = 0.5 * (edges[:-1] + edges[1:])
    med = np.array([np.median(np.abs(dr[(pa >= a) & (pa < b)]))
                    if ((pa >= a) & (pa < b)).sum() > 20 else np.nan
                    for a, b in zip(edges[:-1], edges[1:])])
    ax[1].plot(mid, np.abs((-2 * mid + 90) % 180 - 90), color="0.6", ls="--", lw=1.2,
               label=r"prediction $|{\rm wrap}(-2\theta)|$")
    ax[1].plot(mid, med, "o-", color=BLUE, ms=4, label="measured, binned")
    for v in (0, 90, 180):
        ax[1].axvline(v, color=GREEN, lw=0.8, alpha=0.7)
    for v in (45, 135):
        ax[1].axvline(v, color=RED, lw=0.8, alpha=0.7)
    fp, an = f["fixed_points"], f["antinodes"]
    ax[1].set_xlabel("catalog position angle (deg)")
    ax[1].set_ylabel("|displacement| under the mirror (deg)")
    ax[1].set_title("fixed points at 0 and 90, antinodes at 45 and 135\n"
                    f"nodes {fp['median_abs_displacement_deg']:.2f} deg "
                    f"(predicted {fp['median_abs_predicted_displacement_deg']:.2f}), "
                    f"antinodes {an['median_abs_displacement_deg']:.2f} "
                    f"(predicted {an['median_abs_predicted_displacement_deg']:.2f})",
                    fontsize=9)
    ax[1].set_xticks([0, 45, 90, 135, 180])
    ax[1].legend(fontsize=7.5)
    save(fig, "d2Mirror")


def fig_invariance():
    probes = {k: v for k, v in R["invariance_complement"]["probes"].items() if v["reported"]}
    ops = ["rotate_30", "rotate_60", "rotate_90", "rotate_120", "rotate_150", "rotate_180",
           "mirror", "mirror_then_rotate_30"]
    short = ["rot 30", "rot 60", "rot 90", "rot 120", "rot 150", "rot 180", "mirror",
             "mirror+30"]
    labels = {"colour_g_minus_r": "colour g - r",
              "smooth-or-featured_smooth_fraction": "smooth vote",
              "smooth-or-featured_featured-or-disk_fraction": "featured vote",
              "elpetro_mass_log": "stellar mass"}
    cols = [BLUE, MID, GREEN, RED]

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3))
    w = 0.8 / len(probes)
    xa = np.arange(len(ops))
    for i, (k, v) in enumerate(probes.items()):
        ax[0].bar(xa + i * w - 0.4 + w / 2,
                  [v["under_transform"][o]["delta_r2"] for o in ops], w,
                  color=cols[i], label=f"{labels[k]} (n = {v['n_heldout']:,})")
    for j, o in enumerate(ops):
        if o in GRID_ALIGNED:
            ax[0].axvspan(j - 0.5, j + 0.5, color=GREEN, alpha=0.08, zorder=0)
    ax[0].axhline(0, color="0.3", lw=1)
    ax[0].set_xticks(xa, short, fontsize=8)
    ax[0].set_ylabel(r"change in held-out $R^2$ under the transform")
    ax[0].set_title("physical readouts barely move\n"
                    "shaded: interpolation-free operations (mirror, 90, 180)")
    ax[0].legend(fontsize=7.5)

    for i, (k, v) in enumerate(probes.items()):
        ax[1].plot(xa, [v["under_transform"][o]["median_abs_change_over_prediction_sd"]
                        for o in ops], "o-", color=cols[i], ms=4, label=labels[k])
        ax[1].plot(xa, [v["NULL_matched_norm_random_directions"][o]
                        ["median_abs_change_over_prediction_sd"] for o in ops],
                   "s:", color=cols[i], ms=3, alpha=0.55)
    ax[1].set_xticks(xa, short, fontsize=8)
    ax[1].set_ylabel("median |change| / spread of the readout")
    ax[1].set_ylim(0, None)
    ax[1].set_title("solid: fitted physical probe\n"
                    "dotted: random-direction null (the three full-coverage\n"
                    "labels share held-out rows, so their nulls coincide)", fontsize=9)
    ax[1].legend(fontsize=7.5)
    save(fig, "d2Invariance")


def fig_nulls():
    n = R["nulls"]["matched_norm_random_directions"]
    rot = R["rotation"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.9))

    sl = np.array(n["slopes"])
    ax[0].hist(sl, bins=12, color="0.65", label=f"{len(sl)} random direction pairs")
    ax[0].axvline(0.0, color=GREEN, ls="--", lw=1.2, label="analytic expectation 0")
    ax[0].axvline(n["slope_median"], color=RED, lw=1.4,
                  label=f"null median {n['slope_median']:+.4f}")
    ax[0].annotate(f"fitted probe\n{rot['slope_vs_applied']:.4f}",
                   xy=(0.02, 0.72), xycoords="axes fraction", fontsize=8, color=BLUE,
                   ha="left", va="top",
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=BLUE, lw=1))
    ax[0].set_xlabel("slope of recovered shift against applied rotation")
    ax[0].set_ylabel("random direction pairs")
    ax[0].set_title("the null tracks nothing, and says so")
    ax[0].legend(fontsize=7.5)

    er = np.array(n["errors_deg"])
    ax[1].hist(er, bins=12, color="0.65", label=f"{len(er)} random direction pairs")
    ax[1].axvline(45.0, color=GREEN, ls="--", lw=1.2, label="chance floor 45 deg")
    ax[1].axvline(n["median_circular_error_deg"], color=RED, lw=1.4,
                  label=f"null median {n['median_circular_error_deg']:.2f} deg")
    base = R["nulls"]["unperturbed_baseline"]["median_circular_error_deg"]
    ax[1].axvline(base, color=BLUE, lw=1.4, label=f"untransformed baseline {base:.2f} deg")
    ax[1].set_xlim(0, 50)
    ax[1].set_xlabel("median circular error against the expected shift (deg)")
    ax[1].set_ylabel("random direction pairs")
    ax[1].set_title("both nulls reproduce their analytic values")
    ax[1].legend(fontsize=7.5)
    save(fig, "d2Nulls")


def main():
    fig_rotation()
    fig_mirror()
    fig_invariance()
    fig_nulls()


if __name__ == "__main__":
    main()
