"""Figures for Diagnostic 3. Reads only the results artifact and its saved arrays.

    python paper1/diagnostics/d3ChiralityFigures.py
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

NAME = "d3Chirality"
R = json.loads((C.RESULTS / f"{NAME}.json").read_text())["result"]
A = np.load(C.RESULTS / f"{NAME}Arrays.npz")
C.FIGURES.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 200, "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.25})

BLUE, RED, MID, GREEN, GREY = "#25506e", "#b8442e", "#4878a8", "#5a8f4a", "#8a8a8a"
POOLS = ["spiral_armed", "featured", "edge_on", "smooth"]
LABEL = {"spiral_armed": "spiral armed", "featured": "featured", "edge_on": "edge on",
         "smooth": "smooth (achiral)", "all": "all"}
KINDS = [("EXTENSION_d_pure", "flip against its matched\nrotation control", BLUE),
         ("d_spec", "scoping definition\nE(x) - E(flip)", MID),
         ("NULL_d_resampling", "resampling null\nE(x) - E(sandwich)", RED)]


def save(fig, stem):
    p = C.FIGURES / f"{stem}.png"
    fig.tight_layout()
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


def fig_magnitude():
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    w = 0.8 / len(KINDS)
    xa = np.arange(len(POOLS))
    for i, (key, lab, col) in enumerate(KINDS):
        by = R["difference_magnitudes"][key]["by_pool"]
        y = [by[p]["median_norm"] for p in POOLS]
        lo = [y[j] - by[p]["median_norm_ci"][0] for j, p in enumerate(POOLS)]
        hi = [by[p]["median_norm_ci"][1] - y[j] for j, p in enumerate(POOLS)]
        ax[0].bar(xa + i * w - 0.4 + w / 2, y, w, color=col, label=lab)
        ax[0].errorbar(xa + i * w - 0.4 + w / 2, y, yerr=[lo, hi], fmt="none", ecolor="k", lw=1)
    ax[0].set_xticks(xa, [f"{LABEL[p]}\nn={R['population']['pools'][p]['n']:,}" for p in POOLS],
                     fontsize=8)
    ax[0].set_ylabel("median || difference || (z-scored units)")
    ax[0].set_title("how far the flip moves the embedding")
    ax[0].set_ylim(0, max(R["difference_magnitudes"]["d_spec"]["by_pool"][p]["median_norm"]
                          for p in POOLS) * 1.32)
    ax[0].legend(fontsize=7.5, ncol=3, loc="upper center")

    ratios = R["EXTENSION_confound_budget"]["spiral_over_smooth_ratio"]
    keys = ["d_spec", "d_pure", "d_resampling"]
    names = ["scoping definition", "flip vs matched control", "resampling null"]
    cols = [MID, BLUE, RED]
    ax[1].bar(np.arange(3), [ratios[k] for k in keys], 0.6, color=cols)
    ax[1].axhline(1.0, color="0.3", ls="--", lw=1.2, label="no difference between pools")
    for i, k in enumerate(keys):
        ax[1].text(i, ratios[k] + 0.02, f"{ratios[k]:.2f}", ha="center", fontsize=9)
    ax[1].set_xticks(np.arange(3), names, fontsize=8)
    ax[1].set_ylabel("spiral median / smooth median")
    sw = R["EXTENSION_adversarial_checks"]["ellipticity_threshold_sweep"]["rows"]
    ax[1].set_title("pooled, the interpolation confound is absent (null below 1).\n"
                    "On matched pairs it appears once round objects go: the\n"
                    f"resampling excess runs {sw[0]['resampling_excess']:+.2f} to "
                    f"{sw[-1]['resampling_excess']:+.2f} as ellipticity rises", fontsize=8.5)
    ax[1].legend(fontsize=8, loc="lower left")
    save(fig, "d3Magnitude")


def fig_axis():
    a = R["single_axis"]["spiral_armed"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))

    k = np.arange(1, len(a["variance_fraction_top10"]) + 1)
    ax[0].plot(k, a["variance_fraction_top10"], "o-", color=BLUE, ms=5,
               label=f"chirality difference, spiral pool (n = {a['n']:,})")
    sm = R["single_axis"]["smooth"]
    if sm.get("reported"):
        ax[0].plot(k, sm["variance_fraction_top10"], "s--", color=GREEN, ms=4,
                   label=f"smooth pool (achiral control, n = {sm['n']:,})")
    rs = R["single_axis_NULL_resampling"]["spiral_armed"]
    if rs.get("reported"):
        ax[0].plot(k, rs["variance_fraction_top10"], "^:", color=RED, ms=4,
                   label="resampling null, spiral pool")
    nul = a["NULL_random_direction_variance_fraction"]
    ax[0].axhline(nul["analytic_expectation"], color="0.4", ls="--", lw=1,
                  label=f"random direction, 1/d = {nul['analytic_expectation']:.2e}")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("singular direction")
    ax[0].set_ylabel("fraction of the difference variance")
    ax[0].set_title("is the difference carried by a single axis")
    ax[0].legend(fontsize=7.5)

    proj = A["projection"]
    for p, col in [("spiral_armed", BLUE), ("smooth", GREEN)]:
        m = A[f"pool_{p}"]
        s = proj[m]
        s = s / (np.percentile(np.abs(proj[A["pool_spiral_armed"]]), 99) + 1e-12)
        ax[1].hist(s, bins=80, range=(-2, 2), histtype="step", lw=1.4, color=col,
                   density=True, label=f"{LABEL[p]} (n = {int(m.sum()):,})")
    ax[1].axvline(0, color="0.3", lw=1)
    ax[1].set_xlabel("projection onto the leading axis (scaled)")
    ax[1].set_ylabel("density")
    ax[1].set_title("a handedness axis needs two clusters at plus and minus c.\n"
                    "The distribution is unimodal at zero and the achiral pool\n"
                    "traces it, so the leading axis is not handedness", fontsize=9)
    ax[1].legend(fontsize=7.5)
    save(fig, "d3Axis")


def fig_controls():
    lk = R["EXTENSION_orientation_leak"]["readout_shift_deg"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

    xa = np.arange(len(POOLS))
    y = [lk[p]["median_abs"] for p in POOLS]
    lo = [y[i] - lk[p]["median_abs_ci"][0] for i, p in enumerate(POOLS)]
    hi = [lk[p]["median_abs_ci"][1] - y[i] for i, p in enumerate(POOLS)]
    ax[0].bar(xa, y, 0.6, color=[BLUE, MID, GREY, GREEN])
    ax[0].errorbar(xa, y, yerr=[lo, hi], fmt="none", ecolor="k", lw=1)
    ax[0].set_xticks(xa, [f"{LABEL[p]}\nn={lk[p]['n']:,}" for p in POOLS], fontsize=8)
    ax[0].set_ylabel("median |shift| of the angle readout (deg)")
    ax[0].set_title("the orientation leak is not chirality specific: the achiral\n"
                    "pool leaks nearly as much, and edge-on discs, whose axis is\n"
                    "best determined, leak least", fontsize=9)

    g = R["EXTENSION_ellipticity_grading"]["bins"]
    if g:
        xc = [(b["ellip_lo"] + b["ellip_hi"]) / 2 for b in g]
        ax[1].plot(xc, [b["median_norm_d_pure"] for b in g], "o-", color=BLUE, ms=5,
                   label="median || chirality difference ||")
        a2 = ax[1].twinx()
        a2.plot(xc, [b["median_abs_readout_shift_deg"] for b in g], "s--", color=RED, ms=4,
                label="median |readout shift| (deg)")
        a2.set_ylabel("median |readout shift| (deg)", color=RED)
        a2.grid(False)
        ax[1].set_xlabel("catalog ellipticity")
        ax[1].set_ylabel("median || chirality difference ||", color=BLUE)
        ax[1].set_title("the flip uses the CATALOG axis, whose uncertainty\n"
                        "grows as the object becomes round")
        h1, l1 = ax[1].get_legend_handles_labels()
        h2, l2 = a2.get_legend_handles_labels()
        ax[1].legend(h1 + h2, l1 + l2, fontsize=7.5, loc="upper center")
    save(fig, "d3Controls")


def main():
    fig_magnitude()
    fig_axis()
    fig_controls()


if __name__ == "__main__":
    main()
