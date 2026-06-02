"""10 — Predicted vs true for three image-only probes, to VISUALISE the R^2.

Refit the held-out probe (pc.predict_heldout) on E_img for redshift, mass, sSFR.
1x3 panels: hexbin density of true (x) vs predicted (y), y=x identity line,
equal aspect with a shared per-panel range. Annotate R^2 and n_test.
Faithful: mass & sSFR use the ~4k cross-matched subset; redshift is mostly photo-z.
All image-only.
"""
import sys
sys.path.insert(0, "code/plots")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import plotcommon as pc

df, ok = pc.load_df()
L = pc.labels(df)
ei = np.load("data/E_img.npy")

specs = [
    ("redshift", L["redshift"], "redshift  (photo-z)"),
    ("mass", L["mass"], r"$\log_{10} M_\star / M_\odot$"),
    ("sSFR", L["sSFR"], r"$\log_{10}$ sSFR  [yr$^{-1}$]"),
]

fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))

for ax, (name, (vals, unit, full)) in zip(axes, [(s[0], s[1]) for s in specs]):
    ytrue, ypred, r2, alpha = pc.predict_heldout(ei, vals)
    ntest = len(ytrue)

    # shared symmetric range so identity line is meaningful and aspect equal
    lo = np.percentile(np.concatenate([ytrue, ypred]), 0.5)
    hi = np.percentile(np.concatenate([ytrue, ypred]), 99.5)
    pad = 0.04 * (hi - lo)
    lo, hi = lo - pad, hi + pad

    hb = ax.hexbin(ytrue, ypred, gridsize=45, cmap="viridis", mincnt=1,
                   norm=LogNorm(), extent=(lo, hi, lo, hi), rasterized=True)
    ax.plot([lo, hi], [lo, hi], color="#d62728", lw=1.6, ls="--", label="$y=x$")

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    title_unit = dict(specs and {s[0]: s[2] for s in specs})[name]
    ax.set_xlabel(f"true  {unit}")
    ax.set_ylabel(f"predicted  {unit}")
    ax.set_title(title_unit, fontsize=12)

    ax.text(0.04, 0.96, f"$R^2$ = {r2:.3f}\n$n_{{test}}$ = {ntest:,}",
            transform=ax.transAxes, va="top", ha="left", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#999999", alpha=0.9))
    ax.legend(loc="lower right", fontsize=9.5)

    cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("galaxies per bin", fontsize=9)

fig.suptitle("Image-only probe: predicted vs true (held-out 20% test set)", fontsize=14)

note = ("All from the image-only embedding $E_{img}$.  Mass & sSFR use the ~4k cross-matched "
        "subset; redshift is mostly photo-z.")
fig.text(0.5, -0.02, note, ha="center", va="top", fontsize=9, color="#333333")

pc.save(fig, "10_probe_pred_vs_true.png")
