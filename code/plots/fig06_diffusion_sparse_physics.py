"""06 — diffusion embedding (coords 1x2) coloured by the cross-matched SUBSET labels.

mass, sSFR, sersic come from the GZ-DESI external-catalog / NSA cross-match, available for only a
few thousand of the 48,398 galaxies. All points are shown as grey context; the
finite-label subset is overplotted in colour.
"""
import sys
sys.path.insert(0, "code/plots")
import numpy as np
import matplotlib.pyplot as plt
import plotcommon as pc

df, ok = pc.load_df()
dc = pc.diffcoords("E_full")
x, y = dc[:, 1], dc[:, 2]
L = pc.labels(df)

# Shared axis limits keep ~99% of points (0.5-99.5 percentiles).
xlim = np.nanpercentile(x, [0.5, 99.5])
ylim = np.nanpercentile(y, [0.5, 99.5])

panels = ["mass", "sSFR", "sersic"]

fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
for ax, name in zip(axes, panels):
    vals, unit, _ = L[name]
    m = np.isfinite(vals) & np.isfinite(x) & np.isfinite(y)
    n = int(m.sum())

    # grey context: all 48,398 points
    ax.scatter(x, y, s=3, c="0.7", alpha=0.15, linewidths=0, rasterized=True)

    # coloured finite subset
    vlo, vhi = pc.clip_range(vals)
    pc.scatter_phys(ax, x, y, vals, pc.CMAP[name], vlo, vhi, unit, s=7, alpha=0.7)

    ax.set_title(name, fontsize=12)
    ax.set_xlabel("diffusion coordinate 1")
    ax.set_ylabel("diffusion coordinate 2")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.text(0.03, 0.96, f"n = {n:,} labelled", transform=ax.transAxes,
            ha="left", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.6", alpha=0.85))

fig.suptitle(
    "AION-1 diffusion embedding (coords 1x2) coloured by cross-matched physical property",
    fontsize=15,
)
fig.text(
    0.5, -0.02,
    "Grey = all 48,398 galaxies (context). Coloured points = the GZ-DESI external-catalog / NSA "
    "cross-matched subset (a few thousand each), NOT the full sample. "
    "Shared axis limits = 0.5-99.5 percentiles (keeps ~99% of points); colour scales clipped to 2-98 percentiles.",
    ha="center", va="top", fontsize=9, color="0.35",
)
pc.save(fig, "06_diffusion_sparse_physics.png")
