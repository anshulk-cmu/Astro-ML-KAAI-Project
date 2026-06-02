"""05 — AION-1 diffusion embedding (coords 1x2) coloured by the full-N physical labels."""
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

panels = ["g-r", "r-z", "redshift", "smooth", "featured", "merger"]

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for ax, name in zip(axes.ravel(), panels):
    vals, unit, _ = L[name]
    vlo, vhi = pc.clip_range(vals)
    pc.scatter_phys(ax, x, y, vals, pc.CMAP[name], vlo, vhi, unit)
    ax.set_title(name, fontsize=12)
    ax.set_xlabel("diffusion coordinate 1")
    ax.set_ylabel("diffusion coordinate 2")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

fig.suptitle(
    "AION-1 diffusion embedding (coords 1x2, N=48,398) coloured by physical property",
    fontsize=15,
)
fig.text(
    0.5, -0.005,
    "Shared axis limits = 0.5-99.5 percentiles of each coordinate (keeps ~99% of points). "
    "Redshift is mostly photo-z. Colour scales clipped to 2-98 percentiles.",
    ha="center", va="top", fontsize=9, color="0.35",
)
pc.save(fig, "05_diffusion_full_labels.png")
