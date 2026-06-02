"""Galaxy montage along two discovered manifold axes (real DESI cutouts).

Row 1: sampled at evenly spaced percentiles of g-r colour (blue -> red).
Row 2: sampled at evenly spaced percentiles of diffusion coord 1 (morphology),
       titled with the smooth vote fraction so featured<->smooth is visible.

Representative percentile sampling, not curated extremes (extremes hold
stars/artifacts). RGB = z/r/g with arcsinh stretch.
"""
import sys
sys.path.insert(0, 'code/plots')
import numpy as np
import matplotlib.pyplot as plt
import plotcommon as pc

NCOL = 9
PCTS = np.linspace(5, 95, NCOL)

df, ok = pc.load_df()
imgs = np.load('data/images.npy', mmap_mode='r')
L = pc.labels(df)

gr = L['g-r'][0]
smooth = L['smooth'][0]
dc1 = pc.diffcoords('E_full')[:, 1]


def not_artifact(idx):
    """Reject saturated stars/artifacts: a tiny fraction of very bright pixels."""
    img = np.asarray(imgs[ok[idx]], dtype=np.float64)
    return np.percentile(img, 99.9) < 80.0


def pick_at_value(values, target, used, repr_of=None, pool=60):
    """Pick a galaxy near `values`==`target`. From the `pool` nearest matches,
    drop saturated artifacts, then keep the percentile-sampling principle:
    - if repr_of is None: take the nearest non-artifact (used for colour).
    - else: among the pool, pick the galaxy whose `repr_of` value is nearest the
      LOCAL POPULATION MEAN of repr_of in the pool -- i.e. a representative,
      not a noisy single-vote outlier. (dc1 is highly concentrated, so a single
      galaxy's smooth vote is noisy; the population trend is the signal.)"""
    finite = np.isfinite(values)
    cand = np.where(finite)[0]
    cand = cand[~np.isin(cand, list(used))]
    order = cand[np.argsort(np.abs(values[cand] - target))]
    nb = [i for i in order[:pool] if not_artifact(i)]
    if not nb:
        nb = list(order[:pool])
    if repr_of is None:
        return int(nb[0])
    rv = repr_of[np.array(nb)]
    local_mean = np.nanmean(rv)
    best = nb[int(np.nanargmin(np.abs(rv - local_mean)))]
    return int(best)


def build_row(values, repr_of=None):
    finite = values[np.isfinite(values)]
    targets = np.percentile(finite, PCTS)
    used, picks = set(), []
    for t in targets:
        idx = pick_at_value(values, t, used, repr_of=repr_of)
        used.add(idx)
        picks.append(idx)
    return picks


row_colour = build_row(gr)
row_morph = build_row(dc1, repr_of=smooth)

fig, axes = plt.subplots(2, NCOL, figsize=(15.5, 4.6))

for j, idx in enumerate(row_colour):
    ax = axes[0, j]
    ax.imshow(pc.galaxy_rgb(np.asarray(imgs[ok[idx]], dtype=np.float64)))
    ax.set_title(f"g-r = {gr[idx]:.2f}", fontsize=9, pad=2)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

for j, idx in enumerate(row_morph):
    ax = axes[1, j]
    ax.imshow(pc.galaxy_rgb(np.asarray(imgs[ok[idx]], dtype=np.float64)))
    ax.set_title(f"smooth = {smooth[idx]:.2f}", fontsize=9, pad=2)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

axes[0, 0].set_ylabel("colour (g-r)", fontsize=11)
axes[1, 0].set_ylabel("morphology\n(diffusion coord 1)", fontsize=11)

fig.suptitle("Galaxies sampled along two discovered manifold axes "
             "(g-r colour; diffusion coordinate 1)", fontsize=13)
fig.text(0.5, -0.02,
         "Galaxies sampled at evenly spaced percentiles (5-95) along each axis "
         "(representative, not curated); RGB = z/r/g with arcsinh stretch.\n"
         "Row 1 left->right: bluer -> redder.  Row 2 left->right: spans the "
         "featured<->smooth transition (titled with smooth vote fraction; "
         "diffusion coord 1 also co-varies with redshift).",
         ha='center', va='top', fontsize=8.5)

print('colour g-r:', [round(float(gr[i]), 2) for i in row_colour])
print('morph dc1 :', [round(float(dc1[i]), 4) for i in row_morph])
print('morph smooth:', [round(float(smooth[i]), 2) for i in row_morph])
pc.save(fig, '18_manifold_galaxy_montage.png')
