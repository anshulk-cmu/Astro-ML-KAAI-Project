import sys
sys.path.insert(0, 'code/plots')
import numpy as np
import matplotlib.pyplot as plt
import plotcommon as pc

sae = pc.load_json('sae')
con = sae['concepts']
thr95 = con['align_null_thr95']
thr99 = con['align_null_thr99']
max_align = con['max_align']
n_alien = con['alien_candidates']

align = np.load('results/sae_align.npy')
nov = np.load('results/sae_novelty.npy')
rec = np.load('results/sae_recurrence.npy')

# active mask: features that actually fire (std > 1e-6), computed in column chunks
acts = np.load('results/sae_acts0.npy', mmap_mode='r')
ncol = acts.shape[1]
active = np.zeros(ncol, dtype=bool)
chunk = 256
for j in range(0, ncol, chunk):
    sub = np.asarray(acts[:, j:j + chunk], dtype=np.float64)
    active[j:j + sub.shape[1]] = sub.std(axis=0) > 1e-6

n_active = int(active.sum())
a = align[active]
nv = nov[active]
stable = rec[active] >= 0.5

ratio = max_align / thr95

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# ----- Panel A: alignment histogram vs shuffle null -----
axA = axes[0]
bins = np.linspace(0, max(a.max(), thr99) * 1.02, 60)
axA.hist(a, bins=bins, color='#aac4e0', edgecolor='#5a82ad', lw=0.4)
axA.axvline(thr95, color='#c44', lw=1.6, ls='--',
            label=f'shuffle null thr95 = {thr95:.3f}')
axA.axvline(thr99, color='#7a1f1f', lw=1.6, ls=':',
            label=f'shuffle null thr99 = {thr99:.3f}')
axA.set_yscale('log')
axA.set_xlabel('alignment  |Spearman| to best physical label')
axA.set_ylabel('number of active SAE features  (log)')
axA.set_title(f'(A) Alignment of {n_active} active features vs label-shuffle null')
axA.legend(loc='upper right', fontsize=10)
axA.annotate(
    f'max alignment = {max_align:.3f}\n(~{ratio:.0f}x the thr95 null)',
    xy=(max_align, 1.5), xytext=(max_align * 0.62, 30),
    fontsize=10, color='#a22', ha='center',
    arrowprops=dict(arrowstyle='->', color='#a22', lw=1.1))

# ----- Panel B: alignment vs novelty, coloured by stability -----
axB = axes[1]
axB.scatter(a[~stable], nv[~stable], s=10, c='#bbbbbb', alpha=0.55,
            linewidths=0, rasterized=True, label='not seed-stable (rec < 0.5)')
axB.scatter(a[stable], nv[stable], s=14, c='#2a6db0', alpha=0.7,
            linewidths=0, rasterized=True, label='seed-stable (rec >= 0.5)')

# alien-candidate region: stable & nov>0.7 & align<=thr95
alien = stable & (nv > 0.7) & (a <= thr95)
n_alien_here = int(alien.sum())
axB.axvspan(0, thr95, ymin=(0.7 - axB.get_ylim()[0]), ymax=1, alpha=0.0)  # placeholder
# draw the region rectangle explicitly
from matplotlib.patches import Rectangle
ymax = max(1.02, nv.max() * 1.02)
axB.add_patch(Rectangle((0, 0.7), thr95, ymax - 0.7, facecolor='#f2c14e',
                        alpha=0.22, edgecolor='#c89a20', lw=1.2, zorder=0))
axB.scatter(a[alien], nv[alien], s=22, facecolors='none', edgecolors='#b8860b',
            linewidths=1.1, label=f'alien candidates (n = {n_alien_here})', zorder=4)

axB.axvline(thr95, color='#c44', lw=1.2, ls='--')
axB.set_xlim(0, a.max() * 1.03)
axB.set_ylim(0, ymax)
axB.set_xlabel('alignment  |Spearman| to best physical label')
axB.set_ylabel('novelty  (residual activation-variance frac., 6 labels regressed out)')
axB.set_title('(B) Alignment vs novelty for active features')
axB.legend(loc='lower right', fontsize=9.5)

# inset: zoom the alien-candidate region (thr95 boundary sits very close to 0)
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
axins = inset_axes(axB, width='42%', height='42%', loc='upper right', borderpad=1.4)
x_in = thr95 * 1.8
axins.add_patch(Rectangle((0, 0.7), thr95, ymax - 0.7, facecolor='#f2c14e',
                          alpha=0.22, edgecolor='#c89a20', lw=1.2, zorder=0))
msk = a <= x_in
ms_stable = msk & stable
ms_not = msk & (~stable)
axins.scatter(a[ms_not], nv[ms_not], s=10, c='#bbbbbb', alpha=0.55, linewidths=0)
axins.scatter(a[ms_stable], nv[ms_stable], s=16, c='#2a6db0', alpha=0.7, linewidths=0)
axins.scatter(a[alien], nv[alien], s=26, facecolors='none', edgecolors='#b8860b',
              linewidths=1.2, zorder=4)
axins.axvline(thr95, color='#c44', lw=1.0, ls='--')
axins.axhline(0.7, color='#c89a20', lw=0.8, ls=':')
axins.set_xlim(0, x_in)
axins.set_ylim(0.55, ymax)
axins.set_title('zoom: alien region', fontsize=8.5)
axins.tick_params(labelsize=7.5)
axins.grid(alpha=0.2)
mark_inset(axB, axins, loc1=2, loc2=3, fc='none', ec='#888', lw=0.7)

caveat = (
    'alien candidates: seed-stable, high-novelty, not label-aligned (align <= thr95) '
    '-- CORRELATIONAL only (no causal test).\n'
    f'JSON-reported alien_candidates = {n_alien}; reproduced here = {n_alien_here}.')
fig.text(0.012, -0.03, caveat, fontsize=8.8, color='#555', va='top')

pc.save(fig, '13_sae_alignment_novelty.png')
