import sys
sys.path.insert(0, 'code/plots')
import numpy as np
import matplotlib.pyplot as plt
import plotcommon as pc

sae = pc.load_json('sae')
con = sae['concepts']
max_align = con['max_align']
n_candidates = con['stable_unexplained_not_label_aligned']

align = np.load('results/sae_align.npy')
nov = np.load('results/sae_novelty.npy')
rec = np.load('results/sae_recurrence.npy')
qval = np.load('results/sae_qvalue.npy')

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
aligned = qval[active] <= 0.05

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# ----- Panel A: tie-aware alignment histogram by FDR status -----
axA = axes[0]
bins = np.linspace(0, a.max() * 1.02, 60)
axA.hist(a[~aligned], bins=bins, color='#cccccc', edgecolor='#888888', lw=0.4,
         label='not FDR-aligned')
axA.hist(a[aligned], bins=bins, color='#6baed6', edgecolor='#2a6db0', lw=0.4,
         label='FDR-aligned (q <= 0.05)')
axA.set_yscale('log')
axA.set_xlabel('alignment  |Spearman| to best physical label')
axA.set_ylabel('number of active SAE features  (log)')
axA.set_title(f'(A) Tie-aware alignment of {n_active} active features')
axA.legend(loc='upper right', fontsize=10)
axA.annotate(
    f'max alignment = {max_align:.3f}',
    xy=(max_align, 1.5), xytext=(max_align * 0.62, 30),
    fontsize=10, color='#a22', ha='center',
    arrowprops=dict(arrowstyle='->', color='#a22', lw=1.1))

# ----- Panel B: alignment vs novelty, coloured by stability -----
axB = axes[1]
axB.scatter(a[~stable], nv[~stable], s=10, c='#bbbbbb', alpha=0.55,
            linewidths=0, rasterized=True, label='not seed-stable (rec < 0.5)')
axB.scatter(a[stable], nv[stable], s=14, c='#2a6db0', alpha=0.7,
            linewidths=0, rasterized=True, label='seed-stable (rec >= 0.5)')

# descriptive candidate region: stable, high residual variance, not FDR-aligned
candidate = stable & (nv > 0.7) & ~aligned
n_candidates_here = int(candidate.sum())
ymax = max(1.02, nv.max() * 1.02)
axB.axhspan(0.7, ymax, color='#f2c14e', alpha=0.10, zorder=0)
axB.scatter(a[candidate], nv[candidate], s=22, facecolors='none', edgecolors='#b8860b',
            linewidths=1.1, label=f'stable unexplained candidates (n={n_candidates_here})', zorder=4)
axB.set_xlim(0, a.max() * 1.03)
axB.set_ylim(0, ymax)
axB.set_xlabel('alignment  |Spearman| to best physical label')
axB.set_ylabel('novelty  (residual activation-variance frac., 6 labels regressed out)')
axB.set_title('(B) Alignment vs novelty for active features')
axB.legend(loc='lower right', fontsize=9.5)

caveat = (
    'FDR alignment: average-tie Spearman, Bonferroni over six labels, BH across active features. '
    'Unexplained candidates use a descriptive novelty > 0.7 threshold and are not discoveries.\n'
    f'JSON candidates = {n_candidates}; reproduced here = {n_candidates_here}.')
fig.text(0.012, -0.03, caveat, fontsize=8.8, color='#555', va='top')

pc.save(fig, '13_sae_alignment_novelty.png')
