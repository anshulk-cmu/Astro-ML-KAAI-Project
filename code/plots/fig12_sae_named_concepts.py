import sys
sys.path.insert(0, 'code/plots')
import numpy as np
import matplotlib.pyplot as plt
import plotcommon as pc

sae = pc.load_json('sae')
nc = sae['named_concepts']
thr95 = sae['concepts']['align_null_thr95']

names = list(nc.keys())
n = np.array([nc[k]['n'] for k in names], dtype=float)
n_stable = np.array([nc[k]['n_stable'] for k in names], dtype=float)
top_align = np.array([nc[k]['top_align'] for k in names], dtype=float)

order = np.argsort(n)  # ascending so largest at top of barh
names = [names[i] for i in order]
n = n[order]; n_stable = n_stable[order]; top_align = top_align[order]

y = np.arange(len(names))
h = 0.38

fig, ax = plt.subplots(figsize=(11, 6))
b1 = ax.barh(y + h / 2, n, height=h, color='#aac4e0',
             edgecolor='#5a82ad', label='significantly aligned (n)')
b2 = ax.barh(y - h / 2, n_stable, height=h, color='#2a6db0',
             edgecolor='#19456f', label='seed-stable subset (n_stable)')

# annotate top_align (max |Spearman|) at end of the total-n bar
for yi, ni, ta in zip(y, n, top_align):
    ax.text(ni + 3, yi + h / 2, f'max |rho| = {ta:.2f}', va='center', fontsize=9.5, color='#333')

ax.set_yticks(y)
ax.set_yticklabels(names)
ax.set_xlabel('number of SAE features')
ax.set_ylabel('physical concept')
ax.set_title('SAE features aligned to physical labels, per concept')
ax.set_xlim(0, max(n) * 1.32)
ax.legend(loc='lower right')

caveat = (
    "'aligned' = |Spearman| exceeds a label-shuffle null (thr95 = "
    f"{thr95:.3f}); naming is each feature's top-correlated label (CORRELATIONAL).\n"
    "Redshift is spread over many weak features (low max |rho|), not one strong feature.")
fig.text(0.012, -0.02, caveat, fontsize=8.6, color='#555', va='top')

pc.save(fig, '12_sae_named_concepts.png')
