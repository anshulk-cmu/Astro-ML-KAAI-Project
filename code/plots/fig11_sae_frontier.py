import sys
sys.path.insert(0, 'code/plots')
import numpy as np
import matplotlib.pyplot as plt
import plotcommon as pc

sae = pc.load_json('sae')
frontier = sae['frontier']

ks = np.array(sorted(int(k) for k in frontier), dtype=float)
fvu = np.array([frontier[str(int(k))] for k in ks], dtype=float)

k_op = 32
fvu_op = frontier[str(k_op)]

fig, ax = plt.subplots(figsize=(8.5, 5.5))
ax.plot(ks, fvu, '-', color='#1f5fa8', lw=2, zorder=2)
ax.plot(ks, fvu, 'o', color='#1f5fa8', ms=7, zorder=3)

ax.axvline(k_op, color='#c44', lw=1.4, ls='--', zorder=1)
ax.annotate(
    f'operating point\nk = {k_op},  FVU = {fvu_op:.3f}',
    xy=(k_op, fvu_op), xytext=(k_op * 1.25, fvu_op + 0.045),
    fontsize=11, color='#a22',
    arrowprops=dict(arrowstyle='->', color='#a22', lw=1.2))

ax.set_xscale('log')
ax.set_xticks(ks)
ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
ax.minorticks_off()
ax.set_xlabel('k (active latents per galaxy, L0)')
ax.set_ylabel('FVU')
ax.set_title('SAE reconstruction error vs sparsity (TopK k = L0)')
ax.set_ylim(0, max(fvu) * 1.12)

pc.save(fig, '11_sae_frontier.png')
