"""Figure 15 - Topology of the AION-1 galaxy-embedding manifold.

Panel A (beta0): cutting the longest MST edges peels off single outliers; the
giant component stays ~48k => one connected continuum (beta0 = 1).
Panel B (beta1): max persistence per metric vs significance threshold; ~0 loops
recur across subsamples or survive a metric change => beta1 = 0.
"""
import sys
sys.path.insert(0, "code/plots")
import plotcommon as pc
import numpy as np
import matplotlib.pyplot as plt

T = pc.load_json("topology")
b0 = T["beta0"]
b1 = T["beta1"]
N = 48398

fig, (axA, axB) = plt.subplots(1, 2, figsize=(15, 5.6))

# --- Panel A: beta0 component sizes after k longest-MST-edge cuts ------------
ss = b0["split_sizes"]
cuts = sorted(ss.keys(), key=int)          # "1","2","3"
rows = ["0"] + cuts                          # include the uncut graph
sizes_by_row = {"0": [N]}
for k in cuts:
    sizes_by_row[k] = ss[k]

y = np.arange(len(rows))[::-1]               # top row = most cuts
giant_color = "#1f77b4"
frag_color = "#d6604d"

for yi, k in zip(y, rows):
    comps = sizes_by_row[k]
    giant = comps[0]
    nfrag = len(comps) - 1
    # giant component bar (full width, log-friendly linear is fine at this scale)
    axA.barh(yi, giant, color=giant_color, edgecolor="black", linewidth=0.6,
             height=0.6, zorder=2)
    axA.text(giant * 0.5, yi, f"giant: {giant:,}", ha="center", va="center",
             color="white", fontsize=10, fontweight="bold")
    # the size-1 fragments, stacked just past the giant (drawn enlarged to be visible)
    if nfrag:
        axA.barh(yi, nfrag * 600, left=giant, color=frag_color,
                 edgecolor="black", linewidth=0.6, height=0.6, zorder=2)
        axA.text(giant + nfrag * 600 + 250, yi,
                 f"+{nfrag} x size-1 outlier" + ("s" if nfrag > 1 else ""),
                 ha="left", va="center", color=frag_color, fontsize=9.5)

axA.set_yticks(y)
axA.set_yticklabels([f"{k} edge cut" + ("s" if k != "1" else "") if k != "0"
                     else "uncut kNN graph" for k in rows], fontsize=10)
axA.set_xlabel("component size (number of galaxies)")
axA.set_xlim(0, N * 1.12)
axA.set_title("A. $\\beta_0$: longest-MST-edge cuts peel off single outliers")
axA.grid(axis="y", visible=False)
axA.text(0.5, -0.20,
         "Fragments are drawn enlarged (true size = 1) to be visible. The giant "
         "component stays ~48k:\none connected continuum, not a red/blue "
         "$\\beta_0$=2 split.  Measured: $\\beta_0$ = "
         f"{b0['knn_components']}.",
         transform=axA.transAxes, ha="center", va="top", fontsize=8.7,
         color="#333333")

# --- Panel B: beta1 max persistence per metric ------------------------------
metrics = ["euclidean", "fermat", "diffusion"]
mcolors = {"euclidean": "#1f77b4", "fermat": "#9467bd", "diffusion": "#2ca02c"}
xb = np.arange(len(metrics))
maxp = [b1[m]["max_persistence"] for m in metrics]
sigthr = 0.1

bars = axB.bar(xb, maxp, width=0.55, color=[mcolors[m] for m in metrics],
               edgecolor="black", linewidth=0.6, zorder=2)
axB.axhline(sigthr, color="#cc3333", ls="--", lw=1.4, zorder=3,
            label=f"significance threshold = {sigthr:g}")

for xi, m in zip(xb, metrics):
    mp = b1[m]["max_persistence"]
    slm = b1[m]["sig_loops_mean"]
    rng = b1[m]["sig_loops_range"]
    nbars = b1[m]["mean_bars"]
    axB.text(xi, mp + 0.004, f"{mp:.3f}", ha="center", va="bottom", fontsize=10)
    axB.text(xi, 0.012,
             f"~{nbars:.0f} short\nnoise bars\n\nrecurring loops:\n"
             f"{slm:g}/10\n(range {rng[0]}-{rng[1]})",
             ha="center", va="bottom", fontsize=8.3, color="#222222")

axB.set_xticks(xb)
axB.set_xticklabels([m.capitalize() for m in metrics], fontsize=10)
axB.set_ylabel("max loop persistence  (diameter-normalized)")
axB.set_ylim(0, max(maxp) * 1.35)
axB.set_title("B. $\\beta_1$: longest loop vs significance threshold")
axB.legend(loc="upper right", fontsize=9)
axB.grid(axis="x", visible=False)
axB.text(0.5, -0.20,
         "Thousands of short noise bars per subsample, but ~0 loops recur across "
         "subsamples\nor survive a metric change.  Measured: $\\beta_1$ = 0.",
         transform=axB.transAxes, ha="center", va="top", fontsize=8.7,
         color="#333333")

fig.suptitle("Topology of the AION-1 manifold: one connected component "
             "($\\beta_0$=1), no robust loops ($\\beta_1$=0)", fontsize=14)
print(pc.save(fig, "15_topology.png"))
