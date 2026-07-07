"""
Track B supplementary figure: the corrected angle battery (with CIs), fork
dimensionality, monotonicity, and geodesic-vs-Euclidean.
Reads results/trackB.json. Writes figures/trackB_supplement.png.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

J = json.load(open("results/trackB.json"))

fig = plt.figure(figsize=(14, 8.5))
gs = GridSpec(2, 2, wspace=0.32, hspace=0.5)

# A) corrected angle battery: excess (embed - label_null) with CI, vs 0
axA = fig.add_subplot(gs[0, 0])
keys = ["seq-bar", "seq-spiral", "bar-spiral", "seq-edgeon", "seq-redshift", "seq-sersic"]
exc = [J["angles"][k]["excess"] for k in keys]
cis = [J["angles"][k]["excess_ci"] for k in keys]
cols = ["#2ca02c" if (c[0] > 0 or c[1] < 0) else "#bbbbbb" for c in cis]   # sig if CI excludes 0
y = np.arange(len(keys))
axA.barh(y, exc, color=cols)
for i, (v, c) in enumerate(zip(exc, cis)):
    axA.plot([c[0], c[1]], [i, i], color="black", lw=1.5)
axA.axvline(0, color="0.4", lw=1)
axA.set_yticks(y); axA.set_yticklabels(keys, fontsize=9)
axA.set_xlabel("excess angle vs label-correlation null (deg)")
axA.set_title("Corrected angle battery (bug fixed): bar's\nindependence from the sequence survives (CI excludes 0)", fontsize=10)

# B) fork dimensionality: disk-residual PCA explained variance
axB = fig.add_subplot(gs[0, 1])
evr = J["fork_dimensionality"]["explained_variance_ratio_top10"][:8]
axB.bar(range(1, len(evr) + 1), evr, color="#8e44ad")
axB.set_xlabel("PC of the disk-population residual (main sequence removed)")
axB.set_ylabel("explained variance ratio")
pc1bar = J["fork_dimensionality"]["pc1_aligned_with_bar"]
axB.set_title(f"Fork residual is low-dimensional (PC1+2 = {sum(evr[:2])*100:.0f}%),\n"
              f"but PC1 is orthogonal to the fitted bar direction (|cos|={pc1bar:.2f})", fontsize=9.5)

# C) monotonicity (Spearman) + preferred-axis divergence
axC = fig.add_subplot(gs[1, 0])
mo = J["monotonicity_spearman"]
labs = list(mo.keys())
axC.bar(range(len(labs)), [mo[k] for k in labs], color="#16a085")
for i, k in enumerate(labs):
    axC.text(i, mo[k] + (0.03 if mo[k] > 0 else -0.08), f"{mo[k]:+.2f}", ha="center", fontsize=9)
axC.axhline(0, color="0.6", lw=.8)
axC.set_xticks(range(len(labs))); axC.set_xticklabels(labs, fontsize=9)
axC.set_ylabel("Spearman rank correlation")
pa = J["preferred_axis"]["angle_seq_vs_sersic_direction"]
axC.set_title(f"The ordering IS monotonic in vote labels;\nbut the sersic-only direction sits {pa:.0f} deg from it", fontsize=9.5)

# D) geodesic vs Euclidean, both targets
axD = fig.add_subplot(gs[1, 1])
g1, g2 = J["geodesic_vs_euclidean_sersic"], J["geodesic_vs_euclidean_votes"]
x = np.arange(2)
axD.bar(x - 0.2, [g1["mean_dist_euclidean"], g2["mean_dist_euclidean"]], 0.4,
       color="#1f77b4", label="Euclidean")
axD.bar(x + 0.2, [g1["mean_dist_geodesic"], g2["mean_dist_geodesic"]], 0.4,
       color="#d62728", label="geodesic (diffusion)")
axD.set_xticks(x); axD.set_xticklabels([f"sersic\n({100*g1['frac_geodesic_better']:.0f}% geo better)",
                                        f"vote vector\n({100*g2['frac_geodesic_better']:.0f}% geo better)"], fontsize=9)
axD.set_ylabel("mean neighbour distance (lower = more similar)")
axD.legend(fontsize=8)
axD.set_title("Geodesic (diffusion-map) distance does NOT\ntrack morphology better than raw Euclidean here", fontsize=9.5)

fig.savefig("figures/trackB_supplement.png", dpi=200, bbox_inches="tight")
print("wrote figures/trackB_supplement.png")
