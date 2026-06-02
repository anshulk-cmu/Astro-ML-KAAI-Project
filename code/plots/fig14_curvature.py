"""Figure 14 - Curvature of the AION-1 galaxy-embedding geometry.

Panel A: delta-hyperbolicity (diameter-normalized; lower = more tree-like).
Panel B: Ollivier-Ricci edge curvature (2k subsample, k=10).
Faithful read: mildly more tree-like than a matched-covariance Gaussian but far
from a synthetic tree; locally mostly positive curvature with ~4% negative bridges.
"""
import sys
sys.path.insert(0, "code/plots")
import plotcommon as pc
import numpy as np
import matplotlib.pyplot as plt

C = pc.load_json("curvature")
d = C["delta"]
o = C["ollivier"]["E_full"]

# --- Panel A series, ordered tree-like -> least tree-like -------------------
series = [
    ("Synthetic tree\n(validation)", "tree_validation", "#4c956c"),
    ("AION E_full\n(euclidean)",     "E_full_euclid",    "#1f77b4"),
    ("AION E_img\n(euclidean)",      "E_img_euclid",     "#5599cc"),
    ("Gaussian anchor\n(matched cov)", "gaussian_anchor", "#999999"),
    ("AION E_full\n(cosine)",        "E_full_cosine",    "#3355aa"),
]
labels = [s[0] for s in series]
medians = [d[s[1]]["median"] for s in series]
p95s = [d[s[1]]["p95"] for s in series]
colors = [s[2] for s in series]
x = np.arange(len(series))

fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 5.6))

# light p95 background bars (the spread), median solid on top
axA.bar(x, p95s, width=0.62, color=colors, alpha=0.22,
        label="p95 (upper spread)", zorder=1)
axA.bar(x, medians, width=0.62, color=colors, alpha=0.95,
        edgecolor="black", linewidth=0.6, label="median", zorder=2)

# Gaussian-anchor reference line so AION bars are visibly below it
gauss_med = d["gaussian_anchor"]["median"]
axA.axhline(gauss_med, color="#777777", ls="--", lw=1.3, zorder=3,
            label=f"Gaussian-anchor median = {gauss_med:.4f}")

for xi, m, p in zip(x, medians, p95s):
    axA.text(xi, p + 0.0015, f"{m:.4f}", ha="center", va="bottom",
             fontsize=9, color="black")

axA.set_xticks(x)
axA.set_xticklabels(labels, fontsize=9)
axA.set_ylabel(r"$\delta$-hyperbolicity  (diameter-normalized)")
axA.set_ylim(0, max(p95s) * 1.18)
axA.set_title("A. Gromov $\\delta$-hyperbolicity: lower = more tree-like")
axA.legend(loc="upper left", fontsize=8.5)
axA.grid(axis="x", visible=False)
axA.text(0.5, -0.30,
         "Faithful read: AION sits below the matched-covariance Gaussian anchor\n"
         "(mildly more tree-like than random) but well above the synthetic tree (not a tree).",
         transform=axA.transAxes, ha="center", va="top", fontsize=8.5,
         color="#333333")

# --- Panel B: Ollivier-Ricci -------------------------------------------------
mean = o["mean"]; p5 = o["p5"]; p95 = o["p95"]; fneg = o["frac_negative"]
yerr = np.array([[mean - p5], [p95 - mean]])  # asymmetric whisker p5..p95

axB.axhline(0.0, color="#cc3333", ls="-", lw=1.4, zorder=1,
            label="sign boundary (curvature = 0)")
axB.errorbar([0], [mean], yerr=yerr, fmt="o", ms=11, color="#1f77b4",
             ecolor="#1f77b4", elinewidth=2.2, capsize=10, capthick=2.2,
             zorder=3, label="mean with p5-p95 whisker")

axB.annotate(f"mean = {mean:.3f}", (0, mean), xytext=(0.10, mean),
             fontsize=10, va="center", color="#1f77b4")
axB.annotate(f"p95 = {p95:.3f}", (0, p95), xytext=(0.10, p95),
             fontsize=9, va="center", color="#555555")
axB.annotate(f"p5 = {p5:.3f}", (0, p5), xytext=(0.10, p5),
             fontsize=9, va="center", color="#555555")

axB.text(0.0, -0.06,
         f"frac. negative edges = {fneg*100:.1f}%\n(negative-curvature 'bridges')",
         ha="center", va="top", fontsize=9.5, color="#cc3333")

axB.set_xlim(-0.6, 0.9)
axB.set_ylim(min(p5, 0) - 0.10, p95 * 1.15)
axB.set_xticks([])
axB.set_ylabel("Ollivier-Ricci edge curvature")
axB.set_title("B. Ollivier-Ricci curvature (2k subsample, $k$=10)")
axB.legend(loc="upper right", fontsize=8.5)
axB.grid(axis="x", visible=False)
axB.text(0.5, -0.18,
         "Faithful read: edges are mostly locally positive (clustered),\n"
         "with ~4% negative-curvature bridge edges between clusters.",
         transform=axB.transAxes, ha="center", va="top", fontsize=8.5,
         color="#333333")

fig.suptitle("Curvature of the AION-1 galaxy-embedding geometry "
             "($\\delta$-hyperbolicity and Ollivier-Ricci)", fontsize=14)
print(pc.save(fig, "14_curvature.png"))
