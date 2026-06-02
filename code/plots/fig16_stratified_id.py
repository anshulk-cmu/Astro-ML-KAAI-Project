"""Figure 16 - Heterogeneity: intrinsic dimension stratified by g-r colour.

Panel A: g-r histogram with the GMM cut and the two component means; passive
(red, g-r>cut) vs star-forming (blue, g-r<cut).
Panel B: TwoNN intrinsic dimension (MLE +/- CI) for passive vs SF, with the
Delta-ID that excludes zero.
Faithful note: absolute IDs (~15-17) are small-scale (1st/2nd-NN) and
noise-inflated; the RELATIVE Delta-ID is the signal.
"""
import sys
sys.path.insert(0, "code/plots")
import plotcommon as pc
import numpy as np
import matplotlib.pyplot as plt

S = pc.load_json("stratifiedId")
cut = S["cut"]
m_sf, m_pass = sorted(S["gmm_means"])     # 0.78 (SF), 1.24 (passive)
IDp = S["ID_passive"]; IDs = S["ID_sf"]; dID = S["delta_ID"]

df, _ = pc.load_df()
gr = pc.labels(df)["g-r"][0]
gr = gr[np.isfinite(gr)]
n_pass = int((gr > cut).sum())
n_sf = int((gr < cut).sum())

fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 5.6))

# --- Panel A: g-r histogram with GMM cut + component means ------------------
bins = np.linspace(np.percentile(gr, 0.2), np.percentile(gr, 99.8), 70)
sf_mask = bins[:-1] < cut                  # for per-bin colouring
counts, edges = np.histogram(gr, bins=bins)
centers = 0.5 * (edges[:-1] + edges[1:])
bar_colors = np.where(centers < cut, "#4477cc", "#cc4444")
axA.bar(centers, counts, width=np.diff(edges), color=bar_colors, alpha=0.55,
        edgecolor="none", align="center", zorder=2)

axA.axvline(cut, color="black", ls="--", lw=1.6, zorder=4,
            label=f"GMM cut = {cut:.3f}")
axA.axvline(m_sf, color="#1f3f99", ls=":", lw=1.6, zorder=4,
            label=f"SF mean = {m_sf:.2f}")
axA.axvline(m_pass, color="#992020", ls=":", lw=1.6, zorder=4,
            label=f"passive mean = {m_pass:.2f}")

ymax = counts.max()
axA.set_ylim(0, ymax * 1.15)
axA.text(m_sf - 0.05, ymax * 0.42, f"star-forming\nn = {n_sf:,}",
         ha="right", va="center", color="#1f3f99", fontsize=10, fontweight="bold")
axA.text(m_pass + 0.05, ymax * 0.55, f"passive\nn = {n_pass:,}",
         ha="left", va="center", color="#992020", fontsize=10, fontweight="bold")

axA.set_xlabel(pc.labels(df)["g-r"][1])
axA.set_ylabel("number of galaxies")
axA.set_title("A. $g-r$ colour split by 2-component GMM")
axA.legend(loc="upper right", fontsize=9)
axA.grid(axis="x", visible=False)

# --- Panel B: TwoNN intrinsic dimension per stratum -------------------------
groups = [
    ("Star-forming\n($g-r$ < cut)", IDs, "#1f77b4"),
    ("Passive\n($g-r$ > cut)", IDp, "#cc4444"),
]
xb = np.arange(len(groups))
for xi, (name, d, col) in zip(xb, groups):
    mle = d["mle"]; lo, hi = d["ci"]; n = d["n"]
    yerr = np.array([[mle - lo], [hi - mle]])
    axB.errorbar([xi], [mle], yerr=yerr, fmt="o", ms=12, color=col,
                 ecolor=col, elinewidth=2.4, capsize=10, capthick=2.4, zorder=3)
    axB.text(xi, hi + 0.06, f"{mle:.2f}\n[{lo:.2f}, {hi:.2f}]\nn = {n:,}",
             ha="center", va="bottom", fontsize=9.5, color=col)

axB.set_xticks(xb)
axB.set_xticklabels([g[0] for g in groups], fontsize=10)
axB.set_xlim(-0.6, 1.6)
ylo = min(IDs["ci"][0], IDp["ci"][0]) - 0.6
yhi = max(IDs["ci"][1], IDp["ci"][1]) + 1.1
axB.set_ylim(ylo, yhi)
axB.set_ylabel("TwoNN intrinsic dimension (MLE)")
axB.set_title("B. Intrinsic dimension: passive vs star-forming")
axB.grid(axis="x", visible=False)

# Delta-ID annotation between the two points
exz = "excludes 0" if dID["excludes_zero"] else "includes 0"
axB.annotate("", xy=(1, IDp["mle"]), xytext=(0, IDs["mle"]),
             arrowprops=dict(arrowstyle="<->", color="#444444", lw=1.4))
axB.text(0.5, (IDs["mle"] + IDp["mle"]) / 2 - 0.45,
         f"$\\Delta$ID = {dID['mean']:.2f}\n"
         f"[{dID['ci'][0]:.2f}, {dID['ci'][1]:.2f}]\n({exz})",
         ha="center", va="top", fontsize=10, color="#222222",
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#888888", alpha=0.9))

axB.text(0.5, -0.21,
         "Faithful note: absolute IDs (~15-17) are small-scale (1st/2nd-NN) and "
         "noise-inflated;\nthe RELATIVE $\\Delta$ID is the signal "
         "(passive on a slightly higher-dimensional submanifold).",
         transform=axB.transAxes, ha="center", va="top", fontsize=8.7,
         color="#333333")

fig.suptitle("Heterogeneity of the AION-1 manifold: intrinsic dimension "
             "stratified by $g-r$ colour", fontsize=14)
print(pc.save(fig, "16_stratified_id.png"))
