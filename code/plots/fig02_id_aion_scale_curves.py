"""Fig 02 - AION embedding intrinsic dimension vs neighbour scale (headline ID figure).

Left:  Gride ID vs n1, log-x. ID falls from ~16.5 at the smallest scale to ~10
       at the largest tested scale; the curve is still declining.
Right: local-MLE ID vs neighbourhood size K, also still declining at K=226.
Both panels: PCA participation-ratio (linear anchor), astrophysical prior band 4-10,
and the resolution ceiling log2(N).
"""
import sys
sys.path.insert(0, "code/plots")
import numpy as np
import matplotlib.pyplot as plt
import plotcommon as pc

d = pc.load_json("intrinsicDim")["aion"]

style = {
    "E_full": dict(color="#1f77b4", marker="o", label="E_full (image+g/r/z flux+redshift)"),
    "E_img": dict(color="#d62728", marker="s", label="E_img (image only)"),
}

N = 48398
ceiling = np.log2(N)  # 15.56 resolution ceiling

fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6.2), sharey=True)

# ---- shared backdrop: prior band, ceiling ----
for ax in (axL, axR):
    ax.axhspan(4, 10, color="#2ca02c", alpha=0.08, zorder=0)
    ax.axhline(ceiling, color="gray", ls=":", lw=1.4, zorder=1)

# ---------- LEFT: Gride vs n1 ----------
for key, st in style.items():
    g = d[key]["gride"]
    xs = np.array(sorted(int(k) for k in g))
    ys = np.array([g[str(x)] for x in xs])
    cis = np.array([d[key]["gride_ci"][str(x)] for x in xs])
    axL.plot(xs, ys, color=st["color"], marker=st["marker"], ms=5, lw=1.8,
             label=st["label"], zorder=4)
    axL.fill_between(xs, cis[:, 0], cis[:, 1], color=st["color"], alpha=0.12, zorder=2)
    # PCA-PR linear anchor (dashed, same colour)
    axL.axhline(d[key]["pca_pr"], color=st["color"], ls="--", lw=1.3, alpha=0.8, zorder=2)

axL.set_xscale("log", base=2)
axL.set_xlabel("Gride neighbour scale  $n_1$  (log$_2$)")
axL.set_ylabel("intrinsic dimension")
axL.set_title("Gride ID vs neighbour scale")
axL.set_xticks([1, 2, 4, 8, 16, 32, 64, 128, 256])
axL.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())

# annotate small-scale sensitivity and largest tested scale
axL.annotate("small-scale estimate",
             xy=(1.0, d["E_full"]["gride"]["1"]), xytext=(3.0, 15.4),
             fontsize=9, color="#222", ha="left",
             bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#888", alpha=0.9),
             arrowprops=dict(arrowstyle="->", color="#222", lw=1.1,
                             connectionstyle="arc3,rad=-0.2"))
axL.annotate("largest tested scale\n(still declining)",
             xy=(256, d["E_full"]["gride"]["256"]), xytext=(36, 8.4),
             fontsize=9, color="#222",
             bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85),
             arrowprops=dict(arrowstyle="->", color="#222", lw=1))

# ---------- RIGHT: local-MLE vs K ----------
for key, st in style.items():
    lm = d[key]["local_mle"]
    xs = np.array(sorted(int(k) for k in lm))
    ys = np.array([lm[str(x)] for x in xs])
    cis = np.array([d[key]["local_mle_ci"][str(x)] for x in xs])
    axR.plot(xs, ys, color=st["color"], marker=st["marker"], ms=5, lw=1.8,
             label=st["label"], zorder=4)
    axR.fill_between(xs, cis[:, 0], cis[:, 1], color=st["color"], alpha=0.12, zorder=2)
    axR.axhline(d[key]["pca_pr"], color=st["color"], ls="--", lw=1.3, alpha=0.8, zorder=2)

axR.set_xscale("log", base=2)
axR.set_xlabel("local-MLE neighbourhood size  $K$  (log$_2$)")
axR.set_title("local-MLE ID vs neighbourhood size")
axR.set_xticks([10, 14, 20, 28, 40, 56, 80, 113, 160, 226])
axR.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
axR.tick_params(axis="x", labelrotation=45, labelsize=9)

axR.annotate(f"largest-scale ID {d['E_full']['local_mle']['226']:.1f}",
             xy=(226, d["E_full"]["local_mle"]["226"]), xytext=(28, 9.3),
             fontsize=9, color="#444",
             arrowprops=dict(arrowstyle="->", color="#444", lw=1))

# ---- y range to fit everything incl ceiling line and band ----
for ax in (axL, axR):
    ax.set_ylim(3.5, 17.0)

# ---- combined legend (curves + anchors + band + ceiling) on left panel ----
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
extra = [
    Line2D([0], [0], color="gray", ls="--", lw=1.3, label="PCA participation-ratio (linear anchor)"),
    Patch(facecolor="#2ca02c", alpha=0.18, label="astrophysical prior band (ID 4-10)"),
    Line2D([0], [0], color="gray", ls=":", lw=1.4, label=f"resolution ceiling  log$_2$(N) = {ceiling:.2f}"),
]
h, l = axL.get_legend_handles_labels()
axL.legend(h + extra, l + [e.get_label() for e in extra], loc="lower left", fontsize=8.8)

fig.suptitle("AION galaxy-embedding intrinsic dimension across neighbour scales",
             fontsize=14)

# faithful footer
foot = ("Both estimators decline with scale. At the largest tested scales they read about 10-11.4, "
        "but neither curve has proved a plateau. Shading is a conditional row-bootstrap CI on the fixed neighbour graph. "
        "PCA-PR overlaps this range and is a linear-spread anchor, not an independent manifold-ID estimate.")
fig.text(0.5, -0.02, foot, ha="center", va="top", fontsize=8.5, color="#333")

pc.save(fig, "02_id_aion_scale_curves.png")
