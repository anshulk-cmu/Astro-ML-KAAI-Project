"""03 — Metric concentration diagnostic.

Source: results/metric.json  {E_full,E_img}:{metric}:{rdr, nn_over_mean}.
Panel A: relative distance ratio RDR = (Dmax-Dmin)/Dmin per metric (log y; higher = more contrast).
Panel B: nearest-neighbour-over-mean ratio per metric (lower = more concentrated).
Both grouped E_full vs E_img. 2k subsample, z-scored embeddings, k=15.
"""
import sys
sys.path.insert(0, "code/plots")
import numpy as np
import matplotlib.pyplot as plt
import plotcommon as pc

m = pc.load_json("metric")

# metric order: raw geometry -> intrinsic geometry
metrics = ["euclidean", "cosine", "isomap", "fermat_p2", "diffusion"]
nice = ["Euclidean", "Cosine", "Isomap", "Fermat (p=2)", "Diffusion"]
embs = ["E_full", "E_img"]
emb_nice = {"E_full": "E_full (img+spectra)", "E_img": "E_img (image-only)"}
emb_col = {"E_full": "#1f6fb4", "E_img": "#d2691e"}

rdr = {e: np.array([m[e][mm]["rdr"] for mm in metrics]) for e in embs}
nnm = {e: np.array([m[e][mm]["nn_over_mean"] for mm in metrics]) for e in embs}

x = np.arange(len(metrics))
w = 0.38

fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 5.6))

# ---- Panel A: RDR (log y) ----
for i, e in enumerate(embs):
    off = (i - 0.5) * w
    bars = axA.bar(x + off, rdr[e], w, label=emb_nice[e], color=emb_col[e],
                   edgecolor="black", linewidth=0.5, zorder=3)
    for b, val in zip(bars, rdr[e]):
        axA.text(b.get_x() + b.get_width() / 2, val * 1.05, f"{val:.0f}",
                 ha="center", va="bottom", fontsize=8.5, zorder=4)
axA.set_yscale("log")
axA.set_xticks(x)
axA.set_xticklabels(nice, rotation=18, ha="right")
axA.set_ylabel("RDR  =  (D$_{max}$ - D$_{min}$) / D$_{min}$   (log scale)")
axA.set_title("A. Relative distance ratio — higher = more contrast (less concentrated)")
axA.set_ylim(8, 600)
axA.legend(loc="upper left", fontsize=9.5)
axA.grid(axis="y", which="both", alpha=0.25)

# ---- Panel B: NN/mean ratio ----
for i, e in enumerate(embs):
    off = (i - 0.5) * w
    bars = axB.bar(x + off, nnm[e], w, label=emb_nice[e], color=emb_col[e],
                   edgecolor="black", linewidth=0.5, zorder=3)
    for b, val in zip(bars, nnm[e]):
        axB.text(b.get_x() + b.get_width() / 2, val + 0.006, f"{val:.2f}",
                 ha="center", va="bottom", fontsize=8.5, zorder=4)
axB.set_xticks(x)
axB.set_xticklabels(nice, rotation=18, ha="right")
axB.set_ylabel("NN distance / mean distance   (lower = more concentrated)")
axB.set_title("B. Nearest-neighbour / mean ratio")
axB.set_ylim(0, 0.52)
axB.legend(loc="upper right", fontsize=9.5)
axB.grid(axis="y", alpha=0.25)

fig.suptitle("Pairwise-distance concentration across metrics  (2k subsample, z-scored, k=15)",
             fontsize=14, y=1.04)

note = ("Read: raw Euclidean distances are the most concentrated (lowest RDR, highest NN/mean). "
        "Intrinsic metrics (isomap / Fermat / diffusion) give several-fold more contrast,\n"
        "motivating Fermat (p=2) as the primary downstream metric. E_full and E_img behave almost identically.")
fig.text(0.5, -0.06, note, ha="center", va="top", fontsize=9.5, color="#333333")

print(pc.save(fig, "03_metric_concentration.png"))
