"""
Track B figure: the Hubble tuning fork in AION's image embedding.
Reads results/trackB_fork.npy (proj_seq, proj_bar, featured, bar, smooth, edgeon)
+ results/trackB.json. Writes figures/trackB_fork.png.

Run from the project root:  python code/plots/figTrackB.py
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

F = np.load("results/trackB_fork.npy")
J = json.load(open("results/trackB.json"))
ps, pb, featured, bar, smooth, edgeon = [F[:, i] for i in range(6)]

fig = plt.figure(figsize=(13.5, 4.4))
gs = GridSpec(1, 3, width_ratios=[1.2, 1, 1], wspace=0.45)

# A) the fork: handle (ellipticals) + two prongs (barred / unbarred disks)
axA = fig.add_subplot(gs[0])
disk = (featured > 0.5) & (edgeon < 0.5)
groups = [("ellipticals (handle)", smooth > 0.7, "#c0392b", 4, 0.25),
          ("disks, weak/no bar", disk & (bar < 0.2), "#2980b9", 6, 0.5),
          ("disks, strong bar", disk & (bar > 0.3), "#e67e22", 12, 0.85)]
for lbl, sel, col, sz, al in groups:
    axA.scatter(ps[sel], pb[sel], s=sz, c=col, alpha=al, edgecolors="none", label=f"{lbl} (n={int(sel.sum())})")
axA.set_xlabel("main-sequence axis  (elliptical $\\rightarrow$ disk)")
axA.set_ylabel("bar direction")
axA.set_title("The fork: a handle that opens into\nbarred / unbarred prongs", fontsize=10)
axA.legend(fontsize=7.5, loc="upper left", framealpha=0.9)

# B) the fork opens: spread along the bar direction vs position on the main sequence
axB = fig.add_subplot(gs[1])
qs = np.quantile(ps, np.linspace(0, 1, 11))
xc = 0.5 * (qs[:-1] + qs[1:])
spread = [pb[(ps >= qs[i]) & (ps < qs[i + 1])].std() for i in range(10)]
axB.plot(xc, spread, "o-", color="#8e44ad", lw=2, ms=6)
axB.set_xlabel("main-sequence axis  (elliptical $\\rightarrow$ disk)")
axB.set_ylabel("spread along bar direction")
axB.set_title("Branch point: the bar axis\nfans out only among disks", fontsize=10)

# C) orthogonality vs the label-correlation null (excess = independent DOF)
axC = fig.add_subplot(gs[2])
keys = ["seq-bar", "seq-spiral", "bar-spiral"]
emb = [J["angles"][k]["embed_angle"] for k in keys]
nul = [J["angles"][k]["label_null"] for k in keys]
x = np.arange(len(keys))
axC.bar(x - 0.2, emb, 0.4, color="#16a085", label="embedding")
axC.bar(x + 0.2, nul, 0.4, color="#bbbbbb", label="label-correlation null")
axC.axhline(90, ls="--", color="0.5", lw=1); axC.text(2.1, 91, "90$\\degree$", color="0.5", fontsize=8)
axC.set_xticks(x); axC.set_xticklabels(keys, fontsize=9, rotation=12)
axC.set_ylabel("angle between directions (deg)"); axC.set_ylim(0, 110)
axC.set_title("Bar is more orthogonal than its\nlabel correlation forces", fontsize=10)
axC.legend(fontsize=8, loc="lower right")

fig.savefig("figures/trackB_fork.png", dpi=200, bbox_inches="tight")
print("wrote figures/trackB_fork.png")
