"""
Track D figure: the reducibility landscape (S vs M with the Engels paper's
calibration), example top-cluster projections, the distributed-loop ablation
(the headline), and the physics-vs-instrument feature census.
Reads results/trackD_eimg.json + results/trackD_eimg_clusters.npz.
Writes figures/trackD.png.   Run: python code/plots/figTrackD.py
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

J = json.load(open("results/trackD_eimg.json"))
P = np.load("results/trackD_eimg_clusters.npz")

fig = plt.figure(figsize=(14.5, 9))
gs = GridSpec(2, 2, wspace=0.32, hspace=0.44)

# A) reducibility landscape: S vs M for all tested clusters + paper calibration
axA = fig.add_subplot(gs[0, 0])
S = [e["S"] for e in J["clusters"]]; M = [e["M"] for e in J["clusters"]]
sc = axA.scatter(M, S, c=[e["score"] for e in J["clusters"]], cmap="viridis", s=45)
axA.axvline(0.18, ls=":", color="#2ca02c", lw=1.2); axA.text(0.185, max(S)*0.98, "paper's clean\ncircle (M~0.18)", color="#2ca02c", fontsize=7.5, va="top")
axA.axvline(0.64, ls=":", color="#d62728", lw=1.2); axA.text(0.55, max(S)*0.98, "paper's mixture\n(M~0.64)", color="#d62728", fontsize=7.5, va="top")
fig.colorbar(sc, ax=axA, fraction=0.045, pad=0.03).set_label("(1-M) x S score")
axA.set_xlabel("$\\epsilon$-mixture index M (lower = less mixture-like)")
axA.set_ylabel("separability index S (bits; higher = less separable)")
axA.set_title("Candidate multi-D clusters sit between the paper's\ncalibrations: structured, but no calendar-grade circle", fontsize=10)

# B) two example top-cluster projections
gsB = gs[0, 1].subgridspec(1, 2, wspace=0.35)
tops = [e for e in J["clusters"] if f"cluster{e['cluster']}" in P][:2]
for k, e in enumerate(tops):
    ax = fig.add_subplot(gsB[0, k])
    d = P[f"cluster{e['cluster']}"]
    # colour by whichever of the two STORED labels (g-r col 5, z col 4) fits better
    col, lab = (d[:, 5], "g-r") if (e["r2_gr"] or 0) >= (e["r2_z"] or 0) else (d[:, 4], "z")
    fin = np.isfinite(col)
    ax.scatter(d[fin, 0], d[fin, 1], c=col[fin], s=3, alpha=0.5, cmap="coolwarm")
    ax.set_title(f"#{e['cluster']} (m={e['members']})\nS={e['S']:.2f} M={e['M']:.2f}, colour={lab}", fontsize=8.5)
    ax.set_xticks([]); ax.set_yticks([])

# C) the distributed-loop ablation (headline)
axC = fig.add_subplot(gs[1, 0])
da = J["distributed_loop_ablation"]
Ks = [0, 15, 50, 200]
ab = [da["baseline"]] + [da[f"K{k}"]["ablate_top_loopcorr"] for k in (15, 50, 200)]
ct = [da["baseline"]] + [da[f"K{k}"]["random_control"] for k in (15, 50, 200)]
axC.plot(Ks, ab, "o-", color="#d62728", lw=2, ms=7, label="ablate top-K loop-correlated features")
axC.plot(Ks, ct, "s--", color="#999999", lw=1.5, label="ablate K random features (control)")
axC.axhline(45, ls=":", color="0.6", lw=1); axC.text(2, 46, "chance", color="0.5", fontsize=8)
axC.set_xlabel("K features ablated from the SAE reconstruction")
axC.set_ylabel("PA recovery error (deg)")
axC.set_ylim(0, 50)
axC.set_title("The loop's information is FRACTURED across the dictionary:\nremoving loop-correlated features degrades recovery monotonically\n(information-removal test; energy-matched controls flat)", fontsize=9.5)
axC.legend(fontsize=8)

# D) physics-vs-instrument feature census
axD = fig.add_subplot(gs[1, 1])
fl = J["instrument_flagging"]
pb, ib = fl["physics_breakdown"], fl["instrument_breakdown"]
sb = fl["strong_instrument_gt_0p3"]
names = list(pb) + list(ib)
vals = list(pb.values()) + list(ib.values())
cols = ["#1f77b4"] * len(pb) + ["#d62728"] * len(ib)
x = np.arange(len(names))
axD.bar(x, vals, color=cols)
for i, k in enumerate(ib):
    axD.bar(len(pb) + list(ib).index(k), sb[k], color="#7f0000")
axD.set_xticks(x); axD.set_xticklabels(names, rotation=35, fontsize=8, ha="right")
axD.set_ylabel("features (first-aligned)")
axD.set_title(f"The dictionary contains instrument-identity features:\ni-band-blankness dominates ({ib['iblank']} features, "
              f"{sb['iblank']} strong) --\nthe exclusion list for Phase 2 concept tests", fontsize=9.5)
from matplotlib.patches import Patch
axD.legend(handles=[Patch(color="#1f77b4", label="physics-first"),
                    Patch(color="#d62728", label="instrument-first"),
                    Patch(color="#7f0000", label="instrument, |corr|>0.3")], fontsize=8)

fig.savefig("figures/trackD.png", dpi=200, bbox_inches="tight")
print("wrote figures/trackD.png")
