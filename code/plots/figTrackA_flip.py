"""
Track A mirror-flip figure: the reflection half of the O(2) causal test.
Reads results/trackA_flip_disp.npy (paDeg, predicted displacement -2*theta,
recovered displacement) + results/trackA_flip.json. Writes figures/trackA_flip.png.

Run from the project root:  python code/plots/figTrackA_flip.py
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

D = np.load("results/trackA_flip_disp.npy")        # cols: paDeg, d_pred, d_rec
J = json.load(open("results/trackA_flip.json"))
pa, d_pred, d_rec = D[:, 0], D[:, 1], D[:, 2]
F, C = J["flip"], J["flip_rot30"]

rng = np.random.default_rng(0)
if len(pa) > 6000:
    k = rng.choice(len(pa), 6000, replace=False)
    pa_s, dp_s, dr_s = pa[k], d_pred[k], d_rec[k]
else:
    pa_s, dp_s, dr_s = pa, d_pred, d_rec

fig = plt.figure(figsize=(13, 4.4))
gs = GridSpec(1, 3, width_ratios=[1.05, 1, 1], wspace=0.45)

# A) the reflection law: recovered vs predicted displacement (-2*theta mod 180)
axA = fig.add_subplot(gs[0])
axA.scatter(dp_s, dr_s, c=pa_s, cmap="twilight", s=4, alpha=0.5, vmin=0, vmax=180)
axA.plot([-90, 90], [-90, 90], color="0.3", lw=1, ls="--", label="prediction (slope 1)")
axA.axvline(0, color="0.85", lw=.6); axA.axhline(0, color="0.85", lw=.6)
axA.set_xlabel(r"predicted displacement $-2\theta$ (deg, mod 180)")
axA.set_ylabel("recovered displacement (deg)")
axA.set_title(f"Mirror flip moves the readout by $-2\\theta$\n"
              f"slope {F['disp_slope']:.3f}, median|resid| {F['disp_median_abs_resid']:.1f} deg", fontsize=10)
axA.legend(fontsize=8, loc="upper left")

# B) fixed points vs antinodes: |displacement| against angular distance to PA=0/90
axB = fig.add_subplot(gs[1])
dist_fp = np.minimum(np.abs((pa_s + 90) % 180 - 90), np.abs((pa_s) % 180 - 90))
axB.scatter(dist_fp, np.abs(dr_s), s=4, alpha=0.35, color="#1f77b4")
xx = np.linspace(0, 45, 50)
axB.plot(xx, 2 * xx, color="0.3", lw=1, ls="--", label="prediction $|2\\theta|$")
axB.set_xlabel("angular distance to nearest fixed point (deg)")
axB.set_ylabel("|recovered displacement| (deg)")
axB.set_title(f"Fixed points at PA 0/90 hold still\n"
              f"median |disp|: {F['fixedpoint_median_absdisp_deg']:.1f} deg there, "
              f"{F['antinode_median_absdisp_deg']:.1f} at 45/135", fontsize=10)
axB.legend(fontsize=8, loc="upper left")

# C) O(2) closes: per-op circular error vs expectation, against chance
axC = fig.add_subplot(gs[2])
bars = {"flip\n($-\\theta$)": F["median_err_vs_minus_readout_deg"],
        "flip + rot 30\n($-\\theta-30$)": C["median_err_vs_expected_deg"],
        "baseline\n(untransformed)": J["baseline_median_err_deg"]}
cols = ["#2ca02c", "#2ca02c", "#999999"]
axC.bar(range(len(bars)), list(bars.values()), color=cols, width=0.6)
axC.axhline(45, ls="--", color="0.6", lw=1); axC.text(1.6, 46, "chance", color="0.5", fontsize=8)
axC.set_xticks(range(len(bars))); axC.set_xticklabels(list(bars.keys()), fontsize=9)
axC.set_ylabel("median circular error (deg)"); axC.set_ylim(0, 50)
axC.set_title("Reflection and composition both track\nthe O(2) prediction", fontsize=10)
for i, v in enumerate(bars.values()):
    axC.text(i, v + 1.2, f"{v:.1f}", ha="center", fontsize=9)

fig.savefig("figures/trackA_flip.png", dpi=200, bbox_inches="tight")
print("wrote figures/trackA_flip.png")
