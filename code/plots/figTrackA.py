"""
Track A figure: position angle as a closed loop in AION's image embedding.
Reads results/trackA_loop.npy (paDeg, cos2theta_hat, sin2theta_hat for elongated
galaxies) + results/trackA.json. Writes figures/trackA_loop.png.

Run from the project root:  python code/plots/figTrackA.py
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

loop = np.load("results/trackA_loop.npy")          # cols: paDeg, cos_hat, sin_hat, is_heldout
J = json.load(open("results/trackA.json"))
te = loop[:, 3] > 0.5 if loop.shape[1] > 3 else np.ones(len(loop), bool)
pa, ch, sh = loop[te, 0], loop[te, 1], loop[te, 2]  # held-out rows only (probe never saw them)

rng = np.random.default_rng(0)                      # thin for a legible scatter
if len(pa) > 6000:
    k = rng.choice(len(pa), 6000, replace=False); pa, ch, sh = pa[k], ch[k], sh[k]

fig = plt.figure(figsize=(13, 4.4))
gs = GridSpec(1, 3, width_ratios=[1.05, 1, 1], wspace=0.5)

# A) the loop, coloured cyclically by position angle (cyclic colormap)
axA = fig.add_subplot(gs[0])
sc = axA.scatter(ch, sh, c=pa, cmap="twilight", s=5, alpha=0.55, vmin=0, vmax=180)
axA.set_aspect("equal"); axA.axhline(0, color="0.85", lw=.6); axA.axvline(0, color="0.85", lw=.6)
axA.set_xlabel(r"predicted $\cos 2\theta$"); axA.set_ylabel(r"predicted $\sin 2\theta$")
axA.set_title("Position angle traces a closed loop\n(E_img, held-out elongated galaxies)", fontsize=10)
cb = fig.colorbar(sc, ax=axA, fraction=0.046, pad=0.04); cb.set_label("PA (deg)")

# B) circular error sharpens with elongation
axB = fig.add_subplot(gs[1])
sweep = J.get("pa_vs_ellip", {})
xs = [0.5 * (float(k.split("-")[0]) + float(k.split("-")[1])) for k in sweep]
ys = [v["med_err_deg"] for v in sweep.values()]
axB.plot(xs, ys, "o-", color="#1f77b4", lw=2, ms=7)
axB.axhline(45, ls="--", color="0.6", lw=1); axB.text(xs[0], 46, "chance", color="0.5", fontsize=8)
axB.set_xlabel("ellipticity (more elongated $\\rightarrow$)"); axB.set_ylabel("median PA error (deg)")
axB.set_title("Loop sharpens as the major axis\nbecomes well defined", fontsize=10)
axB.set_ylim(0, 50)

# C) PA loop vs the controls (all doubled-angle, chance ~45 deg)
axC = fig.add_subplot(gs[2])
bars = {"PA loop\n(E_img)": J["pa_loop_Eimg"]["med_err_deg"],
        "top-2 PCs": J["pa_loop_PCA2"]["med_err_deg"],
        "shuffled PA\n(null)": J["pa_null_shuffle"]["med_err_deg"]}
cols = ["#2ca02c", "#999999", "#999999"]
axC.bar(range(len(bars)), list(bars.values()), color=cols, width=0.6)
axC.axhline(45, ls="--", color="0.6", lw=1); axC.text(1.4, 46, "chance", color="0.5", fontsize=8)
axC.set_xticks(range(len(bars))); axC.set_xticklabels(list(bars.keys()), fontsize=9)
axC.set_ylabel("median PA error (deg)"); axC.set_ylim(0, 50)
axC.set_title("Only the fitted embedding directions\nrecover the angle", fontsize=10)
for i, v in enumerate(bars.values()):
    axC.text(i, v + 1.2, f"{v:.0f}", ha="center", fontsize=9)

fig.savefig("figures/trackA_loop.png", dpi=200, bbox_inches="tight")
print("wrote figures/trackA_loop.png")
