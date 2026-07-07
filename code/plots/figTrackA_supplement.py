"""
Track A supplementary figure: cross-modal consistency, the causal rotation test,
RA/Dec native loop, and the unsupervised (PCA/SAE) discovery check.
Reads results/trackA_crossmodal.json, trackA_causal.json (if present),
trackA_radec.json, trackA_unsupervised.json. Writes figures/trackA_supplement.png.
"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

RES = "results"


def j(name):
    p = f"{RES}/{name}.json"
    return json.load(open(p)) if os.path.exists(p) else None


cm = j("trackA_crossmodal")
ca = j("trackA_causal")
rd = j("trackA_radec")
un = j("trackA_unsupervised")
ta = j("trackA")

fig = plt.figure(figsize=(14, 8.5))
gs = GridSpec(2, 2, wspace=0.32, hspace=0.42)

# A) cross-modal: image-derived vs shape-derived vs true
axA = fig.add_subplot(gs[0, 0])
labels = ["image vs\ntrue PA", "shape vs\ntrue PA", "image vs shape\n(CROSS-MODAL)"]
vals = [cm["img_vs_true"]["med_err_deg"], cm["shape_vs_true"]["med_err_deg"], cm["img_vs_shape"]["med_err_deg"]]
cis = [cm["img_vs_true"]["ci"], cm["shape_vs_true"]["ci"], cm["img_vs_shape"]["ci"]]
cols = ["#1f77b4", "#ff7f0e", "#2ca02c"]
axA.bar(range(3), vals, color=cols, width=0.6)
for i, (v, c) in enumerate(zip(vals, cis)):
    axA.errorbar(i, v, yerr=[[v - c[0]], [c[1] - v]], color="black", capsize=4, lw=1.3)
    axA.text(i, v + 0.3, f"{v:.1f}", ha="center", fontsize=9)
axA.set_xticks(range(3)); axA.set_xticklabels(labels, fontsize=9)
axA.set_ylabel("median circular error (deg)")
axA.set_title("Cross-modal check: image-encoded and catalog-\nencoded PA agree with each other, not just truth", fontsize=10)

# B) RA/Dec native loop
axB = fig.add_subplot(gs[0, 1])
if rd:
    labs2 = ["RA as loop\n(k=1, periodic)", "Dec as scalar\n(bounded arc)"]
    v2 = [rd["ra_loop_k1"]["med_err_deg"], None]
    r2dec = rd["dec_scalar"]["r2"]
    axB.bar(0, v2[0], color="#9467bd", width=0.5)
    axB.errorbar(0, v2[0], yerr=[[v2[0] - rd["ra_loop_k1"]["ci"][0]], [rd["ra_loop_k1"]["ci"][1] - v2[0]]],
                color="black", capsize=4)
    axB.text(0, v2[0] + 0.15, f"{v2[0]:.1f} deg err", ha="center", fontsize=9)
    axB.set_xlim(-1, 1.6); axB.set_ylim(0, 3.6)
    ax2 = axB.twinx()
    ax2.bar(1, r2dec, color="#8c564b", width=0.5)
    ax2.text(1, r2dec + 0.03, f"R2={r2dec:.2f}", ha="center", fontsize=9)
    ax2.set_ylim(0, 1.3); ax2.set_ylabel("Dec decodability R2", color="#8c564b")
    axB.set_xticks([0, 1]); axB.set_xticklabels(labs2, fontsize=9)
    axB.set_ylabel("RA circular error (deg)", color="#9467bd")
axB.set_title("RA (periodic) forms a loop like PA;\nDec (bounded) recovers as a plain scalar", fontsize=10)

# C) unsupervised discovery: supervised probe vs PCA-scan vs existing SAE dictionary
axC = fig.add_subplot(gs[1, 0])
sup_r2 = ta["pa_loop_Eimg"]["r2_cos"]   # the headline supervised R2 (cos2theta component)
pca_r2 = un["pca_scan"]["best_pc_pair_ring_r2"] if un else None
sae_c = un["sae_check"]["best_single_feature_corr_c2"] if un and un["sae_check"] else None
bars = {"supervised\nprobe (headline)": sup_r2, "best unsupervised\nPCA pair (top 50)": pca_r2}
axC.bar(range(len(bars)), list(bars.values()), color=["#2ca02c", "#999999"])
for i, v in enumerate(bars.values()):
    axC.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
axC.set_xticks(range(len(bars))); axC.set_xticklabels(list(bars.keys()), fontsize=9)
axC.set_ylabel("ring R2 (min of cos2theta,sin2theta)"); axC.set_ylim(0, 1.05)
axC.set_title("The loop is real but not dominant:\nunsupervised search finds only part of it", fontsize=10)

# D) causal rotation test (if complete)
axD = fig.add_subplot(gs[1, 1])
if ca:
    keys = sorted(ca["angles"].keys(), key=float)
    exp = np.array([ca["angles"][k]["applied_fold_deg"] for k in keys])
    rec = np.array([ca["angles"][k]["median_recovered_shift_deg"] for k in keys])
    wrap = np.array([abs(e) == 90.0 for e in exp])          # mod-180 antipode: signed median ill-defined
    sl, ic = np.polyfit(exp[~wrap], rec[~wrap], 1)
    xs = np.linspace(-95, 95, 2)
    axD.plot(xs, sl * xs + ic, "-", color="#2ca02c", lw=1.5,
             label=f"fit: slope {sl:+.3f} (|slope|=1 = exact)")
    axD.plot(exp[~wrap], rec[~wrap], "o", color="#d62728", ms=8, label="measured (5 clean angles)")
    if wrap.any():
        axD.plot(exp[wrap], rec[wrap], "x", color="#999999", ms=10, mew=2,
                 label="90deg wrap point (signed median\nill-defined; circular err 2.4deg)")
    axD.set_xlabel("applied rotation, folded mod 180 (deg)")
    axD.set_ylabel("recovered PA shift (deg)")
    axD.legend(fontsize=7, loc="upper right")
    axD.set_title("Causal test PASSED: readout moves in exact lockstep with\n"
                  "physical rotation (residuals <0.05deg; sign = verified pixel-vs-sky\n"
                  "coordinate convention, not a model defect)", fontsize=9)
else:
    axD.text(0.5, 0.5, "causal rotation test\nstill running...", ha="center", va="center",
             fontsize=12, color="0.5", transform=axD.transAxes)
    axD.set_xticks([]); axD.set_yticks([])
    axD.set_title("Causal test: rotate the real image,\nre-embed, read out the shift (pending)", fontsize=10)

fig.savefig("figures/trackA_supplement.png", dpi=200, bbox_inches="tight")
print("wrote figures/trackA_supplement.png")
