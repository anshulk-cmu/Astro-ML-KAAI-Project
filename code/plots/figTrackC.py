"""
Track C figure (audit-grade): manifold translation with CI, decodability with CIs,
the age-metallicity degeneracy angle with robustness variants, and the
selection-vs-evolution tests. Reads results/trackC.json + results/trackC_steer.npy.
Writes figures/trackC.png.

Run from the project root:  python code/plots/figTrackC.py
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

J = json.load(open("results/trackC.json"))
S = np.load("results/trackC_steer.npy")
dz, nz_emb, nz_rand = S[:, 0], S[:, 1], S[:, 2]
i0 = int(np.where(dz == 0.0)[0][0])

fig = plt.figure(figsize=(14, 9))
gs = GridSpec(2, 2, wspace=0.34, hspace=0.46)

# A) manifold translation (the genuine causal test), slope with CI
axA = fig.add_subplot(gs[0, 0])
t = J["translation"]
axA.plot(dz, dz, ":", color="#2ca02c", lw=1.3, label="perfectly faithful (slope 1)")
axA.plot(dz, nz_emb - nz_emb[i0], "o-", color="#1f77b4", lw=2,
         label=f"redshift shift: slope {t['neighbor_z_slope']:.2f} "
               f"CI[{t['slope_ci'][0]:.2f},{t['slope_ci'][1]:.2f}]")
axA.plot(dz, nz_rand - nz_rand[i0], "s--", color="#999999", lw=1.5,
         label=f"random dir (slope {t['rand_slope']:.2f})")
axA.axhline(0, color="0.85", lw=.6); axA.axvline(0, color="0.85", lw=.6)
axA.set_xlabel("intended $\\Delta z$"); axA.set_ylabel("neighbours' actual $\\Delta z$")
axA.set_title("Moving along the redshift direction lands you among\n"
              "real higher-z galaxies (partial but genuine translation)", fontsize=10)
axA.legend(fontsize=7.5, loc="upper left")

# B) decodability with CIs
axB = fig.add_subplot(gs[0, 1])
keys = ["redshift", "g_r", "r_z", "featured", "mass", "sSFR", "metal", "age"]
r2 = [J["decode"][k]["r2"] for k in keys]
cis = [J["decode"][k]["ci"] for k in keys]
cols = ["#1f77b4"] * 4 + ["#999999"] * 2 + ["#8e44ad"] * 2
axB.bar(range(len(keys)), r2, color=cols)
for i, (v, c) in enumerate(zip(r2, cis)):
    axB.errorbar(i, v, yerr=[[max(0, v - c[0])], [max(0, c[1] - v)]], color="black", capsize=3, lw=1.2)
    axB.text(i, max(c[1], v) + 0.02, f"{v:.2f}", ha="center", fontsize=8)
axB.set_xticks(range(len(keys))); axB.set_xticklabels(keys, rotation=25, fontsize=8.5)
axB.set_ylabel("decodability $R^2$ (E_img, 95% CI)"); axB.set_ylim(0, 1.08)
axB.set_title("Images barely encode stellar age and only weakly\nmetallicity (all other properties strongly)", fontsize=10)

# C) the degeneracy angle: headline + robustness variants, with CIs, vs each null
axC = fig.add_subplot(gs[1, 0])
am = J["age_metallicity"]
variants = [("headline\n(lightW, log)", am["headline"]), ("mass-weighted", am["massweighted"]),
            ("z $\\geq$ 0.15", am["z_ge_0p15"])]
for i, (nm, v) in enumerate(variants):
    e, ec = v["embed_angle"], v["embed_ci"]
    axC.errorbar(i, e, yerr=[[e - ec[0]], [ec[1] - e]], fmt="o", color="#8e44ad",
                 capsize=4, ms=8, lw=1.5, label="embedding angle (95% CI)" if i == 0 else None)
    axC.plot([i - 0.18, i + 0.18], [v["label_null"]] * 2, color="#444444", lw=2,
             label="true (label) angle" if i == 0 else None)
axC.axhline(90, ls=":", color="0.6", lw=1); axC.text(2.28, 90.5, "orthogonal", color="0.5", fontsize=7.5)
axC.set_xticks(range(len(variants))); axC.set_xticklabels([v[0] for v in variants], fontsize=9)
axC.set_ylabel("age-metallicity angle (deg)")
h = am["headline"]
axC.set_title(f"No consistent extra conflation: the embedding's age-metal angle\n"
              f"matches the true angle (excess {h['excess']:+.1f}$\\degree$ "
              f"CI[{h['excess_ci'][0]:.1f},{h['excess_ci'][1]:.1f}]; variants scatter in sign).\n"
              f"The degeneracy shows as near-zero decodability instead (panel B)", fontsize=9)
axC.legend(fontsize=8, loc="lower right")

# D) selection vs evolution
axD = fig.add_subplot(gs[1, 1])
sel = J["selection"]
tz = sel["z_decode_by_magr_tertile"]
names = ["bright", "mid", "faint"]
vals = [tz[n]["r2"] for n in names]
cis2 = [tz[n]["ci"] for n in names]
x = np.arange(len(names))
axD.bar(x, vals, 0.55, color="#16a085")
for i, (v, c) in enumerate(zip(vals, cis2)):
    axD.errorbar(i, v, yerr=[[max(0, v - c[0])], [max(0, c[1] - v)]], color="black", capsize=3, lw=1.2)
po = sel["z_partial_out"]
axD.bar([3.2], [po["r2_residual_from_Eimg"]], 0.55, color="#e67e22")
c = po["residual_ci"]; v = po["r2_residual_from_Eimg"]
axD.errorbar(3.2, v, yerr=[[max(0, v - c[0])], [max(0, c[1] - v)]], color="black", capsize=3, lw=1.2)
axD.axhline(po["r2_full"], ls="--", color="0.5", lw=1)
axD.text(-0.35, po["r2_full"] + 0.015, f"full-sample z R2 = {po['r2_full']:.2f}", color="0.4", fontsize=8)
axD.set_xticks(list(x) + [3.2])
axD.set_xticklabels(names + ["residual z\n(mag+size+colours\nremoved)"], fontsize=8.5)
axD.set_ylabel("z decodability $R^2$")
axD.set_title("Selection vs evolution: z decoding survives within\n"
              "brightness-matched bins; the residual quantifies what is\n"
              "left beyond the obvious imaging channels", fontsize=9.5)

fig.savefig("figures/trackC.png", dpi=200, bbox_inches="tight")
print("wrote figures/trackC.png")
