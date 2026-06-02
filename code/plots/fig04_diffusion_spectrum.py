"""04 — Diffusion-operator eigenvalue spectrum and gaps.

Source: results/diffusionMap.json  {E_full,E_img}:{eigenvalues:[30], gaps:[29]}.
eigenvalues[0]=1 is the trivial constant mode (greyed).
Panel A: lambda_k vs k (k=1..29) — smooth gradual decay, no large gap.
Panel B: consecutive gaps lambda_k - lambda_{k+1} vs k — all small/comparable, no dominant spectral gap.
"""
import sys
sys.path.insert(0, "code/plots")
import numpy as np
import matplotlib.pyplot as plt
import plotcommon as pc

d = pc.load_json("diffusionMap")
embs = ["E_full", "E_img"]
emb_nice = {"E_full": "E_full (img+spectra)", "E_img": "E_img (image-only)"}
style = {"E_full": ("#1f6fb4", "o", "-"), "E_img": ("#d2691e", "s", "--")}

fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 5.6))

# ---- Panel A: eigenvalue spectrum ----
for e in embs:
    eig = np.array(d[e]["eigenvalues"])
    k = np.arange(len(eig))  # 0..29
    col, mk, ls = style[e]
    # trivial k=0 greyed; non-trivial k>=1 highlighted
    axA.plot(k[1:], eig[1:], ls=ls, marker=mk, ms=5, color=col, lw=1.6,
             label=emb_nice[e], zorder=3)
    axA.scatter(k[0], eig[0], s=55, facecolor="none", edgecolor="grey",
                linewidths=1.4, zorder=4)
axA.text(0.4, 1.001, "k=0 trivial\n(constant) mode", fontsize=8.5, color="grey",
         ha="left", va="center")
axA.set_xlabel("eigenvalue index  k")
axA.set_ylabel(r"diffusion eigenvalue  $\lambda_k$")
axA.set_title("A. Eigenvalue spectrum — gradual decay, no large gap")
axA.set_xlim(-0.8, 29.8)
axA.legend(loc="upper right", fontsize=9.5)
axA.grid(alpha=0.25)

# ---- Panel B: consecutive gaps ----
maxgap = 0.0
for e in embs:
    gaps = np.array(d[e]["gaps"])  # gaps[k] = lambda_k - lambda_{k+1}, k=0..28
    k = np.arange(len(gaps))
    col, mk, ls = style[e]
    axB.plot(k, gaps, ls=ls, marker=mk, ms=5, color=col, lw=1.6,
             label=emb_nice[e], zorder=3)
    maxgap = max(maxgap, gaps.max())
axB.set_xlabel(r"gap index  k   (between $\lambda_k$ and $\lambda_{k+1}$)")
axB.set_ylabel(r"consecutive gap  $\lambda_k - \lambda_{k+1}$")
axB.set_title("B. Consecutive gaps — all small / comparable")
axB.set_xlim(-0.8, 28.8)
axB.set_ylim(0, maxgap * 1.25)
axB.legend(loc="upper right", fontsize=9.5)
axB.grid(alpha=0.25)

fig.suptitle("Diffusion-operator eigenvalue spectrum and gaps  (k=15 graph)",
             fontsize=14, y=1.03)

note = ("Note: no single dominant spectral gap appears — the spectrum decays smoothly. "
        "This is consistent with one connected continuum rather than discrete clusters\n"
        "(the same conclusion the topology arm reaches).")
fig.text(0.5, -0.04, note, ha="center", va="top", fontsize=9.5, color="#333333")

print(pc.save(fig, "04_diffusion_spectrum.png"))
