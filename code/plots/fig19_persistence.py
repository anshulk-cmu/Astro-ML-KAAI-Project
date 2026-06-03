"""Figure 19: H1 persistence diagrams under Euclidean / Fermat / diffusion, for the beta_1 = 0 check.

Reads results/persistence.json + results/persistenceDiag_<metric>.npy (worst-case subsample per
metric, i.e. the one with the longest-lived loop, the strongest case for a real hole).

Top row: the canonical persistence diagram (birth vs death). Bottom row: the same loops in
birth-vs-lifetime coordinates (lifetime = death - birth), which puts the noise band on the horizontal
and makes the gap between noise and any signal directly readable. Both rows show the fixed 0.1
threshold and the bootstrap 95% confidence band (2 c_n). Faithful: axes show every plotted loop; the
verdict is whatever the data gives.
"""
import json
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"d:/AstroML-Project"
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
METRICS = ["euclidean", "fermat", "diffusion"]
TITLE = {"euclidean": "Euclidean (control)", "fermat": "Fermat (primary metric)", "diffusion": "diffusion"}

with open(os.path.join(RES, "persistence.json")) as f:
    S = json.load(f)
dgs = {m: np.load(os.path.join(RES, f"persistenceDiag_{m}.npy")).astype(float) for m in METRICS}

Lmax = max(float(dgs[m][:, 1].max()) for m in METRICS) * 1.04          # canonical-row shared limit
bands = [S[m]["band_persistence"] for m in METRICS]
maxpers = [float((dgs[m][:, 1] - dgs[m][:, 0]).max()) for m in METRICS]
ytop = max(bands + maxpers + [0.1]) * 1.35                            # lifetime-row shared y-limit

fig, axes = plt.subplots(2, 3, figsize=(16.5, 10.4), constrained_layout=True)

for col, m in enumerate(METRICS):
    dg = dgs[m]
    b, d = dg[:, 0], dg[:, 1]
    pers = d - b
    s = S[m]
    band = s["band_persistence"]
    j = int(np.argmax(pers))
    above = int((pers > band).sum())
    verdict = "no loop clears the band" if above == 0 else f"{above} loop(s) clear the band"

    # ---- top row: canonical birth vs death ----
    ax = axes[0, col]
    xs = np.array([0.0, Lmax])
    ax.fill_between(xs, xs, xs + band, color="#cfe3f5", alpha=0.85, zorder=0,
                    label=f"bootstrap 95% band")
    ax.plot(xs, xs + s["threshold"], color="#c44", lw=1.2, ls="--", zorder=2, label="0.1 threshold")
    ax.plot(xs, xs, color="#444", lw=1.0, zorder=2)
    ax.scatter(b, d, s=9, c="#3a6ea5", alpha=0.4, linewidths=0, rasterized=True, zorder=3,
               label=f"H1 loops (n={len(dg)})")
    ax.scatter([b[j]], [d[j]], s=120, facecolors="none", edgecolors="#d62728", linewidths=1.8, zorder=5)
    ax.set_title(f"{TITLE[m]}\nbirth vs death (canonical)", fontsize=11)
    ax.set_xlabel("birth (diameter-normalised radius)")
    ax.set_ylabel("death (diameter-normalised radius)")
    ax.set_xlim(0, Lmax); ax.set_ylim(0, Lmax); ax.set_aspect("equal"); ax.grid(alpha=0.2)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)

    # ---- bottom row: birth vs lifetime (the readable gap view) ----
    ax = axes[1, col]
    ax.axhspan(0, band, color="#cfe3f5", alpha=0.9, zorder=0,
               label=f"bootstrap 95% band (pers $\\leq$ {band:.3f})")
    ax.axhline(s["threshold"], color="#c44", lw=1.2, ls="--", zorder=2, label="0.1 threshold")
    ax.scatter(b, pers, s=10, c="#3a6ea5", alpha=0.45, linewidths=0, rasterized=True, zorder=3,
               label=f"H1 loops (n={len(dg)})")
    ax.scatter([b[j]], [pers[j]], s=130, facecolors="none", edgecolors="#d62728", linewidths=1.9, zorder=5)
    ax.annotate(f"longest loop = {pers[j]:.3f}", (b[j], pers[j]), textcoords="offset points",
                xytext=(8, 6), fontsize=9, color="#d62728")
    ax.set_title(f"max persistence {s['max_persistence']:.3f}  |  loops > 0.1 in "
                 f"{s['sig_loops_mean']:.1f}/10 subsamples", fontsize=10.5)
    ax.set_xlabel("birth (diameter-normalised radius)")
    ax.set_ylabel("loop lifetime  (death $-$ birth)")
    ax.set_xlim(0, b.max() * 1.05); ax.set_ylim(0, ytop); ax.grid(alpha=0.2)
    txt = (f"$c_n$ = {s['bootstrap_cn']:.3f}  (95% bootstrap, B=20)\n"
           f"band = 2$c_n$ = {band:.3f}\n{verdict}")
    ax.text(0.97, 0.97, txt, transform=ax.transAxes, va="top", ha="right", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="#999", alpha=0.92))
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9)

fig.suptitle("H1 persistence diagrams of the AION-1 galaxy embedding (worst-case 2,000-point subsample "
             "per metric): every loop sits inside its noise band, so $\\beta_1 = 0$", fontsize=13)
note = ("Each dot is one loop (an H1 feature) from the most-persistent of ten 2,000-galaxy subsamples, "
        "the single strongest case for a real hole. Top row is the standard persistence diagram: a loop is "
        "born at radius x and dies (fills in) at radius y, so the distance above the 45-degree diagonal is "
        "its lifetime. Bottom row plots that lifetime directly on the vertical axis, which makes the noise "
        "band horizontal and the gap easy to read. Points near zero lifetime are sampling noise. The blue "
        "band is the bootstrap 95% confidence band (a loop must rise above it to count as a real feature); "
        "the red dashed line is the fixed 0.1 threshold used in Section 15; the red ring marks the longest "
        "loop. Under all three metrics, including Fermat (our primary), the longest loop stays inside the "
        "band, with empty space above it, so there is no robust loop: beta_1 = 0.")
fig.text(0.5, -0.015, note, ha="center", va="top", fontsize=8.7, wrap=True)

path = os.path.join(FIG, "19_persistence_diagrams.png")
fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("saved", path, f"({os.path.getsize(path)/1e3:.0f} kB)")
