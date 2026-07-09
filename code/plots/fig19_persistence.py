"""Figure 19: H1 persistence resolution under paired metrics/subsamples.

Reads results/persistence.json + results/persistenceDiag_<metric>.npy (worst-case subsample per
metric, i.e. the one with the longest-lived loop, the strongest case for a real hole).

Top row: worst-replicate birth vs death diagrams. Bottom row: maximum persistence
for each of the ten paired subsamples. The 0.1 line is descriptive, not a
significance threshold. No formal beta1 verdict is claimed.
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
maxpers = [float((dgs[m][:, 1] - dgs[m][:, 0]).max()) for m in METRICS]
ytop = max(maxpers + [max(S[m]["per_rep_max"]) for m in METRICS] + [0.1]) * 1.25

fig, axes = plt.subplots(2, 3, figsize=(16.5, 10.4), constrained_layout=True)

for col, m in enumerate(METRICS):
    dg = dgs[m]
    b, d = dg[:, 0], dg[:, 1]
    pers = d - b
    s = S[m]
    j = int(np.argmax(pers))

    # ---- top row: canonical birth vs death ----
    ax = axes[0, col]
    xs = np.array([0.0, Lmax])
    ax.plot(xs, xs + s["threshold"], color="#c44", lw=1.2, ls="--", zorder=2,
            label="descriptive persistence = 0.1")
    ax.plot(xs, xs, color="#444", lw=1.0, zorder=2)
    ax.scatter(b, d, s=9, c="#3a6ea5", alpha=0.4, linewidths=0, rasterized=True, zorder=3,
               label=f"H1 loops (n={len(dg)})")
    ax.scatter([b[j]], [d[j]], s=120, facecolors="none", edgecolors="#d62728", linewidths=1.8, zorder=5)
    ax.set_title(f"{TITLE[m]}\nbirth vs death (canonical)", fontsize=11)
    ax.set_xlabel("birth (diameter-normalised radius)")
    ax.set_ylabel("death (diameter-normalised radius)")
    ax.set_xlim(0, Lmax); ax.set_ylim(0, Lmax); ax.set_aspect("equal"); ax.grid(alpha=0.2)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)

    # ---- bottom row: paired-replicate maxima ----
    ax = axes[1, col]
    reps = np.arange(1, len(s["per_rep_max"]) + 1)
    vals = np.asarray(s["per_rep_max"])
    ax.plot(reps, vals, "o-", color="#3a6ea5", lw=1.5)
    ax.axhline(s["threshold"], color="#c44", lw=1.2, ls="--", zorder=2,
               label="descriptive 0.1 line")
    ax.set_title(f"paired subsamples above 0.1: {s['subsamples_with_any_over_0p1']}/10; "
                 f"max={s['max_persistence']:.3f}", fontsize=10.5)
    ax.set_xlabel("paired 2,000-galaxy subsample")
    ax.set_ylabel("maximum H1 persistence")
    ax.set_xticks(reps); ax.set_ylim(0, ytop); ax.grid(alpha=0.2)
    txt = ("No formal confidence band\n"
           "0.1 is descriptive only")
    ax.text(0.97, 0.97, txt, transform=ax.transAxes, va="top", ha="right", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="#999", alpha=0.92))
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9)

fig.suptitle("H1 persistence of the AION-1 embedding: paired subsamples reveal "
             "metric-dependent unresolved loop candidates", fontsize=13)
note = ("Top: the most-persistent diagram for each metric. Bottom: maximum lifetime in each of the same "
        "ten paired 2,000-galaxy subsamples. The red 0.1 line is retained only to show the old reference. "
        "The previous 20-draw truncated bootstrap was not a calibrated confidence set and has been removed. "
        "These plots support neither beta1=0 nor a confirmed loop without a valid resolution/null calibration.")
fig.text(0.5, -0.015, note, ha="center", va="top", fontsize=8.7, wrap=True)

path = os.path.join(FIG, "19_persistence_diagrams.png")
fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("saved", path, f"({os.path.getsize(path)/1e3:.0f} kB)")
