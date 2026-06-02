"""09 — Disentanglement: angle between learned probe directions vs the angle the
label correlation alone would force.

Dumbbell per label pair: null angle (arccos of label correlation) marker and the
learned-probe angle theta marker, connected. excess = theta - null. excess>0 means
the learned directions are MORE orthogonal than the labels' correlation requires.
Pairs with excess ~0 / slightly negative (redshift-merger, smooth-merger) shown honestly.
"""
import sys
sys.path.insert(0, "code/plots")
import numpy as np
import matplotlib.pyplot as plt
import plotcommon as pc

p = pc.load_json("probes")
dis = p["disentangle"]


def pretty_pair(k):
    a, b = k.split("-")
    nm = {"redshift": "redshift", "g_r": "g-r", "r_z": "r-z",
          "smooth": "smooth", "merger": "merger"}
    return f"{nm[a]} – {nm[b]}"


rows = [(k, v["null"], v["theta"], v["excess"]) for k, v in dis.items()]
rows.sort(key=lambda r: r[3])  # ascending excess -> largest at top after plotting

labels = [pretty_pair(r[0]) for r in rows]
null = np.array([r[1] for r in rows])
theta = np.array([r[2] for r in rows])
excess = np.array([r[3] for r in rows])

fig, ax = plt.subplots(figsize=(11, 6.5))
y = np.arange(len(rows))

C_NULL = "#9aa0a6"
C_THETA = "#2c6fbb"
C_NEG = "#c0392b"

# connecting lines: blue where excess>0 (more orthogonal), red where <=0
for yi, (n, t, e) in enumerate(zip(null, theta, excess)):
    lc = C_THETA if e > 0 else C_NEG
    ax.plot([n, t], [yi, yi], color=lc, lw=2.2, zorder=1, alpha=0.85)

ax.scatter(null, y, s=85, color=C_NULL, zorder=3, edgecolor="white", linewidth=0.8,
           label="null angle  =  arccos(label correlation)")
ax.scatter(theta, y, s=85, color=C_THETA, zorder=3, edgecolor="white", linewidth=0.8,
           label="learned probe-direction angle  $\\theta$")

ax.axvline(90, color="black", lw=0.9, ls="--", alpha=0.6, zorder=0)
ax.text(90, len(rows) - 0.35, "90$^\\circ$ (orthogonal)", rotation=90, va="top",
        ha="right", fontsize=8.5, color="#555555")

# annotate excess (deg) beyond the right-most marker of each pair
xmax = max(null.max(), theta.max())
for yi, (n, t, e) in enumerate(zip(null, theta, excess)):
    col = C_THETA if e > 0 else C_NEG
    ax.text(max(n, t) + 1.5, yi, f"{e:+.1f}$^\\circ$", va="center", ha="left",
            fontsize=9, color=col, fontweight="bold")

ax.set_yticks(y)
ax.set_yticklabels(labels)
ax.set_xlabel("angle between directions  [degrees]")
ax.set_xlim(20, xmax + 11)
ax.set_title("Disentanglement: learned probe-direction angle vs label-correlation null")
ax.legend(loc="lower left", fontsize=9.5)

note = ("excess (deg) = $\\theta$ - null.  excess > 0 (blue): learned directions are MORE orthogonal "
        "than the labels' correlation forces (disentangled).\n"
        "excess $\\leq$ 0 (red): redshift-merger, smooth-merger show no extra separation beyond the "
        "label null.")
fig.text(0.5, -0.04, note, ha="center", va="top", fontsize=9, color="#222222",
         bbox=dict(boxstyle="round,pad=0.4", fc="#f5f5f5", ec="#bbbbbb", alpha=0.95))

pc.save(fig, "09_disentanglement_angles.png")
