"""07 — Held-out R^2 (with 95% CI) for the image-only embedding E_img.

Horizontal bars sorted by R^2. Two colours separate full-N labels (n~48k) from
small subset labels (n<5k: spiral/bar branch + mass/sSFR/sersic cross-match).
Image-only embedding => leakage-free. Neutral, descriptive.
"""
import sys
sys.path.insert(0, "code/plots")
import numpy as np
import matplotlib.pyplot as plt
import plotcommon as pc

p = pc.load_json("probes")

# pretty names for the labels we show
PRETTY = {
    "redshift": "redshift  (photo-z)", "g_r": "g - r colour", "r_z": "r - z colour",
    "smooth": "smooth fraction", "featured": "featured fraction", "merger": "merger fraction",
    "spiral": "spiral fraction", "bar": "strong-bar fraction",
    "mass": r"stellar mass  $\log M_\star$", "sSFR": r"sSFR", "sersic": r"Sersic index $n$",
}

rows = []  # (key, r2, lo, hi, n)
for k, d in p["E_img"].items():
    rows.append((k, d["r2"], d["ci"][0], d["ci"][1], d["n"]))
for k, d in p["sparse_E_img"].items():
    rows.append((k, d["r2"], d["ci"][0], d["ci"][1], d["n"]))

rows.sort(key=lambda r: r[1])  # ascending so largest is at top after barh

names = [PRETTY[r[0]] for r in rows]
r2 = np.array([r[1] for r in rows])
lo = np.array([r[2] for r in rows])
hi = np.array([r[3] for r in rows])
ns = np.array([r[4] for r in rows])
# asymmetric error from CI, clipped to be non-negative
xerr = np.vstack([np.maximum(r2 - lo, 0), np.maximum(hi - r2, 0)])

is_full = ns > 20000
C_FULL = "#2c6fbb"
C_SUB = "#d1762b"
colors = [C_FULL if f else C_SUB for f in is_full]

fig, ax = plt.subplots(figsize=(11, 6.5))
y = np.arange(len(rows))
ax.barh(y, r2, xerr=xerr, color=colors, height=0.66, alpha=0.92,
        error_kw=dict(ecolor="#333333", elinewidth=1.2, capsize=3))
ax.axvline(0.0, color="black", lw=1.0, zorder=0)

# annotate n beside each bar (just past the CI upper end)
for yi, (r, h, n) in enumerate(zip(r2, hi, ns)):
    ax.text(max(h, r) + 0.012, yi, f"n={n:,}", va="center", ha="left",
            fontsize=9, color="#444444")

ax.set_yticks(y)
ax.set_yticklabels(names)
ax.set_xlabel(r"held-out $R^2$  (RidgeCV, 80/20 split)")
ax.set_xlim(-0.05, 1.13)
ax.set_title("Decodability of physical labels from the image-only embedding $E_{img}$")

# legend by colour
from matplotlib.patches import Patch
handles = [Patch(facecolor=C_FULL, label="full sample  (n $\\approx$ 48k)"),
           Patch(facecolor=C_SUB, label="subset  (n < 5k)")]
ax.legend(handles=handles, loc="lower right", fontsize=10)

note = ("Image-only embedding: no flux/redshift inputs, so this is leakage-free.  "
        "g-r, r-z and (photo-z) redshift are colour-related;\n"
        "mass & sSFR are recovered from the image ALONE — the cleaner evidence that "
        "physical structure is encoded.")
fig.text(0.5, -0.04, note, ha="center", va="top", fontsize=9, color="#222222",
         bbox=dict(boxstyle="round,pad=0.4", fc="#f5f5f5", ec="#bbbbbb", alpha=0.95))

pc.save(fig, "07_probe_decodability.png")
