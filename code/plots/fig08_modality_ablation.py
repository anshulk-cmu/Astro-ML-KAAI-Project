"""08 — Decodability: image-only vs multimodal embedding.

Grouped bars of held-out R^2 for redshift, g-r, r-z comparing E_img and E_full.
Faithful note: E_full ingests g,r,z flux + redshift as INPUTS (leakage), so it is
higher by construction; E_img is the honest measure.
"""
import sys
sys.path.insert(0, "code/plots")
import numpy as np
import matplotlib.pyplot as plt
import plotcommon as pc

p = pc.load_json("probes")
m = p["modality"]

keys = ["redshift", "g_r", "r_z"]
PRETTY = {"redshift": "redshift\n(photo-z)", "g_r": "g - r colour", "r_z": "r - z colour"}

img = np.array([m[k]["E_img"] for k in keys])
full = np.array([m[k]["E_full"] for k in keys])

fig, ax = plt.subplots(figsize=(8.5, 6))
x = np.arange(len(keys))
w = 0.38
C_IMG = "#2c6fbb"
C_FULL = "#9aa0a6"

b1 = ax.bar(x - w / 2, img, w, color=C_IMG, label="$E_{img}$  (image-only, honest)")
b2 = ax.bar(x + w / 2, full, w, color=C_FULL,
            label="$E_{full}$  (multimodal: flux+redshift as inputs)")

for b in list(b1) + list(b2):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.008,
            f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=9.5,
            color="#222222")

ax.set_xticks(x)
ax.set_xticklabels([PRETTY[k] for k in keys])
ax.set_ylabel(r"held-out $R^2$  (RidgeCV, 80/20 split)")
ax.set_ylim(0, 1.06)
ax.set_title("Decodability: image-only vs multimodal embedding")
ax.legend(loc="lower center", fontsize=10)

note = ("$E_{full}$ ingests g, r, z flux and redshift as INPUTS, so its higher score on exactly "
        "those quantities is partly input leakage, not new recovery.\n"
        "$E_{img}$ (image pixels only) is the leakage-free measure of what the model encodes.")
fig.text(0.5, -0.04, note, ha="center", va="top", fontsize=9, color="#222222",
         bbox=dict(boxstyle="round,pad=0.4", fc="#f5f5f5", ec="#bbbbbb", alpha=0.95))

pc.save(fig, "08_modality_ablation.png")
