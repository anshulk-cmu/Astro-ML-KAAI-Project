"""Fig 17 - Linear PCA cumulative explained variance (the linear null for ID).

Recompute PCA on z-scored embeddings and show cumulative explained-variance vs #PCs
for E_img and E_full. Linear PCA needs ~40-45 dims for 95% variance, whereas the
nonlinear ID is ~10-12 -> the embedding geometry is nonlinear. Baseline, not headline.
"""
import sys
sys.path.insert(0, "code/plots")
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import plotcommon as pc

NPC = 200
embeds = {
    "E_img": dict(path="data/E_img.npy", color="#d62728", label="E_img (image only)"),
    "E_full": dict(path="data/E_full.npy", color="#1f77b4", label="E_full (image + spectra)"),
}

thresholds = [0.50, 0.90, 0.95, 0.99]

results = {}
for key, meta in embeds.items():
    X = np.load(meta["path"]).astype(np.float64)
    Z = (X - X.mean(0)) / (X.std(0) + 1e-8)
    pca = PCA(n_components=NPC).fit(Z)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    # #PCs needed for each threshold (1-indexed)
    needed = {t: int(np.searchsorted(cumvar, t) + 1) for t in thresholds}
    results[key] = dict(cumvar=cumvar, needed=needed, meta=meta)

xs = np.arange(1, NPC + 1)

fig, ax = plt.subplots(figsize=(11, 6.4))

for key in ["E_img", "E_full"]:
    r = results[key]
    ax.plot(xs, r["cumvar"] * 100, color=r["meta"]["color"], lw=2.0,
            label=r["meta"]["label"], zorder=3)

# threshold guide lines
for t in thresholds:
    ax.axhline(t * 100, color="gray", ls=":", lw=1.0, alpha=0.7, zorder=1)
    ax.text(NPC * 0.995, t * 100 + 0.6, f"{int(t*100)}%", ha="right", va="bottom",
            fontsize=9, color="gray")

# annotate #PCs needed for 95% and 99% per embedding (the load-bearing comparison)
ann_y = {"E_img": 0.30, "E_full": 0.18}
for key in ["E_img", "E_full"]:
    r = results[key]
    col = r["meta"]["color"]
    for t in [0.95, 0.99]:
        npc = r["needed"][t]
        ax.scatter([npc], [t * 100], color=col, s=40, zorder=4, edgecolor="white", lw=0.6)
    txt = (f"{r['meta']['label'].split(' ')[0]}:  "
           f"95% at {r['needed'][0.95]} PCs,  99% at {r['needed'][0.99]} PCs")
    ax.text(0.97, ann_y[key], txt, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9.5, color=col)

ax.set_xlabel("number of principal components")
ax.set_ylabel("cumulative explained variance  [%]  (z-scored embeddings)")
ax.set_title("Linear PCA cumulative explained variance  -  the linear null for intrinsic dimension")
ax.set_xlim(1, NPC)
ax.set_ylim(0, 101)
ax.legend(loc="lower right", fontsize=10)

# faithful footer: contrast with nonlinear ID, note z-scored vs raw sanity values
san = pc.load_json("sanity")
n95_lo = min(results['E_img']['needed'][0.95], results['E_full']['needed'][0.95])
n95_hi = max(results['E_img']['needed'][0.95], results['E_full']['needed'][0.95])
foot = (
    "Faithful note (this is a BASELINE, not the headline): on z-scored embeddings linear PCA needs "
    f"~{n95_lo}-{n95_hi} dims for 95% variance (computed here), "
    "whereas the nonlinear intrinsic dimension is only ~10-12 - i.e. the geometry is nonlinear.\n"
    f"z-scoring spreads variance across more dims than the raw embeddings: sanity.json stores raw 95% at "
    f"E_full {san['E_full']['pcs_for_variance']['var95']} / E_img {san['E_img']['pcs_for_variance']['var95']} PCs. "
    "The values plotted here are recomputed on z-scored data, as specified."
)
fig.text(0.5, -0.05, foot, ha="center", va="top", fontsize=8.5, color="#333")

pc.save(fig, "17_pca_variance.png")
print("z-scored 95% PCs:", {k: results[k]["needed"][0.95] for k in results})
print("z-scored 99% PCs:", {k: results[k]["needed"][0.99] for k in results})
print("z-scored 50/90 PCs:", {k: (results[k]["needed"][0.50], results[k]["needed"][0.90]) for k in results})
