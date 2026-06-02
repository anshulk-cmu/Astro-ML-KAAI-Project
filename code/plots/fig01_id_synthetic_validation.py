"""Fig 01 - Intrinsic-dimension estimators recover KNOWN d on synthetic manifolds.

Certifies the four ID estimators (TwoNN-MLE, Gride@large-scale, local-MLE plateau,
PCA participation-ratio) before we trust them on AION. Three known-truth sets are
shown; 'twoblobs' (no intrinsic d) is noted as a tight-cluster control only.
"""
import sys
sys.path.insert(0, "code/plots")
import numpy as np
import matplotlib.pyplot as plt
import plotcommon as pc

d = pc.load_json("intrinsicDim")["synthetic"]

# known-truth sets, in display order
sets = [
    ("sphere_id5", "Hypersphere\n(truth d = 5)"),
    ("swissroll_id2", "Swiss roll\n(truth d = 2)"),
    ("linear_id5", "Linear 5-plane\n(truth d = 5)"),
]

# four estimators: (label, extractor, colour)
ests = [
    ("TwoNN-MLE", lambda s: s["twonn"]["mle"], "#1f77b4"),
    ("Gride @ n1=256", lambda s: s["gride"]["256"], "#ff7f0e"),
    ("local-MLE plateau (K=226)", lambda s: s["local_mle"]["226"], "#2ca02c"),
    ("PCA participation-ratio", lambda s: s["pca_pr"], "#9467bd"),
]

fig, ax = plt.subplots(figsize=(11, 6))

n_sets = len(sets)
n_est = len(ests)
group_w = 0.8
bar_w = group_w / n_est
x_centers = np.arange(n_sets)

for j, (elab, efn, ecol) in enumerate(ests):
    vals = []
    yerr_lo, yerr_hi = [], []
    for (skey, _) in sets:
        s = d[skey]
        v = efn(s)
        vals.append(v)
        # TwoNN gets its posterior CI as an asymmetric error bar; others have none
        if elab == "TwoNN-MLE":
            lo, hi = s["twonn"]["ci_post"]
            yerr_lo.append(v - lo)
            yerr_hi.append(hi - v)
        else:
            yerr_lo.append(0.0)
            yerr_hi.append(0.0)
    offs = (j - (n_est - 1) / 2) * bar_w
    err = np.array([yerr_lo, yerr_hi])
    has_err = err.sum() > 0
    ax.bar(x_centers + offs, vals, width=bar_w * 0.92, color=ecol, label=elab,
           yerr=err if has_err else None,
           error_kw=dict(ecolor="black", capsize=3, lw=1.1) if has_err else None,
           zorder=3, edgecolor="white", linewidth=0.4)

# truth reference: a horizontal segment spanning each group
truth_handle = None
for i, (skey, _) in enumerate(sets):
    t = d[skey]["truth"]
    h = ax.hlines(t, x_centers[i] - group_w / 2, x_centers[i] + group_w / 2,
                  color="black", ls="--", lw=1.8, zorder=4)
    truth_handle = h
    ax.text(x_centers[i] + group_w / 2 + 0.02, t, f"truth = {t}",
            va="center", ha="left", fontsize=9, color="black")

ax.set_xticks(x_centers)
ax.set_xticklabels([lab for (_, lab) in sets])
ax.set_ylabel("estimated intrinsic dimension")
ax.set_title("Intrinsic-dimension estimators recover the known dimension of synthetic manifolds")
ax.set_ylim(0, 7.2)
ax.set_xlim(-0.6, n_sets - 1 + 0.95)

# legend: estimators + truth, placed top-left, no overlap with bars (which sit low)
handles, labels = ax.get_legend_handles_labels()
handles.append(truth_handle)
labels.append("known truth")
ax.legend(handles, labels, loc="upper left", fontsize=9.5, ncol=1)

# honest caveat: sphere PCA-PR sits above truth; note the twoblobs control
caveat = ("Faithful note: hypersphere PCA participation-ratio "
          f"({d['sphere_id5']['pca_pr']:.1f}) sits slightly above truth = 5 "
          "(PR is a linear-spread measure, not a manifold ID).\n"
          "'twoblobs' is excluded (no intrinsic d): it is a tight-cluster control where TwoNN reads "
          f"~{d['twoblobs']['twonn']['mle']:.0f} (ambient-noise dominated).")
ax.text(0.5, -0.16, caveat, transform=ax.transAxes, ha="center", va="top",
        fontsize=8.5, color="#333333")

pc.save(fig, "01_id_synthetic_validation.png")
