"""Shared loaders, faithful label handling, colormaps, and savers for the figure set.

Principles (match the project): faithful to data, no overclaim. Sentinel values are
filtered, nulls/baselines are kept, every estimate keeps its uncertainty where we have it.
Embeddings/coords/labels are aligned via ok_index exactly as code/analysis/common.load does.
"""
import json
import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.axisbelow": True,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "figure.constrained_layout.use": True,
})

# SAE label order (from code/analysis/sae.py LABELS)
SAE_LABELS = ["redshift", "g-r", "r-z", "smooth", "featured", "merger"]


def load_json(name):
    with open(os.path.join(RES, f"{name}.json")) as f:
        return json.load(f)


def load_df():
    """48398 rows aligned to embeddings/diffcoords (== common.load order)."""
    ok = np.load(os.path.join(DATA, "ok_index.npy"))
    df = pd.read_parquet(os.path.join(DATA, "sample.parquet")).iloc[ok].reset_index(drop=True)
    return df, ok


def diffcoords(which="E_full"):
    return np.load(os.path.join(RES, f"diffcoords_{which}.npy")).astype(np.float64)


def _finite(v, lo, hi):
    v = np.asarray(v, dtype=np.float64)
    v = np.where(np.isfinite(v) & (v >= lo) & (v <= hi), v, np.nan)
    return v


def labels(df):
    """Faithful physical labels. Sentinels (-inf mags, -99 sSFR, etc.) -> NaN.
    Returns dict: name -> (values[N] with NaN where invalid, units, full-N? bool)."""
    g = _finite(df["mag_g_desi"].values, 5, 30)
    r = _finite(df["mag_r_desi"].values, 5, 30)
    z = _finite(df["mag_z_desi"].values, 5, 30)
    L = {}
    L["g-r"] = (g - r, "g - r  [mag]", True)
    L["r-z"] = (r - z, "r - z  [mag]", True)
    L["redshift"] = (_finite(df["redshift"].values, 0, 1), "redshift  (mostly photo-z)", True)
    L["smooth"] = (df["smooth-or-featured_smooth_fraction"].values.astype(float),
                   "smooth vote fraction", True)
    L["featured"] = (df["smooth-or-featured_featured-or-disk_fraction"].values.astype(float),
                     "featured/disk vote fraction", True)
    L["merger"] = (df["merging_merger_fraction"].values.astype(float),
                   "merger vote fraction", True)
    L["mass"] = (_finite(df["elpetro_mass_log"].values, 6, 12.5),
                 r"$\log_{10} M_\star/M_\odot$", False)
    L["sSFR"] = (_finite(df["total_ssfr_median"].values, -13.5, -7.5),
                 r"$\log_{10}$ sSFR  [yr$^{-1}$]", False)
    L["sersic"] = (_finite(df["sersic_n"].values, 0.3, 6.2), r"Sersic index $n$", False)
    L["spiral"] = (_finite(df["has-spiral-arms_yes_fraction"].values, 0, 1),
                   "spiral vote fraction", False)
    L["bar"] = (_finite(df["bar_strong_fraction"].values, 0, 1), "strong-bar vote fraction", False)
    return L


# colormap per property (physically intuitive; high g-r -> red etc.)
CMAP = {
    "g-r": "coolwarm", "r-z": "coolwarm", "redshift": "viridis",
    "smooth": "cividis", "featured": "cividis_r", "merger": "magma",
    "mass": "plasma", "sSFR": "viridis_r", "sersic": "cividis",
    "spiral": "cividis_r", "bar": "magma",
}


def clip_range(v, lo=2, hi=98):
    v = v[np.isfinite(v)]
    return float(np.percentile(v, lo)), float(np.percentile(v, hi))


def scatter_phys(ax, x, y, c, cmap, vlo, vhi, label, s=4, alpha=0.45):
    """Faithful coloured scatter with a shuffled draw order (no z-order bias)."""
    m = np.isfinite(c) & np.isfinite(x) & np.isfinite(y)
    rng = np.random.default_rng(0)
    order = rng.permutation(int(m.sum()))
    xi, yi, ci = x[m][order], y[m][order], c[m][order]
    sc = ax.scatter(xi, yi, c=ci, cmap=cmap, s=s, alpha=alpha, vmin=vlo, vmax=vhi,
                    linewidths=0, rasterized=True)
    cb = ax.figure.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(label, fontsize=10)
    cb.solids.set_alpha(1.0)
    return sc


def galaxy_rgb(img4, q=8.0, stretch=0.9):
    """g,r,i,z (4,96,96) -> RGB via a Lupton-style arcsinh stretch, per-image scaled.
    R<-z, G<-r, B<-g. Faithful rendering (no per-channel cherry-picking)."""
    g, r, i, z = img4
    R, G, B = z, r, g
    rgb = np.stack([R, G, B], axis=-1).astype(np.float64)
    rgb = np.clip(rgb, 0, None)
    I = rgb.mean(axis=-1) + 1e-9
    scale = arcsinh_scale(I, q, stretch)
    out = rgb * scale[..., None]
    hi = np.percentile(out, 99.5)
    out = np.clip(out / (hi + 1e-9), 0, 1)
    return out


def arcsinh_scale(I, q, stretch):
    ref = np.percentile(I, 99.0) + 1e-9
    return np.arcsinh(q * I / ref) / (q * stretch + 1e-9)


def predict_heldout(X, y, alphas=None):
    """Refit the probe (zscore -> RidgeCV -> 80/20 holdout) to recover predicted-vs-true.
    Matches code/analysis/probes.py exactly (zscore on full, seed 0)."""
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import train_test_split
    if alphas is None:
        alphas = np.logspace(-2, 4, 7)
    Z = ((X - X.mean(0)) / (X.std(0) + 1e-8)).astype(np.float32)
    ok = np.isfinite(y)
    Xtr, Xte, ytr, yte = train_test_split(Z[ok], y[ok], test_size=0.2, random_state=0)
    m = RidgeCV(alphas=alphas).fit(Xtr, ytr)
    p = m.predict(Xte)
    r2 = 1 - ((yte - p) ** 2).sum() / ((yte - yte.mean()) ** 2).sum()
    return yte, p, float(r2), float(m.alpha_)


def save(fig, name):
    path = os.path.join(FIG, name)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    sz = os.path.getsize(path)
    print(f"saved {name}  ({sz/1e3:.0f} kB)")
    return path
