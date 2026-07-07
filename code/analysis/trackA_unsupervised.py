"""
Track A unsupervised method check (Matt's stated method: "SAE or PCA on the
subspace to find the ring" -- distinct from our supervised probe-direction
approach, which proves the loop is LINEARLY DECODABLE but doesn't test whether
it is discoverable WITHOUT using the true PA labels at all).

Two checks:
  1. Broader PCA scan on E_img: does ANY pair among the top-50 PCs (not just the
     top-2, which Track A already showed fails) trace the cos(2PA)/sin(2PA) ring?
  2. The existing Phase 1 SAE dictionary (trained on E_full, R=4 seed=0,
     results/sae_acts0.npy) -- do any two of its 4096 unsupervised concept
     activations jointly correlate with cos(2PA) and sin(2PA), i.e. did the SAE
     already discover pieces of this loop on its own, unprompted? (Exploratory:
     that dictionary was trained on E_full, not E_img, so this is a secondary
     check, not the primary claim.)

Run from the project root:  python code/analysis/trackA_unsupervised.py
"""
import json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from trackUtils import zscore

DATA, RES = "data", "results"
ELLIP_MIN = 0.3


def load():
    ei = np.load(f"{DATA}/E_img.npy")
    ok = np.load(f"{DATA}/ok_index.npy")
    df = pd.read_parquet(f"{DATA}/sample.parquet").iloc[ok].reset_index(drop=True)
    df["dr8_id"] = df["dr8_id"].astype(str)
    sh = pd.read_parquet(f"{DATA}/anchorShapes.parquet")
    sh["dr8_id"] = sh["dr8_id"].astype(str)
    m = df.merge(sh[["dr8_id", "paDeg", "ellip"]].drop_duplicates("dr8_id"), on="dr8_id", how="left")
    assert len(m) == len(df)
    return ei, m


def main():
    ei, m = load()
    Zi = zscore(ei)
    pa = np.radians(m["paDeg"].to_numpy(float))
    ellip = m["ellip"].to_numpy(float)
    elong = np.isfinite(ellip) & (ellip > ELLIP_MIN) & np.isfinite(pa)
    idx = np.where(elong)[0]
    c2, s2 = np.cos(2 * pa[idx]), np.sin(2 * pa[idx])

    out = {}

    # 1. broad PCA scan: best single-PC correlation with c2/s2, and best PC-PAIR
    #    "ring score" (R2 of jointly regressing c2,s2 on that 2-D PC subspace)
    NPC = 50
    pcs = PCA(n_components=NPC, random_state=0).fit_transform(Zi)[idx]
    corr_c2 = np.array([abs(np.corrcoef(pcs[:, i], c2)[0, 1]) for i in range(NPC)])
    corr_s2 = np.array([abs(np.corrcoef(pcs[:, i], s2)[0, 1]) for i in range(NPC)])
    best_single = float(max(corr_c2.max(), corr_s2.max()))
    from itertools import combinations
    from numpy.linalg import lstsq
    best_pair, best_r2 = None, -1.0
    for i, j in combinations(range(NPC), 2):
        X2 = pcs[:, [i, j]]
        Xd = np.column_stack([np.ones(len(X2)), X2])
        r2c = 1 - np.sum((c2 - Xd @ lstsq(Xd, c2, rcond=None)[0]) ** 2) / np.sum((c2 - c2.mean()) ** 2)
        r2s = 1 - np.sum((s2 - Xd @ lstsq(Xd, s2, rcond=None)[0]) ** 2) / np.sum((s2 - s2.mean()) ** 2)
        r2 = min(r2c, r2s)          # both components must be recoverable for a "ring"
        if r2 > best_r2:
            best_r2, best_pair = r2, (i, j)
    out["pca_scan"] = dict(n_pcs_scanned=NPC, best_single_pc_corr=best_single,
                           best_pc_pair=list(best_pair), best_pc_pair_ring_r2=float(best_r2))

    # 2. existing Phase-1 SAE dictionary (E_full, R=4, seed 0) -- unprompted overlap?
    try:
        acts = np.load(f"{RES}/sae_acts0.npy")          # (48398, 4096), E_full-trained
        acts_idx = acts[idx]
        align_c2 = np.array([abs(np.corrcoef(acts_idx[:, k], c2)[0, 1]) if acts_idx[:, k].std() > 0 else 0.0
                             for k in range(acts.shape[1])])
        align_s2 = np.array([abs(np.corrcoef(acts_idx[:, k], s2)[0, 1]) if acts_idx[:, k].std() > 0 else 0.0
                             for k in range(acts.shape[1])])
        out["sae_check"] = dict(n_features=int(acts.shape[1]), embedding="E_full (Phase 1 dictionary)",
                                best_single_feature_corr_c2=float(align_c2.max()),
                                best_single_feature_corr_s2=float(align_s2.max()),
                                n_features_corr_gt_0p3_either=int(((align_c2 > 0.3) | (align_s2 > 0.3)).sum()))
    except FileNotFoundError:
        out["sae_check"] = None

    with open(f"{RES}/trackA_unsupervised.json", "w") as f:
        json.dump(out, f, indent=2, default=float)

    p = out["pca_scan"]
    print(f"PCA scan (top {p['n_pcs_scanned']} PCs): best single-PC |corr| = {p['best_single_pc_corr']:.2f}; "
          f"best PC-PAIR {p['best_pc_pair']} ring-R2 (min of c2,s2) = {p['best_pc_pair_ring_r2']:.2f}")
    if out["sae_check"]:
        s = out["sae_check"]
        print(f"Phase-1 SAE dictionary ({s['embedding']}, {s['n_features']} features): "
              f"best single-feature |corr| c2={s['best_single_feature_corr_c2']:.2f} "
              f"s2={s['best_single_feature_corr_s2']:.2f}; "
              f"features with |corr|>0.3 to either = {s['n_features_corr_gt_0p3_either']}")
    print(f"wrote {RES}/trackA_unsupervised.json")


if __name__ == "__main__":
    main()
