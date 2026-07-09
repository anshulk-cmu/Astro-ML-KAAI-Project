"""
Track B: does the Hubble tuning fork live in AION's image embedding?

On the real anchor (E_img, image-only, leakage-free):
  1. Main sequence (the handle): smooth/elliptical -> featured/disk axis, recovered
     and validated against independent morphology (sersic index, smooth vote).
  2. Bar as an independent prong (the branch): angle to the main sequence, with the
     EXCESS over the label-correlation null.
  3. The fork opens: spread along the bar (and spiral) direction, ellipticals vs disks.
  4. Curved sequence: centroid path/chord, against a MATCHED straight-line null
     (real along-chord centroid spacings + real per-centroid sampling noise) so the
     ratio isn't just estimator bias; plus a noise-floor-corrected ratio.
  5. Geodesic vs Euclidean (Matt's literal ask): reusing Phase 1's diffusion-map
     coordinates (a validated geodesic-distance proxy on the full 48398 cloud), do
     geodesic-nearest neighbours track an INDEPENDENT morphology measure (sersic
     concentration) better than raw-Euclidean-nearest neighbours?
  6. Redshift-contamination check (Matt's explicit caveat: "keep separate from
     redshift"): angle between the sequence axis and the redshift axis, vs label null.
  7. Preferred-axis check: does an independently-fit sersic (concentration) direction
     coincide with the vote-based sequence direction, or diverge?
All headline numbers carry a 200x bootstrap 95% CI (trackUtils).

Run from the project root:  python code/analysis/trackB.py
Needs data/E_img.npy, data/ok_index.npy, data/sample.parquet,
results/diffcoords_E_img.npy (Phase 1 diffusion map, same row order via ok_index).
"""
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from trackUtils import zscore, probe, direction, angle, label_null_angle, angle_with_ci, SEED

DATA, RES = "data", "results"


def matched_straight_line_null(cents, bins_n, bins_var, n_seeds=20, seed=SEED):
    """Known-answer null MATCHED to the real estimation problem: a TRUE straight line
    with the real centroids' along-chord spacings, each centroid perturbed by its real
    sampling noise (within-bin variance / n_bin, per dim). The path/chord ratio a
    genuinely straight sequence would show given our centroid noise."""
    u = cents[-1] - cents[0]; u = u / np.linalg.norm(u)
    s = (cents - cents[0]) @ u
    rng = np.random.default_rng(seed)
    ratios = []
    for _ in range(n_seeds):
        eps = rng.standard_normal(cents.shape) * np.sqrt(bins_var / bins_n[:, None])
        sc = np.outer(s, u) + eps
        seg = np.linalg.norm(np.diff(sc, axis=0), axis=1).sum()
        ratios.append(seg / np.linalg.norm(sc[-1] - sc[0]))
    return float(np.mean(ratios)), float(np.std(ratios))


def main():
    ei = np.load(f"{DATA}/E_img.npy")
    ok = np.load(f"{DATA}/ok_index.npy")
    df = pd.read_parquet(f"{DATA}/sample.parquet").iloc[ok].reset_index(drop=True)
    diffc = np.load(f"{RES}/diffcoords_E_img.npy")          # same row order (ok_index)
    assert len(diffc) == len(df)
    Zi = zscore(ei)
    c = lambda k: df[k].to_numpy(float)
    smooth, featured = c("smooth-or-featured_smooth_fraction"), c("smooth-or-featured_featured-or-disk_fraction")
    spiral, bar = c("has-spiral-arms_yes_fraction"), c("bar_strong_fraction")
    edgeon, sersic, z = c("disk-edge-on_yes_fraction"), c("sersic_n"), c("redshift")
    branch = (featured > 0.5) & (edgeon < 0.5)
    out = {"n_total": int(len(df)), "n_branch_population": int(branch.sum())}

    # 1. decodability
    out["decode"] = {"featured": probe(Zi, featured), "spiral_branch": probe(Zi, spiral, branch),
                     "bar_branch": probe(Zi, bar, branch), "edgeon": probe(Zi, edgeon),
                     "sersic": probe(Zi, sersic)}
    out["n_branch_finite_bar"] = out["decode"]["bar_branch"]["n"]      # the n actually used
    out["n_branch_finite_spiral"] = out["decode"]["spiral_branch"]["n"]

    # 2+6. concept directions + angle battery (with CI), including redshift
    labels = {"seq": featured, "bar": bar, "spiral": spiral, "edgeon": edgeon,
             "sersic": sersic, "redshift": z}
    masks = {"bar": branch, "spiral": branch}   # only bar/spiral are branch-restricted;
                                                 # seq/edgeon/redshift/sersic use their own isfinite
    out["angles"] = {}
    for a, b in [("seq", "bar"), ("seq", "spiral"), ("bar", "spiral"), ("seq", "edgeon"),
                ("seq", "redshift"), ("seq", "sersic")]:
        out["angles"][f"{a}-{b}"] = angle_with_ci(Zi, labels, a, b,
                                                   mask_a=masks.get(a), mask_b=masks.get(b))
    out["angles_note"] = ("the label-correlation contrast (excess) is a DESCRIPTIVE contrast, not a "
                          "calibrated null: arccos(corr) equals the probe angle only in an idealized "
                          "whitened linear model, and the embedding covariance is anisotropic")
    rng = np.random.default_rng(SEED)
    rd = [angle(*(lambda u: (u[0] / np.linalg.norm(u[0]), u[1] / np.linalg.norm(u[1])))(
          rng.standard_normal((2, Zi.shape[1])))) for _ in range(200)]
    out["random_pair_angle"] = dict(median=float(np.median(rd)), p5=float(np.percentile(rd, 5)),
                                    p95=float(np.percentile(rd, 95)))

    v = {"seq": direction(Zi, featured), "bar": direction(Zi, bar, branch),
         "spiral": direction(Zi, spiral, branch)}

    # 3. fork opens
    sm = smooth > 0.7
    fe = (featured > 0.7) & (edgeon < 0.5)
    for nm, vec in [("bar", v["bar"]), ("spiral", v["spiral"])]:
        p = Zi @ vec
        out[f"fork_opens_{nm}"] = dict(std_elliptical=float(p[sm].std()), std_disk=float(p[fe].std()),
                                       ratio=float(p[fe].std() / (p[sm].std() + 1e-9)),
                                       n_ell=int(sm.sum()), n_disk=int(fe.sum()))

    # 3b. fan robustness (in-artifact): (a) directions refit on SHUFFLED bar votes must
    # not fan; (b) split-half -- fit on half the branch, evaluate ONLY held-out disks.
    okb = np.isfinite(bar) & branch
    idxb = np.where(okb)[0]
    rngf = np.random.default_rng(SEED)
    shuf, half = [], []
    for _ in range(5):
        yb = bar.copy()
        yb[idxb] = bar[idxb][rngf.permutation(len(idxb))]
        p = Zi @ direction(Zi, yb, okb)
        shuf.append(float(p[fe].std() / (p[sm].std() + 1e-9)))
    for _ in range(5):
        m_h = np.zeros(len(bar), bool)
        m_h[rngf.choice(idxb, len(idxb) // 2, replace=False)] = True
        p = Zi @ direction(Zi, bar, m_h)
        held = fe & ~m_h
        half.append(float(p[held].std() / (p[sm].std() + 1e-9)))
    out["fork_opens_bar"]["shuffle_null_ratios"] = shuf
    out["fork_opens_bar"]["split_half_heldout_ratios"] = half

    # 4. curved sequence + MATCHED straight-line null (real centroid spacing, real
    # per-centroid sampling noise), plus a noise-floor-corrected path length
    ps = Zi @ v["seq"]
    qs = np.quantile(ps, np.linspace(0, 1, 11))
    sel = [(ps >= qs[i]) & (ps < qs[i + 1] if i < 9 else ps <= qs[i + 1]) for i in range(10)]
    cents = np.array([Zi[s_].mean(0) for s_ in sel])
    bins_n = np.array([int(s_.sum()) for s_ in sel])
    bins_var = np.array([Zi[s_].var(0) for s_ in sel])
    seg = float(np.linalg.norm(np.diff(cents, axis=0), axis=1).sum())
    chord = float(np.linalg.norm(cents[-1] - cents[0]))
    null_mean, null_std = matched_straight_line_null(cents, bins_n, bins_var)
    noise2 = (bins_var / bins_n[:, None]).sum(1)          # E||noise||^2 per centroid
    segs = np.diff(cents, axis=0)
    seg_corr = np.sqrt(np.maximum((segs ** 2).sum(1) - noise2[:-1] - noise2[1:], 0.0))
    out["sequence_curvature"] = dict(
        path_len=seg, chord=chord, ratio=seg / chord,
        ratio_noise_corrected=float(seg_corr.sum() / chord),
        matched_straight_line_null_mean=null_mean, matched_straight_line_null_std=null_std,
        null_note="true straight line with the real along-chord centroid spacings and "
                  "real per-centroid sampling noise (within-bin var/n_bin per dim), 20 seeds",
        null_scope_note="the null spread covers centroid noise draws only (no direction-fit, "
                        "bin-choice, or galaxy-resampling uncertainty) -- do not quote the gap as "
                        "a sigma count; the noise correction also leaves the chord uncorrected")

    # 5. geodesic (diffusion-coordinate) vs Euclidean: which tracks morphological
    # similarity better? Two independent similarity targets: sersic (fine-grained
    # concentration) and a broad GZ vote-fraction vector (global morphology), since
    # diffusion coordinates emphasise large-scale/slow-mixing structure and might
    # not track a fine local property even if they track a global one.
    def geo_vs_euclid(sim_fn, query_idx, nnE, nnG, label):
        from scipy.stats import binomtest
        dE_l, dG_l, win = [], [], 0
        for i in query_idx:
            nbrE = nnE.kneighbors(Zi[i:i + 1], return_distance=False)[0, 1:]
            nbrG = nnG.kneighbors(diffc[i:i + 1], return_distance=False)[0, 1:]
            dE, nE = sim_fn(i, nbrE); dG, nG = sim_fn(i, nbrG)
            if nE < 5 or nG < 5:
                continue
            dE_l.append(dE); dG_l.append(dG); win += dG < dE
        dE_a, dG_a = np.array(dE_l), np.array(dG_l)
        return dict(label=label, n_sampled=int(len(query_idx)), n_query=int(len(dE_a)),
                    mean_dist_euclidean=float(dE_a.mean()),
                    mean_dist_geodesic=float(dG_a.mean()), frac_geodesic_better=float(win / len(dE_a)),
                    n_wins=int(win), sign_test_p=float(binomtest(int(win), len(dE_a)).pvalue),
                    paired_mean_diff=float((dG_a - dE_a).mean()))

    fin = np.isfinite(sersic)
    rng2 = np.random.default_rng(SEED)
    qsamp = rng2.choice(np.where(fin)[0], min(2000, int(fin.sum())), replace=False)
    nnE = NearestNeighbors(n_neighbors=21, algorithm="brute").fit(Zi)
    nnG = NearestNeighbors(n_neighbors=21, algorithm="brute").fit(diffc)

    def sersic_sim(i, nbr):
        nbr = nbr[np.isfinite(sersic[nbr])]
        return (float(np.mean(np.abs(sersic[nbr] - sersic[i]))) if len(nbr) else (np.nan, 0)), len(nbr)
    out["geodesic_vs_euclidean_sersic"] = geo_vs_euclid(sersic_sim, qsamp, nnE, nnG, "sersic")

    votevec = np.column_stack([smooth, featured, spiral, bar, edgeon])   # broad morphology
    votevec_fin = np.isfinite(votevec).all(1)
    qsamp2 = rng2.choice(np.where(votevec_fin)[0], min(2000, int(votevec_fin.sum())), replace=False)

    def vote_sim(i, nbr):
        nbr = nbr[votevec_fin[nbr]]
        return (float(np.mean(np.linalg.norm(votevec[nbr] - votevec[i], axis=1))) if len(nbr) else np.nan), len(nbr)
    out["geodesic_vs_euclidean_votes"] = geo_vs_euclid(vote_sim, qsamp2, nnE, nnG, "vote_vector")
    out["geodesic_note"] = ("'geodesic' here = Euclidean neighbours in the top-10 diffusion-map "
                            "coordinates -- a diffusion-distance PROXY, not shortest-path geodesic "
                            "distance; quote as 'the 10-D diffusion-coordinate proxy'")

    # 7. preferred-axis check: independent sersic direction vs the vote-based sequence
    v_sersic = direction(Zi, sersic)
    out["preferred_axis"] = dict(angle_seq_vs_sersic_direction=angle(v["seq"], v_sersic))

    # validate the sequence axis against independent morphology (Pearson: linear)
    out["seq_validation"] = dict(
        corr_featured=float(np.corrcoef(ps, featured)[0, 1]),
        corr_smooth=float(np.corrcoef(ps, smooth)[0, 1]),
        corr_sersic=float(np.corrcoef(ps[fin], sersic[fin])[0, 1]))

    # "is the ordering monotonic?" (Matt's explicit ask) -- Spearman rank correlation,
    # the direct test of monotonicity (not just linear correlation)
    out["monotonicity_spearman"] = dict(
        featured=float(spearmanr(ps, featured).correlation),
        smooth=float(spearmanr(ps, smooth).correlation),
        sersic=float(spearmanr(ps[fin], sersic[fin]).correlation))

    # "test dimensionality" (Matt's explicit ask): after removing the seq direction,
    # how many dimensions does the disk-specific (fork) structure span? PCA on the
    # residual of the disk population -- a sharp 1-2 dominant-eigenvalue structure
    # means the fork is a genuinely LOW-dimensional branch, not high-dimensional noise.
    resid = Zi[fe] - np.outer(Zi[fe] @ v["seq"], v["seq"])
    pca_res = PCA(n_components=10, random_state=SEED).fit(resid)
    evr = pca_res.explained_variance_ratio_
    out["fork_dimensionality"] = dict(
        n_disk=int(fe.sum()), explained_variance_ratio_top10=evr.tolist(),
        pc1_aligned_with_bar=float(abs(pca_res.components_[0] @ v["bar"])),
        dim_note="descriptive spectrum only: PC1+PC2 capturing ~half the residual variance does "
                 "not establish a 2-D branch (no matched-noise dimension estimate was run)")

    np.save(f"{RES}/trackB_fork.npy",
            np.column_stack([ps, Zi @ v["bar"], featured, bar, smooth, edgeon]))
    with open(f"{RES}/trackB.json", "w") as f:
        json.dump(out, f, indent=2, default=float)

    d = out["decode"]
    print(f"decode R2: featured {d['featured']['r2']:.2f} CI{d['featured']['ci']} | "
          f"bar(branch,n={out['n_branch_finite_bar']}) {d['bar_branch']['r2']:.2f} | "
          f"spiral(branch,n={out['n_branch_finite_spiral']}) {d['spiral_branch']['r2']:.2f} | "
          f"sersic {d['sersic']['r2']:.2f}")
    print(f"main-seq axis: corr featured {out['seq_validation']['corr_featured']:+.2f}, "
          f"smooth {out['seq_validation']['corr_smooth']:+.2f}, sersic {out['seq_validation']['corr_sersic']:+.2f}")
    for k, x in out["angles"].items():
        print(f"  angle {k:14s} embed {x['embed_angle']:5.1f} CI{[round(v,1) for v in x['embed_ci']]} | "
              f"label-null {x['label_null']:5.1f} | excess {x['excess']:+.1f} CI{[round(v,1) for v in x['excess_ci']]} "
              f"(n_a={x['n_a']},n_b={x['n_b']})")
    print(f"  random-pair angle ~{out['random_pair_angle']['median']:.0f} deg")
    fo = out["fork_opens_bar"]
    print(f"fork opens (bar spread): elliptical {fo['std_elliptical']:.2f} -> disk {fo['std_disk']:.2f} "
          f"(ratio {fo['ratio']:.2f})")
    sc = out["sequence_curvature"]
    print(f"sequence curvature: path/chord = {sc['ratio']:.2f} "
          f"(noise-corrected {sc['ratio_noise_corrected']:.2f}; matched straight-line null "
          f"{sc['matched_straight_line_null_mean']:.3f} +/- {sc['matched_straight_line_null_std']:.3f})")
    fo_r = out["fork_opens_bar"]
    print(f"fan robustness: shuffle-null ratios {[round(r,2) for r in fo_r['shuffle_null_ratios']]} | "
          f"split-half held-out {[round(r,2) for r in fo_r['split_half_heldout_ratios']]}")
    for key in ["geodesic_vs_euclidean_sersic", "geodesic_vs_euclidean_votes"]:
        gv = out[key]
        print(f"geodesic vs Euclidean ({gv['label']}, n={gv['n_query']}): "
              f"euclidean={gv['mean_dist_euclidean']:.3f} vs geodesic={gv['mean_dist_geodesic']:.3f} "
              f"(geodesic better in {100*gv['frac_geodesic_better']:.0f}% of queries)")
    print(f"preferred axis: angle(seq, independent-sersic-direction) = {out['preferred_axis']['angle_seq_vs_sersic_direction']:.1f} deg")
    mo = out["monotonicity_spearman"]
    print(f"monotonicity (Spearman rank corr of seq axis): featured {mo['featured']:+.2f}, "
          f"smooth {mo['smooth']:+.2f}, sersic {mo['sersic']:+.2f}")
    fd = out["fork_dimensionality"]
    print(f"fork dimensionality (disk-residual PCA, n={fd['n_disk']}): top-5 explained-var ratio = "
          f"{[round(v,3) for v in fd['explained_variance_ratio_top10'][:5]]}, "
          f"|cos(PC1, bar-direction)| = {fd['pc1_aligned_with_bar']:.2f}")
    print(f"wrote {RES}/trackB.json + {RES}/trackB_fork.npy")


if __name__ == "__main__":
    main()
