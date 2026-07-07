"""
Track D: do the SAE dictionaries contain irreducibly multi-dimensional features
that map to astrophysical concepts? (Engels arXiv:2405.14860, method re-read and
implemented per the paper's Appendix B.2 / Alg. 1 / Appendix F.)

Pipeline (per embedding; E_img primary -- the loop-bearing, leakage-free space):
 1. TopK activations for the retrained Phase 1-protocol SAE (results/trackD_sae/).
 2. Cluster decoder columns: the paper's Mistral graph method (directed kNN k=2 by
    cosine sim -> undirected -> prune < tau -> connected components), tau sweep
    {0.4, 0.5, 0.6}, primary tau=0.5.
 3. Per cluster (>=2 members, >=500 active points): cluster-restricted
    reconstruction r = z[:,R] @ W_dec[:,R]^T on points where the cluster fires;
    PCA; 2D projections (PC pairs (0,1),(1,2),(2,3)).
 4. TestIrreducible per Definition 3, paper recipes:
    - Separability S = min over 1000 rotation angles of binned mutual information
      (normalize: mean-subtract, RMS-norm x sqrt2, clip 6x6, 40x40 bins).
    - eps-mixture M (eps=0.1) = max over bands (v,c) of the active fraction within
      |v.f + c| < eps*RMS(v.f+c). DEVIATION (documented): the paper maximizes by
      sigmoid-softened GD (10k steps); we use a dense deterministic global grid
      (360 angles x 61 quantile-anchored offsets + local refine) -- equally valid
      for 2D and reproducible. Paper calibration: circle M~0.18, mixture M~0.64.
    Rank clusters by (1-M) x S (the paper's metric), best PC pair.
 5. Concept matching (Matt's candidates): the PA loop (principal angles between
    the cluster's 2-plane and Track A's supervised loop plane + direct circular
    fit), colour-redshift tracks, bar/spiral (branch population), AGN proxy
    (WISE W1-W2, if fetched), plus Phase 1-style naming on standard labels.
 6. Instrument flagging (Matt's diagnostic): every feature rank-correlated against
    physics labels AND instrument covariates (psfsize, psfdepth, EBV,
    north/south, i-blank) with a 20x shuffle null -> physics-first vs
    instrument-first counts; clusters containing instrument-first features flagged.
 7. Causal confirmation by ablation: remove the top clusters' contribution from
    the embedding, retrain the PA probe on the ablated embedding -> does loop
    recovery collapse? Controls: 5 random same-size feature sets.
 8. Cross-seed stability of top clusters (decoder-cosine matching vs seeds 1-4).
Line systems + image-vs-spectrum dictionary comparison: DEFERRED (need spectra).

Run:  python code/analysis/trackD.py [--embedding eimg|efull]
Writes results/trackD_{tag}.json + results/trackD_{tag}_clusters.npz.
"""
import argparse
import json
import time
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

DEV = "cuda"
DATA, RES = "data", "results"
SEED = 0
TAUS = [0.4, 0.5, 0.6]
TAU = 0.5
MIN_MEMBERS, MIN_ACTIVE = 2, 500
N_ANGLES_MI, N_BINS, CLIP = 1000, 40, 3.0
EPS = 0.1
PAIRS = [(0, 1), (1, 2), (2, 3)]


def zscore(E):
    return ((E - E.mean(0)) / (E.std(0) + 1e-8)).astype(np.float32)


def topk_acts(Z, W_enc, b_enc, b_pre, k=32, batch=8192):
    """TopK activations, computed in batches on GPU, returned on CPU (n x m)."""
    n = len(Z)
    We = torch.from_numpy(W_enc).to(DEV); be = torch.from_numpy(b_enc).to(DEV)
    bp = torch.from_numpy(b_pre).to(DEV)
    out = np.zeros((n, We.shape[0]), np.float32)
    with torch.no_grad():
        for s in range(0, n, batch):
            x = torch.from_numpy(Z[s:s + batch]).to(DEV)
            zp = torch.relu((x - bp) @ We.t() + be)
            v, i = zp.topk(k, dim=1)
            z = torch.zeros_like(zp).scatter_(1, i, v)
            out[s:s + len(x)] = z.cpu().numpy()
    return out


def graph_clusters(D, tau, knn=2):
    """The paper's Mistral method: directed kNN by cosine sim -> undirected ->
    prune < tau -> connected components. D: (d x m) decoder, columns unit-norm."""
    Dt = torch.from_numpy(D).to(DEV)
    S = (Dt.t() @ Dt).cpu().numpy()                      # columns are unit norm
    np.fill_diagonal(S, -1)
    m = S.shape[0]
    nbr = np.argsort(-S, axis=1)[:, :knn]
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    rows, cols = [], []
    for i in range(m):
        for j in nbr[i]:
            if S[i, j] >= tau:
                rows.append(i); cols.append(j)
    A = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(m, m))
    A = ((A + A.T) > 0).astype(np.int8)
    ncomp, lab = connected_components(A, directed=False)
    clusters = [np.where(lab == c)[0] for c in range(ncomp)]
    return [c for c in clusters if len(c) >= MIN_MEMBERS]


def normalize_2d(F):
    """Paper B.2.1: mean-subtract, divide by RMS norm, x sqrt2."""
    F = F - F.mean(0, keepdims=True)
    rms = np.sqrt((F ** 2).sum(1).mean()) + 1e-12
    return F / rms * np.sqrt(2.0)


def separability(F, n_angles=N_ANGLES_MI, chunk=100):
    """min over rotations of binned MI (40x40, clip 6x6). F: (n,2) normalized."""
    Ft = torch.from_numpy(F.astype(np.float32)).to(DEV)
    thetas = torch.linspace(0, np.pi, n_angles + 1, device=DEV)[:-1]
    best = float("inf")
    for s in range(0, n_angles, chunk):
        th = thetas[s:s + chunk]
        R = torch.stack([torch.stack([th.cos(), -th.sin()], -1),
                         torch.stack([th.sin(), th.cos()], -1)], -2)     # (c,2,2)
        X = torch.einsum("nd,cde->cne", Ft, R)                            # (c,n,2)
        inside = (X.abs() < CLIP).all(-1)
        ix = ((X[..., 0] + CLIP) / (2 * CLIP) * N_BINS).long().clamp(0, N_BINS - 1)
        iy = ((X[..., 1] + CLIP) / (2 * CLIP) * N_BINS).long().clamp(0, N_BINS - 1)
        flat = ix * N_BINS + iy
        C = th.shape[0]
        offs = (torch.arange(C, device=DEV)[:, None] * N_BINS * N_BINS)
        h = torch.bincount((flat + offs)[inside], minlength=C * N_BINS * N_BINS)
        H = h.view(C, N_BINS, N_BINS).double()
        P = H / H.sum((1, 2), keepdim=True).clamp(min=1)
        pa, pb = P.sum(2, keepdim=True), P.sum(1, keepdim=True)
        mi = (P * (P.clamp(min=1e-12).log() - pa.clamp(min=1e-12).log()
                   - pb.clamp(min=1e-12).log())).sum((1, 2))
        best = min(best, float(mi.min()))
    return best / float(np.log(2))                       # nats -> bits (paper reports bits)


def mixture(F, eps=EPS, n_phi=360, n_c=61):
    """max over bands (v,c) of active fraction within |v.f+c| < eps*RMS(v.f+c).
    Dense global grid + one local refine (deterministic; paper used GD)."""
    Ft = torch.from_numpy(F.astype(np.float32)).to(DEV)

    def evaluate(phis, n_offsets, blk=64):               # chunk phis: torch.quantile size cap
        best_frac, best_phi = 0.0, 0.0
        for s in range(0, len(phis), blk):
            ph = phis[s:s + blk]
            v = torch.stack([ph.cos(), ph.sin()], -1)                    # (p,2)
            P = Ft @ v.t()                                                # (n,p)
            mu, var = P.mean(0), P.var(0)
            qs = torch.quantile(P, torch.linspace(0.005, 0.995, n_offsets, device=DEV), dim=0)
            for ci in range(n_offsets):
                c = -qs[ci]                                               # band centred at quantile
                t = eps * torch.sqrt(var + (mu + c) ** 2)
                frac = ((P + c).abs() < t).float().mean(0)                # (p,)
                j = int(frac.argmax())
                if float(frac[j]) > best_frac:
                    best_frac, best_phi = float(frac[j]), float(ph[j])
        return best_frac, best_phi

    phis = torch.linspace(0, np.pi, n_phi + 1, device=DEV)[:-1]
    f0, p0 = evaluate(phis, n_c)
    fine = torch.linspace(p0 - np.pi / n_phi, p0 + np.pi / n_phi, 40, device=DEV)
    f1, _ = evaluate(fine, 121)
    return max(f0, f1)


def spearman_gpu(A, y, col_chunk=512):
    """|rank correlation| of each column of A vs y (finite mask; column-chunked).
    Binary/discrete y (<=2 distinct values) uses centred y directly (rank-biserial:
    Pearson of feature ranks vs the group indicator) -- arbitrary tie-broken ranks
    on a binary label are invalid."""
    fin = np.isfinite(y)
    a, yy = A[fin], y[fin]
    if len(np.unique(yy)) <= 2:
        ry = torch.from_numpy(yy.astype(np.float32)).to(DEV)
    else:
        ry = torch.from_numpy(yy).to(DEV).argsort(0).argsort(0).float()
    ry = (ry - ry.mean()) / (ry.std() + 1e-9)
    out = np.zeros(A.shape[1], np.float32)
    for s in range(0, A.shape[1], col_chunk):
        ra = torch.from_numpy(a[:, s:s + col_chunk]).to(DEV).argsort(0).argsort(0).float()
        ra = (ra - ra.mean(0)) / (ra.std(0) + 1e-9)
        out[s:s + ra.shape[1]] = ((ra * ry[:, None]).mean(0)).abs().cpu().numpy()
    return out


def circ_fit_2d(P2, theta2):
    """closed-form linear fit (a,b)->(cos,sin), circular median error (deg, PA scale)."""
    X = np.column_stack([np.ones(len(P2)), P2])
    c, s = np.cos(theta2), np.sin(theta2)
    bc, *_ = np.linalg.lstsq(X, c, rcond=None)
    bs, *_ = np.linalg.lstsq(X, s, rcond=None)
    d = np.angle(np.exp(1j * (np.arctan2(X @ bs, X @ bc) - np.arctan2(s, c))))
    return float(np.median(np.abs(np.degrees(d))) / 2)


def principal_angles(A, B):
    """deg angles between column spaces of orthonormal A (d,ka), B (d,kb)."""
    s = np.linalg.svd(A.T @ B, compute_uv=False)
    return np.degrees(np.arccos(np.clip(s, -1, 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedding", default="eimg", choices=["eimg", "efull"])
    args = ap.parse_args()
    tag = args.embedding
    t0 = time.time()

    E = np.load(f"{DATA}/{'E_img' if tag == 'eimg' else 'E_full'}.npy")
    Z = zscore(E)
    ok = np.load(f"{DATA}/ok_index.npy")
    df = pd.read_parquet(f"{DATA}/sample.parquet").iloc[ok].reset_index(drop=True)
    df["dr8_id"] = df["dr8_id"].astype(str)
    for extra, cols in [("anchorShapes", ["paDeg", "ellip"]),
                        ("anchorCovariates", ["psfsize_r", "psfdepth_r", "psfdepth_i", "ebv", "footprint"]),
                        ("anchorWise", ["w1w2Vega"])]:
        try:
            x = pd.read_parquet(f"{DATA}/{extra}.parquet"); x["dr8_id"] = x["dr8_id"].astype(str)
            df = df.merge(x[["dr8_id"] + cols].drop_duplicates("dr8_id"), on="dr8_id", how="left")
        except FileNotFoundError:
            print(f"({extra} not available -- skipping its columns)", flush=True)
    c = lambda k: df[k].to_numpy(float) if k in df else np.full(len(df), np.nan)

    w = np.load(f"{RES}/trackD_sae/{tag}_s0.npz")
    acts = topk_acts(Z, w["W_enc"], w["b_enc"], w["b_pre"])
    D = w["W_dec"]                                        # (d, m), unit columns
    print(f"[{time.time()-t0:.0f}s] activations {acts.shape}", flush=True)

    out = {"embedding": tag, "tau_sweep": {}}
    for tau in TAUS:
        cl = graph_clusters(D, tau)
        out["tau_sweep"][str(tau)] = dict(n_clusters=len(cl),
                                          largest=int(max((len(x) for x in cl), default=0)))
    clusters = graph_clusters(D, TAU)
    print(f"[{time.time()-t0:.0f}s] tau={TAU}: {len(clusters)} clusters "
          f"(sizes {sorted([len(x) for x in clusters])[-8:]} ...)", flush=True)

    # labels for matching
    pa = np.radians(c("paDeg")); ellip = c("ellip")
    elong = np.isfinite(ellip) & (ellip > 0.3) & np.isfinite(pa)
    g_r = c("mag_g_desi") - c("mag_r_desi"); rz = c("mag_r_desi") - c("mag_z_desi")
    z_lab = c("redshift")
    featured = c("smooth-or-featured_featured-or-disk_fraction")
    branch = (featured > 0.5) & (c("disk-edge-on_yes_fraction") < 0.5)
    # Track A supervised loop plane (Ridge directions, orthonormalized)
    wc = Ridge(alpha=100.0).fit(Z[elong], np.cos(2 * pa[elong])).coef_
    ws = Ridge(alpha=100.0).fit(Z[elong], np.sin(2 * pa[elong])).coef_
    P_loop, _ = np.linalg.qr(np.column_stack([wc, ws]))

    results, proj_store = [], {}
    for ci, R in enumerate(clusters):
        za = acts[:, R]
        active = za.max(1) > 0
        if active.sum() < MIN_ACTIVE:
            continue
        rec = za[active] @ D[:, R].T                      # (n_act, d)
        rec = rec - rec.mean(0, keepdims=True)
        Rt = torch.from_numpy(rec).to(DEV)
        U, S_, V = torch.pca_lowrank(Rt, q=min(4, len(R)))   # rank <= |cluster|
        proj = (Rt @ V).cpu().numpy()                     # (n_act, <=4)
        npc = proj.shape[1]
        best = dict(score=-1)
        for (i, j) in PAIRS:
            if j >= npc:
                continue
            F = normalize_2d(proj[:, [i, j]])
            S_idx = separability(F)
            M_idx = mixture(F)
            score = (1 - M_idx) * S_idx
            if score > best["score"]:
                best = dict(score=score, pair=(i, j), S=S_idx, M=M_idx, F=F)
        if best["score"] < 0:
            continue
        # concept matching on this cluster
        act_idx = np.where(active)[0]
        Vnp = V.cpu().numpy()[:, list(best["pair"])]
        P_cl, _ = np.linalg.qr(Vnp)
        pa_ang = principal_angles(P_loop, P_cl)
        el_a = elong[act_idx]
        loop_err = circ_fit_2d(best["F"][el_a], 2 * pa[act_idx][el_a]) if el_a.sum() > 300 else None
        X1 = np.column_stack([np.ones(active.sum()), best["F"]])
        def r2_of(y):
            fin = np.isfinite(y[act_idx])
            if fin.sum() < 300:
                return None
            b, *_ = np.linalg.lstsq(X1[fin], y[act_idx][fin], rcond=None)
            resid = y[act_idx][fin] - X1[fin] @ b
            return float(1 - resid.var() / y[act_idx][fin].var())
        entry = dict(cluster=ci, members=len(R), n_active=int(active.sum()),
                     pair=list(best["pair"]), S=best["S"], M=best["M"], score=best["score"],
                     loop_plane_angles=[float(x) for x in pa_ang],
                     loop_fit_err_deg=loop_err,
                     r2_gr=r2_of(g_r), r2_z=r2_of(z_lab), r2_featured=r2_of(featured),
                     r2_bar=r2_of(np.where(branch, c("bar_strong_fraction"), np.nan)),
                     r2_w1w2=r2_of(c("w1w2Vega")))
        results.append(entry)
        proj_store[f"cluster{ci}"] = np.column_stack(
            [best["F"], pa[act_idx], ellip[act_idx], z_lab[act_idx], g_r[act_idx]]).astype(np.float32)
    results.sort(key=lambda e: -e["score"])
    keep = {f"cluster{e['cluster']}" for e in results[:10]}          # top-10 by score
    lr = [e for e in results if e["loop_fit_err_deg"] is not None]
    if lr:                                                           # + the best loop match
        keep.add(f"cluster{min(lr, key=lambda e: e['loop_fit_err_deg'])['cluster']}")
    proj_store = {k: v for k, v in proj_store.items() if k in keep}
    out["n_tested"] = len(results)
    out["clusters"] = results[:40]
    print(f"[{time.time()-t0:.0f}s] tested {len(results)} clusters; top-5 by (1-M)xS:", flush=True)
    for e in results[:5]:
        print(f"  #{e['cluster']} m={e['members']} act={e['n_active']} S={e['S']:.3f} M={e['M']:.3f} "
              f"score={e['score']:.3f} loopPlane={e['loop_plane_angles'][0]:.0f}deg "
              f"loopErr={e['loop_fit_err_deg']}", flush=True)

    # 6. feature-level physics-vs-instrument flagging (all m features)
    phys = {"redshift": z_lab, "g_r": g_r, "r_z": rz,
            "smooth": c("smooth-or-featured_smooth_fraction"), "featured": featured,
            "merger": c("merging_merger_fraction")}
    inst = {"psfsize_r": c("psfsize_r"), "psfdepth_r": c("psfdepth_r"), "ebv": c("ebv"),
            "north": (df["footprint"] == "north").astype(float).to_numpy() if "footprint" in df else np.full(len(df), np.nan),
            "iblank": (c("psfdepth_i") <= 0).astype(float)}
    activeF = acts.std(0) > 1e-6
    corr = {k: spearman_gpu(acts, v) for k, v in {**phys, **inst}.items()}
    # per-label-type nulls: shuffle a continuous label AND a binary label separately
    rng = np.random.default_rng(SEED)
    def null_thr(y, n_shuf=20):
        fin = np.isfinite(y); yv = y[fin]; aF = acts[fin]
        vals = [spearman_gpu(aF, yv[rng.permutation(len(yv))]) for _ in range(n_shuf)]
        return float(np.percentile(np.concatenate(vals), 95))
    thr_cont = null_thr(z_lab)
    thr_bin = null_thr(inst["north"])
    thr_of = {k: (thr_bin if len(np.unique(v[np.isfinite(v)])) <= 2 else thr_cont)
              for k, v in {**phys, **inst}.items()}
    Pmat = np.stack([corr[k] for k in phys], 0)
    Imat = np.stack([corr[k] for k in inst], 0)
    p_top, p_val = Pmat.argmax(0), Pmat.max(0)
    i_top, i_val = Imat.argmax(0), Imat.max(0)
    p_names, i_names = list(phys), list(inst)
    p_sig = np.array([p_val[f] > thr_of[p_names[p_top[f]]] for f in range(len(p_val))]) & activeF
    i_sig = np.array([i_val[f] > thr_of[i_names[i_top[f]]] for f in range(len(i_val))]) & activeF
    inst_first = i_sig & (i_val > p_val)                 # instrument beats best physics label
    phys_first = p_sig & ~inst_first
    out["instrument_flagging"] = dict(
        thr_continuous=thr_cont, thr_binary=thr_bin, n_active_features=int(activeF.sum()),
        n_physics_first=int(phys_first.sum()), n_instrument_first=int(inst_first.sum()),
        n_unaligned=int((activeF & ~p_sig & ~i_sig).sum()),
        median_best_phys_corr=float(np.median(p_val[activeF])),
        median_best_inst_corr=float(np.median(i_val[activeF])),
        instrument_breakdown={k: int((inst_first & (i_top == i_names.index(k))).sum()) for k in inst},
        physics_breakdown={k: int((phys_first & (p_top == p_names.index(k))).sum()) for k in phys},
        strong_instrument_gt_0p3={k: int((inst_first & (i_top == i_names.index(k))
                                          & (i_val > 0.3)).sum()) for k in inst})
    f_ = out["instrument_flagging"]
    print(f"[{time.time()-t0:.0f}s] features: {f_['n_active_features']} active | physics-first "
          f"{f_['n_physics_first']} {f_['physics_breakdown']} | instrument-first "
          f"{f_['n_instrument_first']} {f_['instrument_breakdown']} (strong>0.3: "
          f"{f_['strong_instrument_gt_0p3']}) | unaligned {f_['n_unaligned']} | median best-corr "
          f"phys {f_['median_best_phys_corr']:.2f} vs inst {f_['median_best_inst_corr']:.2f}", flush=True)

    # 7. ablation: does removing the best loop cluster destroy the loop?
    def pa_err_of(ZX):
        tr, te = train_test_split(np.where(elong)[0], test_size=0.2, random_state=SEED)
        mc = Ridge(alpha=100.0).fit(ZX[tr], np.cos(2 * pa[tr]))
        ms = Ridge(alpha=100.0).fit(ZX[tr], np.sin(2 * pa[tr]))
        d = np.angle(np.exp(1j * (np.arctan2(ms.predict(ZX[te]), mc.predict(ZX[te])) - 2 * pa[te])))
        return float(np.median(np.abs(np.degrees(d))) / 2)
    loop_ranked = sorted([e for e in results if e["loop_fit_err_deg"] is not None],
                         key=lambda e: e["loop_fit_err_deg"])
    out["ablation"] = {"baseline_err_deg": pa_err_of(Z)}
    if loop_ranked:
        bestloop = loop_ranked[0]
        R = clusters[bestloop["cluster"]]
        Zabl = Z - (acts[:, R] @ D[:, R].T)
        out["ablation"]["best_loop_cluster"] = bestloop["cluster"]
        out["ablation"]["cluster_size"] = len(R)
        out["ablation"]["ablated_err_deg"] = pa_err_of(Zabl.astype(np.float32))
        ctrl = []
        pool = np.where(activeF)[0]
        for s in range(5):
            Rr = rng.choice(np.setdiff1d(pool, R), len(R), replace=False)
            ctrl.append(pa_err_of((Z - acts[:, Rr] @ D[:, Rr].T).astype(np.float32)))
        out["ablation"]["random_controls_err_deg"] = ctrl
        print(f"[{time.time()-t0:.0f}s] ablation: baseline {out['ablation']['baseline_err_deg']:.1f} -> "
              f"ablate loop-cluster #{bestloop['cluster']} ({len(R)} feats): "
              f"{out['ablation']['ablated_err_deg']:.1f} deg | random controls "
              f"{[round(x,1) for x in ctrl]}", flush=True)

    # distributed-loop test: ablate the K MOST loop-correlated features (union of
    # |corr| with cos2PA and sin2PA on elongated galaxies), vs matched random sets
    c2 = spearman_gpu(acts[elong], np.cos(2 * pa[elong]))
    s2 = spearman_gpu(acts[elong], np.sin(2 * pa[elong]))
    loop_corr = np.maximum(c2, s2)
    order = np.argsort(-loop_corr)
    out["distributed_loop_ablation"] = {"baseline": out["ablation"]["baseline_err_deg"],
                                        "top_feature_loopcorrs": [float(x) for x in loop_corr[order[:10]]]}
    pool = np.where(activeF)[0]
    for K in (15, 50, 200):
        Rk = order[:K]
        errK = pa_err_of((Z - acts[:, Rk] @ D[:, Rk].T).astype(np.float32))
        Rr = rng.choice(np.setdiff1d(pool, Rk), K, replace=False)
        errR = pa_err_of((Z - acts[:, Rr] @ D[:, Rr].T).astype(np.float32))
        out["distributed_loop_ablation"][f"K{K}"] = dict(ablate_top_loopcorr=errK, random_control=errR)
        print(f"  distributed-loop: ablate top-{K} loop-corr feats -> {errK:.1f} deg "
              f"(random-{K} control {errR:.1f})", flush=True)

    # 8. cross-seed stability of the top-10 clusters
    stab = {}
    for s in range(1, 5):
        Ds = np.load(f"{RES}/trackD_sae/{tag}_s{s}.npz")["W_dec"]
        Dt0 = torch.from_numpy(D).to(DEV); Dts = torch.from_numpy(Ds).to(DEV)
        M_ = (Dt0.t() @ Dts).abs()
        for e in results[:10]:
            R = clusters[e["cluster"]]
            mx = M_[torch.from_numpy(np.asarray(R)).to(DEV)].max(1).values
            stab.setdefault(str(e["cluster"]), []).append(float((mx >= 0.6).float().mean()))
    out["stability_top10"] = {k: dict(mean_frac_matched=float(np.mean(v))) for k, v in stab.items()}

    np.savez(f"{RES}/trackD_{tag}_clusters.npz", **proj_store)
    with open(f"{RES}/trackD_{tag}.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"[{time.time()-t0:.0f}s] wrote {RES}/trackD_{tag}.json + clusters npz", flush=True)


if __name__ == "__main__":
    main()
